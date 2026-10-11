use crate::{NpuError, Result};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const DMA_HEAP_IOCTL_ALLOC: libc::c_ulong = 0xC018_4800;
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200;
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_WRITE: u64 = 2;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 4;

const HEAPS: &[&str] = &["/dev/dma_heap/qcom,system", "/dev/dma_heap/system"];

#[repr(C)]
struct HeapAlloc {
    len: u64,
    fd: u32,
    fd_flags: u32,
    heap_flags: u64,
}

pub(crate) type Release = Arc<dyn Fn(u32) + Send + Sync>;

pub struct NpuBuffer {
    fd: OwnedFd,
    ptr: *mut u8,
    len: usize,
    dmabuf: bool,
    uid: u64,
    pub(crate) handle: u32,
    release: Option<Release>,
}

unsafe impl Send for NpuBuffer {}
unsafe impl Sync for NpuBuffer {}

static NEXT_UID: AtomicU64 = AtomicU64::new(1);

fn alloc_dmabuf(len: usize) -> Option<OwnedFd> {
    for h in HEAPS {
        let path = std::ffi::CString::new(*h).unwrap();
        let heap = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if heap < 0 {
            continue;
        }
        let mut a = HeapAlloc {
            len: len as u64,
            fd: 0,
            fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            heap_flags: 0,
        };
        let r = unsafe { libc::ioctl(heap, DMA_HEAP_IOCTL_ALLOC as _, &mut a) };
        unsafe { libc::close(heap) };
        if r == 0 {
            return Some(unsafe { OwnedFd::from_raw_fd(a.fd as RawFd) });
        }
    }
    None
}

fn alloc_memfd(len: usize) -> Result<OwnedFd> {
    let name = std::ffi::CString::new("synaptix-npu").unwrap();
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(fd)
}

impl NpuBuffer {
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(NpuError::Graph("буфер нулевого размера".into()));
        }
        let len = len.next_multiple_of(4096);
        let (fd, dmabuf) = match alloc_dmabuf(len) {
            Some(fd) => (fd, true),
            None => (alloc_memfd(len)?, false),
        };
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self {
            fd,
            ptr: ptr as *mut u8,
            len,
            dmabuf,
            uid: NEXT_UID.fetch_add(1, Ordering::Relaxed),
            handle: 0,
            release: None,
        })
    }

    pub(crate) fn attach(&mut self, handle: u32, release: Release) {
        self.handle = handle;
        self.release = Some(release);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_dmabuf(&self) -> bool {
        self.dmabuf
    }

    pub fn uid(&self) -> u64 {
        self.uid
    }

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn typed<T: bytemuck::Pod>(&self, offset: usize, n: usize) -> &[T] {
        assert!(offset % std::mem::align_of::<T>() == 0 && offset + n * std::mem::size_of::<T>() <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset) as *const T, n) }
    }

    pub fn typed_mut<T: bytemuck::Pod>(&mut self, offset: usize, n: usize) -> &mut [T] {
        assert!(offset % std::mem::align_of::<T>() == 0 && offset + n * std::mem::size_of::<T>() <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset) as *mut T, n) }
    }

    fn sync(&self, flags: u64) {
        if self.dmabuf {
            let f = flags;
            unsafe { libc::ioctl(self.fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC as _, &f) };
        }
    }

    pub fn cpu_begin(&self) {
        self.sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE);
    }

    pub fn cpu_end(&self) {
        self.sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE);
    }
}

impl Drop for NpuBuffer {
    fn drop(&mut self) {
        if let Some(r) = self.release.take() {
            r(self.handle);
        }
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}
