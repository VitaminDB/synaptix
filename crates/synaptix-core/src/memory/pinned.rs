//! Pinned host memory.
//!
//! Под фичей `cuda` — реальная page-locked память через `cuMemHostAlloc(PORTABLE)` (доступна
//! из всех CUDA-контекстов; нужна для async HtoD/DtoH copy, не уезжает в swap, ×2 быстрее
//! pageable host memory). Без фичи — fallback на обычный `alloc_zeroed` (Vec-like) с
//! 64-byte alignment.

pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
    /// `true` если ptr получен из `cuMemHostAlloc` (Drop через `cuMemFreeHost`).
    /// `false` если из `std::alloc::alloc_zeroed` (Drop через `std::alloc::dealloc`).
    is_cuda_pinned: bool,
}

unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// Аллокация. Под `cuda` фичей пытается page-locked; если CUDA недоступна на runtime
    /// (нет драйвера / нет устройства / cuInit упал) — молча падает обратно на
    /// `alloc_zeroed`. Это безопасно: API ниже всё равно отдаёт обычный slice u8.
    pub fn new(len: usize) -> Self {
        let allocated = len.max(1);
        {
            if let Ok(ptr) = try_cuda_alloc(allocated) {
                // cuMemHostAlloc не гарантирует zero-init, обнулим явно — это
                // совместимо с прошлым контрактом `alloc_zeroed`.
                unsafe { std::ptr::write_bytes(ptr, 0, allocated) };
                return Self { ptr, len, is_cuda_pinned: true };
            }
        }
        let layout = std::alloc::Layout::from_size_align(allocated, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        Self { ptr, len, is_cuda_pinned: false }
    }

    /// Как [`PinnedBuf::new`], но БЕЗ обнуления (для буферов, заполняемых сразу
    /// целиком — pinned-кэш offload-весов: обнулять десятки GB накладно и незачем).
    pub fn new_uninit(len: usize) -> Self {
        let allocated = len.max(1);
        {
            if let Ok(ptr) = try_cuda_alloc(allocated) {
                return Self { ptr, len, is_cuda_pinned: true };
            }
        }
        let layout = std::alloc::Layout::from_size_align(allocated, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        Self { ptr, len, is_cuda_pinned: false }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// True если буфер физически page-locked (cuMemHostAlloc). Полезно для
    /// диагностики/тестов.
    pub fn is_pinned(&self) -> bool { self.is_cuda_pinned }

    /// Raw pointer на данные. Используется при cuMemcpyHtoDAsync если нужно работать с
    /// page-locked памятью напрямую.
    pub fn as_ptr(&self) -> *const u8 { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        if self.is_cuda_pinned {
            unsafe {
                let _ = cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void);
            }
            return;
        }
        let layout = std::alloc::Layout::from_size_align(self.len.max(1), 64).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

fn try_cuda_alloc(num_bytes: usize) -> Result<*mut u8, ()> {
    // cuMemHostAlloc требует initialized CUDA context. Возвращаем Err если что-то не так
    // — тогда вызов upgradeитcя на std::alloc fallback. Никогда не паникуем.
    let ctx = crate::device::cuda::get(0).map_err(|_| ())?;
    ctx.bind_to_thread().map_err(|_| ())?;
    // flag = PORTABLE (1) — page-locked доступен из любого CUDA-контекста, что важно
    // для multi-GPU / multi-thread inference.
    let ptr = unsafe { cudarc::driver::result::malloc_host(num_bytes, 0x01) }.map_err(|_| ())?;
    Ok(ptr as *mut u8)
}
