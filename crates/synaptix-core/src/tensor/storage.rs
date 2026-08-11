use crate::device::Device;

pub struct CpuBuf {
    pub(crate) bytes: Vec<u8>,
}

impl CpuBuf {
    pub fn alloc_zeros(n_bytes: usize) -> Self {
        Self { bytes: vec![0u8; n_bytes] }
    }

    pub fn from_vec(bytes: Vec<u8>) -> Self { Self { bytes } }

    pub fn byte_len(&self) -> usize { self.bytes.len() }

    pub fn as_bytes(&self) -> &[u8] { &self.bytes }
    pub fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.bytes }
    pub fn as_ptr(&self) -> *const u8 { self.bytes.as_ptr() }
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.bytes.as_mut_ptr() }
}

impl std::fmt::Debug for CpuBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBuf")
            .field("byte_len", &self.bytes.len())
            .finish()
    }
}

pub struct CudaBuf {
    pub(crate) device: std::sync::Arc<cudarc::driver::CudaContext>,
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
    pub(crate) buf: cudarc::driver::CudaSlice<u8>,
    pub(crate) ordinal: usize,
}

impl CudaBuf {
    pub fn new(
        device: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        buf: cudarc::driver::CudaSlice<u8>,
        ordinal: usize,
    ) -> Self {
        crate::memory::cuda_pool::record_cuda_alloc(buf.len());
        Self { device, stream, buf, ordinal }
    }

    pub fn byte_len(&self) -> usize { self.buf.len() }
    pub fn device(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> { &self.device }
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> { &self.stream }
    pub fn slice(&self) -> &cudarc::driver::CudaSlice<u8> { &self.buf }
    pub fn slice_mut(&mut self) -> &mut cudarc::driver::CudaSlice<u8> { &mut self.buf }
    pub fn ordinal(&self) -> usize { self.ordinal }
}

impl Drop for CudaBuf {
    fn drop(&mut self) {
        crate::memory::cuda_pool::record_cuda_free(self.buf.len());
    }
}

impl std::fmt::Debug for CudaBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuf")
            .field("byte_len", &self.buf.len())
            .field("ordinal", &self.ordinal)
            .finish()
    }
}




#[derive(Debug)]
pub enum Storage {
    Cpu(CpuBuf),
    Cuda(CudaBuf),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Cuda(b) => Device::Cuda(b.ordinal),
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Storage::Cpu(b) => b.byte_len(),
            Storage::Cuda(b) => b.byte_len(),
        }
    }

    pub fn as_cpu(&self) -> Option<&CpuBuf> {
        match self {
            Storage::Cpu(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuBuf> {
        match self {
            Storage::Cpu(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_cuda(&self) -> Option<&CudaBuf> {
        match self {
            Storage::Cuda(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_cuda_mut(&mut self) -> Option<&mut CudaBuf> {
        match self {
            Storage::Cuda(b) => Some(b),
            _ => None,
        }
    }
}
