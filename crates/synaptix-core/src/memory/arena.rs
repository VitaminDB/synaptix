use crate::tensor::storage::CpuBuf;

pub fn alloc_zeros_cpu(n_bytes: usize) -> CpuBuf { CpuBuf::alloc_zeros(n_bytes) }

pub fn alloc_uninit_cpu(n_bytes: usize) -> CpuBuf {
    let mut v = Vec::with_capacity(n_bytes);
    #[allow(clippy::uninit_vec)]
    unsafe {
        v.set_len(n_bytes);
    }
    CpuBuf::from_vec(v)
}
