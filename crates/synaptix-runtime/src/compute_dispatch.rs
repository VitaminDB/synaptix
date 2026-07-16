use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputeDType {
    Auto = 0,
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    FP8 = 4,
    NVFP4 = 5,
}

static COMPUTE_DTYPE: AtomicU8 = AtomicU8::new(0);

pub fn compute_dtype() -> ComputeDType {
    match COMPUTE_DTYPE.load(Ordering::Relaxed) {
        1 => ComputeDType::F32,
        2 => ComputeDType::F16,
        3 => ComputeDType::BF16,
        4 => ComputeDType::FP8,
        5 => ComputeDType::NVFP4,
        _ => ComputeDType::Auto,
    }
}

pub fn set_compute_dtype(d: ComputeDType) {
    COMPUTE_DTYPE.store(d as u8, Ordering::Relaxed);
}

impl ComputeDType {
    pub fn to_synaptix_dtype(self) -> Option<synaptix_core::dtype::DType> {
        match self {
            Self::F32 => Some(synaptix_core::dtype::DType::F32),
            Self::F16 => Some(synaptix_core::dtype::DType::F16),
            Self::BF16 => Some(synaptix_core::dtype::DType::BF16),
            Self::FP8 => Some(synaptix_core::dtype::DType::MXFP8),
            Self::NVFP4 => Some(synaptix_core::dtype::DType::NVFP4),
            Self::Auto => None,
        }
    }
}
