use std::sync::atomic::{AtomicBool, Ordering};

static NVFP4_GEMV_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_nvfp4_gemv_enabled(on: bool) {
    NVFP4_GEMV_ENABLED.store(on, Ordering::Relaxed);
}

pub fn nvfp4_gemv_enabled() -> bool {
    NVFP4_GEMV_ENABLED.load(Ordering::Relaxed)
}

static NVFP4_MMA_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_nvfp4_mma_enabled(on: bool) {
    NVFP4_MMA_ENABLED.store(on, Ordering::Relaxed);
}

pub fn nvfp4_mma_enabled() -> bool {
    NVFP4_MMA_ENABLED.load(Ordering::Relaxed)
}
