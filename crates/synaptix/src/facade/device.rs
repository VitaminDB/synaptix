//! Возможности CUDA-карт для потребителей фасада (synthos, CLI): compute
//! capability, цель NVRTC, наличие block-scale MMA — то, от чего зависит,
//! исполняется ли квант-формат нативно или обходом «деквант → плотный GEMM».
//! См. `synaptix_kernels_cuda::caps`.

use std::sync::Arc;

pub use synaptix_kernels_cuda::caps::{DeviceCaps, Feature};
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;

/// Возможности карты `ordinal` (контекст создаётся при первом обращении).
pub fn cuda_caps(ordinal: usize) -> Result<Arc<DeviceCaps>> {
    DeviceCaps::for_ordinal(ordinal)
}

/// Число CUDA-карт; 0 — драйвера нет.
pub fn cuda_device_count() -> usize {
    synaptix_kernels_cuda::caps::device_count()
}

/// Исполняется ли квант-формат весов нативно на карте `ordinal`.
pub fn quant_native(ordinal: usize, dtype: DType) -> bool {
    cuda_caps(ordinal).map(|c| c.quant_native(dtype)).unwrap_or(false)
}

/// Короткая строка о карте для логов и панели настроек; `None` — CUDA
/// недоступна.
pub fn cuda_summary(ordinal: usize) -> Option<String> {
    cuda_caps(ordinal).ok().map(|c| c.summary())
}
