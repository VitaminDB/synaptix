pub mod avx2;
pub mod avx512;
pub mod naive;
pub mod neon;
pub mod simd_dot;

use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

pub fn matmul_dispatch(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    match a_lo.dtype() {
        DType::F32 => naive::matmul_f32(a, a_lo, b, b_lo, dst, dst_lo),
        DType::F64 => naive::matmul_f64(a, a_lo, b, b_lo, dst, dst_lo),
        DType::F16 => naive::matmul_f16(a, a_lo, b, b_lo, dst, dst_lo),
        DType::BF16 => naive::matmul_bf16(a, a_lo, b, b_lo, dst, dst_lo),
        _ => Err(SynaptixError::Unsupported("matmul dtype")),
    }
}

/// `out = x @ wᵀ` с весом `w` в натуральном `[N, K]` layout (без транспонирования).
/// F32/F16/BF16 → параллельный кэш-оптимальный путь; прочие dtype → `Unsupported`
/// (вызывающий код падает в общий `matmul(wᵀ)`).
pub fn linear_dispatch(
    x: &CpuBuf,
    x_lo: &Layout,
    w: &CpuBuf,
    w_lo: &Layout,
    out: &mut CpuBuf,
    out_lo: &Layout,
) -> Result<()> {
    match x_lo.dtype() {
        DType::F32 => naive::linear_f32(x, x_lo, w, w_lo, out, out_lo),
        DType::F16 => naive::linear_f16(x, x_lo, w, w_lo, out, out_lo),
        DType::BF16 => naive::linear_bf16(x, x_lo, w, w_lo, out, out_lo),
        _ => Err(SynaptixError::Unsupported("linear dtype")),
    }
}
