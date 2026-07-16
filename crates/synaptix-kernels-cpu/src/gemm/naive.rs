use rayon::prelude::*;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

/// Скаляр для GEMM с F32-аккумулятором (F32/F16/BF16). Порядок суммирования по K
/// фиксирован (последовательный внутри ячейки) — параллелизм только по
/// независимым выходным ячейкам, поэтому результат bit-identical однопоточному.
pub trait GemmScalar: Copy + Send + Sync + bytemuck::Pod {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    /// `Σ x[i]·w[i]` (равные длины, непрерывны) — SIMD-ускоренный (см. [`super::simd_dot`]).
    fn dot(x: &[Self], w: &[Self]) -> f32;
}

impl GemmScalar for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn dot(x: &[Self], w: &[Self]) -> f32 {
        super::simd_dot::dot_f32(x, w)
    }
}

impl GemmScalar for half::f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::f16::from_f32(v)
    }
    #[inline]
    fn dot(x: &[Self], w: &[Self]) -> f32 {
        super::simd_dot::dot_f16(x, w)
    }
}

impl GemmScalar for half::bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }
    #[inline]
    fn dot(x: &[Self], w: &[Self]) -> f32 {
        super::simd_dot::dot_bf16(x, w)
    }
}

#[inline]
fn dot_f32<T: GemmScalar>(a_s: &[T], b_s: &[T], a_row: usize, b_base: usize, j: usize, k: usize, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for kk in 0..k {
        acc += a_s[a_row + kk].to_f32() * b_s[b_base + kk * n + j].to_f32();
    }
    acc
}

/// Параллельный GEMM с F32-аккумулятором. Над выходом `[batch_d, m, n]`:
/// при достаточном числе выходных строк (`batch_d*m >= потоки`, типично prefill)
/// параллелизм по строкам (хорошая локальность B); иначе (decode M=1 → GEMV)
/// параллелизм по колонкам `n`. Оба варианта — независимые ячейки.
fn matmul_t<T: GemmScalar>(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    let (batch_a, batch_b, batch_d, m, k, n) = matmul_shapes(a_lo, b_lo, dst_lo)?;
    let a_s: &[T] = bytemuck::cast_slice(a.as_bytes());
    let b_s: &[T] = bytemuck::cast_slice(b.as_bytes());
    let d_s: &mut [T] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let a_off = a_lo.offset();
    let b_off = b_lo.offset();
    let d_off = dst_lo.offset();
    let rows = batch_d * m;
    let out = &mut d_s[d_off..d_off + rows * n];
    let nthreads = rayon::current_num_threads().max(1);

    if rows >= nthreads {
        out.par_chunks_mut(n).enumerate().for_each(|(r, orow)| {
            let batch_i = r / m;
            let i = r % m;
            let a_row = a_off + (batch_i % batch_a) * m * k + i * k;
            let b_base = b_off + (batch_i % batch_b) * k * n;
            for (j, cell) in orow.iter_mut().enumerate() {
                *cell = T::from_f32(dot_f32(a_s, b_s, a_row, b_base, j, k, n));
            }
        });
    } else {
        for r in 0..rows {
            let batch_i = r / m;
            let i = r % m;
            let a_row = a_off + (batch_i % batch_a) * m * k + i * k;
            let b_base = b_off + (batch_i % batch_b) * k * n;
            out[r * n..r * n + n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, cell)| {
                    *cell = T::from_f32(dot_f32(a_s, b_s, a_row, b_base, j, k, n));
                });
        }
    }
    Ok(())
}

pub fn matmul_f32(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    matmul_t::<f32>(a, a_lo, b, b_lo, dst, dst_lo)
}

pub fn matmul_f16(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    matmul_t::<half::f16>(a, a_lo, b, b_lo, dst, dst_lo)
}

pub fn matmul_bf16(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    matmul_t::<half::bf16>(a, a_lo, b, b_lo, dst, dst_lo)
}

/// `out = x @ wᵀ`, где `w` — `[N, K]` в натуральном Linear-layout. Считает
/// `out[m, n] = Σ_k x[m,k]·w[n,k]` напрямую, БЕЗ транспонирования `w`: строка
/// `w[n]` непрерывна в `[N,K]`, как и строка `x[m]` → внутренний цикл по K читает
/// оба операнда последовательно (кэш-оптимально, автовекторизуется), в отличие от
/// `matmul(wᵀ)` со strided-чтением колонок. Параллелизм: по строкам `m` при
/// `m >= потоки` (prefill), иначе по выходным `n` (decode M=1, GEMV).
fn linear_t<T: GemmScalar>(
    x: &CpuBuf,
    x_lo: &Layout,
    w: &CpuBuf,
    w_lo: &Layout,
    out: &mut CpuBuf,
    out_lo: &Layout,
) -> Result<()> {
    if !x_lo.is_contiguous() || !w_lo.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let w_dims = w_lo.dims();
    if w_dims.len() != 2 {
        return Err(SynaptixError::Unsupported("linear: w must be rank-2 [N, K]"));
    }
    let n = w_dims[0];
    let k = w_dims[1];
    if k == 0 {
        return Err(SynaptixError::Unsupported("linear: K == 0"));
    }
    let x_dims = x_lo.dims();
    if x_dims.is_empty() || *x_dims.last().unwrap() != k {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![n, k],
            got: x_dims.to_vec(),
        });
    }
    let m = x_lo.numel() / k;
    if out_lo.numel() != m * n {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![m, n],
            got: out_lo.dims().to_vec(),
        });
    }
    let x_s: &[T] = bytemuck::cast_slice(x.as_bytes());
    let w_s: &[T] = bytemuck::cast_slice(w.as_bytes());
    let o_s: &mut [T] = bytemuck::cast_slice_mut(out.as_bytes_mut());
    let x_off = x_lo.offset();
    let w_off = w_lo.offset();
    let o_off = out_lo.offset();
    let o = &mut o_s[o_off..o_off + m * n];
    let nthreads = rayon::current_num_threads().max(1);

    let row_dot = |xb: usize, nn: usize| -> f32 {
        let wb = w_off + nn * k;
        T::dot(&x_s[xb..xb + k], &w_s[wb..wb + k])
    };

    if m >= nthreads {
        o.par_chunks_mut(n).enumerate().for_each(|(row, orow)| {
            let xb = x_off + row * k;
            for (nn, cell) in orow.iter_mut().enumerate() {
                *cell = T::from_f32(row_dot(xb, nn));
            }
        });
    } else {
        // Мало строк (decode GEMV): параллелим выходы `n` блоками ≈ по числу потоков
        // (фикс. число задач вместо поэлементного over-split — меньше fork/join).
        let col_chunk = n.div_ceil(nthreads).max(1);
        for row in 0..m {
            let xb = x_off + row * k;
            o[row * n..row * n + n]
                .par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(ci, blk)| {
                    let n0 = ci * col_chunk;
                    for (off, cell) in blk.iter_mut().enumerate() {
                        *cell = T::from_f32(row_dot(xb, n0 + off));
                    }
                });
        }
    }
    Ok(())
}

pub fn linear_f32(
    x: &CpuBuf,
    x_lo: &Layout,
    w: &CpuBuf,
    w_lo: &Layout,
    out: &mut CpuBuf,
    out_lo: &Layout,
) -> Result<()> {
    linear_t::<f32>(x, x_lo, w, w_lo, out, out_lo)
}

pub fn linear_f16(
    x: &CpuBuf,
    x_lo: &Layout,
    w: &CpuBuf,
    w_lo: &Layout,
    out: &mut CpuBuf,
    out_lo: &Layout,
) -> Result<()> {
    linear_t::<half::f16>(x, x_lo, w, w_lo, out, out_lo)
}

pub fn linear_bf16(
    x: &CpuBuf,
    x_lo: &Layout,
    w: &CpuBuf,
    w_lo: &Layout,
    out: &mut CpuBuf,
    out_lo: &Layout,
) -> Result<()> {
    linear_t::<half::bf16>(x, x_lo, w, w_lo, out, out_lo)
}

#[inline]
fn dot_f64(a_s: &[f64], b_s: &[f64], a_row: usize, b_base: usize, j: usize, k: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for kk in 0..k {
        acc += a_s[a_row + kk] * b_s[b_base + kk * n + j];
    }
    acc
}

pub fn matmul_f64(
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    let (batch_a, batch_b, batch_d, m, k, n) = matmul_shapes(a_lo, b_lo, dst_lo)?;
    let a_s: &[f64] = bytemuck::cast_slice(a.as_bytes());
    let b_s: &[f64] = bytemuck::cast_slice(b.as_bytes());
    let d_s: &mut [f64] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let a_off = a_lo.offset();
    let b_off = b_lo.offset();
    let d_off = dst_lo.offset();
    let rows = batch_d * m;
    let out = &mut d_s[d_off..d_off + rows * n];
    let nthreads = rayon::current_num_threads().max(1);

    if rows >= nthreads {
        out.par_chunks_mut(n).enumerate().for_each(|(r, orow)| {
            let batch_i = r / m;
            let i = r % m;
            let a_row = a_off + (batch_i % batch_a) * m * k + i * k;
            let b_base = b_off + (batch_i % batch_b) * k * n;
            for (j, cell) in orow.iter_mut().enumerate() {
                *cell = dot_f64(a_s, b_s, a_row, b_base, j, k, n);
            }
        });
    } else {
        for r in 0..rows {
            let batch_i = r / m;
            let i = r % m;
            let a_row = a_off + (batch_i % batch_a) * m * k + i * k;
            let b_base = b_off + (batch_i % batch_b) * k * n;
            out[r * n..r * n + n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, cell)| {
                    *cell = dot_f64(a_s, b_s, a_row, b_base, j, k, n);
                });
        }
    }
    Ok(())
}

fn matmul_shapes(a: &Layout, b: &Layout, dst: &Layout) -> Result<(usize, usize, usize, usize, usize, usize)> {
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let a_dims = a.dims();
    let b_dims = b.dims();
    let d_dims = dst.dims();
    if a_dims.len() < 2 || b_dims.len() < 2 || d_dims.len() < 2 {
        return Err(SynaptixError::RankMismatch { expected: 2, got: 0 });
    }
    let m = a_dims[a_dims.len() - 2];
    let k = a_dims[a_dims.len() - 1];
    let kk = b_dims[b_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    if k != kk {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![kk, n],
        });
    }
    let batch_a = a.numel() / (m * k);
    let batch_b = b.numel() / (k * n);
    let batch_d = dst.numel() / (m * n);
    if batch_a == 0 || batch_b == 0 || batch_d == 0 {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![batch_d, m, n],
            got: d_dims.to_vec(),
        });
    }
    if batch_d % batch_a != 0 || batch_d % batch_b != 0 {
        return Err(SynaptixError::ShapeMismatch {
            expected: vec![batch_d, m, n],
            got: d_dims.to_vec(),
        });
    }
    Ok((batch_a, batch_b, batch_d, m, k, n))
}
