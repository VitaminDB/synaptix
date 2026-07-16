use synaptix_core::backend::ReduceOp;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

pub fn reduce_dispatch(
    op: ReduceOp,
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
    dims: &[usize],
) -> Result<()> {
    match (op, src_lo.dtype()) {
        (ReduceOp::ArgMax, _) => reduce_argmax(src, src_lo, dst, dst_lo, dims),
        (_, DType::F32) => reduce_typed_g::<f32, F32Ops>(op, src, src_lo, dst, dst_lo, dims),
        (_, DType::F64) => reduce_typed_g::<f64, F64Ops>(op, src, src_lo, dst, dst_lo, dims),
        (_, DType::F16) => reduce_typed_g::<half::f16, F16Ops>(op, src, src_lo, dst, dst_lo, dims),
        (_, DType::BF16) => reduce_typed_g::<half::bf16, BF16Ops>(op, src, src_lo, dst, dst_lo, dims),
        _ => Err(SynaptixError::Unsupported("reduce dtype")),
    }
}

#[derive(Clone, Copy)] pub(crate) struct F32Ops;
#[derive(Clone, Copy)] pub(crate) struct F64Ops;
#[derive(Clone, Copy)] pub(crate) struct F16Ops;
#[derive(Clone, Copy)] pub(crate) struct BF16Ops;

pub(crate) trait NumericOps<T> {
    fn zero() -> T;
    fn neg_inf() -> T;
    fn add(a: T, b: T) -> T;
    fn from_f32(v: f32) -> T;
    fn to_f32(v: T) -> f32;
}

impl NumericOps<f32> for F32Ops {
    fn zero() -> f32 { 0.0 }
    fn neg_inf() -> f32 { f32::NEG_INFINITY }
    fn add(a: f32, b: f32) -> f32 { a + b }
    fn from_f32(v: f32) -> f32 { v }
    fn to_f32(v: f32) -> f32 { v }
}
impl NumericOps<f64> for F64Ops {
    fn zero() -> f64 { 0.0 }
    fn neg_inf() -> f64 { f64::NEG_INFINITY }
    fn add(a: f64, b: f64) -> f64 { a + b }
    fn from_f32(v: f32) -> f64 { v as f64 }
    fn to_f32(v: f64) -> f32 { v as f32 }
}
impl NumericOps<half::f16> for F16Ops {
    fn zero() -> half::f16 { half::f16::ZERO }
    fn neg_inf() -> half::f16 { half::f16::NEG_INFINITY }
    fn add(a: half::f16, b: half::f16) -> half::f16 { a + b }
    fn from_f32(v: f32) -> half::f16 { half::f16::from_f32(v) }
    fn to_f32(v: half::f16) -> f32 { v.to_f32() }
}
impl NumericOps<half::bf16> for BF16Ops {
    fn zero() -> half::bf16 { half::bf16::ZERO }
    fn neg_inf() -> half::bf16 { half::bf16::NEG_INFINITY }
    fn add(a: half::bf16, b: half::bf16) -> half::bf16 { a + b }
    fn from_f32(v: f32) -> half::bf16 { half::bf16::from_f32(v) }
    fn to_f32(v: half::bf16) -> f32 { v.to_f32() }
}

fn reduce_argmax(
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
    dims: &[usize],
) -> Result<()> {
    if dims.len() != 1 {
        return Err(SynaptixError::Unsupported("argmax over multiple dims"));
    }
    let dim = dims[0];
    let d_slc: &mut [u32] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    match src_lo.dtype() {
        DType::F32 => argmax_t::<f32>(src, src_lo, d_slc, dst_lo, dim),
        DType::F16 => argmax_t::<half::f16>(src, src_lo, d_slc, dst_lo, dim),
        DType::BF16 => argmax_t::<half::bf16>(src, src_lo, d_slc, dst_lo, dim),
        _ => Err(SynaptixError::Unsupported("argmax dtype")),
    }
}

fn argmax_t<T: bytemuck::Pod + Copy + PartialOrd>(
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut [u32],
    dst_lo: &Layout,
    dim: usize,
) -> Result<()> {
    if !src_lo.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let s: &[T] = bytemuck::cast_slice(src.as_bytes());
    let dims = src_lo.dims();
    let outer: usize = dims[..dim].iter().product();
    let axis = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0u32;
            let mut best = s[o * axis * inner + i];
            for a in 1..axis {
                let v = s[o * axis * inner + a * inner + i];
                if v > best {
                    best = v;
                    best_idx = a as u32;
                }
            }
            dst[dst_lo.offset() + o * inner + i] = best_idx;
        }
    }
    Ok(())
}

pub(crate) fn reduce_typed_g<T: bytemuck::Pod + Copy + PartialOrd, OPS: NumericOps<T>>(
    op: ReduceOp,
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
    dims: &[usize],
) -> Result<()> {
    if !src_lo.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let s: &[T] = bytemuck::cast_slice(src.as_bytes());
    let d: &mut [T] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let src_dims = src_lo.dims();
    let src_strides_row: Vec<usize> = {
        let mut sr = vec![1usize; src_dims.len()];
        for i in (0..src_dims.len().saturating_sub(1)).rev() {
            sr[i] = sr[i + 1] * src_dims[i + 1];
        }
        sr
    };
    let mut keep_axes: Vec<usize> = Vec::new();
    for i in 0..src_dims.len() {
        if !dims.contains(&i) { keep_axes.push(i); }
    }
    let out_numel = dst_lo.numel();
    let reduce_count: usize = dims.iter().map(|&d| src_dims[d]).product();
    let out_dims = dst_lo.dims();
    let mut keep_dims_sizes: Vec<usize> = Vec::with_capacity(keep_axes.len());
    let mut keep_dims_strides_out: Vec<usize> = Vec::with_capacity(keep_axes.len());
    {
        let mut sr_out = vec![1usize; out_dims.len()];
        for i in (0..out_dims.len().saturating_sub(1)).rev() {
            sr_out[i] = sr_out[i + 1] * out_dims[i + 1];
        }
        let out_compact = if dims.len() + keep_axes.len() == out_dims.len() {
            (0..out_dims.len()).collect::<Vec<_>>()
        } else {
            (0..keep_axes.len()).collect::<Vec<_>>()
        };
        for (k, &ka) in keep_axes.iter().enumerate() {
            keep_dims_sizes.push(src_dims[ka]);
            keep_dims_strides_out.push(sr_out[out_compact[k]]);
        }
    }
    for out_lin in 0..out_numel {
        let mut acc = match op {
            ReduceOp::Sum | ReduceOp::Mean => OPS::zero(),
            ReduceOp::Max => OPS::neg_inf(),
            ReduceOp::ArgMax => unreachable!(),
        };
        let mut keep_idx = vec![0usize; keep_axes.len()];
        {
            let mut rem = out_lin;
            for (k, &ksz) in keep_dims_sizes.iter().enumerate().rev() {
                keep_idx[k] = rem % ksz;
                rem /= ksz;
            }
        }
        let mut src_base = 0usize;
        for (k, &ka) in keep_axes.iter().enumerate() {
            src_base += keep_idx[k] * src_strides_row[ka];
        }
        let mut red_idx = vec![0usize; dims.len()];
        for _ in 0..reduce_count {
            let mut src_pos = src_base;
            for (k, &rd) in dims.iter().enumerate() {
                src_pos += red_idx[k] * src_strides_row[rd];
            }
            let v = s[src_pos];
            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => OPS::add(acc, v),
                ReduceOp::Max => if v > acc { v } else { acc },
                _ => acc,
            };
            for k in (0..dims.len()).rev() {
                red_idx[k] += 1;
                if red_idx[k] < src_dims[dims[k]] { break; }
                red_idx[k] = 0;
            }
        }
        if matches!(op, ReduceOp::Mean) {
            let inv = 1.0_f32 / (reduce_count as f32);
            acc = OPS::from_f32(OPS::to_f32(acc) * inv);
        }
        d[dst_lo.offset() + out_lin] = acc;
    }
    Ok(())
}
