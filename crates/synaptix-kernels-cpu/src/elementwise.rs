use synaptix_core::backend::{BinaryOp, UnaryOp};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

pub fn unary_dispatch(
    op: UnaryOp,
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    match src_lo.dtype() {
        DType::F32 => unary_typed::<f32, FloatOpsF32>(op, src, src_lo, dst, dst_lo),
        DType::F64 => unary_typed::<f64, FloatOpsF64>(op, src, src_lo, dst, dst_lo),
        DType::F16 => unary_typed::<half::f16, FloatOpsF16>(op, src, src_lo, dst, dst_lo),
        DType::BF16 => unary_typed::<half::bf16, FloatOpsBF16>(op, src, src_lo, dst, dst_lo),
        _ => Err(SynaptixError::Unsupported("unary on non-float dtype")),
    }
}

pub fn binary_dispatch(
    op: BinaryOp,
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    match dst_lo.dtype() {
        DType::F32 => binary_typed::<f32, FloatOpsF32>(op, a, a_lo, b, b_lo, dst, dst_lo),
        DType::F64 => binary_typed::<f64, FloatOpsF64>(op, a, a_lo, b, b_lo, dst, dst_lo),
        DType::F16 => binary_typed::<half::f16, FloatOpsF16>(op, a, a_lo, b, b_lo, dst, dst_lo),
        DType::BF16 => binary_typed::<half::bf16, FloatOpsBF16>(op, a, a_lo, b, b_lo, dst, dst_lo),
        _ => Err(SynaptixError::Unsupported("binary on non-float dtype")),
    }
}

pub fn cast_dispatch(
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    if src_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    let numel = src_lo.numel();
    let src_offset = src_lo.offset();
    let dst_offset = dst_lo.offset();
    macro_rules! cast {
        ($from:ty, $to:ty) => {{
            let s: &[$from] = bytemuck::cast_slice(src.as_bytes());
            let d: &mut [$to] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
            for i in 0..numel {
                d[dst_offset + i] = cast_scalar::<$from, $to>(s[src_offset + i]);
            }
            Ok(())
        }};
    }
    match (src_lo.dtype(), dst_lo.dtype()) {
        (DType::F32, DType::F16) => cast!(f32, half::f16),
        (DType::F32, DType::BF16) => cast!(f32, half::bf16),
        (DType::F16, DType::F32) => cast!(half::f16, f32),
        (DType::BF16, DType::F32) => cast!(half::bf16, f32),
        (DType::F32, DType::F64) => cast!(f32, f64),
        (DType::F64, DType::F32) => cast!(f64, f32),
        (DType::F16, DType::BF16) => cast!(half::f16, half::bf16),
        (DType::BF16, DType::F16) => cast!(half::bf16, half::f16),
        (DType::U32, DType::F32) => cast!(u32, f32),
        (DType::F32, DType::U32) => cast!(f32, u32),
        (DType::U32, DType::I64) => cast!(u32, i64),
        (DType::I64, DType::U32) => cast!(i64, u32),
        _ => Err(SynaptixError::Unsupported("cast: dtype pair")),
    }
}

trait CastScalar<T> { fn cast_from(v: T) -> Self; }

fn cast_scalar<F: Copy, T>(v: F) -> T
where
    T: CastScalar<F>,
{
    T::cast_from(v)
}

impl CastScalar<f32> for half::f16 { fn cast_from(v: f32) -> Self { half::f16::from_f32(v) } }
impl CastScalar<f32> for half::bf16 { fn cast_from(v: f32) -> Self { half::bf16::from_f32(v) } }
impl CastScalar<f32> for f64 { fn cast_from(v: f32) -> Self { v as f64 } }
impl CastScalar<f32> for u32 { fn cast_from(v: f32) -> Self { v as u32 } }
impl CastScalar<half::f16> for f32 { fn cast_from(v: half::f16) -> Self { v.to_f32() } }
impl CastScalar<half::bf16> for f32 { fn cast_from(v: half::bf16) -> Self { v.to_f32() } }
impl CastScalar<f64> for f32 { fn cast_from(v: f64) -> Self { v as f32 } }
impl CastScalar<u32> for f32 { fn cast_from(v: u32) -> Self { v as f32 } }
impl CastScalar<half::f16> for half::bf16 {
    fn cast_from(v: half::f16) -> Self { half::bf16::from_f32(v.to_f32()) }
}
impl CastScalar<half::bf16> for half::f16 {
    fn cast_from(v: half::bf16) -> Self { half::f16::from_f32(v.to_f32()) }
}
impl CastScalar<u32> for i64 { fn cast_from(v: u32) -> Self { v as i64 } }
impl CastScalar<i64> for u32 { fn cast_from(v: i64) -> Self { v as u32 } }

trait FloatOps<T> {
    fn neg(x: T) -> T;
    fn abs(x: T) -> T;
    fn sqrt(x: T) -> T;
    fn sqr(x: T) -> T;
    fn recip(x: T) -> T;
    fn exp(x: T) -> T;
    fn log(x: T) -> T;
    fn sin(x: T) -> T;
    fn cos(x: T) -> T;
    fn tanh(x: T) -> T;
    fn silu(x: T) -> T;
    fn gelu_tanh(x: T) -> T;
    fn gelu_exact(x: T) -> T;
    fn erf(x: T) -> T;
    fn sigmoid(x: T) -> T;
    fn clamp(x: T, lo: f32, hi: f32) -> T;
    fn powf(x: T, e: f32) -> T;
    fn affine(x: T, mul: f32, add: f32) -> T;
    fn relu(x: T) -> T;
    fn relu2(x: T) -> T;
    fn leaky_relu(x: T, alpha: f32) -> T;
    fn sign(x: T) -> T;
    fn step_gt_zero(x: T) -> T;
    fn round(x: T) -> T;
    fn floor(x: T) -> T;
    fn ceil(x: T) -> T;

    fn add(a: T, b: T) -> T;
    fn sub(a: T, b: T) -> T;
    fn mul(a: T, b: T) -> T;
    fn div(a: T, b: T) -> T;
    fn max(a: T, b: T) -> T;
    fn min(a: T, b: T) -> T;
}

struct FloatOpsF32;
struct FloatOpsF64;
struct FloatOpsF16;
struct FloatOpsBF16;

macro_rules! impl_ops_f32_like {
    ($s:ty, $t:ty) => {
        impl FloatOps<$t> for $s {
            fn neg(x: $t) -> $t { -x }
            fn abs(x: $t) -> $t { x.abs() }
            fn sqrt(x: $t) -> $t { x.sqrt() }
            fn sqr(x: $t) -> $t { x * x }
            fn recip(x: $t) -> $t { 1.0 / x }
            fn exp(x: $t) -> $t { x.exp() }
            fn log(x: $t) -> $t { x.ln() }
            fn sin(x: $t) -> $t { x.sin() }
            fn cos(x: $t) -> $t { x.cos() }
            fn tanh(x: $t) -> $t { x.tanh() }
            fn silu(x: $t) -> $t { x / (1.0 + (-x).exp()) }
            fn gelu_tanh(x: $t) -> $t {
                let c = (2.0 / std::f64::consts::PI).sqrt() as $t;
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            }
            fn gelu_exact(x: $t) -> $t {
                let inv_sqrt2 = (1.0 / (2.0_f64).sqrt()) as $t;
                let e = erf_f64((x * inv_sqrt2) as f64) as $t;
                0.5 * x * (1.0 + e)
            }
            fn erf(x: $t) -> $t { erf_f64(x as f64) as $t }
            fn sigmoid(x: $t) -> $t { 1.0 / (1.0 + (-x).exp()) }
            fn clamp(x: $t, lo: f32, hi: f32) -> $t {
                let lo = lo as $t; let hi = hi as $t;
                if x < lo { lo } else if x > hi { hi } else { x }
            }
            fn powf(x: $t, e: f32) -> $t { x.powf(e as $t) }
            fn affine(x: $t, mul: f32, add: f32) -> $t { x * (mul as $t) + (add as $t) }
            fn relu(x: $t) -> $t { if x > 0.0 { x } else { 0.0 } }
            fn relu2(x: $t) -> $t { if x > 0.0 { x * x } else { 0.0 } }
            fn leaky_relu(x: $t, alpha: f32) -> $t {
                if x > 0.0 { x } else { x * (alpha as $t) }
            }
            fn sign(x: $t) -> $t {
                if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
            }
            fn step_gt_zero(x: $t) -> $t { if x > 0.0 { 1.0 } else { 0.0 } }
            fn round(x: $t) -> $t { x.round() }
            fn floor(x: $t) -> $t { x.floor() }
            fn ceil(x: $t) -> $t { x.ceil() }
            fn add(a: $t, b: $t) -> $t { a + b }
            fn sub(a: $t, b: $t) -> $t { a - b }
            fn mul(a: $t, b: $t) -> $t { a * b }
            fn div(a: $t, b: $t) -> $t { a / b }
            fn max(a: $t, b: $t) -> $t { if a > b { a } else { b } }
            fn min(a: $t, b: $t) -> $t { if a < b { a } else { b } }
        }
    };
}

impl_ops_f32_like!(FloatOpsF32, f32);
impl_ops_f32_like!(FloatOpsF64, f64);

impl FloatOps<half::f16> for FloatOpsF16 {
    fn neg(x: half::f16) -> half::f16 { -x }
    fn abs(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().abs()) }
    fn sqrt(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().sqrt()) }
    fn sqr(x: half::f16) -> half::f16 { x * x }
    fn recip(x: half::f16) -> half::f16 { half::f16::from_f32(1.0 / x.to_f32()) }
    fn exp(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().exp()) }
    fn log(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().ln()) }
    fn sin(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().sin()) }
    fn cos(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().cos()) }
    fn tanh(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().tanh()) }
    fn silu(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(v / (1.0 + (-v).exp()))
    }
    fn gelu_tanh(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        half::f16::from_f32(0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
    }
    fn gelu_exact(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        let inv_sqrt2 = (1.0_f32 / 2.0_f32.sqrt()) as f64;
        let e = erf_f64((v as f64) * inv_sqrt2) as f32;
        half::f16::from_f32(0.5 * v * (1.0 + e))
    }
    fn erf(x: half::f16) -> half::f16 {
        half::f16::from_f32(erf_f64(x.to_f32() as f64) as f32)
    }
    fn sigmoid(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(1.0 / (1.0 + (-v).exp()))
    }
    fn clamp(x: half::f16, lo: f32, hi: f32) -> half::f16 {
        half::f16::from_f32(x.to_f32().clamp(lo, hi))
    }
    fn powf(x: half::f16, e: f32) -> half::f16 { half::f16::from_f32(x.to_f32().powf(e)) }
    fn affine(x: half::f16, mul: f32, add: f32) -> half::f16 {
        half::f16::from_f32(x.to_f32() * mul + add)
    }
    fn relu(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(if v > 0.0 { v } else { 0.0 })
    }
    fn relu2(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(if v > 0.0 { v * v } else { 0.0 })
    }
    fn leaky_relu(x: half::f16, alpha: f32) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(if v > 0.0 { v } else { v * alpha })
    }
    fn sign(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 })
    }
    fn step_gt_zero(x: half::f16) -> half::f16 {
        let v = x.to_f32();
        half::f16::from_f32(if v > 0.0 { 1.0 } else { 0.0 })
    }
    fn round(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().round()) }
    fn floor(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().floor()) }
    fn ceil(x: half::f16) -> half::f16 { half::f16::from_f32(x.to_f32().ceil()) }
    fn add(a: half::f16, b: half::f16) -> half::f16 { a + b }
    fn sub(a: half::f16, b: half::f16) -> half::f16 { a - b }
    fn mul(a: half::f16, b: half::f16) -> half::f16 { a * b }
    fn div(a: half::f16, b: half::f16) -> half::f16 { a / b }
    fn max(a: half::f16, b: half::f16) -> half::f16 { if a > b { a } else { b } }
    fn min(a: half::f16, b: half::f16) -> half::f16 { if a < b { a } else { b } }
}

impl FloatOps<half::bf16> for FloatOpsBF16 {
    fn neg(x: half::bf16) -> half::bf16 { -x }
    fn abs(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().abs()) }
    fn sqrt(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().sqrt()) }
    fn sqr(x: half::bf16) -> half::bf16 { x * x }
    fn recip(x: half::bf16) -> half::bf16 { half::bf16::from_f32(1.0 / x.to_f32()) }
    fn exp(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().exp()) }
    fn log(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().ln()) }
    fn sin(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().sin()) }
    fn cos(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().cos()) }
    fn tanh(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().tanh()) }
    fn silu(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(v / (1.0 + (-v).exp()))
    }
    fn gelu_tanh(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        half::bf16::from_f32(0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
    }
    fn gelu_exact(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        let inv_sqrt2 = (1.0_f32 / 2.0_f32.sqrt()) as f64;
        let e = erf_f64((v as f64) * inv_sqrt2) as f32;
        half::bf16::from_f32(0.5 * v * (1.0 + e))
    }
    fn erf(x: half::bf16) -> half::bf16 {
        half::bf16::from_f32(erf_f64(x.to_f32() as f64) as f32)
    }
    fn sigmoid(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(1.0 / (1.0 + (-v).exp()))
    }
    fn clamp(x: half::bf16, lo: f32, hi: f32) -> half::bf16 {
        half::bf16::from_f32(x.to_f32().clamp(lo, hi))
    }
    fn powf(x: half::bf16, e: f32) -> half::bf16 { half::bf16::from_f32(x.to_f32().powf(e)) }
    fn affine(x: half::bf16, mul: f32, add: f32) -> half::bf16 {
        half::bf16::from_f32(x.to_f32() * mul + add)
    }
    fn relu(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(if v > 0.0 { v } else { 0.0 })
    }
    fn relu2(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(if v > 0.0 { v * v } else { 0.0 })
    }
    fn leaky_relu(x: half::bf16, alpha: f32) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(if v > 0.0 { v } else { v * alpha })
    }
    fn sign(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 })
    }
    fn round(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().round()) }
    fn floor(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().floor()) }
    fn ceil(x: half::bf16) -> half::bf16 { half::bf16::from_f32(x.to_f32().ceil()) }
    fn step_gt_zero(x: half::bf16) -> half::bf16 {
        let v = x.to_f32();
        half::bf16::from_f32(if v > 0.0 { 1.0 } else { 0.0 })
    }
    fn add(a: half::bf16, b: half::bf16) -> half::bf16 { a + b }
    fn sub(a: half::bf16, b: half::bf16) -> half::bf16 { a - b }
    fn mul(a: half::bf16, b: half::bf16) -> half::bf16 { a * b }
    fn div(a: half::bf16, b: half::bf16) -> half::bf16 { a / b }
    fn max(a: half::bf16, b: half::bf16) -> half::bf16 { if a > b { a } else { b } }
    fn min(a: half::bf16, b: half::bf16) -> half::bf16 { if a < b { a } else { b } }
}

fn unary_typed<T: bytemuck::Pod + Copy, OPS: FloatOps<T>>(
    op: UnaryOp,
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    if src_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    let s: &[T] = bytemuck::cast_slice(src.as_bytes());
    let d: &mut [T] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let strides = src_lo.strides().as_slice().to_vec();
    let dims = src_lo.dims().to_vec();
    let s_off = src_lo.offset();
    let d_off = dst_lo.offset();
    iter_indices(&dims, |idx, lin_dst| {
        let mut s_idx = s_off as isize;
        for k in 0..idx.len() {
            s_idx += idx[k] as isize * strides[k];
        }
        d[d_off + lin_dst] = apply_unary::<T, OPS>(op, s[s_idx as usize]);
    });
    Ok(())
}

fn binary_typed<T: bytemuck::Pod + Copy, OPS: FloatOps<T>>(
    op: BinaryOp,
    a: &CpuBuf,
    a_lo: &Layout,
    b: &CpuBuf,
    b_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    let dims = dst_lo.dims().to_vec();
    let a_slc: &[T] = bytemuck::cast_slice(a.as_bytes());
    let b_slc: &[T] = bytemuck::cast_slice(b.as_bytes());
    let d_slc: &mut [T] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let a_strides = a_lo.strides().as_slice().to_vec();
    let b_strides = b_lo.strides().as_slice().to_vec();
    let a_off = a_lo.offset();
    let b_off = b_lo.offset();
    let d_off = dst_lo.offset();
    iter_indices(&dims, |idx, lin_dst| {
        let mut ai = a_off as isize;
        let mut bi = b_off as isize;
        for k in 0..idx.len() {
            ai += idx[k] as isize * a_strides[k];
            bi += idx[k] as isize * b_strides[k];
        }
        d_slc[d_off + lin_dst] = apply_binary::<T, OPS>(op, a_slc[ai as usize], b_slc[bi as usize]);
    });
    Ok(())
}

fn iter_indices(dims: &[usize], mut f: impl FnMut(&[usize], usize)) {
    let rank = dims.len();
    if rank == 0 {
        f(&[], 0);
        return;
    }
    let numel: usize = dims.iter().product();
    let mut idx = vec![0usize; rank];
    for lin in 0..numel {
        f(&idx, lin);
        for k in (0..rank).rev() {
            idx[k] += 1;
            if idx[k] < dims[k] { break; }
            idx[k] = 0;
        }
    }
}

fn apply_unary<T: Copy, OPS: FloatOps<T>>(op: UnaryOp, x: T) -> T {
    match op {
        UnaryOp::Neg => OPS::neg(x),
        UnaryOp::Abs => OPS::abs(x),
        UnaryOp::Sqrt => OPS::sqrt(x),
        UnaryOp::Sqr => OPS::sqr(x),
        UnaryOp::Recip => OPS::recip(x),
        UnaryOp::Exp => OPS::exp(x),
        UnaryOp::Log => OPS::log(x),
        UnaryOp::Sin => OPS::sin(x),
        UnaryOp::Cos => OPS::cos(x),
        UnaryOp::Silu => OPS::silu(x),
        UnaryOp::GeluTanh => OPS::gelu_tanh(x),
        UnaryOp::GeluExact => OPS::gelu_exact(x),
        UnaryOp::Tanh => OPS::tanh(x),
        UnaryOp::Clamp(lo, hi) => OPS::clamp(x, lo, hi),
        UnaryOp::Powf(e) => OPS::powf(x, e),
        UnaryOp::Identity => x,
        UnaryOp::Affine(mul, add) => OPS::affine(x, mul, add),
        UnaryOp::Erf => OPS::erf(x),
        UnaryOp::Sigmoid => OPS::sigmoid(x),
        UnaryOp::Relu => OPS::relu(x),
        UnaryOp::Relu2 => OPS::relu2(x),
        UnaryOp::LeakyRelu(alpha) => OPS::leaky_relu(x, alpha),
        UnaryOp::Sign => OPS::sign(x),
        UnaryOp::Round => OPS::round(x),
        UnaryOp::Floor => OPS::floor(x),
        UnaryOp::Ceil => OPS::ceil(x),
        UnaryOp::StepGtZero => OPS::step_gt_zero(x),
    }
}

fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn apply_binary<T: Copy, OPS: FloatOps<T>>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => OPS::add(a, b),
        BinaryOp::Sub => OPS::sub(a, b),
        BinaryOp::Mul => OPS::mul(a, b),
        BinaryOp::Div => OPS::div(a, b),
        BinaryOp::Max => OPS::max(a, b),
        BinaryOp::Min => OPS::min(a, b),
    }
}
