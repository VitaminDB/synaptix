use synaptix_core::backend::{BinaryOp, UnaryOp};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

use crate::common::{bf16_bits_to_f32, elem_bytes, f32_to_bf16_bits, from_f32, par_chunks, to_f32};

pub(crate) const BLK: usize = 1024;
const UNARY_MIN_CHUNK: usize = 1 << 12;
const BINARY_MIN_CHUNK: usize = 1 << 12;
const CAST_MIN_CHUNK: usize = 1 << 14;

#[derive(Clone, Copy)]
pub(crate) struct SyncPtr(*mut u8);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

impl SyncPtr {
    pub(crate) fn new(p: *mut u8) -> Self {
        Self(p)
    }
    #[inline]
    pub(crate) fn get(&self) -> *mut u8 {
        self.0
    }
}

pub(crate) fn coalesce(dims: &[usize], strides: &mut [Vec<isize>]) -> Vec<usize> {
    let mut out_dims: Vec<usize> = Vec::with_capacity(dims.len());
    let mut out_st: Vec<Vec<isize>> = vec![Vec::with_capacity(dims.len()); strides.len()];
    for (k, &d) in dims.iter().enumerate() {
        if d == 1 {
            continue;
        }
        if let Some(&last) = out_dims.last() {
            let mergeable = strides
                .iter()
                .enumerate()
                .all(|(n, st)| *out_st[n].last().unwrap() == st[k] * d as isize);
            if mergeable {
                *out_dims.last_mut().unwrap() = last * d;
                for (n, st) in strides.iter().enumerate() {
                    *out_st[n].last_mut().unwrap() = st[k];
                }
                continue;
            }
        }
        out_dims.push(d);
        for (n, st) in strides.iter().enumerate() {
            out_st[n].push(st[k]);
        }
    }
    if out_dims.is_empty() {
        out_dims.push(1);
        for st in out_st.iter_mut() {
            st.push(1);
        }
    }
    for (n, st) in out_st.into_iter().enumerate() {
        strides[n] = st;
    }
    out_dims
}

pub(crate) fn walk<const N: usize>(
    dims: &[usize],
    strides: &[&[isize]; N],
    base: [isize; N],
    start: usize,
    end: usize,
    mut f: impl FnMut(&[isize; N], usize),
) {
    if start >= end {
        return;
    }
    let rank = dims.len();
    let last = dims[rank - 1];
    let mut idx = vec![0usize; rank];
    let mut rem = start;
    for k in (0..rank).rev() {
        idx[k] = rem % dims[k];
        rem /= dims[k];
    }
    let mut offs = base;
    for n in 0..N {
        for k in 0..rank {
            offs[n] += idx[k] as isize * strides[n][k];
        }
    }
    let mut pos = start;
    while pos < end {
        let seg = (last - idx[rank - 1]).min(end - pos);
        f(&offs, seg);
        pos += seg;
        idx[rank - 1] += seg;
        for n in 0..N {
            offs[n] += seg as isize * strides[n][rank - 1];
        }
        if idx[rank - 1] == last {
            idx[rank - 1] = 0;
            for n in 0..N {
                offs[n] -= last as isize * strides[n][rank - 1];
            }
            let mut k = rank - 1;
            while k > 0 {
                k -= 1;
                idx[k] += 1;
                for n in 0..N {
                    offs[n] += strides[n][k];
                }
                if idx[k] < dims[k] {
                    break;
                }
                idx[k] = 0;
                for n in 0..N {
                    offs[n] -= dims[k] as isize * strides[n][k];
                }
            }
        }
    }
}

pub(crate) fn check_bounds(lo: &Layout, esz: usize, byte_len: usize, what: &'static str) -> Result<()> {
    if lo.numel() == 0 {
        return Ok(());
    }
    let mut lo_idx = lo.offset() as isize;
    let mut hi_idx = lo.offset() as isize;
    for (&d, &s) in lo.dims().iter().zip(lo.strides().as_slice()) {
        let span = (d as isize - 1) * s;
        if span >= 0 {
            hi_idx += span;
        } else {
            lo_idx += span;
        }
    }
    if lo_idx < 0 || (hi_idx as usize + 1) * esz > byte_len {
        return Err(SynaptixError::Other(format!("{what}: layout out of buffer bounds")));
    }
    Ok(())
}

#[inline]
fn read_f32(dt: DType, bytes: &[u8], idx: usize) -> f32 {
    match dt {
        DType::F32 => f32::from_le_bytes(bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
        DType::F16 => half::f16::from_bits(u16::from_le_bytes(bytes[idx * 2..idx * 2 + 2].try_into().unwrap())).to_f32(),
        DType::BF16 => bf16_bits_to_f32(u16::from_le_bytes(bytes[idx * 2..idx * 2 + 2].try_into().unwrap())),
        DType::F64 => f64::from_le_bytes(bytes[idx * 8..idx * 8 + 8].try_into().unwrap()) as f32,
        _ => 0.0,
    }
}

#[inline]
unsafe fn write_f32(dt: DType, dp: *mut u8, idx: isize, v: f32) {
    match dt {
        DType::F32 => (dp as *mut f32).offset(idx).write_unaligned(v),
        DType::F16 => (dp as *mut u16).offset(idx).write_unaligned(half::f16::from_f32(v).to_bits()),
        DType::BF16 => (dp as *mut u16).offset(idx).write_unaligned(f32_to_bf16_bits(v)),
        _ => {}
    }
}

pub(crate) fn load_f32(dt: DType, bytes: &[u8], off: isize, st: isize, buf: &mut [f32]) {
    if st == 1 {
        to_f32(dt, bytes, off as usize, buf).expect("load_f32: dtype");
    } else if st == 0 {
        let v = read_f32(dt, bytes, off as usize);
        buf.fill(v);
    } else {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = read_f32(dt, bytes, (off + i as isize * st) as usize);
        }
    }
}

unsafe fn store_f32(dt: DType, dp: *mut u8, off: isize, st: isize, buf: &[f32]) {
    if st == 1 {
        let esz = elem_bytes(dt);
        let d = std::slice::from_raw_parts_mut(dp.offset(off * esz as isize), buf.len() * esz);
        from_f32(dt, buf, d, 0).expect("store_f32: dtype");
    } else {
        for (i, &v) in buf.iter().enumerate() {
            write_f32(dt, dp, off + i as isize * st, v);
        }
    }
}

pub fn unary_dispatch(
    op: UnaryOp,
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    if src_lo.dims() != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    let dt = src_lo.dtype();
    if dst_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, dst_lo.dtype()));
    }
    if !matches!(dt, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("unary on non-float dtype"));
    }
    let numel = src_lo.numel();
    if numel == 0 {
        return Ok(());
    }
    let esz = elem_bytes(dt);
    check_bounds(src_lo, esz, src.byte_len(), "unary src")?;
    check_bounds(dst_lo, esz, dst.byte_len(), "unary dst")?;
    let mut st = vec![src_lo.strides().as_slice().to_vec(), dst_lo.strides().as_slice().to_vec()];
    let dims = coalesce(src_lo.dims(), &mut st);
    let base = [src_lo.offset() as isize, dst_lo.offset() as isize];
    let s_st = *st[0].last().unwrap();
    let d_st = *st[1].last().unwrap();
    let sb = src.as_bytes();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    let min_chunk = if unary_cheap(op) { UNARY_MIN_CHUNK } else { UNARY_MIN_CHUNK / 4 };
    par_chunks(numel, min_chunk, |a, b| {
        let mut scratch = vec![0f32; 2 * BLK];
        let d = dp.get();
        walk(&dims, &[&st[0], &st[1]], base, a, b, |offs, len| unsafe {
            unary_segment(dt, op, sb, offs[0], s_st, d, offs[1], d_st, len, &mut scratch);
        });
    });
    Ok(())
}

fn unary_cheap(op: UnaryOp) -> bool {
    !matches!(
        op,
        UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tanh
            | UnaryOp::Silu
            | UnaryOp::GeluTanh
            | UnaryOp::GeluExact
            | UnaryOp::Erf
            | UnaryOp::Sigmoid
            | UnaryOp::Powf(_)
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn unary_segment(
    dt: DType,
    op: UnaryOp,
    sb: &[u8],
    s_off: isize,
    s_st: isize,
    dp: *mut u8,
    d_off: isize,
    d_st: isize,
    len: usize,
    scratch: &mut [f32],
) {
    match dt {
        DType::F64 => {
            let s: &[f64] = bytemuck::cast_slice(sb);
            let d = dp as *mut f64;
            for i in 0..len {
                let x = s[(s_off + i as isize * s_st) as usize];
                d.offset(d_off + i as isize * d_st).write_unaligned(unary_scalar_f64(op, x));
            }
        }
        DType::F32 if s_st == 1 && d_st == 1 => {
            let s: &[f32] = bytemuck::cast_slice(&sb[s_off as usize * 4..(s_off as usize + len) * 4]);
            let d: &mut [f32] =
                bytemuck::cast_slice_mut(std::slice::from_raw_parts_mut(dp.offset(d_off * 4), len * 4));
            unary_f32(op, false, s, d);
        }
        _ => {
            let (bin, bout) = scratch.split_at_mut(BLK);
            let half = dt != DType::F32;
            let mut i = 0;
            while i < len {
                let n = (len - i).min(BLK);
                load_f32(dt, sb, s_off + i as isize * s_st, s_st, &mut bin[..n]);
                unary_f32(op, half, &bin[..n], &mut bout[..n]);
                store_f32(dt, dp, d_off + i as isize * d_st, d_st, &bout[..n]);
                i += n;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum VUnary {
    Neg,
    Abs,
    Sqr,
    Affine(f32, f32),
    Relu,
    Clamp(f32, f32),
    Sqrt,
    Recip,
    Sign,
    StepGtZero,
    Round,
    Floor,
    Ceil,
}

fn vec_unary(op: UnaryOp) -> Option<VUnary> {
    Some(match op {
        UnaryOp::Neg => VUnary::Neg,
        UnaryOp::Abs => VUnary::Abs,
        UnaryOp::Sqr => VUnary::Sqr,
        UnaryOp::Affine(m, a) => VUnary::Affine(m, a),
        UnaryOp::Relu => VUnary::Relu,
        UnaryOp::Clamp(lo, hi) => VUnary::Clamp(lo, hi),
        UnaryOp::Sqrt => VUnary::Sqrt,
        UnaryOp::Recip => VUnary::Recip,
        UnaryOp::Sign => VUnary::Sign,
        UnaryOp::StepGtZero => VUnary::StepGtZero,
        UnaryOp::Round => VUnary::Round,
        UnaryOp::Floor => VUnary::Floor,
        UnaryOp::Ceil => VUnary::Ceil,
        _ => return None,
    })
}

pub(crate) fn unary_f32(op: UnaryOp, half_consts: bool, s: &[f32], d: &mut [f32]) {
    if matches!(op, UnaryOp::Identity) {
        d.copy_from_slice(s);
        return;
    }
    if let Some(v) = vec_unary(op) {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma {
                unsafe { x86::unary(v, s, d) };
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { arm::unary(v, s, d) };
            return;
        }
        #[allow(unreachable_code)]
        {
            scalar_unary_v(v, s, d, 0);
            return;
        }
    }
    for (o, &x) in d.iter_mut().zip(s) {
        *o = unary_scalar_f32(op, half_consts, x);
    }
}

fn scalar_unary_v(op: VUnary, s: &[f32], d: &mut [f32], from: usize) {
    for i in from..s.len() {
        d[i] = vunary_scalar(op, s[i]);
    }
}

#[inline]
fn vunary_scalar(op: VUnary, x: f32) -> f32 {
    match op {
        VUnary::Neg => -x,
        VUnary::Abs => x.abs(),
        VUnary::Sqr => x * x,
        VUnary::Affine(m, a) => x * m + a,
        VUnary::Relu => if x > 0.0 { x } else { 0.0 },
        VUnary::Clamp(lo, hi) => if x < lo { lo } else if x > hi { hi } else { x },
        VUnary::Sqrt => x.sqrt(),
        VUnary::Recip => 1.0 / x,
        VUnary::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        VUnary::StepGtZero => if x > 0.0 { 1.0 } else { 0.0 },
        VUnary::Round => x.round(),
        VUnary::Floor => x.floor(),
        VUnary::Ceil => x.ceil(),
    }
}

pub(crate) fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

#[inline]
pub(crate) fn unary_scalar_f32(op: UnaryOp, half_consts: bool, x: f32) -> f32 {
    match op {
        UnaryOp::Identity => x,
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Sqr => x * x,
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Silu => x / (1.0 + (-x).exp()),
        UnaryOp::GeluTanh => {
            let c = if half_consts {
                (2.0_f32 / std::f32::consts::PI).sqrt()
            } else {
                (2.0 / std::f64::consts::PI).sqrt() as f32
            };
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
        UnaryOp::GeluExact => {
            let e = if half_consts {
                let inv_sqrt2 = (1.0_f32 / 2.0_f32.sqrt()) as f64;
                erf_f64((x as f64) * inv_sqrt2) as f32
            } else {
                let inv_sqrt2 = (1.0 / (2.0_f64).sqrt()) as f32;
                erf_f64((x * inv_sqrt2) as f64) as f32
            };
            0.5 * x * (1.0 + e)
        }
        UnaryOp::Erf => erf_f64(x as f64) as f32,
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOp::Clamp(lo, hi) => if x < lo { lo } else if x > hi { hi } else { x },
        UnaryOp::Powf(e) => x.powf(e),
        UnaryOp::Affine(m, a) => x * m + a,
        UnaryOp::Relu => if x > 0.0 { x } else { 0.0 },
        UnaryOp::Relu2 => if x > 0.0 { x * x } else { 0.0 },
        UnaryOp::LeakyRelu(alpha) => if x > 0.0 { x } else { x * alpha },
        UnaryOp::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        UnaryOp::StepGtZero => if x > 0.0 { 1.0 } else { 0.0 },
        UnaryOp::Round => x.round(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
    }
}

#[inline]
pub(crate) fn unary_scalar_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Identity => x,
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Sqr => x * x,
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Silu => x / (1.0 + (-x).exp()),
        UnaryOp::GeluTanh => {
            let c = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
        UnaryOp::GeluExact => {
            let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
            let e = erf_f64(x * inv_sqrt2);
            0.5 * x * (1.0 + e)
        }
        UnaryOp::Erf => erf_f64(x),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOp::Clamp(lo, hi) => {
            let lo = lo as f64;
            let hi = hi as f64;
            if x < lo { lo } else if x > hi { hi } else { x }
        }
        UnaryOp::Powf(e) => x.powf(e as f64),
        UnaryOp::Affine(m, a) => x * (m as f64) + (a as f64),
        UnaryOp::Relu => if x > 0.0 { x } else { 0.0 },
        UnaryOp::Relu2 => if x > 0.0 { x * x } else { 0.0 },
        UnaryOp::LeakyRelu(alpha) => if x > 0.0 { x } else { x * (alpha as f64) },
        UnaryOp::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        UnaryOp::StepGtZero => if x > 0.0 { 1.0 } else { 0.0 },
        UnaryOp::Round => x.round(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
    }
}

#[inline]
pub(crate) fn binary_scalar_f32(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => if a > b { a } else { b },
        BinaryOp::Min => if a < b { a } else { b },
    }
}

#[inline]
fn binary_scalar_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => if a > b { a } else { b },
        BinaryOp::Min => if a < b { a } else { b },
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
    let dt = dst_lo.dtype();
    if a_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, a_lo.dtype()));
    }
    if b_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, b_lo.dtype()));
    }
    if !matches!(dt, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("binary on non-float dtype"));
    }
    let dims = dst_lo.dims();
    if a_lo.dims() != dims {
        return Err(SynaptixError::shape_mismatch(dims, a_lo.dims()));
    }
    if b_lo.dims() != dims {
        return Err(SynaptixError::shape_mismatch(dims, b_lo.dims()));
    }
    let numel = dst_lo.numel();
    if numel == 0 {
        return Ok(());
    }
    let esz = elem_bytes(dt);
    check_bounds(a_lo, esz, a.byte_len(), "binary a")?;
    check_bounds(b_lo, esz, b.byte_len(), "binary b")?;
    check_bounds(dst_lo, esz, dst.byte_len(), "binary dst")?;
    let mut st = vec![
        a_lo.strides().as_slice().to_vec(),
        b_lo.strides().as_slice().to_vec(),
        dst_lo.strides().as_slice().to_vec(),
    ];
    let cdims = coalesce(dims, &mut st);
    let base = [a_lo.offset() as isize, b_lo.offset() as isize, dst_lo.offset() as isize];
    let a_st = *st[0].last().unwrap();
    let b_st = *st[1].last().unwrap();
    let d_st = *st[2].last().unwrap();
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    par_chunks(numel, BINARY_MIN_CHUNK, |lo, hi| {
        let mut scratch = vec![0f32; 3 * BLK];
        let d = dp.get();
        walk(&cdims, &[&st[0], &st[1], &st[2]], base, lo, hi, |offs, len| unsafe {
            binary_segment(dt, op, ab, offs[0], a_st, bb, offs[1], b_st, d, offs[2], d_st, len, &mut scratch);
        });
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn binary_segment(
    dt: DType,
    op: BinaryOp,
    ab: &[u8],
    a_off: isize,
    a_st: isize,
    bb: &[u8],
    b_off: isize,
    b_st: isize,
    dp: *mut u8,
    d_off: isize,
    d_st: isize,
    len: usize,
    scratch: &mut [f32],
) {
    match dt {
        DType::F64 => {
            let a: &[f64] = bytemuck::cast_slice(ab);
            let b: &[f64] = bytemuck::cast_slice(bb);
            let d = dp as *mut f64;
            for i in 0..len {
                let x = a[(a_off + i as isize * a_st) as usize];
                let y = b[(b_off + i as isize * b_st) as usize];
                d.offset(d_off + i as isize * d_st).write_unaligned(binary_scalar_f64(op, x, y));
            }
        }
        DType::F32 if d_st == 1 && a_st == 1 && b_st == 1 => {
            let a: &[f32] = bytemuck::cast_slice(&ab[a_off as usize * 4..(a_off as usize + len) * 4]);
            let b: &[f32] = bytemuck::cast_slice(&bb[b_off as usize * 4..(b_off as usize + len) * 4]);
            let d: &mut [f32] =
                bytemuck::cast_slice_mut(std::slice::from_raw_parts_mut(dp.offset(d_off * 4), len * 4));
            binary_f32_vv(op, a, b, d);
        }
        DType::F32 if d_st == 1 && a_st == 1 && b_st == 0 => {
            let a: &[f32] = bytemuck::cast_slice(&ab[a_off as usize * 4..(a_off as usize + len) * 4]);
            let bs = read_f32(DType::F32, bb, b_off as usize);
            let d: &mut [f32] =
                bytemuck::cast_slice_mut(std::slice::from_raw_parts_mut(dp.offset(d_off * 4), len * 4));
            binary_f32_vs(op, a, bs, d);
        }
        DType::F32 if d_st == 1 && a_st == 0 && b_st == 1 => {
            let as_ = read_f32(DType::F32, ab, a_off as usize);
            let b: &[f32] = bytemuck::cast_slice(&bb[b_off as usize * 4..(b_off as usize + len) * 4]);
            let d: &mut [f32] =
                bytemuck::cast_slice_mut(std::slice::from_raw_parts_mut(dp.offset(d_off * 4), len * 4));
            binary_f32_sv(op, as_, b, d);
        }
        _ => {
            let (ba, rest) = scratch.split_at_mut(BLK);
            let (bbuf, bout) = rest.split_at_mut(BLK);
            let mut i = 0;
            while i < len {
                let n = (len - i).min(BLK);
                let ao = a_off + i as isize * a_st;
                let bo = b_off + i as isize * b_st;
                if a_st == 0 && b_st != 0 {
                    load_f32(dt, bb, bo, b_st, &mut bbuf[..n]);
                    binary_f32_sv(op, read_f32(dt, ab, ao as usize), &bbuf[..n], &mut bout[..n]);
                } else if b_st == 0 && a_st != 0 {
                    load_f32(dt, ab, ao, a_st, &mut ba[..n]);
                    binary_f32_vs(op, &ba[..n], read_f32(dt, bb, bo as usize), &mut bout[..n]);
                } else {
                    load_f32(dt, ab, ao, a_st, &mut ba[..n]);
                    load_f32(dt, bb, bo, b_st, &mut bbuf[..n]);
                    binary_f32_vv(op, &ba[..n], &bbuf[..n], &mut bout[..n]);
                }
                store_f32(dt, dp, d_off + i as isize * d_st, d_st, &bout[..n]);
                i += n;
            }
        }
    }
}

pub(crate) fn binary_f32_vv(op: BinaryOp, a: &[f32], b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::binary_vv(op, a, b, d) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::binary_vv(op, a, b, d) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..d.len() {
        d[i] = binary_scalar_f32(op, a[i], b[i]);
    }
}

pub(crate) fn binary_f32_vs(op: BinaryOp, a: &[f32], b: f32, d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::binary_vs(op, a, b, d) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::binary_vs(op, a, b, d) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..d.len() {
        d[i] = binary_scalar_f32(op, a[i], b);
    }
}

pub(crate) fn binary_f32_sv(op: BinaryOp, a: f32, b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::binary_sv(op, a, b, d) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::binary_sv(op, a, b, d) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..d.len() {
        d[i] = binary_scalar_f32(op, a, b[i]);
    }
}

fn cast_supported(s: DType, d: DType) -> bool {
    use DType::*;
    matches!(
        (s, d),
        (F32, F16)
            | (F32, BF16)
            | (F16, F32)
            | (BF16, F32)
            | (F32, F64)
            | (F64, F32)
            | (F16, BF16)
            | (BF16, F16)
            | (U32, F32)
            | (F32, U32)
            | (U32, I64)
            | (I64, U32)
            | (U8, F32)
            | (F32, U8)
            | (I32, F32)
            | (F32, I32)
            | (I64, F32)
            | (F32, I64)
            | (U32, I32)
            | (I32, U32)
            | (F16, U32)
            | (BF16, U32)
    )
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
    let sdt = src_lo.dtype();
    let ddt = dst_lo.dtype();
    if sdt == ddt {
        return crate::copy::copy_strided(src, src_lo, dst, dst_lo);
    }
    if !cast_supported(sdt, ddt) {
        return Err(SynaptixError::Unsupported("cast: dtype pair"));
    }
    let numel = src_lo.numel();
    if numel == 0 {
        return Ok(());
    }
    let se = elem_bytes(sdt);
    let de = elem_bytes(ddt);
    check_bounds(src_lo, se, src.byte_len(), "cast src")?;
    check_bounds(dst_lo, de, dst.byte_len(), "cast dst")?;
    let mut st = vec![src_lo.strides().as_slice().to_vec(), dst_lo.strides().as_slice().to_vec()];
    let dims = coalesce(src_lo.dims(), &mut st);
    let base = [src_lo.offset() as isize, dst_lo.offset() as isize];
    let s_st = *st[0].last().unwrap();
    let d_st = *st[1].last().unwrap();
    let sb = src.as_bytes();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    par_chunks(numel, CAST_MIN_CHUNK, |lo, hi| {
        let mut sbuf = vec![0u64; BLK];
        let mut dbuf = vec![0u64; BLK];
        let mut fbuf = vec![0f32; BLK];
        let d = dp.get();
        walk(&dims, &[&st[0], &st[1]], base, lo, hi, |offs, len| unsafe {
            if s_st == 1 && d_st == 1 {
                let s = &sb[offs[0] as usize * se..(offs[0] as usize + len) * se];
                let dd = std::slice::from_raw_parts_mut(d.offset(offs[1] * de as isize), len * de);
                cast_block(sdt, ddt, s, dd, &mut fbuf);
            } else {
                let sbytes: &mut [u8] = bytemuck::cast_slice_mut(&mut sbuf);
                let dbytes: &mut [u8] = bytemuck::cast_slice_mut(&mut dbuf);
                let mut i = 0;
                while i < len {
                    let n = (len - i).min(BLK);
                    for j in 0..n {
                        let si = (offs[0] + (i + j) as isize * s_st) as usize * se;
                        sbytes[j * se..j * se + se].copy_from_slice(&sb[si..si + se]);
                    }
                    cast_block(sdt, ddt, &sbytes[..n * se], &mut dbytes[..n * de], &mut fbuf);
                    for j in 0..n {
                        let di = (offs[1] + (i + j) as isize * d_st) * de as isize;
                        std::ptr::copy_nonoverlapping(dbytes.as_ptr().add(j * de), d.offset(di), de);
                    }
                    i += n;
                }
            }
        });
    });
    Ok(())
}

fn cast_block(sdt: DType, ddt: DType, s: &[u8], d: &mut [u8], fbuf: &mut [f32]) {
    use DType::*;
    macro_rules! conv {
        ($from:ty, $to:ty) => {{
            let s: &[$from] = bytemuck::cast_slice(s);
            let d: &mut [$to] = bytemuck::cast_slice_mut(d);
            for (o, &v) in d.iter_mut().zip(s) {
                *o = v as $to;
            }
        }};
    }
    match (sdt, ddt) {
        (F32, F16) => crate::common::f32_to_f16(bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d)),
        (F32, BF16) => crate::common::f32_to_bf16(bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d)),
        (F16, F32) => crate::common::f16_to_f32(bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d)),
        (BF16, F32) => crate::common::bf16_to_f32(bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d)),
        (F16, BF16) | (BF16, F16) | (F16, U32) | (BF16, U32) => {
            let n = s.len() / 2;
            let mut i = 0;
            while i < n {
                let m = (n - i).min(fbuf.len());
                to_f32(sdt, s, i, &mut fbuf[..m]).expect("cast_block: half");
                match ddt {
                    U32 => {
                        let o: &mut [u32] = bytemuck::cast_slice_mut(&mut d[i * 4..(i + m) * 4]);
                        for (o, &v) in o.iter_mut().zip(&fbuf[..m]) {
                            *o = v as u32;
                        }
                    }
                    _ => from_f32(ddt, &fbuf[..m], d, i).expect("cast_block: half"),
                }
                i += m;
            }
        }
        (F32, F64) => conv!(f32, f64),
        (F64, F32) => conv!(f64, f32),
        (U32, F32) => conv!(u32, f32),
        (F32, U32) => conv!(f32, u32),
        (U32, I64) => conv!(u32, i64),
        (I64, U32) => conv!(i64, u32),
        (U8, F32) => conv!(u8, f32),
        (F32, U8) => conv!(f32, u8),
        (I32, F32) => conv!(i32, f32),
        (F32, I32) => conv!(f32, i32),
        (I64, F32) => conv!(i64, f32),
        (F32, I64) => conv!(f32, i64),
        (U32, I32) => conv!(u32, i32),
        (I32, U32) => conv!(i32, u32),
        _ => unreachable!("cast_block: unsupported pair"),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{binary_scalar_f32, vunary_scalar, VUnary};
    use std::arch::x86_64::*;
    use synaptix_core::backend::BinaryOp;

    macro_rules! vloop {
        ($n:expr, $sp:expr, $dp:expr, $x:ident => $v:expr, $op:expr) => {{
            let mut i = 0;
            while i + 8 <= $n {
                let $x = _mm256_loadu_ps($sp.add(i));
                _mm256_storeu_ps($dp.add(i), $v);
                i += 8;
            }
            while i < $n {
                *$dp.add(i) = vunary_scalar($op, *$sp.add(i));
                i += 1;
            }
        }};
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn unary(op: VUnary, s: &[f32], d: &mut [f32]) {
        let n = s.len();
        let sp = s.as_ptr();
        let dp = d.as_mut_ptr();
        let sign = _mm256_set1_ps(-0.0);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        match op {
            VUnary::Neg => vloop!(n, sp, dp, x => _mm256_xor_ps(x, sign), op),
            VUnary::Abs => vloop!(n, sp, dp, x => _mm256_andnot_ps(sign, x), op),
            VUnary::Sqr => vloop!(n, sp, dp, x => _mm256_mul_ps(x, x), op),
            VUnary::Affine(m, a) => {
                let mv = _mm256_set1_ps(m);
                let av = _mm256_set1_ps(a);
                vloop!(n, sp, dp, x => _mm256_add_ps(_mm256_mul_ps(x, mv), av), op)
            }
            VUnary::Relu => vloop!(n, sp, dp, x => _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(x, zero), x), op),
            VUnary::Clamp(lo, hi) => {
                let lov = _mm256_set1_ps(lo);
                let hiv = _mm256_set1_ps(hi);
                vloop!(n, sp, dp, x => {
                    let r = _mm256_blendv_ps(x, hiv, _mm256_cmp_ps::<_CMP_GT_OQ>(x, hiv));
                    _mm256_blendv_ps(r, lov, _mm256_cmp_ps::<_CMP_LT_OQ>(x, lov))
                }, op)
            }
            VUnary::Sqrt => vloop!(n, sp, dp, x => _mm256_sqrt_ps(x), op),
            VUnary::Recip => vloop!(n, sp, dp, x => _mm256_div_ps(one, x), op),
            VUnary::Sign => {
                let neg_one = _mm256_set1_ps(-1.0);
                vloop!(n, sp, dp, x => _mm256_or_ps(
                    _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(x, zero), one),
                    _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(x, zero), neg_one)
                ), op)
            }
            VUnary::StepGtZero => vloop!(n, sp, dp, x => _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(x, zero), one), op),
            VUnary::Round => {
                let half = _mm256_set1_ps(0.5);
                vloop!(n, sp, dp, x => {
                    let t = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
                    let diff = _mm256_andnot_ps(sign, _mm256_sub_ps(x, t));
                    let m = _mm256_cmp_ps::<_CMP_GE_OQ>(diff, half);
                    let step = _mm256_or_ps(_mm256_and_ps(x, sign), one);
                    _mm256_blendv_ps(t, _mm256_add_ps(t, step), m)
                }, op)
            }
            VUnary::Floor => vloop!(n, sp, dp, x => _mm256_floor_ps(x), op),
            VUnary::Ceil => vloop!(n, sp, dp, x => _mm256_ceil_ps(x), op),
        }
    }

    macro_rules! bloop {
        ($n:expr, $load_a:expr, $load_b:expr, $sa:expr, $sb:expr, $dp:expr, $a:ident, $b:ident => $v:expr, $op:expr) => {{
            let mut i = 0;
            while i + 8 <= $n {
                let $a = $load_a(i);
                let $b = $load_b(i);
                _mm256_storeu_ps($dp.add(i), $v);
                i += 8;
            }
            while i < $n {
                *$dp.add(i) = binary_scalar_f32($op, $sa(i), $sb(i));
                i += 1;
            }
        }};
    }

    macro_rules! bops {
        ($op:expr, $n:expr, $la:expr, $lb:expr, $sa:expr, $sb:expr, $dp:expr) => {
            match $op {
                BinaryOp::Add => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_add_ps(a, b), $op),
                BinaryOp::Sub => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_sub_ps(a, b), $op),
                BinaryOp::Mul => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_mul_ps(a, b), $op),
                BinaryOp::Div => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_div_ps(a, b), $op),
                BinaryOp::Max => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_max_ps(a, b), $op),
                BinaryOp::Min => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => _mm256_min_ps(a, b), $op),
            }
        };
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn binary_vv(op: BinaryOp, a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp = d.as_mut_ptr();
        bops!(op, n, |i| _mm256_loadu_ps(ap.add(i)), |i| _mm256_loadu_ps(bp.add(i)), |i| *ap.add(i), |i| *bp.add(i), dp)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn binary_vs(op: BinaryOp, a: &[f32], b: f32, d: &mut [f32]) {
        let n = d.len();
        let ap = a.as_ptr();
        let dp = d.as_mut_ptr();
        let bv = _mm256_set1_ps(b);
        bops!(op, n, |i| _mm256_loadu_ps(ap.add(i)), |_| bv, |i| *ap.add(i), |_| b, dp)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn binary_sv(op: BinaryOp, a: f32, b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let bp = b.as_ptr();
        let dp = d.as_mut_ptr();
        let av = _mm256_set1_ps(a);
        bops!(op, n, |_| av, |i| _mm256_loadu_ps(bp.add(i)), |_| a, |i| *bp.add(i), dp)
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{binary_scalar_f32, vunary_scalar, VUnary};
    use std::arch::aarch64::*;
    use synaptix_core::backend::BinaryOp;

    macro_rules! vloop {
        ($n:expr, $sp:expr, $dp:expr, $x:ident => $v:expr, $op:expr) => {{
            let mut i = 0;
            while i + 4 <= $n {
                let $x = vld1q_f32($sp.add(i));
                vst1q_f32($dp.add(i), $v);
                i += 4;
            }
            while i < $n {
                *$dp.add(i) = vunary_scalar($op, *$sp.add(i));
                i += 1;
            }
        }};
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn unary(op: VUnary, s: &[f32], d: &mut [f32]) {
        let n = s.len();
        let sp = s.as_ptr();
        let dp = d.as_mut_ptr();
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        match op {
            VUnary::Neg => vloop!(n, sp, dp, x => vnegq_f32(x), op),
            VUnary::Abs => vloop!(n, sp, dp, x => vabsq_f32(x), op),
            VUnary::Sqr => vloop!(n, sp, dp, x => vmulq_f32(x, x), op),
            VUnary::Affine(m, a) => {
                let mv = vdupq_n_f32(m);
                let av = vdupq_n_f32(a);
                vloop!(n, sp, dp, x => vaddq_f32(vmulq_f32(x, mv), av), op)
            }
            VUnary::Relu => vloop!(n, sp, dp, x => vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(x, zero), vreinterpretq_u32_f32(x))), op),
            VUnary::Clamp(lo, hi) => {
                let lov = vdupq_n_f32(lo);
                let hiv = vdupq_n_f32(hi);
                vloop!(n, sp, dp, x => {
                    let r = vbslq_f32(vcgtq_f32(x, hiv), hiv, x);
                    vbslq_f32(vcltq_f32(x, lov), lov, r)
                }, op)
            }
            VUnary::Sqrt => vloop!(n, sp, dp, x => vsqrtq_f32(x), op),
            VUnary::Recip => vloop!(n, sp, dp, x => vdivq_f32(one, x), op),
            VUnary::Sign => {
                let neg_one = vdupq_n_f32(-1.0);
                vloop!(n, sp, dp, x => vreinterpretq_f32_u32(vorrq_u32(
                    vandq_u32(vcgtq_f32(x, zero), vreinterpretq_u32_f32(one)),
                    vandq_u32(vcltq_f32(x, zero), vreinterpretq_u32_f32(neg_one))
                )), op)
            }
            VUnary::StepGtZero => vloop!(n, sp, dp, x => vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(x, zero), vreinterpretq_u32_f32(one))), op),
            VUnary::Round => vloop!(n, sp, dp, x => vrndaq_f32(x), op),
            VUnary::Floor => vloop!(n, sp, dp, x => vrndmq_f32(x), op),
            VUnary::Ceil => vloop!(n, sp, dp, x => vrndpq_f32(x), op),
        }
    }

    macro_rules! bloop {
        ($n:expr, $load_a:expr, $load_b:expr, $sa:expr, $sb:expr, $dp:expr, $a:ident, $b:ident => $v:expr, $op:expr) => {{
            let mut i = 0;
            while i + 4 <= $n {
                let $a = $load_a(i);
                let $b = $load_b(i);
                vst1q_f32($dp.add(i), $v);
                i += 4;
            }
            while i < $n {
                *$dp.add(i) = binary_scalar_f32($op, $sa(i), $sb(i));
                i += 1;
            }
        }};
    }

    macro_rules! bops {
        ($op:expr, $n:expr, $la:expr, $lb:expr, $sa:expr, $sb:expr, $dp:expr) => {
            match $op {
                BinaryOp::Add => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vaddq_f32(a, b), $op),
                BinaryOp::Sub => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vsubq_f32(a, b), $op),
                BinaryOp::Mul => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vmulq_f32(a, b), $op),
                BinaryOp::Div => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vdivq_f32(a, b), $op),
                BinaryOp::Max => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vbslq_f32(vcgtq_f32(a, b), a, b), $op),
                BinaryOp::Min => bloop!($n, $la, $lb, $sa, $sb, $dp, a, b => vbslq_f32(vcltq_f32(a, b), a, b), $op),
            }
        };
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn binary_vv(op: BinaryOp, a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp = d.as_mut_ptr();
        bops!(op, n, |i| vld1q_f32(ap.add(i)), |i| vld1q_f32(bp.add(i)), |i| *ap.add(i), |i| *bp.add(i), dp)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn binary_vs(op: BinaryOp, a: &[f32], b: f32, d: &mut [f32]) {
        let n = d.len();
        let ap = a.as_ptr();
        let dp = d.as_mut_ptr();
        let bv = vdupq_n_f32(b);
        bops!(op, n, |i| vld1q_f32(ap.add(i)), |_| bv, |i| *ap.add(i), |_| b, dp)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn binary_sv(op: BinaryOp, a: f32, b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let bp = b.as_ptr();
        let dp = d.as_mut_ptr();
        let av = vdupq_n_f32(a);
        bops!(op, n, |_| av, |i| vld1q_f32(bp.add(i)), |_| a, |i| *bp.add(i), dp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::tensor::shape::Shape;

    fn fill_src(dt: DType, n: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let u = (state as f32) / (u32::MAX as f32);
            u * 12.0 - 6.0
        };
        let mut vals: Vec<f32> = (0..n).map(|_| next()).collect();
        let specials = [0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 1e-3, f32::INFINITY, f32::NEG_INFINITY, 65504.0];
        for (i, s) in specials.iter().enumerate() {
            if i * 97 + 3 < n {
                vals[i * 97 + 3] = *s;
            }
        }
        match dt {
            DType::F32 => bytemuck::cast_slice(&vals).to_vec(),
            DType::F64 => bytemuck::cast_slice(&vals.iter().map(|&v| v as f64).collect::<Vec<_>>()).to_vec(),
            DType::F16 => bytemuck::cast_slice(&vals.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>()).to_vec(),
            DType::BF16 => bytemuck::cast_slice(&vals.iter().map(|&v| half::bf16::from_f32(v)).collect::<Vec<_>>()).to_vec(),
            _ => unreachable!(),
        }
    }

    fn half_unary_ref(op: UnaryOp, v: f32) -> f32 {
        match op {
            UnaryOp::Identity => v,
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Sqr => v * v,
            UnaryOp::Recip => 1.0 / v,
            UnaryOp::Exp => v.exp(),
            UnaryOp::Log => v.ln(),
            UnaryOp::Sin => v.sin(),
            UnaryOp::Cos => v.cos(),
            UnaryOp::Tanh => v.tanh(),
            UnaryOp::Silu => v / (1.0 + (-v).exp()),
            UnaryOp::GeluTanh => {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
            }
            UnaryOp::GeluExact => {
                let inv_sqrt2 = (1.0_f32 / 2.0_f32.sqrt()) as f64;
                let e = erf_f64((v as f64) * inv_sqrt2) as f32;
                0.5 * v * (1.0 + e)
            }
            UnaryOp::Erf => erf_f64(v as f64) as f32,
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            UnaryOp::Clamp(lo, hi) => v.clamp(lo, hi),
            UnaryOp::Powf(e) => v.powf(e),
            UnaryOp::Affine(m, a) => v * m + a,
            UnaryOp::Relu => if v > 0.0 { v } else { 0.0 },
            UnaryOp::Relu2 => if v > 0.0 { v * v } else { 0.0 },
            UnaryOp::LeakyRelu(a) => if v > 0.0 { v } else { v * a },
            UnaryOp::Sign => if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 },
            UnaryOp::StepGtZero => if v > 0.0 { 1.0 } else { 0.0 },
            UnaryOp::Round => v.round(),
            UnaryOp::Floor => v.floor(),
            UnaryOp::Ceil => v.ceil(),
        }
    }

    fn iter_indices(dims: &[usize], mut f: impl FnMut(&[usize], usize)) {
        let rank = dims.len();
        let numel: usize = dims.iter().product();
        let mut idx = vec![0usize; rank];
        for lin in 0..numel {
            f(&idx, lin);
            for k in (0..rank).rev() {
                idx[k] += 1;
                if idx[k] < dims[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    fn lin(lo: &Layout, idx: &[usize]) -> usize {
        let mut o = lo.offset() as isize;
        for (k, &i) in idx.iter().enumerate() {
            o += i as isize * lo.strides().as_slice()[k];
        }
        o as usize
    }

    fn naive_unary(op: UnaryOp, dt: DType, src: &[u8], src_lo: &Layout, dst: &mut [u8], dst_lo: &Layout) {
        iter_indices(src_lo.dims(), |idx, _| {
            let si = lin(src_lo, idx);
            let di = lin(dst_lo, idx);
            match dt {
                DType::F32 => {
                    let s: &[f32] = bytemuck::cast_slice(src);
                    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    d[di] = unary_scalar_f32(op, false, s[si]);
                }
                DType::F64 => {
                    let s: &[f64] = bytemuck::cast_slice(src);
                    let d: &mut [f64] = bytemuck::cast_slice_mut(dst);
                    d[di] = unary_scalar_f64(op, s[si]);
                }
                DType::F16 => {
                    let s: &[half::f16] = bytemuck::cast_slice(src);
                    let d: &mut [half::f16] = bytemuck::cast_slice_mut(dst);
                    d[di] = match op {
                        UnaryOp::Neg => -s[si],
                        UnaryOp::Sqr => s[si] * s[si],
                        _ => half::f16::from_f32(half_unary_ref(op, s[si].to_f32())),
                    };
                }
                DType::BF16 => {
                    let s: &[half::bf16] = bytemuck::cast_slice(src);
                    let d: &mut [half::bf16] = bytemuck::cast_slice_mut(dst);
                    d[di] = match op {
                        UnaryOp::Neg => -s[si],
                        UnaryOp::Sqr => s[si] * s[si],
                        _ => half::bf16::from_f32(half_unary_ref(op, s[si].to_f32())),
                    };
                }
                _ => unreachable!(),
            }
        });
    }

    fn assert_bytes_eq(dt: DType, got: &[u8], want: &[u8], ctx: &str) {
        assert_eq!(got.len(), want.len());
        match dt {
            DType::F32 => {
                let g: &[f32] = bytemuck::cast_slice(got);
                let w: &[f32] = bytemuck::cast_slice(want);
                for i in 0..g.len() {
                    let ok = g[i].to_bits() == w[i].to_bits() || (g[i].is_nan() && w[i].is_nan());
                    assert!(ok, "{ctx}: idx {i} got {} want {}", g[i], w[i]);
                }
            }
            DType::F64 => {
                let g: &[f64] = bytemuck::cast_slice(got);
                let w: &[f64] = bytemuck::cast_slice(want);
                for i in 0..g.len() {
                    let ok = g[i].to_bits() == w[i].to_bits() || (g[i].is_nan() && w[i].is_nan());
                    assert!(ok, "{ctx}: idx {i} got {} want {}", g[i], w[i]);
                }
            }
            DType::F16 | DType::BF16 => {
                let g: &[u16] = bytemuck::cast_slice(got);
                let w: &[u16] = bytemuck::cast_slice(want);
                for i in 0..g.len() {
                    let (gf, wf) = if dt == DType::F16 {
                        (half::f16::from_bits(g[i]).to_f32(), half::f16::from_bits(w[i]).to_f32())
                    } else {
                        (bf16_bits_to_f32(g[i]), bf16_bits_to_f32(w[i]))
                    };
                    let ok = g[i] == w[i] || (gf.is_nan() && wf.is_nan());
                    assert!(ok, "{ctx}: idx {i} got {gf} ({:#x}) want {wf} ({:#x})", g[i], w[i]);
                }
            }
            _ => assert_eq!(got, want, "{ctx}"),
        }
    }

    const ALL_UNARY: [UnaryOp; 27] = [
        UnaryOp::Identity,
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sqrt,
        UnaryOp::Sqr,
        UnaryOp::Recip,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Silu,
        UnaryOp::GeluTanh,
        UnaryOp::GeluExact,
        UnaryOp::Tanh,
        UnaryOp::Clamp(-1.25, 2.5),
        UnaryOp::Powf(1.7),
        UnaryOp::Affine(0.37, -1.2),
        UnaryOp::Erf,
        UnaryOp::Sigmoid,
        UnaryOp::Relu,
        UnaryOp::Relu2,
        UnaryOp::LeakyRelu(0.1),
        UnaryOp::Sign,
        UnaryOp::StepGtZero,
        UnaryOp::Round,
        UnaryOp::Floor,
        UnaryOp::Ceil,
    ];

    const FLOATS: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];

    fn check_unary(dt: DType, src_lo: &Layout, dst_lo: &Layout, src_elems: usize, dst_elems: usize) {
        let esz = elem_bytes(dt);
        let src_bytes = fill_src(dt, src_elems, 7);
        let src = CpuBuf::from_vec(src_bytes.clone());
        for op in ALL_UNARY {
            let mut want = vec![0u8; dst_elems * esz];
            naive_unary(op, dt, &src_bytes, src_lo, &mut want, dst_lo);
            let mut dst = CpuBuf::alloc_zeros(dst_elems * esz);
            unary_dispatch(op, &src, src_lo, &mut dst, dst_lo).unwrap();
            assert_bytes_eq(dt, dst.as_bytes(), &want, &format!("{dt:?} {op:?} {:?}", src_lo.dims()));
        }
    }

    #[test]
    fn unary_contiguous_all_ops_all_dtypes() {
        for dt in FLOATS {
            for n in [1usize, 7, 8, 9, 1023, 4097, 70001] {
                let lo = Layout::contiguous(Shape::new(vec![n]), dt);
                check_unary(dt, &lo, &lo, n, n);
            }
        }
    }

    #[test]
    fn unary_strided_sources() {
        for dt in FLOATS {
            let base = Layout::contiguous(Shape::new(vec![37, 53]), dt);
            let src = base.transpose(0, 1).unwrap();
            let dst = Layout::contiguous(Shape::new(vec![53, 37]), dt);
            check_unary(dt, &src, &dst, 37 * 53, 37 * 53);
            let src = base.narrow(0, 3, 20).unwrap().narrow(1, 5, 17).unwrap();
            let dst = Layout::contiguous(Shape::new(vec![20, 17]), dt);
            check_unary(dt, &src, &dst, 37 * 53, 20 * 17);
            let src = Layout::contiguous(Shape::new(vec![1, 9]), dt).expand(&Shape::new(vec![2000, 9])).unwrap();
            let dst = Layout::contiguous(Shape::new(vec![2000, 9]), dt);
            check_unary(dt, &src, &dst, 9, 2000 * 9);
            let big = Layout::contiguous(Shape::new(vec![300, 700]), dt).transpose(0, 1).unwrap();
            let dst = Layout::contiguous(Shape::new(vec![700, 300]), dt);
            check_unary(dt, &big, &dst, 300 * 700, 300 * 700);
        }
    }

    #[test]
    fn unary_strided_destination() {
        for dt in FLOATS {
            let src = Layout::contiguous(Shape::new(vec![20, 17]), dt);
            let dst = Layout::contiguous(Shape::new(vec![37, 53]), dt).narrow(0, 3, 20).unwrap().narrow(1, 5, 17).unwrap();
            check_unary(dt, &src, &dst, 20 * 17, 37 * 53);
            let dst_t = Layout::contiguous(Shape::new(vec![17, 20]), dt).transpose(0, 1).unwrap();
            check_unary(dt, &src, &dst_t, 20 * 17, 20 * 17);
        }
    }

    fn naive_binary(op: BinaryOp, dt: DType, a: &[u8], a_lo: &Layout, b: &[u8], b_lo: &Layout, dst: &mut [u8], dst_lo: &Layout) {
        iter_indices(dst_lo.dims(), |idx, _| {
            let ai = lin(a_lo, idx);
            let bi = lin(b_lo, idx);
            let di = lin(dst_lo, idx);
            match dt {
                DType::F32 => {
                    let a: &[f32] = bytemuck::cast_slice(a);
                    let b: &[f32] = bytemuck::cast_slice(b);
                    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    d[di] = binary_scalar_f32(op, a[ai], b[bi]);
                }
                DType::F64 => {
                    let a: &[f64] = bytemuck::cast_slice(a);
                    let b: &[f64] = bytemuck::cast_slice(b);
                    let d: &mut [f64] = bytemuck::cast_slice_mut(dst);
                    d[di] = binary_scalar_f64(op, a[ai], b[bi]);
                }
                DType::F16 => {
                    let a: &[half::f16] = bytemuck::cast_slice(a);
                    let b: &[half::f16] = bytemuck::cast_slice(b);
                    let d: &mut [half::f16] = bytemuck::cast_slice_mut(dst);
                    let (x, y) = (a[ai], b[bi]);
                    d[di] = match op {
                        BinaryOp::Add => x + y,
                        BinaryOp::Sub => x - y,
                        BinaryOp::Mul => x * y,
                        BinaryOp::Div => x / y,
                        BinaryOp::Max => if x > y { x } else { y },
                        BinaryOp::Min => if x < y { x } else { y },
                    };
                }
                DType::BF16 => {
                    let a: &[half::bf16] = bytemuck::cast_slice(a);
                    let b: &[half::bf16] = bytemuck::cast_slice(b);
                    let d: &mut [half::bf16] = bytemuck::cast_slice_mut(dst);
                    let (x, y) = (a[ai], b[bi]);
                    d[di] = match op {
                        BinaryOp::Add => x + y,
                        BinaryOp::Sub => x - y,
                        BinaryOp::Mul => x * y,
                        BinaryOp::Div => x / y,
                        BinaryOp::Max => if x > y { x } else { y },
                        BinaryOp::Min => if x < y { x } else { y },
                    };
                }
                _ => unreachable!(),
            }
        });
    }

    const ALL_BINARY: [BinaryOp; 6] = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Max, BinaryOp::Min];

    fn check_binary(dt: DType, a_lo: &Layout, a_elems: usize, b_lo: &Layout, b_elems: usize, dst_lo: &Layout, dst_elems: usize) {
        let esz = elem_bytes(dt);
        let a_bytes = fill_src(dt, a_elems, 11);
        let b_bytes = fill_src(dt, b_elems, 23);
        let a = CpuBuf::from_vec(a_bytes.clone());
        let b = CpuBuf::from_vec(b_bytes.clone());
        for op in ALL_BINARY {
            let mut want = vec![0u8; dst_elems * esz];
            naive_binary(op, dt, &a_bytes, a_lo, &b_bytes, b_lo, &mut want, dst_lo);
            let mut dst = CpuBuf::alloc_zeros(dst_elems * esz);
            binary_dispatch(op, &a, a_lo, &b, b_lo, &mut dst, dst_lo).unwrap();
            assert_bytes_eq(dt, dst.as_bytes(), &want, &format!("{dt:?} {op:?} a={:?} b={:?}", a_lo.strides(), b_lo.strides()));
        }
    }

    #[test]
    fn binary_same_shape_contiguous() {
        for dt in FLOATS {
            for n in [1usize, 5, 8, 17, 1000, 4099, 50001] {
                let lo = Layout::contiguous(Shape::new(vec![n]), dt);
                check_binary(dt, &lo, n, &lo, n, &lo, n);
            }
        }
    }

    #[test]
    fn binary_broadcast_cases() {
        for dt in FLOATS {
            let out = Shape::new(vec![1000, 33]);
            let dst = Layout::contiguous(out.clone(), dt);
            let full = Layout::contiguous(out.clone(), dt);
            let scalar = Layout::contiguous(Shape::new(vec![1, 1]), dt).expand(&out).unwrap();
            let row = Layout::contiguous(Shape::new(vec![1, 33]), dt).expand(&out).unwrap();
            let col = Layout::contiguous(Shape::new(vec![1000, 1]), dt).expand(&out).unwrap();
            check_binary(dt, &full, 33000, &scalar, 1, &dst, 33000);
            check_binary(dt, &scalar, 1, &full, 33000, &dst, 33000);
            check_binary(dt, &full, 33000, &row, 33, &dst, 33000);
            check_binary(dt, &row, 33, &full, 33000, &dst, 33000);
            check_binary(dt, &full, 33000, &col, 1000, &dst, 33000);
            check_binary(dt, &col, 1000, &full, 33000, &dst, 33000);
            check_binary(dt, &row, 33, &col, 1000, &dst, 33000);
            check_binary(dt, &scalar, 1, &scalar, 1, &dst, 33000);
            let out3 = Shape::new(vec![6, 70, 9]);
            let dst3 = Layout::contiguous(out3.clone(), dt);
            let a_t = Layout::contiguous(Shape::new(vec![6, 9, 70]), dt).transpose(1, 2).unwrap();
            let b_mid = Layout::contiguous(Shape::new(vec![6, 1, 9]), dt).expand(&out3).unwrap();
            check_binary(dt, &a_t, 6 * 70 * 9, &b_mid, 6 * 9, &dst3, 6 * 70 * 9);
            check_binary(dt, &b_mid, 6 * 9, &a_t, 6 * 70 * 9, &dst3, 6 * 70 * 9);
            let bias = Layout::contiguous(Shape::new(vec![9]), dt).expand(&out3).unwrap();
            check_binary(dt, &a_t, 6 * 70 * 9, &bias, 9, &dst3, 6 * 70 * 9);
            let dst_off = Layout::contiguous(Shape::new(vec![8, 80, 9]), dt).narrow(0, 1, 6).unwrap().narrow(1, 5, 70).unwrap();
            check_binary(dt, &Layout::contiguous(out3.clone(), dt), 6 * 70 * 9, &bias, 9, &dst_off, 8 * 80 * 9);
        }
    }

    fn cast_ref(sdt: DType, ddt: DType, s: &[u8], n: usize) -> Vec<u8> {
        fn rd(dt: DType, s: &[u8], i: usize) -> f64 {
            match dt {
                DType::F32 => bytemuck::cast_slice::<u8, f32>(s)[i] as f64,
                DType::F64 => bytemuck::cast_slice::<u8, f64>(s)[i],
                DType::F16 => bytemuck::cast_slice::<u8, half::f16>(s)[i].to_f32() as f64,
                DType::BF16 => bytemuck::cast_slice::<u8, half::bf16>(s)[i].to_f32() as f64,
                DType::U8 => s[i] as f64,
                DType::U32 => bytemuck::cast_slice::<u8, u32>(s)[i] as f64,
                DType::I32 => bytemuck::cast_slice::<u8, i32>(s)[i] as f64,
                DType::I64 => bytemuck::cast_slice::<u8, i64>(s)[i] as f64,
                _ => unreachable!(),
            }
        }
        let mut out = vec![0u8; n * elem_bytes(ddt)];
        for i in 0..n {
            let v = rd(sdt, s, i);
            match (sdt, ddt) {
                (DType::F32, DType::F16) => bytemuck::cast_slice_mut::<u8, half::f16>(&mut out)[i] = half::f16::from_f32(v as f32),
                (DType::F32, DType::BF16) => bytemuck::cast_slice_mut::<u8, half::bf16>(&mut out)[i] = half::bf16::from_f32(v as f32),
                (DType::F16, DType::BF16) => bytemuck::cast_slice_mut::<u8, half::bf16>(&mut out)[i] = half::bf16::from_f32(v as f32),
                (DType::BF16, DType::F16) => bytemuck::cast_slice_mut::<u8, half::f16>(&mut out)[i] = half::f16::from_f32(v as f32),
                (_, DType::F32) => {
                    let w = match sdt {
                        DType::F64 => bytemuck::cast_slice::<u8, f64>(s)[i] as f32,
                        DType::U32 => bytemuck::cast_slice::<u8, u32>(s)[i] as f32,
                        DType::I32 => bytemuck::cast_slice::<u8, i32>(s)[i] as f32,
                        DType::I64 => bytemuck::cast_slice::<u8, i64>(s)[i] as f32,
                        DType::U8 => s[i] as f32,
                        _ => v as f32,
                    };
                    bytemuck::cast_slice_mut::<u8, f32>(&mut out)[i] = w;
                }
                (DType::F32, DType::F64) => bytemuck::cast_slice_mut::<u8, f64>(&mut out)[i] = v,
                (DType::F32, DType::U32) | (DType::F16, DType::U32) | (DType::BF16, DType::U32) => {
                    bytemuck::cast_slice_mut::<u8, u32>(&mut out)[i] = (v as f32) as u32
                }
                (DType::F32, DType::U8) => out[i] = (v as f32) as u8,
                (DType::F32, DType::I32) => bytemuck::cast_slice_mut::<u8, i32>(&mut out)[i] = (v as f32) as i32,
                (DType::F32, DType::I64) => bytemuck::cast_slice_mut::<u8, i64>(&mut out)[i] = (v as f32) as i64,
                (DType::U32, DType::I64) => bytemuck::cast_slice_mut::<u8, i64>(&mut out)[i] = bytemuck::cast_slice::<u8, u32>(s)[i] as i64,
                (DType::I64, DType::U32) => bytemuck::cast_slice_mut::<u8, u32>(&mut out)[i] = bytemuck::cast_slice::<u8, i64>(s)[i] as u32,
                (DType::U32, DType::I32) => bytemuck::cast_slice_mut::<u8, i32>(&mut out)[i] = bytemuck::cast_slice::<u8, u32>(s)[i] as i32,
                (DType::I32, DType::U32) => bytemuck::cast_slice_mut::<u8, u32>(&mut out)[i] = bytemuck::cast_slice::<u8, i32>(s)[i] as u32,
                _ => unreachable!(),
            }
        }
        out
    }

    fn int_src(dt: DType, n: usize) -> Vec<u8> {
        match dt {
            DType::U8 => (0..n).map(|i| (i * 37 % 256) as u8).collect(),
            DType::U32 => bytemuck::cast_slice(&(0..n).map(|i| (i as u32).wrapping_mul(2654435761) >> 3).collect::<Vec<u32>>()).to_vec(),
            DType::I32 => bytemuck::cast_slice(&(0..n).map(|i| (i as i32).wrapping_mul(-1103515245)).collect::<Vec<i32>>()).to_vec(),
            DType::I64 => bytemuck::cast_slice(&(0..n).map(|i| (i as i64 - 5000).wrapping_mul(6364136223846793005)).collect::<Vec<i64>>()).to_vec(),
            _ => fill_src(dt, n, 5),
        }
    }

    #[test]
    fn cast_all_pairs_contiguous_and_strided() {
        use DType::*;
        let pairs = [
            (F32, F16), (F32, BF16), (F16, F32), (BF16, F32), (F32, F64), (F64, F32), (F16, BF16), (BF16, F16),
            (U32, F32), (F32, U32), (U32, I64), (I64, U32), (U8, F32), (F32, U8), (I32, F32), (F32, I32),
            (I64, F32), (F32, I64), (U32, I32), (I32, U32), (F16, U32), (BF16, U32),
        ];
        for (sdt, ddt) in pairs {
            for n in [1usize, 9, 1029, 40003] {
                let s = int_src(sdt, n);
                let want = cast_ref(sdt, ddt, &s, n);
                let src = CpuBuf::from_vec(s);
                let mut dst = CpuBuf::alloc_zeros(n * elem_bytes(ddt));
                let slo = Layout::contiguous(Shape::new(vec![n]), sdt);
                let dlo = Layout::contiguous(Shape::new(vec![n]), ddt);
                cast_dispatch(&src, &slo, &mut dst, &dlo).unwrap();
                assert_bytes_eq(ddt, dst.as_bytes(), &want, &format!("cast {sdt:?}->{ddt:?} n={n}"));
            }
            let (r, c) = (41, 67);
            let s = int_src(sdt, r * c);
            let src = CpuBuf::from_vec(s.clone());
            let slo = Layout::contiguous(Shape::new(vec![r, c]), sdt).transpose(0, 1).unwrap();
            let dlo = Layout::contiguous(Shape::new(vec![c, r]), ddt);
            let mut dst = CpuBuf::alloc_zeros(r * c * elem_bytes(ddt));
            cast_dispatch(&src, &slo, &mut dst, &dlo).unwrap();
            let mut gathered = vec![0u8; r * c * elem_bytes(sdt)];
            let se = elem_bytes(sdt);
            iter_indices(&[c, r], |idx, l| {
                let si = lin(&slo, idx) * se;
                gathered[l * se..l * se + se].copy_from_slice(&s[si..si + se]);
            });
            let want = cast_ref(sdt, ddt, &gathered, r * c);
            assert_bytes_eq(ddt, dst.as_bytes(), &want, &format!("cast strided {sdt:?}->{ddt:?}"));
        }
    }

    #[test]
    fn coalesce_merges_contiguous_axes() {
        let mut st = vec![vec![12isize, 4, 1], vec![0, 0, 1]];
        let dims = coalesce(&[2, 3, 4], &mut st);
        assert_eq!(dims, vec![6, 4]);
        assert_eq!(st[0], vec![4, 1]);
        assert_eq!(st[1], vec![0, 1]);
        let mut st = vec![vec![12isize, 4, 1]];
        let dims = coalesce(&[2, 3, 4], &mut st);
        assert_eq!(dims, vec![24]);
        let mut st = vec![vec![0isize, 1, 0]];
        let dims = coalesce(&[1, 5, 1], &mut st);
        assert_eq!(dims, vec![5]);
        assert_eq!(st[0], vec![1]);
    }

    #[test]
    #[ignore]
    fn bench_unary_binary() {
        let n = 16 * 1024 * 1024;
        let src = CpuBuf::from_vec(fill_src(DType::F32, n, 3));
        let lo = Layout::contiguous(Shape::new(vec![n]), DType::F32);
        let mut dst = CpuBuf::alloc_zeros(n * 4);
        for op in [UnaryOp::Silu, UnaryOp::Relu, UnaryOp::Affine(2.0, 1.0)] {
            unary_dispatch(op, &src, &lo, &mut dst, &lo).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                unary_dispatch(op, &src, &lo, &mut dst, &lo).unwrap();
            }
            println!("unary {op:?} 64MB f32: {:?}/iter", t.elapsed() / 5);
        }
        let mut naive_out = vec![0u8; n * 4];
        let t = std::time::Instant::now();
        naive_unary(UnaryOp::Silu, DType::F32, src.as_bytes(), &lo, &mut naive_out, &lo);
        println!("naive (old) unary Silu 64MB f32: {:?}", t.elapsed());
        let out = Shape::new(vec![4096, 4096]);
        let a_lo = Layout::contiguous(out.clone(), DType::F32);
        let b_lo = Layout::contiguous(Shape::new(vec![4096]), DType::F32).expand(&out).unwrap();
        let b = CpuBuf::from_vec(fill_src(DType::F32, 4096, 9));
        binary_dispatch(BinaryOp::Add, &src, &a_lo, &b, &b_lo, &mut dst, &a_lo).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            binary_dispatch(BinaryOp::Add, &src, &a_lo, &b, &b_lo, &mut dst, &a_lo).unwrap();
        }
        println!("binary broadcast add [4096,4096]+[4096]: {:?}/iter", t.elapsed() / 5);
        let t = std::time::Instant::now();
        naive_binary(BinaryOp::Add, DType::F32, src.as_bytes(), &a_lo, b.as_bytes(), &b_lo, &mut naive_out, &a_lo);
        println!("naive (old) binary broadcast add: {:?}", t.elapsed());
        let src_t = Layout::contiguous(Shape::new(vec![4096, 4096]), DType::F32).transpose(0, 1).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            unary_dispatch(UnaryOp::Identity, &src, &src_t, &mut dst, &a_lo).unwrap();
        }
        println!("unary Identity transposed [4096,4096]: {:?}/iter", t.elapsed() / 5);
        let t = std::time::Instant::now();
        for _ in 0..5 {
            crate::copy::copy_strided(&src, &src_t, &mut dst, &a_lo).unwrap();
        }
        println!("copy_strided transposed [4096,4096]: {:?}/iter", t.elapsed() / 5);
        let hsrc = CpuBuf::from_vec(fill_src(DType::BF16, n, 3));
        let hlo = Layout::contiguous(Shape::new(vec![n]), DType::BF16);
        let mut hdst = CpuBuf::alloc_zeros(n * 2);
        let t = std::time::Instant::now();
        for _ in 0..5 {
            unary_dispatch(UnaryOp::Silu, &hsrc, &hlo, &mut hdst, &hlo).unwrap();
        }
        println!("unary Silu 32MB bf16: {:?}/iter", t.elapsed() / 5);
    }
}
