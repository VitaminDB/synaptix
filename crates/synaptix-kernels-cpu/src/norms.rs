use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;
use wide::f32x8;

use crate::common::{cpu, cpu_mut, elem_bytes, row_from_f32, row_to_f32, same_dtype, threads};

pub(crate) fn dense_offset(lo: &Layout) -> Result<usize> {
    if lo.strides().is_contiguous(lo.shape()) {
        Ok(lo.offset())
    } else {
        Err(SynaptixError::NonContiguous)
    }
}

pub(crate) fn check_float(dt: DType) -> Result<()> {
    if matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        Ok(())
    } else {
        Err(SynaptixError::Unsupported("cpu fused: dtype must be F32/F16/BF16"))
    }
}

pub(crate) fn to_f32_vec(st: &Storage, lo: &Layout, what: &'static str) -> Result<Vec<f32>> {
    let off = dense_offset(lo)?;
    let buf = cpu(st, what)?;
    let mut v = vec![0.0; lo.numel()];
    crate::common::to_f32(lo.dtype(), buf.as_bytes(), off, &mut v)?;
    Ok(v)
}

pub(crate) fn in_bytes<'a>(st: &'a Storage, lo: &Layout, what: &'static str) -> Result<(&'a [u8], usize)> {
    let off = dense_offset(lo)?;
    Ok((cpu(st, what)?.as_bytes(), off))
}

pub(crate) fn out_bytes<'a>(st: &'a mut Storage, lo: &Layout, what: &'static str) -> Result<&'a mut [u8]> {
    let off = dense_offset(lo)?;
    let eb = elem_bytes(lo.dtype());
    let n = lo.numel();
    let b = cpu_mut(st, what)?.as_bytes_mut();
    b.get_mut(off * eb..(off + n) * eb).ok_or(SynaptixError::Unsupported("cpu fused: out buffer too small"))
}

pub(crate) struct Rows<'a> {
    dt: DType,
    bytes: &'a [u8],
    off: usize,
    buf: Vec<f32>,
}

impl<'a> Rows<'a> {
    pub(crate) fn new(dt: DType, bytes: &'a [u8], off: usize, width: usize) -> Self {
        let buf = if dt == DType::F32 { Vec::new() } else { vec![0.0; width] };
        Self { dt, bytes, off, buf }
    }

    pub(crate) fn get(&mut self, elem_off: usize, n: usize) -> &[f32] {
        let off = self.off + elem_off;
        if self.dt == DType::F32 {
            bytemuck::cast_slice(&self.bytes[off * 4..(off + n) * 4])
        } else {
            row_to_f32(self.dt, self.bytes, off, &mut self.buf[..n]);
            &self.buf[..n]
        }
    }
}

pub(crate) fn out_row(dt: DType, dst: &mut [u8], tmp: &mut [f32], f: impl FnOnce(&mut [f32])) {
    if dt == DType::F32 {
        f(bytemuck::cast_slice_mut(dst));
    } else {
        let n = dst.len() / elem_bytes(dt);
        f(&mut tmp[..n]);
        row_from_f32(dt, &tmp[..n], dst, 0);
    }
}

pub(crate) fn roundtrip(dt: DType, v: &mut [f32], scratch: &mut Vec<u8>) {
    if dt == DType::F32 {
        return;
    }
    scratch.resize(v.len() * elem_bytes(dt), 0);
    row_from_f32(dt, v, scratch, 0);
    row_to_f32(dt, scratch, 0, v);
}

pub(crate) fn min_rows(width: usize) -> usize {
    (1usize << 14) / width.max(1) + 1
}

pub(crate) fn par_rows(out: &mut [u8], row_bytes: usize, min_rows: usize, f: impl Fn(usize, &mut [u8]) + Sync) {
    if row_bytes == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_bytes;
    let chunk = rows.div_ceil(threads()).max(min_rows).max(1);
    if rows <= chunk {
        f(0, out);
        return;
    }
    use rayon::prelude::*;
    out.par_chunks_mut(chunk * row_bytes).enumerate().for_each(|(i, blk)| f(i * chunk, blk));
}

pub(crate) fn par_rows2(
    a: &mut [u8],
    b: &mut [u8],
    row_bytes: usize,
    min_rows: usize,
    f: impl Fn(usize, &mut [u8], &mut [u8]) + Sync,
) {
    if row_bytes == 0 || a.is_empty() {
        return;
    }
    let rows = a.len() / row_bytes;
    let chunk = rows.div_ceil(threads()).max(min_rows).max(1);
    if rows <= chunk {
        f(0, a, b);
        return;
    }
    use rayon::prelude::*;
    a.par_chunks_mut(chunk * row_bytes)
        .zip(b.par_chunks_mut(chunk * row_bytes))
        .enumerate()
        .for_each(|(i, (x, y))| f(i * chunk, x, y));
}

#[derive(Clone, Copy)]
pub(crate) struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    pub(crate) fn new(p: *mut u8) -> Self {
        Self(p)
    }

    pub(crate) fn get(self) -> *mut u8 {
        self.0
    }
}

macro_rules! simd_fn {
    ($vis:vis fn $name:ident($($arg:ident : $ty:ty),*) $(-> $ret:ty)? = $body:ident) => {
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                let f = crate::dispatch::cpu_features();
                if f.avx2 && f.fma {
                    #[target_feature(enable = "avx2,fma")]
                    #[allow(clippy::too_many_arguments)]
                    unsafe fn fast($($arg: $ty),*) $(-> $ret)? {
                        $body($($arg),*)
                    }
                    return unsafe { fast($($arg),*) };
                }
            }
            $body($($arg),*)
        }
    };
}
pub(crate) use simd_fn;

#[inline(always)]
pub(crate) fn ld(s: &[f32]) -> f32x8 {
    let mut a = [0.0f32; 8];
    a.copy_from_slice(s);
    f32x8::new(a)
}

#[inline(always)]
pub(crate) fn st(v: f32x8, d: &mut [f32]) {
    d.copy_from_slice(&v.to_array());
}

#[inline(always)]
pub(crate) fn sigmoid8(v: f32x8) -> f32x8 {
    let e = (-v).min(f32x8::splat(80.0)).exp();
    f32x8::ONE / (f32x8::ONE + e)
}

#[inline(always)]
fn sum_body(x: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        acc += ld(c);
    }
    let mut s = acc.reduce_add();
    for &v in it.remainder() {
        s += v;
    }
    s
}

#[inline(always)]
fn sum_sq_body(x: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        let v = ld(c);
        acc = v.mul_add(v, acc);
    }
    let mut s = acc.reduce_add();
    for &v in it.remainder() {
        s += v * v;
    }
    s
}

#[inline(always)]
fn sum_sq_centered_body(x: &[f32], mean: f32) -> f32 {
    let m = f32x8::splat(mean);
    let mut acc = f32x8::ZERO;
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        let v = ld(c) - m;
        acc = v.mul_add(v, acc);
    }
    let mut s = acc.reduce_add();
    for &v in it.remainder() {
        let d = v - mean;
        s += d * d;
    }
    s
}

#[inline(always)]
fn norm_row_body(x: &[f32], mean: f32, inv: f32, w: &[f32], b: Option<&[f32]>, out: &mut [f32]) {
    let m = f32x8::splat(mean);
    let s = f32x8::splat(inv);
    let n = out.len();
    let main = n - n % 8;
    match b {
        Some(b) => {
            for i in (0..main).step_by(8) {
                let v = (ld(&x[i..i + 8]) - m) * s * ld(&w[i..i + 8]) + ld(&b[i..i + 8]);
                st(v, &mut out[i..i + 8]);
            }
            for i in main..n {
                out[i] = (x[i] - mean) * inv * w[i] + b[i];
            }
        }
        None => {
            for i in (0..main).step_by(8) {
                let v = (ld(&x[i..i + 8]) - m) * s * ld(&w[i..i + 8]);
                st(v, &mut out[i..i + 8]);
            }
            for i in main..n {
                out[i] = (x[i] - mean) * inv * w[i];
            }
        }
    }
}

#[inline(always)]
fn norm_scalar_body(x: &[f32], mean: f32, inv: f32, wc: f32, bc: f32, out: &mut [f32]) {
    let m = f32x8::splat(mean);
    let s = f32x8::splat(inv);
    let w = f32x8::splat(wc);
    let b = f32x8::splat(bc);
    let n = out.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let v = ((ld(&x[i..i + 8]) - m) * s).mul_add(w, b);
        st(v, &mut out[i..i + 8]);
    }
    for i in main..n {
        out[i] = (x[i] - mean) * inv * wc + bc;
    }
}

#[inline(always)]
fn norm_vec_body(x: &[f32], mean: &[f32], inv: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let v = ((ld(&x[i..i + 8]) - ld(&mean[i..i + 8])) * ld(&inv[i..i + 8])).mul_add(ld(&w[i..i + 8]), ld(&b[i..i + 8]));
        st(v, &mut out[i..i + 8]);
    }
    for i in main..n {
        out[i] = (x[i] - mean[i]) * inv[i] * w[i] + b[i];
    }
}

#[inline(always)]
fn scale_vec_body(x: &[f32], inv: &[f32], out: &mut [f32]) {
    let n = out.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        st(ld(&x[i..i + 8]) * ld(&inv[i..i + 8]), &mut out[i..i + 8]);
    }
    for i in main..n {
        out[i] = x[i] * inv[i];
    }
}

#[inline(always)]
fn silu_inplace_body(v: &mut [f32]) {
    let n = v.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let x = ld(&v[i..i + 8]);
        st(x * sigmoid8(x), &mut v[i..i + 8]);
    }
    for x in &mut v[main..] {
        *x = crate::common::silu_f32(*x);
    }
}

simd_fn!(pub(crate) fn sum(x: &[f32]) -> f32 = sum_body);
simd_fn!(pub(crate) fn sum_sq(x: &[f32]) -> f32 = sum_sq_body);
simd_fn!(pub(crate) fn sum_sq_centered(x: &[f32], mean: f32) -> f32 = sum_sq_centered_body);
simd_fn!(pub(crate) fn norm_row(x: &[f32], mean: f32, inv: f32, w: &[f32], b: Option<&[f32]>, out: &mut [f32]) = norm_row_body);
simd_fn!(pub(crate) fn norm_scalar(x: &[f32], mean: f32, inv: f32, wc: f32, bc: f32, out: &mut [f32]) = norm_scalar_body);
simd_fn!(pub(crate) fn norm_vec(x: &[f32], mean: &[f32], inv: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) = norm_vec_body);
simd_fn!(pub(crate) fn scale_vec(x: &[f32], inv: &[f32], out: &mut [f32]) = scale_vec_body);
simd_fn!(pub(crate) fn silu_inplace(v: &mut [f32]) = silu_inplace_body);

fn last_dim(lo: &Layout, what: &'static str) -> Result<usize> {
    match lo.dims().last() {
        Some(&h) if h > 0 => Ok(h),
        _ => Err(SynaptixError::Unsupported(what)),
    }
}

fn gain_vec(w: (&Storage, &Layout), h: usize, qwen_gain: bool, what: &'static str) -> Result<Vec<f32>> {
    if w.1.numel() != h {
        return Err(SynaptixError::Unsupported(what));
    }
    let mut g = to_f32_vec(w.0, w.1, what)?;
    if qwen_gain {
        for v in &mut g {
            *v += 1.0;
        }
    }
    Ok(g)
}

pub fn rms_norm(
    x: (&Storage, &Layout),
    w: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    eps: f32,
    qwen_gain: bool,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, w.1.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    let h = last_dim(x_lo, "cpu rms_norm: scalar x")?;
    if o_lo.dims() != x_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu rms_norm: out shape"));
    }
    let gain = gain_vec(w, h, qwen_gain, "cpu rms_norm: w")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu rms_norm: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu rms_norm: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, h * eb, min_rows(h), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, h);
        let mut tmp = vec![0.0; h];
        for (i, dst) in blk.chunks_exact_mut(h * eb).enumerate() {
            let xr = rows.get((r0 + i) * h, h);
            let inv = 1.0 / (sum_sq(xr) / h as f32 + eps).sqrt();
            out_row(dt, dst, &mut tmp, |o| norm_row(xr, 0.0, inv, &gain, None, o));
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual(
    x: (&Storage, &Layout),
    residual: (&Storage, &Layout),
    w: (&Storage, &Layout),
    hidden_out: (&mut Storage, &Layout),
    y: (&mut Storage, &Layout),
    eps: f32,
    qwen_gain: bool,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (r_st, r_lo) = residual;
    let (h_st, h_lo) = hidden_out;
    let (y_st, y_lo) = y;
    let dt = x_lo.dtype();
    check_float(dt)?;
    for lo in [r_lo, w.1, h_lo, y_lo] {
        same_dtype(dt, lo.dtype())?;
    }
    let h = last_dim(x_lo, "cpu rms_norm_residual: scalar x")?;
    if r_lo.dims() != x_lo.dims() || h_lo.dims() != x_lo.dims() || y_lo.dims() != x_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu rms_norm_residual: shape"));
    }
    let gain = gain_vec(w, h, qwen_gain, "cpu rms_norm_residual: w")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu rms_norm_residual: x")?;
    let (rb, r_off) = in_bytes(r_st, r_lo, "cpu rms_norm_residual: residual")?;
    let hb = out_bytes(h_st, h_lo, "cpu rms_norm_residual: hidden_out")?;
    let yb = out_bytes(y_st, y_lo, "cpu rms_norm_residual: y")?;
    let eb = elem_bytes(dt);
    par_rows2(hb, yb, h * eb, min_rows(h), |r0, hblk, yblk| {
        let mut xrows = Rows::new(dt, xb, x_off, h);
        let mut rrows = Rows::new(dt, rb, r_off, h);
        let mut tmp = vec![0.0; h];
        let mut hid = vec![0.0; h];
        for (i, (hd, yd)) in hblk.chunks_exact_mut(h * eb).zip(yblk.chunks_exact_mut(h * eb)).enumerate() {
            let r = (r0 + i) * h;
            let xr = xrows.get(r, h);
            let rr = rrows.get(r, h);
            out_row(dt, hd, &mut tmp, |o| {
                for ((o, &a), &b) in o.iter_mut().zip(xr).zip(rr) {
                    *o = a + b;
                }
            });
            row_to_f32(dt, hd, 0, &mut hid);
            let inv = 1.0 / (sum_sq(&hid) / h as f32 + eps).sqrt();
            out_row(dt, yd, &mut tmp, |o| norm_row(&hid, 0.0, inv, &gain, None, o));
        }
    });
    Ok(())
}

pub fn layer_norm(
    x: (&Storage, &Layout),
    w: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    eps: f32,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, w.1.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    let h = last_dim(x_lo, "cpu layer_norm: scalar x")?;
    if o_lo.dims() != x_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu layer_norm: out shape"));
    }
    let wv = gain_vec(w, h, false, "cpu layer_norm: w")?;
    let bv = match bias {
        Some(b) => {
            same_dtype(dt, b.1.dtype())?;
            Some(gain_vec(b, h, false, "cpu layer_norm: bias")?)
        }
        None => None,
    };
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu layer_norm: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu layer_norm: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, h * eb, min_rows(h), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, h);
        let mut tmp = vec![0.0; h];
        for (i, dst) in blk.chunks_exact_mut(h * eb).enumerate() {
            let xr = rows.get((r0 + i) * h, h);
            let mean = sum(xr) / h as f32;
            let var = sum_sq_centered(xr, mean) / h as f32;
            let inv = 1.0 / (var + eps).sqrt();
            out_row(dt, dst, &mut tmp, |o| norm_row(xr, mean, inv, &wv, bv.as_deref(), o));
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn group_rms(
    x: (&Storage, &Layout),
    w: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    rows: usize,
    groups: usize,
    group: usize,
    eps: f32,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, w.1.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    let width = groups * group;
    if group == 0 || groups == 0 || x_lo.numel() != rows * width || o_lo.numel() != rows * width {
        return Err(SynaptixError::Unsupported("cpu group_rms: shape"));
    }
    let wv = gain_vec(w, width, false, "cpu group_rms: w")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu group_rms: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu group_rms: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, width * eb, min_rows(width), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, width);
        let mut tmp = vec![0.0; width];
        for (i, dst) in blk.chunks_exact_mut(width * eb).enumerate() {
            let xr = rows.get((r0 + i) * width, width);
            out_row(dt, dst, &mut tmp, |o| {
                for g in 0..groups {
                    let s = g * group..(g + 1) * group;
                    let inv = 1.0 / (sum_sq(&xr[s.clone()]) / group as f32 + eps).sqrt();
                    norm_row(&xr[s.clone()], 0.0, inv, &wv[s.clone()], None, &mut o[s]);
                }
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm(
    x: (&Storage, &Layout),
    weight: Option<(&Storage, &Layout)>,
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    num_groups: usize,
    eps: f32,
    silu: bool,
    nhwc: bool,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    let d = x_lo.dims();
    if d.len() < 2 || o_lo.dims() != d {
        return Err(SynaptixError::Unsupported("cpu group_norm: shape"));
    }
    let (c, hw) = if nhwc {
        (d[d.len() - 1], d[1..d.len() - 1].iter().product::<usize>().max(1))
    } else {
        (d[1], d[2..].iter().product::<usize>().max(1))
    };
    if num_groups == 0 || !c.is_multiple_of(num_groups) || c == 0 {
        return Err(SynaptixError::Unsupported("cpu group_norm: c % num_groups"));
    }
    if weight.is_none() && bias.is_some() {
        return Err(SynaptixError::Unsupported("cpu group_norm: bias без weight"));
    }
    let pg = c / num_groups;
    let wv = match weight {
        Some(w) => {
            same_dtype(dt, w.1.dtype())?;
            gain_vec(w, c, false, "cpu group_norm: weight")?
        }
        None => vec![1.0; c],
    };
    let bv = match bias {
        Some(bb) => {
            same_dtype(dt, bb.1.dtype())?;
            gain_vec(bb, c, false, "cpu group_norm: bias")?
        }
        None => vec![0.0; c],
    };
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu group_norm: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu group_norm: out")?;
    let eb = elem_bytes(dt);
    let gn = (pg * hw) as f32;
    if !nhwc {
        let block = pg * hw;
        par_rows(ob, block * eb, 1, |b0, blk| {
            let mut rows = Rows::new(dt, xb, x_off, hw);
            let mut tmp = vec![0.0; hw];
            for (i, dst) in blk.chunks_exact_mut(block * eb).enumerate() {
                let idx = b0 + i;
                let g = idx % num_groups;
                let base = idx * block;
                let mut s = 0.0;
                for j in 0..pg {
                    s += sum(rows.get(base + j * hw, hw));
                }
                let mean = s / gn;
                let mut sq = 0.0;
                for j in 0..pg {
                    sq += sum_sq_centered(rows.get(base + j * hw, hw), mean);
                }
                let inv = 1.0 / (sq / gn + eps).sqrt();
                for (j, seg) in dst.chunks_exact_mut(hw * eb).enumerate() {
                    let ch = g * pg + j;
                    let xr = rows.get(base + j * hw, hw);
                    out_row(dt, seg, &mut tmp, |o| {
                        norm_scalar(xr, mean, inv, wv[ch], bv[ch], o);
                        if silu {
                            silu_inplace(o);
                        }
                    });
                }
            }
        });
        return Ok(());
    }
    use rayon::prelude::*;
    let sp_chunk = (hw.div_ceil(threads() * 4)).max(1);
    let n_chunks = hw.div_ceil(sp_chunk);
    let mut mean_c = vec![0.0f32; c];
    let mut inv_c = vec![0.0f32; c];
    for (bi, oblk) in ob.chunks_exact_mut(hw * c * eb).enumerate() {
        let base = bi * hw * c;
        let sums = (0..n_chunks)
            .into_par_iter()
            .map(|ci| {
                let mut rows = Rows::new(dt, xb, x_off, c);
                let mut acc = vec![0.0f32; c];
                for s in ci * sp_chunk..((ci + 1) * sp_chunk).min(hw) {
                    for (a, &v) in acc.iter_mut().zip(rows.get(base + s * c, c)) {
                        *a += v;
                    }
                }
                acc
            })
            .reduce(
                || vec![0.0f32; c],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        for g in 0..num_groups {
            let m = sums[g * pg..(g + 1) * pg].iter().sum::<f32>() / gn;
            mean_c[g * pg..(g + 1) * pg].fill(m);
        }
        let sq = (0..n_chunks)
            .into_par_iter()
            .map(|ci| {
                let mut rows = Rows::new(dt, xb, x_off, c);
                let mut acc = vec![0.0f32; c];
                for s in ci * sp_chunk..((ci + 1) * sp_chunk).min(hw) {
                    for ((a, &v), &m) in acc.iter_mut().zip(rows.get(base + s * c, c)).zip(&mean_c) {
                        let dlt = v - m;
                        *a += dlt * dlt;
                    }
                }
                acc
            })
            .reduce(
                || vec![0.0f32; c],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        for g in 0..num_groups {
            let var = sq[g * pg..(g + 1) * pg].iter().sum::<f32>() / gn;
            inv_c[g * pg..(g + 1) * pg].fill(1.0 / (var + eps).sqrt());
        }
        let (mean_c, inv_c) = (&mean_c, &inv_c);
        par_rows(oblk, c * eb, min_rows(c), |s0, blk| {
            let mut rows = Rows::new(dt, xb, x_off, c);
            let mut tmp = vec![0.0; c];
            for (i, dst) in blk.chunks_exact_mut(c * eb).enumerate() {
                let xr = rows.get(base + (s0 + i) * c, c);
                out_row(dt, dst, &mut tmp, |o| {
                    norm_vec(xr, mean_c, inv_c, &wv, &bv, o);
                    if silu {
                        silu_inplace(o);
                    }
                });
            }
        });
    }
    Ok(())
}

pub fn pixel_norm(
    x: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    c: usize,
    eps: f32,
    silu: bool,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    let d = x_lo.dims();
    if d.len() < 2 || d[1] != c || c == 0 || o_lo.dims() != d {
        return Err(SynaptixError::Unsupported("cpu pixel_norm: shape"));
    }
    let b = d[0];
    let s = d[2..].iter().product::<usize>().max(1);
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu pixel_norm: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu pixel_norm: out")?;
    let eb = elem_bytes(dt);
    let optr = SendPtr::new(ob.as_mut_ptr());
    let chunk = 1024usize;
    for bi in 0..b {
        let base = bi * c * s;
        crate::common::par_chunks(s, chunk, |s0, s1| {
            let len = s1 - s0;
            let mut buf = vec![0.0f32; len];
            let mut ss = vec![0.0f32; len];
            let mut tmp = vec![0.0f32; len];
            for ch in 0..c {
                row_to_f32(dt, xb, x_off + base + ch * s + s0, &mut buf);
                for (a, &v) in ss.iter_mut().zip(&buf) {
                    *a += v * v;
                }
            }
            for v in &mut ss {
                *v = 1.0 / (*v / c as f32 + eps).sqrt();
            }
            for ch in 0..c {
                let off = base + ch * s + s0;
                row_to_f32(dt, xb, x_off + off, &mut buf);
                let dst = unsafe { std::slice::from_raw_parts_mut(optr.get().add(off * eb), len * eb) };
                out_row(dt, dst, &mut tmp, |o| {
                    scale_vec(&buf, &ss, o);
                    if silu {
                        silu_inplace(o);
                    }
                });
            }
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    use synaptix_core::dtype::DType;
    use synaptix_core::tensor::layout::Layout;
    use synaptix_core::tensor::shape::Shape;
    use synaptix_core::tensor::storage::{CpuBuf, Storage};

    pub(crate) const DTS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

    pub(crate) fn tol(dt: DType) -> f32 {
        match dt {
            DType::F32 => 1e-5,
            DType::F16 => 1e-2,
            _ => 2e-2,
        }
    }

    pub(crate) fn round(dt: DType, v: f32) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    pub(crate) fn gen(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    pub(crate) fn tensor(dt: DType, dims: &[usize], data: &[f32]) -> (Storage, Layout, Vec<f32>) {
        assert_eq!(dims.iter().product::<usize>(), data.len());
        let rounded: Vec<f32> = data.iter().map(|&v| round(dt, v)).collect();
        let mut bytes = vec![0u8; dt.bytes_for_numel(data.len())];
        crate::common::from_f32(dt, &rounded, &mut bytes, 0).unwrap();
        (Storage::Cpu(CpuBuf::from_vec(bytes)), Layout::contiguous(Shape::new(dims.to_vec()), dt), rounded)
    }

    pub(crate) fn empty(dt: DType, dims: &[usize]) -> (Storage, Layout) {
        let n = dims.iter().product::<usize>();
        (
            Storage::Cpu(CpuBuf::alloc_zeros(dt.bytes_for_numel(n))),
            Layout::contiguous(Shape::new(dims.to_vec()), dt),
        )
    }

    pub(crate) fn read(st: &Storage, lo: &Layout) -> Vec<f32> {
        let mut v = vec![0.0; lo.numel()];
        crate::common::to_f32(lo.dtype(), st.as_cpu().unwrap().as_bytes(), lo.offset(), &mut v).unwrap();
        v
    }

    pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: len");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            assert!(err <= tol * (1.0 + w.abs()), "{what}[{i}]: got {g}, want {w}");
        }
    }

    pub(crate) fn stream() -> synaptix_core::stream::Stream {
        synaptix_core::stream::Stream::default_for(synaptix_core::device::Device::Cpu).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    fn ref_rms(x: &[f32], w: &[f32], h: usize, eps: f32, qwen: bool) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        for r in 0..x.len() / h {
            let row = &x[r * h..(r + 1) * h];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..h {
                let g = if qwen { w[i] + 1.0 } else { w[i] };
                out[r * h + i] = row[i] * inv * g;
            }
        }
        out
    }

    #[test]
    fn rms_norm_matches_reference() {
        for dt in DTS {
            for (dims, qwen) in [(vec![3usize, 7], false), (vec![2, 5, 33], true), (vec![4, 1024], false), (vec![1, 1], true)] {
                let h = *dims.last().unwrap();
                let n: usize = dims.iter().product();
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 1, 3.0));
                let (ws, wl, wv) = tensor(dt, &[h], &gen(h, 2, 1.0));
                let (mut os, ol) = empty(dt, &dims);
                rms_norm((&xs, &xl), (&ws, &wl), (&mut os, &ol), 1e-5, qwen, &stream()).unwrap();
                assert_close(&read(&os, &ol), &ref_rms(&xv, &wv, h, 1e-5, qwen), tol(dt), "rms_norm");
            }
        }
    }

    #[test]
    fn rms_norm_residual_matches_reference() {
        for dt in DTS {
            for dims in [vec![3usize, 7], vec![2, 5, 33], vec![9, 300]] {
                let h = *dims.last().unwrap();
                let n: usize = dims.iter().product();
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 3, 2.0));
                let (rs, rl, rv) = tensor(dt, &dims, &gen(n, 4, 2.0));
                let (ws, wl, wv) = tensor(dt, &[h], &gen(h, 5, 1.0));
                let (mut hs, hl) = empty(dt, &dims);
                let (mut ys, yl) = empty(dt, &dims);
                rms_norm_residual((&xs, &xl), (&rs, &rl), (&ws, &wl), (&mut hs, &hl), (&mut ys, &yl), 1e-6, true, &stream())
                    .unwrap();
                let hid: Vec<f32> = xv.iter().zip(&rv).map(|(a, b)| round(dt, a + b)).collect();
                assert_close(&read(&hs, &hl), &hid, tol(dt), "hidden");
                assert_close(&read(&ys, &yl), &ref_rms(&hid, &wv, h, 1e-6, true), tol(dt), "y");
            }
        }
    }

    #[test]
    fn layer_norm_matches_reference() {
        for dt in DTS {
            for (dims, with_bias) in [(vec![3usize, 7], true), (vec![2, 5, 33], false), (vec![6, 513], true)] {
                let h = *dims.last().unwrap();
                let n: usize = dims.iter().product();
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 6, 4.0));
                let (ws, wl, wv) = tensor(dt, &[h], &gen(h, 7, 1.0));
                let (bs, bl, bv) = tensor(dt, &[h], &gen(h, 8, 0.5));
                let (mut os, ol) = empty(dt, &dims);
                let bias = if with_bias { Some((&bs, &bl)) } else { None };
                layer_norm((&xs, &xl), (&ws, &wl), bias, (&mut os, &ol), 1e-5, &stream()).unwrap();
                let mut want = vec![0.0; n];
                for r in 0..n / h {
                    let row = &xv[r * h..(r + 1) * h];
                    let mean = row.iter().sum::<f32>() / h as f32;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
                    let inv = 1.0 / (var + 1e-5).sqrt();
                    for i in 0..h {
                        want[r * h + i] = (row[i] - mean) * inv * wv[i] + if with_bias { bv[i] } else { 0.0 };
                    }
                }
                assert_close(&read(&os, &ol), &want, tol(dt), "layer_norm");
            }
        }
    }

    #[test]
    fn group_rms_matches_reference() {
        for dt in DTS {
            let (rows, groups, group) = (5usize, 3usize, 11usize);
            let width = groups * group;
            let (xs, xl, xv) = tensor(dt, &[rows, width], &gen(rows * width, 9, 2.0));
            let (ws, wl, wv) = tensor(dt, &[width], &gen(width, 10, 1.0));
            let (mut os, ol) = empty(dt, &[rows, width]);
            group_rms((&xs, &xl), (&ws, &wl), (&mut os, &ol), rows, groups, group, 1e-6, &stream()).unwrap();
            let mut want = vec![0.0; rows * width];
            for r in 0..rows {
                for g in 0..groups {
                    let seg = &xv[r * width + g * group..r * width + (g + 1) * group];
                    let inv = 1.0 / (seg.iter().map(|v| v * v).sum::<f32>() / group as f32 + 1e-6).sqrt();
                    for i in 0..group {
                        want[r * width + g * group + i] = seg[i] * inv * wv[g * group + i];
                    }
                }
            }
            assert_close(&read(&os, &ol), &want, tol(dt), "group_rms");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ref_group_norm(
        x: &[f32],
        dims: &[usize],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        ng: usize,
        eps: f32,
        silu: bool,
        nhwc: bool,
    ) -> Vec<f32> {
        let bsz = dims[0];
        let (c, hw) = if nhwc {
            (dims[dims.len() - 1], dims[1..dims.len() - 1].iter().product::<usize>())
        } else {
            (dims[1], dims[2..].iter().product::<usize>())
        };
        let pg = c / ng;
        let idx = |bi: usize, ch: usize, s: usize| if nhwc { (bi * hw + s) * c + ch } else { (bi * c + ch) * hw + s };
        let mut out = vec![0.0; x.len()];
        for bi in 0..bsz {
            for g in 0..ng {
                let mut vals = Vec::new();
                for ch in g * pg..(g + 1) * pg {
                    for s in 0..hw {
                        vals.push(x[idx(bi, ch, s)]);
                    }
                }
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for ch in g * pg..(g + 1) * pg {
                    for s in 0..hw {
                        let mut v = (x[idx(bi, ch, s)] - mean) * inv;
                        if let Some(w) = w {
                            v = v * w[ch] + b.map_or(0.0, |b| b[ch]);
                        }
                        if silu {
                            v = v / (1.0 + (-v).exp());
                        }
                        out[idx(bi, ch, s)] = v;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn group_norm_matches_reference() {
        for dt in DTS {
            for (dims, ng, nhwc, affine, with_bias, silu) in [
                (vec![2usize, 6, 5, 3], 3usize, false, true, true, false),
                (vec![1, 8, 7], 4, false, true, false, true),
                (vec![2, 4, 9], 2, false, false, false, true),
                (vec![2, 5, 3, 6], 3, true, true, true, true),
                (vec![1, 7, 8], 4, true, true, false, false),
                (vec![2, 6], 2, true, false, false, false),
                (vec![3, 6], 3, false, true, true, false),
            ] {
                let n: usize = dims.iter().product();
                let c = if nhwc { dims[dims.len() - 1] } else { dims[1] };
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 11, 3.0));
                let (ws, wl, wv) = tensor(dt, &[c], &gen(c, 12, 1.0));
                let (bs, bl, bv) = tensor(dt, &[c], &gen(c, 13, 0.5));
                let (mut os, ol) = empty(dt, &dims);
                let w = if affine { Some((&ws, &wl)) } else { None };
                let b = if with_bias { Some((&bs, &bl)) } else { None };
                group_norm((&xs, &xl), w, b, (&mut os, &ol), ng, 1e-5, silu, nhwc, &stream()).unwrap();
                let want = ref_group_norm(
                    &xv,
                    &dims,
                    affine.then_some(&wv[..]),
                    with_bias.then_some(&bv[..]),
                    ng,
                    1e-5,
                    silu,
                    nhwc,
                );
                assert_close(&read(&os, &ol), &want, tol(dt), &format!("group_norm {dims:?} nhwc={nhwc}"));
            }
        }
    }

    #[test]
    fn group_norm_rejects_bias_without_weight() {
        let dt = DType::F32;
        let (xs, xl, _) = tensor(dt, &[1, 4, 3], &gen(12, 1, 1.0));
        let (bs, bl, _) = tensor(dt, &[4], &gen(4, 2, 1.0));
        let (mut os, ol) = empty(dt, &[1, 4, 3]);
        let r = group_norm((&xs, &xl), None, Some((&bs, &bl)), (&mut os, &ol), 2, 1e-5, false, false, &stream());
        assert!(matches!(r, Err(SynaptixError::Unsupported(_))));
    }

    #[test]
    fn pixel_norm_matches_reference() {
        for dt in DTS {
            for (dims, silu) in [(vec![2usize, 5, 7], false), (vec![1, 3, 4, 5], true), (vec![2, 6, 1100], true)] {
                let n: usize = dims.iter().product();
                let c = dims[1];
                let s: usize = dims[2..].iter().product();
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 14, 2.0));
                let (mut os, ol) = empty(dt, &dims);
                pixel_norm((&xs, &xl), (&mut os, &ol), c, 1e-6, silu, &stream()).unwrap();
                let mut want = vec![0.0; n];
                for bi in 0..dims[0] {
                    for p in 0..s {
                        let ss = (0..c).map(|ch| xv[(bi * c + ch) * s + p].powi(2)).sum::<f32>() / c as f32;
                        let inv = 1.0 / (ss + 1e-6).sqrt();
                        for ch in 0..c {
                            let mut v = xv[(bi * c + ch) * s + p] * inv;
                            if silu {
                                v = v / (1.0 + (-v).exp());
                            }
                            want[(bi * c + ch) * s + p] = v;
                        }
                    }
                }
                assert_close(&read(&os, &ol), &want, tol(dt), "pixel_norm");
            }
        }
    }

    #[test]
    fn rejects_non_contiguous_and_bad_shapes() {
        let dt = DType::F32;
        let (xs, xl, _) = tensor(dt, &[4, 6], &gen(24, 1, 1.0));
        let (mut os, ol) = empty(dt, &[4, 6]);
        let xt = xl.transpose(0, 1).unwrap();
        let (mut ot, olt) = empty(dt, &[6, 4]);
        let (ws4, wl4, _) = tensor(dt, &[4], &gen(4, 2, 1.0));
        assert!(matches!(
            rms_norm((&xs, &xt), (&ws4, &wl4), (&mut ot, &olt), 1e-5, false, &stream()),
            Err(SynaptixError::NonContiguous)
        ));
        let (ws2, wl2, _) = tensor(dt, &[5], &gen(5, 2, 1.0));
        assert!(matches!(
            rms_norm((&xs, &xl), (&ws2, &wl2), (&mut os, &ol), 1e-5, false, &stream()),
            Err(SynaptixError::Unsupported(_))
        ));
    }
}
