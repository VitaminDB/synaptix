use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;
use wide::f32x8;

use crate::common::{cpu, cpu_mut, elem_bytes, row_to_f32, same_dtype};
use crate::norms::{
    check_float, dense_offset, in_bytes, ld, min_rows, out_bytes, out_row, par_rows, roundtrip, sigmoid8, simd_fn, st,
    to_f32_vec, Rows,
};

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

#[inline]
fn gelu_exact_f32(x: f32) -> f32 {
    let inv_sqrt2 = (1.0 / (2.0_f64).sqrt()) as f32;
    let e = erf_f64((x * inv_sqrt2) as f64) as f32;
    0.5 * x * (1.0 + e)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn silu_mul_body(g: &[f32], u: &[f32], o: &mut [f32]) {
    let n = o.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let gv = ld(&g[i..i + 8]);
        st(gv * sigmoid8(gv) * ld(&u[i..i + 8]), &mut o[i..i + 8]);
    }
    for i in main..n {
        o[i] = crate::common::silu_f32(g[i]) * u[i];
    }
}

#[inline(always)]
fn scale_act_body(x: &[f32], scale: f32, act: u32, o: &mut [f32]) {
    let n = o.len();
    let main = n - n % 8;
    let sc = f32x8::splat(scale);
    let two = f32x8::splat(2.0);
    for i in (0..main).step_by(8) {
        let v = ld(&x[i..i + 8]) * sc;
        let s = sigmoid8(v);
        let r = match act {
            0 => v * s,
            1 => s,
            _ => two * s,
        };
        st(r, &mut o[i..i + 8]);
    }
    for i in main..n {
        let v = x[i] * scale;
        let s = sigmoid_f32(v);
        o[i] = match act {
            0 => v * s,
            1 => s,
            _ => 2.0 * s,
        };
    }
}

#[inline(always)]
fn sigmoid_mul_acc_body(u: &[f32], nv: &[f32], acc: &mut [f32]) {
    let n = acc.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let a = ld(&acc[i..i + 8]);
        let s = sigmoid8(ld(&u[i..i + 8]));
        st(s.mul_add(ld(&nv[i..i + 8]), a), &mut acc[i..i + 8]);
    }
    for i in main..n {
        acc[i] += sigmoid_f32(u[i]) * nv[i];
    }
}

#[inline(always)]
fn axpy_body(a: f32, x: &[f32], y: &mut [f32]) {
    let n = y.len();
    let main = n - n % 8;
    let av = f32x8::splat(a);
    for i in (0..main).step_by(8) {
        st(av.mul_add(ld(&x[i..i + 8]), ld(&y[i..i + 8])), &mut y[i..i + 8]);
    }
    for i in main..n {
        y[i] += a * x[i];
    }
}

#[inline(always)]
fn mul_body(a: &[f32], b: &[f32], o: &mut [f32]) {
    let n = o.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        st(ld(&a[i..i + 8]) * ld(&b[i..i + 8]), &mut o[i..i + 8]);
    }
    for i in main..n {
        o[i] = a[i] * b[i];
    }
}

#[inline(always)]
fn add_body(a: &[f32], b: &[f32], o: &mut [f32]) {
    let n = o.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        st(ld(&a[i..i + 8]) + ld(&b[i..i + 8]), &mut o[i..i + 8]);
    }
    for i in main..n {
        o[i] = a[i] + b[i];
    }
}

simd_fn!(pub(crate) fn silu_mul(g: &[f32], u: &[f32], o: &mut [f32]) = silu_mul_body);
simd_fn!(pub(crate) fn scale_act_row(x: &[f32], scale: f32, act: u32, o: &mut [f32]) = scale_act_body);
simd_fn!(pub(crate) fn sigmoid_mul_acc(u: &[f32], nv: &[f32], acc: &mut [f32]) = sigmoid_mul_acc_body);
simd_fn!(pub(crate) fn axpy(a: f32, x: &[f32], y: &mut [f32]) = axpy_body);
simd_fn!(pub(crate) fn mul(a: &[f32], b: &[f32], o: &mut [f32]) = mul_body);
simd_fn!(pub(crate) fn add(a: &[f32], b: &[f32], o: &mut [f32]) = add_body);

const ELEM_CHUNK: usize = 4096;

pub fn silu_and_mul(
    gate: (&Storage, &Layout),
    up: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (g_st, g_lo) = gate;
    let (u_st, u_lo) = up;
    let (o_st, o_lo) = out;
    let dt = g_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, u_lo.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    if g_lo.dims() != u_lo.dims() || g_lo.dims() != o_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu silu_and_mul: shape"));
    }
    let (gb, g_off) = in_bytes(g_st, g_lo, "cpu silu_and_mul: gate")?;
    let (ub, u_off) = in_bytes(u_st, u_lo, "cpu silu_and_mul: up")?;
    let ob = out_bytes(o_st, o_lo, "cpu silu_and_mul: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, eb, ELEM_CHUNK, |e0, blk| {
        let m = blk.len() / eb;
        let mut gr = Rows::new(dt, gb, g_off, m);
        let mut ur = Rows::new(dt, ub, u_off, m);
        let mut tmp = vec![0.0; m];
        let g = gr.get(e0, m);
        let u = ur.get(e0, m);
        out_row(dt, blk, &mut tmp, |o| silu_mul(g, u, o));
    });
    Ok(())
}

pub fn geglu_split(inp: (&Storage, &Layout), out: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
    let (i_st, i_lo) = inp;
    let (o_st, o_lo) = out;
    let dt = i_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    let d = i_lo.dims();
    let r = d.len();
    if r == 0 || o_lo.dims().len() != r || !d[r - 1].is_multiple_of(2) || d[r - 1] == 0 {
        return Err(SynaptixError::Unsupported("cpu geglu_split: shape"));
    }
    let inner = d[r - 1] / 2;
    if o_lo.dims()[..r - 1] != d[..r - 1] || o_lo.dims()[r - 1] != inner {
        return Err(SynaptixError::Unsupported("cpu geglu_split: out shape"));
    }
    let (ib, i_off) = in_bytes(i_st, i_lo, "cpu geglu_split: inp")?;
    let ob = out_bytes(o_st, o_lo, "cpu geglu_split: out")?;
    let eb = elem_bytes(dt);
    let w = 2 * inner;
    par_rows(ob, inner * eb, min_rows(inner), |r0, blk| {
        let mut rows = Rows::new(dt, ib, i_off, w);
        let mut tmp = vec![0.0; inner];
        for (i, dst) in blk.chunks_exact_mut(inner * eb).enumerate() {
            let xr = rows.get((r0 + i) * w, w);
            out_row(dt, dst, &mut tmp, |o| {
                for (j, o) in o.iter_mut().enumerate() {
                    *o = xr[j] * gelu_exact_f32(xr[inner + j]);
                }
            });
        }
    });
    Ok(())
}

pub fn scale_act(
    x: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    n: usize,
    scale: f32,
    act: u32,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    if act > 2 {
        return Err(SynaptixError::Unsupported("cpu scale_act: act"));
    }
    if n > x_lo.numel() || n > o_lo.numel() {
        return Err(SynaptixError::Unsupported("cpu scale_act: n"));
    }
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu scale_act: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu scale_act: out")?;
    let eb = elem_bytes(dt);
    par_rows(&mut ob[..n * eb], eb, ELEM_CHUNK, |e0, blk| {
        let m = blk.len() / eb;
        let mut xr = Rows::new(dt, xb, x_off, m);
        let mut tmp = vec![0.0; m];
        let xs = xr.get(e0, m);
        out_row(dt, blk, &mut tmp, |o| scale_act_row(xs, scale, act, o));
    });
    Ok(())
}

pub fn ternary_fused(
    kind: u8,
    x: (&Storage, &Layout),
    b: (&Storage, &Layout),
    c: (&Storage, &Layout),
    dst: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (b_st, b_lo) = b;
    let (c_st, c_lo) = c;
    let (d_st, d_lo) = dst;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, b_lo.dtype())?;
    same_dtype(dt, c_lo.dtype())?;
    same_dtype(dt, d_lo.dtype())?;
    if x_lo.dims() != d_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu ternary_fused: x/dst shape"));
    }
    let d = *d_lo.dims().last().unwrap_or(&0);
    if d == 0 {
        return Err(SynaptixError::Unsupported("cpu ternary_fused: scalar"));
    }
    let full = |lo: &Layout| lo.dims() == d_lo.dims();
    let row = |lo: &Layout| lo.numel() == d;
    let (b_full, c_full) = match kind {
        0 => (true, true),
        1 => (true, false),
        2 => (false, false),
        3 => (true, true),
        _ => return Err(SynaptixError::Unsupported("cpu ternary_fused: kind")),
    };
    if (b_full && !full(b_lo)) || (!b_full && !row(b_lo)) || (c_full && !full(c_lo)) || (!c_full && !row(c_lo)) {
        return Err(SynaptixError::Unsupported("cpu ternary_fused: b/c shape"));
    }
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu ternary_fused: x")?;
    let (bb, b_off) = in_bytes(b_st, b_lo, "cpu ternary_fused: b")?;
    let (cb, c_off) = in_bytes(c_st, c_lo, "cpu ternary_fused: c")?;
    let ob = out_bytes(d_st, d_lo, "cpu ternary_fused: dst")?;
    let eb = elem_bytes(dt);
    let modulate = kind >= 2;
    let mut shared_b = Vec::new();
    let mut shared_c = Vec::new();
    if !b_full {
        shared_b = to_f32_vec(b_st, b_lo, "cpu ternary_fused: b")?;
        if modulate {
            for v in &mut shared_b {
                *v += 1.0;
            }
            roundtrip(dt, &mut shared_b, &mut Vec::new());
        }
    }
    if !c_full {
        shared_c = to_f32_vec(c_st, c_lo, "cpu ternary_fused: c")?;
    }
    par_rows(ob, d * eb, min_rows(d), |r0, blk| {
        let mut xr = Rows::new(dt, xb, x_off, d);
        let mut br = Rows::new(dt, bb, b_off, d);
        let mut cr = Rows::new(dt, cb, c_off, d);
        let mut tmp = vec![0.0; d];
        let mut t1 = vec![0.0; d];
        let mut s1 = vec![0.0; d];
        let mut scratch = Vec::new();
        for (i, dst) in blk.chunks_exact_mut(d * eb).enumerate() {
            let r = (r0 + i) * d;
            let x = xr.get(r, d);
            let bv = if b_full { br.get(r, d) } else { &shared_b[..] };
            let cv = if c_full { cr.get(r, d) } else { &shared_c[..] };
            if modulate {
                let s = if b_full {
                    for (s, &b) in s1.iter_mut().zip(bv) {
                        *s = 1.0 + b;
                    }
                    roundtrip(dt, &mut s1, &mut scratch);
                    &s1[..]
                } else {
                    bv
                };
                mul(x, s, &mut t1);
                roundtrip(dt, &mut t1, &mut scratch);
                out_row(dt, dst, &mut tmp, |o| add(&t1, cv, o));
            } else {
                mul(bv, cv, &mut t1);
                roundtrip(dt, &mut t1, &mut scratch);
                out_row(dt, dst, &mut tmp, |o| add(x, &t1, o));
            }
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn snake(
    x: (&Storage, &Layout),
    a: (&Storage, &Layout),
    binv: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    c: usize,
    t_len: usize,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    let n = x_lo.numel();
    if c == 0 || t_len == 0 || !n.is_multiple_of(c * t_len) || o_lo.numel() != n {
        return Err(SynaptixError::Unsupported("cpu snake: shape"));
    }
    if a.1.numel() != c || binv.1.numel() != c {
        return Err(SynaptixError::Unsupported("cpu snake: a/binv must be [C]"));
    }
    let av = to_f32_vec(a.0, a.1, "cpu snake: a")?;
    let bv = to_f32_vec(binv.0, binv.1, "cpu snake: binv")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu snake: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu snake: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, t_len * eb, min_rows(t_len), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, t_len);
        let mut tmp = vec![0.0; t_len];
        for (i, dst) in blk.chunks_exact_mut(t_len * eb).enumerate() {
            let r = r0 + i;
            let (ac, bc) = (av[r % c], bv[r % c]);
            let xr = rows.get(r * t_len, t_len);
            out_row(dt, dst, &mut tmp, |o| {
                for (o, &v) in o.iter_mut().zip(xr) {
                    let s = (ac * v).sin();
                    *o = v + s * s * bc;
                }
            });
        }
    });
    Ok(())
}

pub fn linear_epilogue(
    x: (&Storage, &Layout),
    w: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    residual: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (w_st, w_lo) = w;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, w_lo.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    let xr = x_lo.dims().len();
    if xr < 1 || w_lo.dims().len() != 2 {
        return Err(SynaptixError::Unsupported("cpu linear_epilogue: rank"));
    }
    let (n, k) = (w_lo.dims()[0], w_lo.dims()[1]);
    if x_lo.dims()[xr - 1] != k || o_lo.dims()[..xr - 1] != x_lo.dims()[..xr - 1] || o_lo.dims()[xr - 1] != n || n == 0 {
        return Err(SynaptixError::Unsupported("cpu linear_epilogue: shape"));
    }
    dense_offset(x_lo)?;
    dense_offset(w_lo)?;
    dense_offset(o_lo)?;
    let bv = match bias {
        Some((b_st, b_lo)) => {
            if b_lo.numel() != n {
                return Err(SynaptixError::Unsupported("cpu linear_epilogue: bias shape"));
            }
            Some(to_f32_vec(b_st, b_lo, "cpu linear_epilogue: bias")?)
        }
        None => None,
    };
    let res = match residual {
        Some((r_st, r_lo)) => {
            if r_lo.dims() != o_lo.dims() {
                return Err(SynaptixError::Unsupported("cpu linear_epilogue: residual shape"));
            }
            check_float(r_lo.dtype())?;
            let (rb, r_off) = in_bytes(r_st, r_lo, "cpu linear_epilogue: residual")?;
            Some((rb, r_off, r_lo.dtype()))
        }
        None => None,
    };
    let x_buf = cpu(x_st, "cpu linear_epilogue: x")?;
    let w_buf = cpu(w_st, "cpu linear_epilogue: w")?;
    let o_buf = cpu_mut(o_st, "cpu linear_epilogue: out")?;
    crate::gemm::linear_dispatch(x_buf, x_lo, w_buf, w_lo, o_buf, o_lo)?;
    if bv.is_none() && res.is_none() {
        return Ok(());
    }
    let ob = out_bytes(o_st, o_lo, "cpu linear_epilogue: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, n * eb, min_rows(n), |r0, blk| {
        let mut rrows = res.map(|(rb, r_off, rdt)| Rows::new(rdt, rb, r_off, n));
        let mut cur = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        for (i, dst) in blk.chunks_exact_mut(n * eb).enumerate() {
            row_to_f32(dt, dst, 0, &mut cur);
            if let Some(b) = &bv {
                for (c, &b) in cur.iter_mut().zip(b) {
                    *c += b;
                }
            }
            if let Some(rr) = rrows.as_mut() {
                let r = rr.get((r0 + i) * n, n);
                for (c, &v) in cur.iter_mut().zip(r) {
                    *c += v;
                }
            }
            out_row(dt, dst, &mut tmp, |o| o.copy_from_slice(&cur));
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn hc_mix(
    up: (&Storage, &Layout),
    normed: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    rows: usize,
    hc: usize,
    h: usize,
    _stream: &Stream,
) -> Result<()> {
    let (u_st, u_lo) = up;
    let (n_st, n_lo) = normed;
    let (o_st, o_lo) = out;
    let dt = u_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, n_lo.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    let width = hc * h;
    if hc == 0 || h == 0 || u_lo.numel() != rows * width || n_lo.numel() != rows * width || o_lo.numel() != rows * h {
        return Err(SynaptixError::Unsupported("cpu hc_mix: shape"));
    }
    let (ub, u_off) = in_bytes(u_st, u_lo, "cpu hc_mix: up")?;
    let (nb, n_off) = in_bytes(n_st, n_lo, "cpu hc_mix: normed")?;
    let ob = out_bytes(o_st, o_lo, "cpu hc_mix: out")?;
    let eb = elem_bytes(dt);
    let inv_hc = 1.0 / hc as f32;
    par_rows(ob, h * eb, min_rows(width), |r0, blk| {
        let mut ur = Rows::new(dt, ub, u_off, width);
        let mut nr = Rows::new(dt, nb, n_off, width);
        let mut acc = vec![0.0; h];
        let mut tmp = vec![0.0; h];
        for (i, dst) in blk.chunks_exact_mut(h * eb).enumerate() {
            let r = (r0 + i) * width;
            let u = ur.get(r, width);
            let nv = nr.get(r, width);
            acc.fill(0.0);
            for c in 0..hc {
                sigmoid_mul_acc(&u[c * h..(c + 1) * h], &nv[c * h..(c + 1) * h], &mut acc);
            }
            out_row(dt, dst, &mut tmp, |o| {
                for (o, &a) in o.iter_mut().zip(&acc) {
                    *o = a * inv_hc;
                }
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn hc_inject(
    hyper: (&Storage, &Layout),
    block: (&Storage, &Layout),
    w: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    rows: usize,
    hc: usize,
    h: usize,
    _stream: &Stream,
) -> Result<()> {
    let (h_st, h_lo) = hyper;
    let (b_st, b_lo) = block;
    let (o_st, o_lo) = out;
    let dt = h_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, b_lo.dtype())?;
    same_dtype(dt, o_lo.dtype())?;
    check_float(w.1.dtype())?;
    let width = hc * h;
    if hc == 0
        || h == 0
        || h_lo.numel() != rows * width
        || o_lo.numel() != rows * width
        || b_lo.numel() != rows * h
        || w.1.numel() != rows * hc
    {
        return Err(SynaptixError::Unsupported("cpu hc_inject: shape"));
    }
    let wv = to_f32_vec(w.0, w.1, "cpu hc_inject: w")?;
    let (hb, h_off) = in_bytes(h_st, h_lo, "cpu hc_inject: hyper")?;
    let (bb, b_off) = in_bytes(b_st, b_lo, "cpu hc_inject: block")?;
    let ob = out_bytes(o_st, o_lo, "cpu hc_inject: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, width * eb, min_rows(width), |r0, blk| {
        let mut hr = Rows::new(dt, hb, h_off, width);
        let mut br = Rows::new(dt, bb, b_off, h);
        let mut tmp = vec![0.0; width];
        for (i, dst) in blk.chunks_exact_mut(width * eb).enumerate() {
            let r = r0 + i;
            let hy = hr.get(r * width, width);
            let bl = br.get(r * h, h);
            out_row(dt, dst, &mut tmp, |o| {
                for c in 0..hc {
                    let seg = c * h..(c + 1) * h;
                    o[seg.clone()].copy_from_slice(&hy[seg]);
                    axpy(wv[r * hc + c], bl, &mut o[c * h..(c + 1) * h]);
                }
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn weighted_rows_sum(
    parts: (&Storage, &Layout),
    w: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    t: usize,
    k: usize,
    h: usize,
    _stream: &Stream,
) -> Result<()> {
    let (p_st, p_lo) = parts;
    let (o_st, o_lo) = out;
    let dt = p_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    check_float(w.1.dtype())?;
    if k == 0 || h == 0 || p_lo.numel() != t * k * h || o_lo.numel() != t * h || w.1.numel() != t * k {
        return Err(SynaptixError::Unsupported("cpu weighted_rows_sum: shape"));
    }
    let wv = to_f32_vec(w.0, w.1, "cpu weighted_rows_sum: w")?;
    let (pb, p_off) = in_bytes(p_st, p_lo, "cpu weighted_rows_sum: parts")?;
    let ob = out_bytes(o_st, o_lo, "cpu weighted_rows_sum: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, h * eb, min_rows(h * k), |r0, blk| {
        let mut pr = Rows::new(dt, pb, p_off, h);
        let mut tmp = vec![0.0; h];
        for (i, dst) in blk.chunks_exact_mut(h * eb).enumerate() {
            let row = r0 + i;
            out_row(dt, dst, &mut tmp, |o| {
                o.fill(0.0);
                for j in 0..k {
                    let p = row * k + j;
                    axpy(wv[p], pr.get(p * h, h), o);
                }
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn moe_combine(
    y: (&Storage, &Layout),
    inv: (&Storage, &Layout),
    w: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    t: usize,
    k: usize,
    d: usize,
    _stream: &Stream,
) -> Result<()> {
    let (y_st, y_lo) = y;
    let (i_st, i_lo) = inv;
    let (o_st, o_lo) = out;
    let dt = y_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    check_float(w.1.dtype())?;
    if i_lo.dtype() != DType::U32 {
        return Err(SynaptixError::Unsupported("cpu moe_combine: inv must be U32"));
    }
    if k == 0 || d == 0 || !y_lo.numel().is_multiple_of(d) || o_lo.numel() != t * d || i_lo.numel() != t * k || w.1.numel() != t * k {
        return Err(SynaptixError::Unsupported("cpu moe_combine: shape"));
    }
    let rows = y_lo.numel() / d;
    let wv = to_f32_vec(w.0, w.1, "cpu moe_combine: w")?;
    let i_off = dense_offset(i_lo)?;
    let ib = cpu(i_st, "cpu moe_combine: inv")?.as_bytes();
    let iv: Vec<usize> = (0..t * k)
        .map(|i| {
            let o = (i_off + i) * 4;
            u32::from_ne_bytes([ib[o], ib[o + 1], ib[o + 2], ib[o + 3]]) as usize
        })
        .collect();
    if iv.iter().any(|&r| r >= rows) {
        return Err(SynaptixError::Unsupported("cpu moe_combine: inv out of range"));
    }
    let (yb, y_off) = in_bytes(y_st, y_lo, "cpu moe_combine: y")?;
    let ob = out_bytes(o_st, o_lo, "cpu moe_combine: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, d * eb, min_rows(d * k), |r0, blk| {
        let mut yr = Rows::new(dt, yb, y_off, d);
        let mut tmp = vec![0.0; d];
        for (i, dst) in blk.chunks_exact_mut(d * eb).enumerate() {
            let row = r0 + i;
            out_row(dt, dst, &mut tmp, |o| {
                o.fill(0.0);
                for s in 0..k {
                    let p = row * k + s;
                    axpy(wv[p], yr.get(iv[p] * d, d), o);
                }
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norms::testutil::*;
    use synaptix_core::tensor::shape::Shape;
    use synaptix_core::tensor::storage::CpuBuf;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn silu_and_mul_matches_reference() {
        for dt in DTS {
            for dims in [vec![3usize, 7], vec![2, 5, 1031], vec![1, 9000]] {
                let n: usize = dims.iter().product();
                let (gs, gl, gv) = tensor(dt, &dims, &gen(n, 1, 6.0));
                let (us, ul, uv) = tensor(dt, &dims, &gen(n, 2, 2.0));
                let (mut os, ol) = empty(dt, &dims);
                silu_and_mul((&gs, &gl), (&us, &ul), (&mut os, &ol), &stream()).unwrap();
                let want: Vec<f32> = gv.iter().zip(&uv).map(|(&g, &u)| silu(g) * u).collect();
                assert_close(&read(&os, &ol), &want, tol(dt), "silu_and_mul");
            }
        }
    }

    #[test]
    fn silu_extremes_are_finite() {
        let dt = DType::F32;
        let data = [-200.0f32, -90.0, -30.0, 0.0, 30.0, 90.0, 200.0, 1.0];
        let (gs, gl, _) = tensor(dt, &[8], &data);
        let (us, ul, _) = tensor(dt, &[8], &[1.0; 8]);
        let (mut os, ol) = empty(dt, &[8]);
        silu_and_mul((&gs, &gl), (&us, &ul), (&mut os, &ol), &stream()).unwrap();
        let want: Vec<f32> = data.iter().map(|&g| silu(g)).collect();
        assert_close(&read(&os, &ol), &want, 1e-5, "silu extremes");
    }

    #[test]
    fn geglu_split_matches_reference() {
        for dt in DTS {
            for dims in [vec![3usize, 14], vec![2, 3, 66]] {
                let n: usize = dims.iter().product();
                let inner = dims[dims.len() - 1] / 2;
                let (is, il, iv) = tensor(dt, &dims, &gen(n, 3, 3.0));
                let mut od = dims.clone();
                *od.last_mut().unwrap() = inner;
                let (mut os, ol) = empty(dt, &od);
                geglu_split((&is, &il), (&mut os, &ol), &stream()).unwrap();
                let mut want = vec![0.0; n / 2];
                for r in 0..n / (2 * inner) {
                    for i in 0..inner {
                        let a = iv[r * 2 * inner + i];
                        let b = iv[r * 2 * inner + inner + i];
                        let e = erf_f64((b as f64) / 2f64.sqrt()) as f32;
                        want[r * inner + i] = a * (0.5 * b * (1.0 + e));
                    }
                }
                assert_close(&read(&os, &ol), &want, tol(dt), "geglu_split");
            }
        }
    }

    #[test]
    fn scale_act_matches_reference() {
        for dt in DTS {
            for act in 0..3u32 {
                let n = 5003usize;
                let (xs, xl, xv) = tensor(dt, &[n], &gen(n, 4, 4.0));
                let (mut os, ol) = empty(dt, &[n]);
                scale_act((&xs, &xl), (&mut os, &ol), n, 0.7, act, &stream()).unwrap();
                let want: Vec<f32> = xv
                    .iter()
                    .map(|&x| {
                        let v = x * 0.7;
                        let s = 1.0 / (1.0 + (-v).exp());
                        match act {
                            0 => v * s,
                            1 => s,
                            _ => 2.0 * s,
                        }
                    })
                    .collect();
                assert_close(&read(&os, &ol), &want, tol(dt), "scale_act");
            }
        }
        let (xs, xl, _) = tensor(DType::F32, &[4], &[1.0; 4]);
        let (mut os, ol) = empty(DType::F32, &[4]);
        assert!(matches!(
            scale_act((&xs, &xl), (&mut os, &ol), 4, 1.0, 7, &stream()),
            Err(SynaptixError::Unsupported(_))
        ));
    }

    #[test]
    fn ternary_fused_is_bit_exact() {
        for dt in DTS {
            let dims = [3usize, 2, 13];
            let n: usize = dims.iter().product();
            let d = 13;
            let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 5, 2.0));
            let (bfs, bfl, bfv) = tensor(dt, &dims, &gen(n, 6, 1.0));
            let (cfs, cfl, cfv) = tensor(dt, &dims, &gen(n, 7, 1.0));
            let (brs, brl, brv) = tensor(dt, &[1, 1, d], &gen(d, 8, 1.0));
            let (crs, crl, crv) = tensor(dt, &[d], &gen(d, 9, 1.0));
            for kind in 0..4u8 {
                let (mut os, ol) = empty(dt, &dims);
                let (b, c): ((&Storage, &Layout), (&Storage, &Layout)) = match kind {
                    0 => ((&bfs, &bfl), (&cfs, &cfl)),
                    1 => ((&bfs, &bfl), (&crs, &crl)),
                    2 => ((&brs, &brl), (&crs, &crl)),
                    _ => ((&bfs, &bfl), (&cfs, &cfl)),
                };
                ternary_fused(kind, (&xs, &xl), b, c, (&mut os, &ol), &stream()).unwrap();
                let got = read(&os, &ol);
                for i in 0..n {
                    let j = i % d;
                    let want = match kind {
                        0 => round(dt, xv[i] + round(dt, bfv[i] * cfv[i])),
                        1 => round(dt, xv[i] + round(dt, bfv[i] * crv[j])),
                        2 => round(dt, round(dt, xv[i] * round(dt, 1.0 + brv[j])) + crv[j]),
                        _ => round(dt, round(dt, xv[i] * round(dt, 1.0 + bfv[i])) + cfv[i]),
                    };
                    assert_eq!(got[i].to_bits(), want.to_bits(), "kind {kind} {dt:?} [{i}]: {} vs {want}", got[i]);
                }
            }
            let (mut os, ol) = empty(dt, &dims);
            assert!(matches!(
                ternary_fused(2, (&xs, &xl), (&bfs, &bfl), (&crs, &crl), (&mut os, &ol), &stream()),
                Err(SynaptixError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn snake_matches_reference() {
        for dt in DTS {
            let (b, c, t) = (2usize, 3usize, 17usize);
            let n = b * c * t;
            let (xs, xl, xv) = tensor(dt, &[b, c, t], &gen(n, 10, 2.0));
            let (as_, al, av) = tensor(DType::F32, &[c], &gen(c, 11, 1.0));
            let (bs, bl, bv) = tensor(DType::F32, &[c], &gen(c, 12, 1.0));
            let (mut os, ol) = empty(dt, &[b, c, t]);
            snake((&xs, &xl), (&as_, &al), (&bs, &bl), (&mut os, &ol), c, t, &stream()).unwrap();
            let want: Vec<f32> = (0..n)
                .map(|i| {
                    let ch = (i / t) % c;
                    let s = (av[ch] * xv[i]).sin();
                    xv[i] + s * s * bv[ch]
                })
                .collect();
            assert_close(&read(&os, &ol), &want, tol(dt), "snake");
        }
    }

    #[test]
    fn linear_epilogue_matches_reference() {
        for dt in DTS {
            for (m, with_bias, with_res) in [(3usize, true, true), (12, false, true), (2, true, false), (9, false, false)] {
                let (k, n) = (11usize, 5usize);
                let (xs, xl, xv) = tensor(dt, &[2, m, k], &gen(2 * m * k, 13, 1.0));
                let (ws, wl, wv) = tensor(dt, &[n, k], &gen(n * k, 14, 1.0));
                let (bs, bl, bv) = tensor(dt, &[n], &gen(n, 15, 1.0));
                let (rs, rl, rv) = tensor(dt, &[2, m, n], &gen(2 * m * n, 16, 1.0));
                let (mut os, ol) = empty(dt, &[2, m, n]);
                let bias = with_bias.then_some((&bs, &bl));
                let res = with_res.then_some((&rs, &rl));
                linear_epilogue((&xs, &xl), (&ws, &wl), bias, res, (&mut os, &ol), &stream()).unwrap();
                let mut want = vec![0.0; 2 * m * n];
                for r in 0..2 * m {
                    for j in 0..n {
                        let mut acc = (0..k).map(|q| xv[r * k + q] * wv[j * k + q]).sum::<f32>();
                        acc = round(dt, acc);
                        if with_bias {
                            acc += bv[j];
                        }
                        if with_res {
                            acc += rv[r * n + j];
                        }
                        want[r * n + j] = acc;
                    }
                }
                assert_close(&read(&os, &ol), &want, tol(dt) * 2.0, "linear_epilogue");
            }
        }
    }

    #[test]
    fn hc_mix_and_inject_match_reference() {
        for dt in DTS {
            let (rows, hc, h) = (4usize, 3usize, 21usize);
            let w = hc * h;
            let (us, ul, uv) = tensor(dt, &[rows, w], &gen(rows * w, 17, 3.0));
            let (ns, nl, nv) = tensor(dt, &[rows, w], &gen(rows * w, 18, 2.0));
            let (mut os, ol) = empty(dt, &[rows, h]);
            hc_mix((&us, &ul), (&ns, &nl), (&mut os, &ol), rows, hc, h, &stream()).unwrap();
            let mut want = vec![0.0; rows * h];
            for r in 0..rows {
                for i in 0..h {
                    let mut acc = 0.0;
                    for c in 0..hc {
                        let p = r * w + c * h + i;
                        acc += (1.0 / (1.0 + (-uv[p]).exp())) * nv[p];
                    }
                    want[r * h + i] = acc / hc as f32;
                }
            }
            assert_close(&read(&os, &ol), &want, tol(dt), "hc_mix");

            let (bs, bl, bv) = tensor(dt, &[rows, h], &gen(rows * h, 19, 1.0));
            let (ws, wl, wv) = tensor(dt, &[rows, hc], &gen(rows * hc, 20, 1.0));
            let (mut os2, ol2) = empty(dt, &[rows, w]);
            hc_inject((&us, &ul), (&bs, &bl), (&ws, &wl), (&mut os2, &ol2), rows, hc, h, &stream()).unwrap();
            let mut want2 = vec![0.0; rows * w];
            for r in 0..rows {
                for c in 0..hc {
                    for i in 0..h {
                        want2[r * w + c * h + i] = uv[r * w + c * h + i] + bv[r * h + i] * wv[r * hc + c];
                    }
                }
            }
            assert_close(&read(&os2, &ol2), &want2, tol(dt), "hc_inject");
        }
    }

    #[test]
    fn weighted_rows_sum_and_moe_combine_match_reference() {
        for dt in DTS {
            let (t, k, h) = (5usize, 3usize, 19usize);
            let (ps, pl, pv) = tensor(dt, &[t * k, h], &gen(t * k * h, 21, 2.0));
            let (ws, wl, wv) = tensor(DType::F32, &[t * k], &gen(t * k, 22, 1.0));
            let (mut os, ol) = empty(dt, &[t, h]);
            weighted_rows_sum((&ps, &pl), (&ws, &wl), (&mut os, &ol), t, k, h, &stream()).unwrap();
            let mut want = vec![0.0; t * h];
            for r in 0..t {
                for i in 0..h {
                    want[r * h + i] = (0..k).map(|j| wv[r * k + j] * pv[(r * k + j) * h + i]).sum();
                }
            }
            assert_close(&read(&os, &ol), &want, tol(dt), "weighted_rows_sum");

            let rows = t * k + 2;
            let (ys, yl, yv) = tensor(dt, &[rows, h], &gen(rows * h, 23, 2.0));
            let inv: Vec<u32> = (0..t * k).map(|i| ((i * 7 + 3) % rows) as u32).collect();
            let inv_bytes: Vec<u8> = inv.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let is = Storage::Cpu(CpuBuf::from_vec(inv_bytes));
            let il = Layout::contiguous(Shape::new(vec![t * k]), DType::U32);
            let (mut os2, ol2) = empty(dt, &[t, h]);
            moe_combine((&ys, &yl), (&is, &il), (&ws, &wl), (&mut os2, &ol2), t, k, h, &stream()).unwrap();
            let mut want2 = vec![0.0; t * h];
            for r in 0..t {
                for i in 0..h {
                    want2[r * h + i] = (0..k).map(|s| wv[r * k + s] * yv[inv[r * k + s] as usize * h + i]).sum();
                }
            }
            assert_close(&read(&os2, &ol2), &want2, tol(dt), "moe_combine");

            let bad: Vec<u8> = (0..t * k).flat_map(|_| (rows as u32).to_ne_bytes()).collect();
            let bs = Storage::Cpu(CpuBuf::from_vec(bad));
            assert!(matches!(
                moe_combine((&ys, &yl), (&bs, &il), (&ws, &wl), (&mut os2, &ol2), t, k, h, &stream()),
                Err(SynaptixError::Unsupported(_))
            ));
        }
    }
}
