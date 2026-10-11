use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;

use crate::common::{cpu, elem_bytes, same_dtype};
use crate::norms::{check_float, dense_offset, in_bytes, ld, min_rows, out_bytes, out_row, par_rows, simd_fn, st, to_f32_vec, Rows};

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn rotate_body(x1: &[f32], x2: &[f32], c1: &[f32], s1: &[f32], c2: &[f32], s2: &[f32], o1: &mut [f32], o2: &mut [f32]) {
    let n = o1.len();
    let main = n - n % 8;
    for i in (0..main).step_by(8) {
        let a = ld(&x1[i..i + 8]);
        let b = ld(&x2[i..i + 8]);
        st(a * ld(&c1[i..i + 8]) - b * ld(&s1[i..i + 8]), &mut o1[i..i + 8]);
        st(b * ld(&c2[i..i + 8]) + a * ld(&s2[i..i + 8]), &mut o2[i..i + 8]);
    }
    for i in main..n {
        o1[i] = x1[i] * c1[i] - x2[i] * s1[i];
        o2[i] = x2[i] * c2[i] + x1[i] * s2[i];
    }
}

simd_fn!(pub(crate) fn rotate(x1: &[f32], x2: &[f32], c1: &[f32], s1: &[f32], c2: &[f32], s2: &[f32], o1: &mut [f32], o2: &mut [f32]) = rotate_body);

fn rotate_row(x: &[f32], c: &[f32], s: &[f32], rot: usize, o: &mut [f32]) {
    let half = rot / 2;
    let (o1, rest) = o.split_at_mut(half);
    let (o2, tail) = rest.split_at_mut(half);
    rotate(&x[..half], &x[half..rot], &c[..half], &s[..half], &c[..half], &s[..half], o1, o2);
    tail.copy_from_slice(&x[rot..]);
}

fn check_tables(cos: &Layout, sin: &Layout, numel: usize, what: &'static str) -> Result<()> {
    if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        return Err(SynaptixError::Unsupported(what));
    }
    if cos.numel() != numel || sin.numel() != numel {
        return Err(SynaptixError::Unsupported(what));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn split_impl(
    x: (&Storage, &Layout),
    cos: (&Storage, &Layout),
    sin: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    rot_dim: usize,
    s_len: usize,
    pos_div: usize,
    what: &'static str,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    let d = *x_lo.dims().last().unwrap_or(&0);
    let half = rot_dim / 2;
    let cv = to_f32_vec(cos.0, cos.1, what)?;
    let sv = to_f32_vec(sin.0, sin.1, what)?;
    let (xb, x_off) = in_bytes(x_st, x_lo, what)?;
    let ob = out_bytes(o_st, o_lo, what)?;
    let eb = elem_bytes(dt);
    par_rows(ob, d * eb, min_rows(d), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, d);
        let mut tmp = vec![0.0; d];
        for (i, dst) in blk.chunks_exact_mut(d * eb).enumerate() {
            let r = r0 + i;
            let pos = (r / pos_div) % s_len;
            let xr = rows.get(r * d, d);
            let c = &cv[pos * half..(pos + 1) * half];
            let s = &sv[pos * half..(pos + 1) * half];
            out_row(dt, dst, &mut tmp, |o| rotate_row(xr, c, s, rot_dim, o));
        }
    });
    Ok(())
}

pub fn rope_split(
    x: (&Storage, &Layout),
    cos: (&Storage, &Layout),
    sin: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let x_lo = x.1;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, out.1.dtype())?;
    let xr = x_lo.dims().len();
    if xr < 2 || out.1.dims() != x_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu rope_split: shape"));
    }
    let d = x_lo.dims()[xr - 1];
    let s = x_lo.dims()[xr - 2];
    if d == 0 || !d.is_multiple_of(2) || s == 0 {
        return Err(SynaptixError::Unsupported("cpu rope_split: D must be even"));
    }
    check_tables(cos.1, sin.1, s * (d / 2), "cpu rope_split: cos/sin must be F32 [S, D/2]")?;
    split_impl(x, cos, sin, out, d, s, 1, "cpu rope_split")
}

pub fn rope_split_partial(
    x: (&Storage, &Layout),
    cos: (&Storage, &Layout),
    sin: (&Storage, &Layout),
    rot_dim: usize,
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let x_lo = x.1;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, out.1.dtype())?;
    let xr = x_lo.dims().len();
    if xr < 2 || out.1.dims() != x_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu rope_split_partial: shape"));
    }
    let d = x_lo.dims()[xr - 1];
    let mut s_len = x_lo.dims()[xr - 2];
    let mut pos_div = 1usize;
    if rot_dim == 0 || !rot_dim.is_multiple_of(2) || rot_dim > d {
        return Err(SynaptixError::Unsupported("cpu rope_split_partial: rot_dim"));
    }
    let cd = cos.1.dims();
    if cd.len() != 2 || cd[1] * 2 != rot_dim {
        return Err(SynaptixError::Unsupported("cpu rope_split_partial: cos shape"));
    }
    if cd[0] != s_len {
        if xr >= 3 && cd[0] == x_lo.dims()[xr - 3] {
            pos_div = s_len;
            s_len = cd[0];
        } else {
            return Err(SynaptixError::Unsupported("cpu rope_split_partial: cos shape"));
        }
    }
    if sin.1.dims() != cd || s_len == 0 || pos_div == 0 {
        return Err(SynaptixError::Unsupported("cpu rope_split_partial: sin shape"));
    }
    check_tables(cos.1, sin.1, s_len * (rot_dim / 2), "cpu rope_split_partial: cos/sin must be F32")?;
    split_impl(x, cos, sin, out, rot_dim, s_len, pos_div, "cpu rope_split_partial")
}

pub fn rope_interleaved(
    x: (&Storage, &Layout),
    cos: (&Storage, &Layout),
    sin: (&Storage, &Layout),
    h: usize,
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    let dims = x_lo.dims();
    if dims.len() != 4 || o_lo.dims() != dims || h == 0 || dims[2] != h {
        return Err(SynaptixError::Unsupported("cpu rope_interleaved: x must be [B,S,H,D]"));
    }
    let (s, d) = (dims[1], dims[3]);
    if d == 0 || !d.is_multiple_of(2) || s == 0 {
        return Err(SynaptixError::Unsupported("cpu rope_interleaved: D must be even"));
    }
    check_tables(cos.1, sin.1, s * d, "cpu rope_interleaved: cos/sin must be F32 [S, D]")?;
    let cv = to_f32_vec(cos.0, cos.1, "cpu rope_interleaved: cos")?;
    let sv = to_f32_vec(sin.0, sin.1, "cpu rope_interleaved: sin")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu rope_interleaved: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu rope_interleaved: out")?;
    let eb = elem_bytes(dt);
    par_rows(ob, d * eb, min_rows(d), |r0, blk| {
        let mut rows = Rows::new(dt, xb, x_off, d);
        let mut tmp = vec![0.0; d];
        for (i, dst) in blk.chunks_exact_mut(d * eb).enumerate() {
            let r = r0 + i;
            let pos = (r / h) % s;
            let xr = rows.get(r * d, d);
            let c = &cv[pos * d..(pos + 1) * d];
            let sn = &sv[pos * d..(pos + 1) * d];
            out_row(dt, dst, &mut tmp, |o| {
                for j in 0..d / 2 {
                    let (e, od) = (2 * j, 2 * j + 1);
                    o[e] = xr[e] * c[e] - xr[od] * sn[e];
                    o[od] = xr[od] * c[od] + xr[e] * sn[od];
                }
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_apply_dev(
    x: (&Storage, &Layout),
    cos: (&Storage, &Layout),
    sin: (&Storage, &Layout),
    start_pos: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    rotary_dim: usize,
    _stream: &Stream,
) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let (sp_st, sp_lo) = start_pos;
    let dt = x_lo.dtype();
    check_float(dt)?;
    same_dtype(dt, o_lo.dtype())?;
    same_dtype(dt, cos.1.dtype())?;
    same_dtype(dt, sin.1.dtype())?;
    if sp_lo.dtype() != DType::U32 {
        return Err(SynaptixError::Unsupported("cpu rope_apply_dev: start_pos must be U32"));
    }
    let dims = x_lo.dims();
    if dims.len() != 4 || o_lo.dims() != dims {
        return Err(SynaptixError::Unsupported("cpu rope_apply_dev: x must be [b,h,t,head_dim]"));
    }
    let (b, h, t, hd) = (dims[0], dims[1], dims[2], dims[3]);
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > hd {
        return Err(SynaptixError::Unsupported("cpu rope_apply_dev: rotary_dim"));
    }
    if cos.1.numel() != sin.1.numel() || !cos.1.numel().is_multiple_of(rotary_dim) {
        return Err(SynaptixError::Unsupported("cpu rope_apply_dev: cos/sin table"));
    }
    let max_seq = cos.1.numel() / rotary_dim;
    let n_pos = sp_lo.numel().clamp(1, b.max(1));
    let sp_off = dense_offset(sp_lo)?;
    let spb = cpu(sp_st, "cpu rope_apply_dev: start_pos")?.as_bytes();
    let sp: Vec<usize> = (0..n_pos)
        .map(|i| {
            let o = (sp_off + i) * 4;
            u32::from_ne_bytes([spb[o], spb[o + 1], spb[o + 2], spb[o + 3]]) as usize
        })
        .collect();
    let rows = b * h * t;
    if rows > 0 && sp.iter().any(|&p| p + t > max_seq) {
        return Err(SynaptixError::Unsupported("cpu rope_apply_dev: position beyond cos table"));
    }
    let (cb, c_off) = in_bytes(cos.0, cos.1, "cpu rope_apply_dev: cos")?;
    let (sb, s_off) = in_bytes(sin.0, sin.1, "cpu rope_apply_dev: sin")?;
    let (xb, x_off) = in_bytes(x_st, x_lo, "cpu rope_apply_dev: x")?;
    let ob = out_bytes(o_st, o_lo, "cpu rope_apply_dev: out")?;
    let eb = elem_bytes(dt);
    let half = rotary_dim / 2;
    par_rows(ob, hd * eb, min_rows(hd), |r0, blk| {
        let mut xrows = Rows::new(dt, xb, x_off, hd);
        let mut crows = Rows::new(dt, cb, c_off, rotary_dim);
        let mut srows = Rows::new(dt, sb, s_off, rotary_dim);
        let mut tmp = vec![0.0; hd];
        for (i, dst) in blk.chunks_exact_mut(hd * eb).enumerate() {
            let r = r0 + i;
            let bi = r / (h * t);
            let pos = sp[if bi < n_pos { bi } else { 0 }] + r % t;
            let xr = xrows.get(r * hd, hd);
            let c = crows.get(pos * rotary_dim, rotary_dim);
            let s = srows.get(pos * rotary_dim, rotary_dim);
            out_row(dt, dst, &mut tmp, |o| {
                let (o1, rest) = o.split_at_mut(half);
                let (o2, tail) = rest.split_at_mut(half);
                rotate(&xr[..half], &xr[half..rotary_dim], &c[..half], &s[..half], &c[half..], &s[half..], o1, o2);
                tail.copy_from_slice(&xr[rotary_dim..]);
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norms::testutil::*;
    use synaptix_core::tensor::layout::Layout;
    use synaptix_core::tensor::shape::Shape;
    use synaptix_core::tensor::storage::CpuBuf;

    fn ref_split(x: &[f32], d: usize, rot: usize, pos_of: impl Fn(usize) -> usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let half = rot / 2;
        let mut out = x.to_vec();
        for r in 0..x.len() / d {
            let p = pos_of(r);
            let row = &x[r * d..(r + 1) * d];
            for i in 0..half {
                let (c, s) = (cos[p * half + i], sin[p * half + i]);
                out[r * d + i] = row[i] * c - row[i + half] * s;
                out[r * d + half + i] = row[i + half] * c + row[i] * s;
            }
        }
        out
    }

    #[test]
    fn rope_split_matches_reference() {
        for dt in DTS {
            for dims in [vec![1usize, 2, 5, 6], vec![3, 4, 34], vec![2, 1, 7, 128]] {
                let n: usize = dims.iter().product();
                let d = dims[dims.len() - 1];
                let s = dims[dims.len() - 2];
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 1, 2.0));
                let (cs, cl, cv) = tensor(DType::F32, &[s, d / 2], &gen(s * d / 2, 2, 1.0));
                let (ss, sl, sv) = tensor(DType::F32, &[s, d / 2], &gen(s * d / 2, 3, 1.0));
                let (mut os, ol) = empty(dt, &dims);
                rope_split((&xs, &xl), (&cs, &cl), (&ss, &sl), (&mut os, &ol), &stream()).unwrap();
                let want = ref_split(&xv, d, d, |r| r % s, &cv, &sv);
                assert_close(&read(&os, &ol), &want, tol(dt), "rope_split");
            }
        }
    }

    #[test]
    fn rope_split_partial_matches_reference() {
        for dt in DTS {
            let (hh, s, d, rot) = (3usize, 5usize, 12usize, 8usize);
            let dims = [hh, s, d];
            let n = hh * s * d;
            let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 4, 2.0));
            let (cs, cl, cv) = tensor(DType::F32, &[s, rot / 2], &gen(s * rot / 2, 5, 1.0));
            let (ss, sl, sv) = tensor(DType::F32, &[s, rot / 2], &gen(s * rot / 2, 6, 1.0));
            let (mut os, ol) = empty(dt, &dims);
            rope_split_partial((&xs, &xl), (&cs, &cl), (&ss, &sl), rot, (&mut os, &ol), &stream()).unwrap();
            let want = ref_split(&xv, d, rot, |r| r % s, &cv, &sv);
            assert_close(&read(&os, &ol), &want, tol(dt), "rope_split_partial");

            let dims2 = [s, hh, d];
            let (xs2, xl2, xv2) = tensor(dt, &dims2, &gen(n, 7, 2.0));
            let (mut os2, ol2) = empty(dt, &dims2);
            rope_split_partial((&xs2, &xl2), (&cs, &cl), (&ss, &sl), rot, (&mut os2, &ol2), &stream()).unwrap();
            let want2 = ref_split(&xv2, d, rot, |r| (r / hh) % s, &cv, &sv);
            assert_close(&read(&os2, &ol2), &want2, tol(dt), "rope_split_partial pos_div");
        }
    }

    #[test]
    fn rope_interleaved_matches_reference() {
        for dt in DTS {
            let (b, s, h, d) = (2usize, 3usize, 4usize, 10usize);
            let dims = [b, s, h, d];
            let n = b * s * h * d;
            let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 8, 2.0));
            let mut cos = vec![0.0; s * d];
            let mut sin = vec![0.0; s * d];
            let base = gen(s * d / 2, 9, 1.0);
            for p in 0..s {
                for j in 0..d / 2 {
                    let a = base[p * d / 2 + j] * 3.0;
                    cos[p * d + 2 * j] = a.cos();
                    cos[p * d + 2 * j + 1] = a.cos();
                    sin[p * d + 2 * j] = a.sin();
                    sin[p * d + 2 * j + 1] = a.sin();
                }
            }
            let (cs, cl, cv) = tensor(DType::F32, &[1, s, 1, d], &cos);
            let (ss, sl, sv) = tensor(DType::F32, &[1, s, 1, d], &sin);
            let (mut os, ol) = empty(dt, &dims);
            rope_interleaved((&xs, &xl), (&cs, &cl), (&ss, &sl), h, (&mut os, &ol), &stream()).unwrap();
            let mut want = vec![0.0; n];
            for r in 0..n / d {
                let p = (r / h) % s;
                for j in 0..d / 2 {
                    let (e, o) = (2 * j, 2 * j + 1);
                    want[r * d + e] = xv[r * d + e] * cv[p * d + e] - xv[r * d + o] * sv[p * d + e];
                    want[r * d + o] = xv[r * d + o] * cv[p * d + o] + xv[r * d + e] * sv[p * d + o];
                }
            }
            assert_close(&read(&os, &ol), &want, tol(dt), "rope_interleaved");
        }
    }

    #[test]
    fn rope_apply_dev_matches_reference() {
        for dt in DTS {
            for (b, starts) in [(1usize, vec![3u32]), (2, vec![1u32, 4]), (2, vec![2u32])] {
                let (h, t, hd, rot, max_seq) = (2usize, 3usize, 12usize, 8usize, 16usize);
                let dims = [b, h, t, hd];
                let n = b * h * t * hd;
                let (xs, xl, xv) = tensor(dt, &dims, &gen(n, 10, 2.0));
                let half = rot / 2;
                let mut cos = vec![0.0; max_seq * rot];
                let mut sin = vec![0.0; max_seq * rot];
                for p in 0..max_seq {
                    for i in 0..half {
                        let a = p as f32 * 0.37 * (i as f32 + 1.0);
                        cos[p * rot + i] = a.cos();
                        cos[p * rot + half + i] = a.cos();
                        sin[p * rot + i] = a.sin();
                        sin[p * rot + half + i] = a.sin();
                    }
                }
                let (cs, cl, cv) = tensor(dt, &[max_seq, rot], &cos);
                let (ss, sl, sv) = tensor(dt, &[max_seq, rot], &sin);
                let sp_bytes: Vec<u8> = starts.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let sps = Storage::Cpu(CpuBuf::from_vec(sp_bytes));
                let spl = Layout::contiguous(Shape::new(vec![starts.len()]), DType::U32);
                let (mut os, ol) = empty(dt, &dims);
                rope_apply_dev((&xs, &xl), (&cs, &cl), (&ss, &sl), (&sps, &spl), (&mut os, &ol), rot, &stream()).unwrap();
                let mut want = xv.clone();
                for r in 0..n / hd {
                    let bi = r / (h * t);
                    let pos = starts[if bi < starts.len() { bi } else { 0 }] as usize + r % t;
                    for i in 0..half {
                        let (c1, s1) = (cv[pos * rot + i], sv[pos * rot + i]);
                        let (c2, s2) = (cv[pos * rot + half + i], sv[pos * rot + half + i]);
                        want[r * hd + i] = xv[r * hd + i] * c1 - xv[r * hd + half + i] * s1;
                        want[r * hd + half + i] = xv[r * hd + half + i] * c2 + xv[r * hd + i] * s2;
                    }
                }
                assert_close(&read(&os, &ol), &want, tol(dt), "rope_apply_dev");
            }
        }
    }

    #[test]
    fn rope_rejects_bad_tables() {
        let dt = DType::F32;
        let (xs, xl, _) = tensor(dt, &[2, 4, 6], &gen(48, 1, 1.0));
        let (cs, cl, _) = tensor(DType::F16, &[4, 3], &gen(12, 2, 1.0));
        let (mut os, ol) = empty(dt, &[2, 4, 6]);
        assert!(matches!(
            rope_split((&xs, &xl), (&cs, &cl), (&cs, &cl), (&mut os, &ol), &stream()),
            Err(SynaptixError::Unsupported(_))
        ));
        let (cs2, cl2, _) = tensor(DType::F32, &[5, 3], &gen(15, 2, 1.0));
        assert!(matches!(
            rope_split((&xs, &xl), (&cs2, &cl2), (&cs2, &cl2), (&mut os, &ol), &stream()),
            Err(SynaptixError::Unsupported(_))
        ));
    }
}
