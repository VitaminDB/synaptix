use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;

use crate::common::{check_contig, cpu, cpu_mut, elem_bytes, row_from_f32, row_to_f32};

#[derive(Debug, Clone, Copy)]
pub struct Im2ColParams2d {
    pub batch: usize,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

impl Im2ColParams2d {
    pub fn out_h(&self) -> usize {
        let eff = self.dilation_h * (self.kernel_h - 1) + 1;
        (self.in_h + 2 * self.pad_h - eff) / self.stride_h + 1
    }
    pub fn out_w(&self) -> usize {
        let eff = self.dilation_w * (self.kernel_w - 1) + 1;
        (self.in_w + 2 * self.pad_w - eff) / self.stride_w + 1
    }
    pub fn check(&self) -> Result<()> {
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(SynaptixError::Unsupported("im2col: zero kernel"));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(SynaptixError::Unsupported("im2col: zero stride"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Im2ColRows {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub kh: usize,
    pub kw: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub m_offset: usize,
    pub m_count: usize,
}

pub fn im2col_rows<T: bytemuck::Pod + Send + Sync>(src: &[u8], src_off: usize, dst: &mut [u8], dst_off: usize, p: Im2ColRows) {
    use rayon::prelude::*;
    let src: &[T] = &bytemuck::cast_slice::<u8, T>(src)[src_off..];
    let k = p.c * p.kh * p.kw;
    let dst: &mut [T] = &mut bytemuck::cast_slice_mut::<u8, T>(dst)[dst_off..dst_off + p.m_count * k];
    let plane = p.h * p.w;
    let per_img = p.h_out * p.w_out;
    let zero = T::zeroed();
    let (ph, pw) = p.padding;
    dst.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
        let m = p.m_offset + r;
        let (b, rem) = (m / per_img, m % per_img);
        let (oy, ox) = (rem / p.w_out, rem % p.w_out);
        let x0 = ox * p.stride.1;
        let kx_lo = pw.saturating_sub(x0).min(p.kw);
        let kx_hi = (p.w + pw).saturating_sub(x0).min(p.kw).max(kx_lo);
        let base = b * p.c * plane;
        for ci in 0..p.c {
            let cb = base + ci * plane;
            for ky in 0..p.kh {
                let seg = &mut row[(ci * p.kh + ky) * p.kw..(ci * p.kh + ky + 1) * p.kw];
                let y = oy * p.stride.0 + ky;
                if y < ph || y - ph >= p.h {
                    seg.fill(zero);
                    continue;
                }
                let s = cb + (y - ph) * p.w + x0 + kx_lo - pw;
                seg[..kx_lo].fill(zero);
                seg[kx_lo..kx_hi].copy_from_slice(&src[s..s + (kx_hi - kx_lo)]);
                seg[kx_hi..].fill(zero);
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
struct ConvGeom {
    cin: usize,
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize, usize),
    pad: (usize, usize, usize),
    od: usize,
    oh: usize,
    ow: usize,
}

impl ConvGeom {
    fn k(&self) -> usize {
        self.cin * self.kd * self.kh * self.kw
    }
    fn out_plane(&self) -> usize {
        self.od * self.oh * self.ow
    }
    fn in_plane(&self) -> usize {
        self.cin * self.d * self.h * self.w
    }
    fn direct(&self) -> bool {
        self.kd == 1 && self.kh == 1 && self.kw == 1 && self.stride == (1, 1, 1) && self.pad == (0, 0, 0)
    }
}

fn out_len(n: usize, k: usize, s: usize, p: usize) -> Option<usize> {
    (n + 2 * p >= k).then(|| (n + 2 * p - k) / s + 1)
}

fn im2col_kmajor(x: &[f32], col: &mut [f32], g: &ConvGeom, p0: usize, p1: usize) {
    use rayon::prelude::*;
    let mt = p1 - p0;
    let plane_o = g.oh * g.ow;
    let (sd, sh, sw) = g.stride;
    let (pd, ph, pw) = g.pad;
    col[..g.k() * mt].par_chunks_mut(mt).enumerate().for_each(|(kr, row)| {
        let kx = kr % g.kw;
        let t = kr / g.kw;
        let ky = t % g.kh;
        let t = t / g.kh;
        let kz = t % g.kd;
        let ci = t / g.kd;
        let src_c = &x[ci * g.d * g.h * g.w..(ci + 1) * g.d * g.h * g.w];
        let lo_all = pw.saturating_sub(kx).div_ceil(sw);
        let hi_all = (g.w + pw).saturating_sub(kx).div_ceil(sw);
        let mut p = p0;
        while p < p1 {
            let od = p / plane_o;
            let rem = p % plane_o;
            let oy = rem / g.ow;
            let ox0 = rem % g.ow;
            let seg = (g.ow - ox0).min(p1 - p);
            let dst = &mut row[p - p0..p - p0 + seg];
            let z = od * sd + kz;
            let y = oy * sh + ky;
            if z < pd || z - pd >= g.d || y < ph || y - ph >= g.h {
                dst.fill(0.0);
            } else {
                let src_row = &src_c[((z - pd) * g.h + (y - ph)) * g.w..][..g.w];
                let lo = lo_all.max(ox0).min(ox0 + seg);
                let hi = hi_all.min(ox0 + seg).max(lo);
                dst[..lo - ox0].fill(0.0);
                dst[hi - ox0..].fill(0.0);
                if hi > lo && sw == 1 {
                    let s = lo + kx - pw;
                    dst[lo - ox0..hi - ox0].copy_from_slice(&src_row[s..s + (hi - lo)]);
                } else if hi > lo {
                    for ox in lo..hi {
                        dst[ox - ox0] = src_row[ox * sw + kx - pw];
                    }
                }
            }
            p += seg;
        }
    });
}

const CONV_TILE_ELEMS: usize = 16 * 1024 * 1024;

#[allow(clippy::too_many_arguments)]
fn conv_gemm(
    input: (&Storage, &Layout),
    weight: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    g: &ConvGeom,
    batch: usize,
    cout: usize,
    tile_elems: usize,
) -> Result<()> {
    use rayon::prelude::*;
    let (x_st, x_lo) = input;
    let (w_st, w_lo) = weight;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    if !matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("cpu conv: dtype"));
    }
    if w_lo.dtype() != dt || o_lo.dtype() != dt {
        return Err(SynaptixError::Unsupported("cpu conv: dtype mismatch"));
    }
    check_contig(x_lo)?;
    check_contig(w_lo)?;
    check_contig(o_lo)?;
    let k = g.k();
    let p_out = g.out_plane();
    if k == 0 || cout == 0 {
        return Err(SynaptixError::Unsupported("cpu conv: empty kernel"));
    }
    let bias_f = match bias {
        Some((b_st, b_lo)) => {
            if b_lo.dtype() != dt || b_lo.dims() != [cout] {
                return Err(SynaptixError::Unsupported("cpu conv: bias shape/dtype"));
            }
            check_contig(b_lo)?;
            let mut v = vec![0f32; cout];
            row_to_f32(dt, cpu(b_st, "cpu conv: bias non-cpu")?.as_bytes(), b_lo.offset(), &mut v);
            Some(v)
        }
        None => None,
    };
    let xb = cpu(x_st, "cpu conv: input non-cpu")?.as_bytes();
    let wb = cpu(w_st, "cpu conv: weight non-cpu")?.as_bytes();
    let mut wf = vec![0f32; cout * k];
    row_to_f32(dt, wb, w_lo.offset(), &mut wf);
    let esz = elem_bytes(dt);
    let ob = cpu_mut(o_st, "cpu conv: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + batch * cout * p_out) * esz];
    if p_out == 0 || batch == 0 {
        return Ok(());
    }
    let in_plane = g.in_plane();
    let mt_max = (tile_elems / (k + cout)).clamp(1, p_out);
    let direct = g.direct();
    let mut col = if direct && mt_max == p_out { Vec::new() } else { vec![0f32; k * mt_max] };
    let mut cbuf = vec![0f32; cout * mt_max];
    let mut xconv = if dt == DType::F32 { Vec::new() } else { vec![0f32; in_plane] };
    for (b, ob_b) in ob.chunks_mut(cout * p_out * esz).enumerate() {
        let xf: &[f32] = if dt == DType::F32 {
            let all: &[f32] = bytemuck::cast_slice(&xb[..(x_lo.offset() + batch * in_plane) * 4]);
            &all[x_lo.offset() + b * in_plane..x_lo.offset() + (b + 1) * in_plane]
        } else {
            row_to_f32(dt, xb, x_lo.offset() + b * in_plane, &mut xconv);
            &xconv
        };
        let mut p0 = 0;
        while p0 < p_out {
            let p1 = (p0 + mt_max).min(p_out);
            let mt = p1 - p0;
            let bmat: &[f32] = if direct && mt == p_out {
                xf
            } else {
                im2col_kmajor(xf, &mut col, g, p0, p1);
                &col[..k * mt]
            };
            let c = &mut cbuf[..cout * mt];
            crate::gemm::blocked::sgemm(cout, mt, k, &wf, bmat, c);
            c.par_chunks_mut(mt).zip(ob_b.par_chunks_mut(p_out * esz)).enumerate().for_each(|(co, (crow, orow))| {
                if let Some(bv) = &bias_f {
                    let add = bv[co];
                    for v in crow.iter_mut() {
                        *v += add;
                    }
                }
                row_from_f32(dt, crow, orow, p0);
            });
            p0 = p1;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: (&Storage, &Layout),
    weight: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    stride: (usize, usize),
    padding: (usize, usize),
    _stream: &Stream,
) -> Result<()> {
    let (x_lo, w_lo, o_lo) = (input.1, weight.1, out.1);
    if x_lo.dims().len() != 4 || w_lo.dims().len() != 4 || o_lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported("cpu conv2d: ranks [4,4,4]"));
    }
    let (b, cin, h, w) = (x_lo.dims()[0], x_lo.dims()[1], x_lo.dims()[2], x_lo.dims()[3]);
    let (cout, cin_w, kh, kw) = (w_lo.dims()[0], w_lo.dims()[1], w_lo.dims()[2], w_lo.dims()[3]);
    if cin_w != cin {
        return Err(SynaptixError::Unsupported("cpu conv2d: c_in mismatch (grouped conv unsupported)"));
    }
    let (sh, sw) = (stride.0.max(1), stride.1.max(1));
    let (ph, pw) = padding;
    let (Some(oh), Some(ow)) = (out_len(h, kh, sh, ph), out_len(w, kw, sw, pw)) else {
        return Err(SynaptixError::Unsupported("cpu conv2d: input too small"));
    };
    if o_lo.dims() != [b, cout, oh, ow] {
        return Err(SynaptixError::shape_mismatch(&[b, cout, oh, ow], o_lo.dims()));
    }
    let g = ConvGeom { cin, d: 1, h, w, kd: 1, kh, kw, stride: (1, sh, sw), pad: (0, ph, pw), od: 1, oh, ow };
    conv_gemm(input, weight, bias, out, &g, b, cout, CONV_TILE_ELEMS)
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    input: (&Storage, &Layout),
    weight: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    _stream: &Stream,
) -> Result<()> {
    let (x_lo, w_lo, o_lo) = (input.1, weight.1, out.1);
    if x_lo.dims().len() != 5 || w_lo.dims().len() != 5 || o_lo.dims().len() != 5 {
        return Err(SynaptixError::Unsupported("cpu conv3d: ranks [5,5,5]"));
    }
    let (b, cin, d, h, w) = (x_lo.dims()[0], x_lo.dims()[1], x_lo.dims()[2], x_lo.dims()[3], x_lo.dims()[4]);
    let (cout, cin_w, kd, kh, kw) = (w_lo.dims()[0], w_lo.dims()[1], w_lo.dims()[2], w_lo.dims()[3], w_lo.dims()[4]);
    if cin_w != cin {
        return Err(SynaptixError::Unsupported("cpu conv3d: c_in mismatch (grouped conv unsupported)"));
    }
    let (sd, sh, sw) = (stride.0.max(1), stride.1.max(1), stride.2.max(1));
    let (pd, ph, pw) = padding;
    let (Some(od), Some(oh), Some(ow)) = (out_len(d, kd, sd, pd), out_len(h, kh, sh, ph), out_len(w, kw, sw, pw)) else {
        return Err(SynaptixError::Unsupported("cpu conv3d: input too small"));
    };
    if o_lo.dims() != [b, cout, od, oh, ow] {
        return Err(SynaptixError::shape_mismatch(&[b, cout, od, oh, ow], o_lo.dims()));
    }
    let g = ConvGeom { cin, d, h, w, kd, kh, kw, stride: (sd, sh, sw), pad: (pd, ph, pw), od, oh, ow };
    conv_gemm(input, weight, bias, out, &g, b, cout, CONV_TILE_ELEMS)
}

#[allow(clippy::too_many_arguments)]
pub fn dwconv1d(
    input: (&Storage, &Layout),
    weight: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    stride: usize,
    padding: usize,
    transpose: bool,
    _stream: &Stream,
) -> Result<()> {
    use rayon::prelude::*;
    let (x_st, x_lo) = input;
    let (w_st, w_lo) = weight;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    if !matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("cpu dwconv1d: dtype"));
    }
    if w_lo.dtype() != dt || o_lo.dtype() != dt {
        return Err(SynaptixError::Unsupported("cpu dwconv1d: dtype mismatch"));
    }
    if x_lo.dims().len() != 3 || w_lo.dims().len() != 3 || o_lo.dims().len() != 3 || w_lo.dims()[1] != 1 {
        return Err(SynaptixError::Unsupported("cpu dwconv1d: input [B,C,L], weight [C,1,K]"));
    }
    check_contig(x_lo)?;
    check_contig(w_lo)?;
    check_contig(o_lo)?;
    let (b, c, l) = (x_lo.dims()[0], x_lo.dims()[1], x_lo.dims()[2]);
    let k = w_lo.dims()[2];
    if w_lo.dims()[0] != c || k == 0 {
        return Err(SynaptixError::Unsupported("cpu dwconv1d: weight C/K"));
    }
    let s = stride.max(1);
    let lo = if transpose {
        if l == 0 {
            return Err(SynaptixError::Unsupported("cpu dwconv1d: empty input"));
        }
        (l - 1) * s + k
    } else {
        match out_len(l, k, s, padding) {
            Some(v) => v,
            None => return Err(SynaptixError::Unsupported("cpu dwconv1d: input too small")),
        }
    };
    if o_lo.dims() != [b, c, lo] {
        return Err(SynaptixError::shape_mismatch(&[b, c, lo], o_lo.dims()));
    }
    let bias_f = match bias {
        Some((b_st, b_lo)) => {
            if b_lo.dtype() != dt || b_lo.dims() != [c] {
                return Err(SynaptixError::Unsupported("cpu dwconv1d: bias shape/dtype"));
            }
            check_contig(b_lo)?;
            let mut v = vec![0f32; c];
            row_to_f32(dt, cpu(b_st, "cpu dwconv1d: bias non-cpu")?.as_bytes(), b_lo.offset(), &mut v);
            Some(v)
        }
        None => None,
    };
    let xb = cpu(x_st, "cpu dwconv1d: input non-cpu")?.as_bytes();
    let wb = cpu(w_st, "cpu dwconv1d: weight non-cpu")?.as_bytes();
    let mut wf = vec![0f32; c * k];
    row_to_f32(dt, wb, w_lo.offset(), &mut wf);
    let esz = elem_bytes(dt);
    let ob = cpu_mut(o_st, "cpu dwconv1d: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + b * c * lo) * esz];
    if lo == 0 || b * c == 0 {
        return Ok(());
    }
    ob.par_chunks_mut(lo * esz).enumerate().for_each(|(bc, orow)| {
        let ch = bc % c;
        let mut x = vec![0f32; l];
        row_to_f32(dt, xb, x_lo.offset() + bc * l, &mut x);
        let wr = &wf[ch * k..(ch + 1) * k];
        let mut acc = vec![bias_f.as_ref().map_or(0.0, |v| v[ch]); lo];
        if transpose {
            for (kk, &wv) in wr.iter().enumerate() {
                if s == 1 {
                    for (a, &xv) in acc[kk..kk + l].iter_mut().zip(x.iter()) {
                        *a += wv * xv;
                    }
                } else {
                    for (i, &xv) in x.iter().enumerate() {
                        acc[i * s + kk] += wv * xv;
                    }
                }
            }
        } else {
            for (kk, &wv) in wr.iter().enumerate() {
                let t_lo = padding.saturating_sub(kk).div_ceil(s).min(lo);
                let t_hi = (l + padding).saturating_sub(kk).div_ceil(s).min(lo).max(t_lo);
                if s == 1 {
                    let base = t_lo + kk - padding;
                    for (a, &xv) in acc[t_lo..t_hi].iter_mut().zip(x[base..base + (t_hi - t_lo)].iter()) {
                        *a += wv * xv;
                    }
                } else {
                    for t in t_lo..t_hi {
                        acc[t] += wv * x[t * s + kk - padding];
                    }
                }
            }
        }
        row_from_f32(dt, &acc, orow, 0);
    });
    Ok(())
}

pub fn im2col_2d_dispatch(
    dtype: DType,
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams2d,
) -> Result<(usize, usize)> {
    p.check()?;
    let oh = p.out_h();
    let ow = p.out_w();
    let patches_per_image = oh * ow;
    let patch_size = p.channels * p.kernel_h * p.kernel_w;
    let needed = p.batch * patches_per_image * patch_size * elem_size(dtype)?;
    if dst.len() < needed {
        return Err(SynaptixError::Other(format!(
            "im2col: dst too small, expected {} bytes, got {}",
            needed,
            dst.len()
        )));
    }
    match dtype {
        DType::F32 => im2col_2d_typed::<f32>(src, dst, p, oh, ow),
        DType::F64 => im2col_2d_typed::<f64>(src, dst, p, oh, ow),
        DType::F16 => im2col_2d_typed::<half::f16>(src, dst, p, oh, ow),
        DType::BF16 => im2col_2d_typed::<half::bf16>(src, dst, p, oh, ow),
        _ => Err(SynaptixError::Unsupported("im2col: dtype")),
    }?;
    Ok((p.batch * patches_per_image, patch_size))
}

fn im2col_2d_typed<T: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams2d,
    oh: usize,
    ow: usize,
) -> Result<()> {
    let src_t: &[T] = bytemuck::cast_slice(src);
    let dst_t: &mut [T] = bytemuck::cast_slice_mut(dst);
    let patch_size = p.channels * p.kernel_h * p.kernel_w;
    let patches_per_image = oh * ow;
    let zero = T::zeroed();
    for n in 0..p.batch {
        for ph in 0..oh {
            for pw in 0..ow {
                let patch_row = (n * patches_per_image + ph * ow + pw) * patch_size;
                let mut idx = 0usize;
                for c in 0..p.channels {
                    for ki in 0..p.kernel_h {
                        for kj in 0..p.kernel_w {
                            let h = ph * p.stride_h + ki * p.dilation_h;
                            let w = pw * p.stride_w + kj * p.dilation_w;
                            let in_h_pad = h as isize - p.pad_h as isize;
                            let in_w_pad = w as isize - p.pad_w as isize;
                            let v = if in_h_pad < 0
                                || in_w_pad < 0
                                || in_h_pad as usize >= p.in_h
                                || in_w_pad as usize >= p.in_w
                            {
                                zero
                            } else {
                                let src_off = ((n * p.channels + c) * p.in_h + in_h_pad as usize)
                                    * p.in_w
                                    + in_w_pad as usize;
                                src_t[src_off]
                            };
                            dst_t[patch_row + idx] = v;
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Im2ColParams3d {
    pub batch: usize,
    pub channels: usize,
    pub in_t: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_t: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_t: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl Im2ColParams3d {
    pub fn out_t(&self) -> usize { (self.in_t - self.kernel_t) / self.stride_t + 1 }
    pub fn out_h(&self) -> usize { (self.in_h - self.kernel_h) / self.stride_h + 1 }
    pub fn out_w(&self) -> usize { (self.in_w - self.kernel_w) / self.stride_w + 1 }
}

pub fn im2col_3d_dispatch(
    dtype: DType,
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams3d,
) -> Result<(usize, usize)> {
    if p.kernel_t == 0 || p.kernel_h == 0 || p.kernel_w == 0 {
        return Err(SynaptixError::Unsupported("im2col3d: zero kernel"));
    }
    if p.stride_t == 0 || p.stride_h == 0 || p.stride_w == 0 {
        return Err(SynaptixError::Unsupported("im2col3d: zero stride"));
    }
    let ot = p.out_t();
    let oh = p.out_h();
    let ow = p.out_w();
    let patches_per_image = ot * oh * ow;
    let patch_size = p.channels * p.kernel_t * p.kernel_h * p.kernel_w;
    let needed = p.batch * patches_per_image * patch_size * elem_size(dtype)?;
    if dst.len() < needed {
        return Err(SynaptixError::Other(format!(
            "im2col3d: dst too small, expected {} bytes, got {}",
            needed,
            dst.len()
        )));
    }
    match dtype {
        DType::F32 => im2col_3d_typed::<f32>(src, dst, p, ot, oh, ow),
        DType::F64 => im2col_3d_typed::<f64>(src, dst, p, ot, oh, ow),
        DType::F16 => im2col_3d_typed::<half::f16>(src, dst, p, ot, oh, ow),
        DType::BF16 => im2col_3d_typed::<half::bf16>(src, dst, p, ot, oh, ow),
        _ => Err(SynaptixError::Unsupported("im2col3d: dtype")),
    }?;
    Ok((p.batch * patches_per_image, patch_size))
}

fn im2col_3d_typed<T: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    src: &[u8],
    dst: &mut [u8],
    p: Im2ColParams3d,
    ot: usize,
    oh: usize,
    ow: usize,
) -> Result<()> {
    let src_t: &[T] = bytemuck::cast_slice(src);
    let dst_t: &mut [T] = bytemuck::cast_slice_mut(dst);
    let patches_per_image = ot * oh * ow;
    let patch_size = p.channels * p.kernel_t * p.kernel_h * p.kernel_w;
    for n in 0..p.batch {
        for pt in 0..ot {
            for ph in 0..oh {
                for pw in 0..ow {
                    let patch_row =
                        (n * patches_per_image + (pt * oh + ph) * ow + pw) * patch_size;
                    let mut idx = 0usize;
                    for c in 0..p.channels {
                        for kt in 0..p.kernel_t {
                            for ki in 0..p.kernel_h {
                                for kj in 0..p.kernel_w {
                                    let t = pt * p.stride_t + kt;
                                    let h = ph * p.stride_h + ki;
                                    let w = pw * p.stride_w + kj;
                                    let src_off = (((n * p.channels + c) * p.in_t + t) * p.in_h
                                        + h)
                                        * p.in_w
                                        + w;
                                    dst_t[patch_row + idx] = src_t[src_off];
                                    idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn elem_size(dtype: DType) -> Result<usize> {
    if dtype.is_sub_byte() || dtype.is_quantized() {
        return Err(SynaptixError::Unsupported("im2col: sub-byte/quant dtype"));
    }
    Ok((dtype.size_in_bits() / 8).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::tensor::shape::Shape;
    use synaptix_core::tensor::storage::CpuBuf;

    fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (x >> 33) as u32;
                ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn round(v: &[f32], dt: DType) -> Vec<f32> {
        let mut b = vec![0u8; v.len() * elem_bytes(dt)];
        row_from_f32(dt, v, &mut b, 0);
        let mut o = vec![0f32; v.len()];
        row_to_f32(dt, &b, 0, &mut o);
        o
    }

    fn tensor(v: &[f32], dims: Vec<usize>, dt: DType) -> (Storage, Layout) {
        let mut bytes = vec![0u8; v.len() * elem_bytes(dt)];
        row_from_f32(dt, v, &mut bytes, 0);
        (Storage::Cpu(CpuBuf::from_vec(bytes)), Layout::contiguous(Shape::new(dims), dt))
    }

    fn empty(dims: Vec<usize>, dt: DType) -> (Storage, Layout) {
        let lo = Layout::contiguous(Shape::new(dims), dt);
        (Storage::Cpu(CpuBuf::alloc_zeros(dt.bytes_for_numel(lo.numel()))), lo)
    }

    fn to_vec(st: &Storage, lo: &Layout) -> Vec<f32> {
        let mut v = vec![0f32; lo.numel()];
        row_to_f32(lo.dtype(), st.as_cpu().unwrap().as_bytes(), lo.offset(), &mut v);
        v
    }

    fn tol(dt: DType) -> f32 {
        if dt == DType::F32 { 1e-4 } else { 2e-2 }
    }

    fn check(got: &[f32], want: &[f32], dt: DType, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol(dt) * (1.0 + w.abs()), "{what} @{i}: {g} vs {w}");
        }
    }

    const DTS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

    type Dims3 = (usize, usize, usize);
    type Case2d = (usize, usize, usize, usize, usize, usize, usize, (usize, usize), (usize, usize), bool);
    type Case3d = (usize, usize, Dims3, usize, Dims3, Dims3, Dims3, bool);
    type CaseDw = (usize, usize, usize, usize, usize, usize, bool, bool);

    #[allow(clippy::too_many_arguments)]
    fn naive_conv3d(
        x: &[f32], w: &[f32], bias: Option<&[f32]>,
        b: usize, cin: usize, (d, h, wd): (usize, usize, usize),
        cout: usize, (kd, kh, kw): (usize, usize, usize),
        (sd, sh, sw): (usize, usize, usize), (pd, ph, pw): (usize, usize, usize),
    ) -> Vec<f32> {
        let od = (d + 2 * pd - kd) / sd + 1;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (wd + 2 * pw - kw) / sw + 1;
        let mut out = vec![0f32; b * cout * od * oh * ow];
        for bi in 0..b {
            for co in 0..cout {
                for z in 0..od {
                    for y in 0..oh {
                        for xx in 0..ow {
                            let mut acc = bias.map_or(0.0, |v| v[co]);
                            for ci in 0..cin {
                                for kz in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let iz = (z * sd + kz) as isize - pd as isize;
                                            let iy = (y * sh + ky) as isize - ph as isize;
                                            let ix = (xx * sw + kx) as isize - pw as isize;
                                            if iz < 0 || iy < 0 || ix < 0 || iz as usize >= d || iy as usize >= h || ix as usize >= wd {
                                                continue;
                                            }
                                            let xi = (((bi * cin + ci) * d + iz as usize) * h + iy as usize) * wd + ix as usize;
                                            let wi = (((co * cin + ci) * kd + kz) * kh + ky) * kw + kx;
                                            acc += x[xi] * w[wi];
                                        }
                                    }
                                }
                            }
                            out[(((bi * cout + co) * od + z) * oh + y) * ow + xx] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn conv2d_matches_naive() {
        let cases: [Case2d; 7] = [
            (1, 1, 5, 5, 1, 3, 3, (1, 1), (1, 1), false),
            (2, 3, 7, 5, 4, 3, 3, (1, 1), (1, 1), true),
            (2, 3, 9, 7, 5, 3, 3, (2, 2), (1, 1), true),
            (1, 4, 6, 11, 3, 2, 4, (2, 3), (0, 2), false),
            (2, 5, 8, 8, 6, 1, 1, (1, 1), (0, 0), true),
            (1, 2, 7, 9, 3, 1, 1, (2, 2), (0, 0), false),
            (1, 3, 4, 4, 2, 5, 5, (1, 1), (3, 2), true),
        ];
        for dt in DTS {
            for (ci_case, &(b, cin, h, w, cout, kh, kw, stride, pad, has_bias)) in cases.iter().enumerate() {
                let x = round(&det(ci_case as u64 + 1, b * cin * h * w, 1.0), dt);
                let wt = round(&det(ci_case as u64 + 100, cout * cin * kh * kw, 0.5), dt);
                let bias = round(&det(ci_case as u64 + 200, cout, 1.0), dt);
                let want = naive_conv3d(&x, &wt, has_bias.then_some(&bias[..]), b, cin, (1, h, w), cout, (1, kh, kw), (1, stride.0, stride.1), (0, pad.0, pad.1));
                let oh = (h + 2 * pad.0 - kh) / stride.0 + 1;
                let ow = (w + 2 * pad.1 - kw) / stride.1 + 1;
                let (xs, xl) = tensor(&x, vec![b, cin, h, w], dt);
                let (ws, wl) = tensor(&wt, vec![cout, cin, kh, kw], dt);
                let (bs, bl) = tensor(&bias, vec![cout], dt);
                let (mut os, ol) = empty(vec![b, cout, oh, ow], dt);
                conv2d((&xs, &xl), (&ws, &wl), has_bias.then_some((&bs, &bl)), (&mut os, &ol), stride, pad, &Stream::CpuNoop).unwrap();
                check(&to_vec(&os, &ol), &want, dt, &format!("conv2d {dt:?} case{ci_case}"));
                let g = ConvGeom { cin, d: 1, h, w, kd: 1, kh, kw, stride: (1, stride.0, stride.1), pad: (0, pad.0, pad.1), od: 1, oh, ow };
                for tile in [1usize, 3, 7] {
                    let (mut os2, ol2) = empty(vec![b, cout, oh, ow], dt);
                    conv_gemm((&xs, &xl), (&ws, &wl), has_bias.then_some((&bs, &bl)), (&mut os2, &ol2), &g, b, cout, tile * (g.k() + cout)).unwrap();
                    check(&to_vec(&os2, &ol2), &want, dt, &format!("conv2d tiled {dt:?} case{ci_case} tile{tile}"));
                }
            }
        }
    }

    #[test]
    fn conv3d_matches_naive() {
        let cases: [Case3d; 5] = [
            (1, 2, (4, 5, 5), 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), true),
            (2, 3, (5, 6, 7), 4, (2, 3, 2), (2, 1, 2), (0, 1, 1), false),
            (1, 1, (3, 4, 9), 2, (1, 1, 1), (1, 1, 1), (0, 0, 0), true),
            (1, 4, (6, 3, 4), 5, (3, 2, 2), (3, 2, 1), (1, 0, 1), true),
            (2, 2, (2, 7, 3), 3, (1, 3, 3), (1, 2, 1), (0, 2, 1), false),
        ];
        for dt in DTS {
            for (ci_case, &(b, cin, dims, cout, ks, stride, pad, has_bias)) in cases.iter().enumerate() {
                let (d, h, w) = dims;
                let x = round(&det(ci_case as u64 + 11, b * cin * d * h * w, 1.0), dt);
                let wt = round(&det(ci_case as u64 + 111, cout * cin * ks.0 * ks.1 * ks.2, 0.5), dt);
                let bias = round(&det(ci_case as u64 + 211, cout, 1.0), dt);
                let want = naive_conv3d(&x, &wt, has_bias.then_some(&bias[..]), b, cin, dims, cout, ks, stride, pad);
                let od = (d + 2 * pad.0 - ks.0) / stride.0 + 1;
                let oh = (h + 2 * pad.1 - ks.1) / stride.1 + 1;
                let ow = (w + 2 * pad.2 - ks.2) / stride.2 + 1;
                let (xs, xl) = tensor(&x, vec![b, cin, d, h, w], dt);
                let (ws, wl) = tensor(&wt, vec![cout, cin, ks.0, ks.1, ks.2], dt);
                let (bs, bl) = tensor(&bias, vec![cout], dt);
                let (mut os, ol) = empty(vec![b, cout, od, oh, ow], dt);
                conv3d((&xs, &xl), (&ws, &wl), has_bias.then_some((&bs, &bl)), (&mut os, &ol), stride, pad, &Stream::CpuNoop).unwrap();
                check(&to_vec(&os, &ol), &want, dt, &format!("conv3d {dt:?} case{ci_case}"));
                let g = ConvGeom { cin, d, h, w, kd: ks.0, kh: ks.1, kw: ks.2, stride, pad, od, oh, ow };
                let (mut os2, ol2) = empty(vec![b, cout, od, oh, ow], dt);
                conv_gemm((&xs, &xl), (&ws, &wl), has_bias.then_some((&bs, &bl)), (&mut os2, &ol2), &g, b, cout, 5 * (g.k() + cout)).unwrap();
                check(&to_vec(&os2, &ol2), &want, dt, &format!("conv3d tiled {dt:?} case{ci_case}"));
            }
        }
    }

    #[test]
    fn dwconv1d_matches_naive() {
        let cases: [CaseDw; 8] = [
            (1, 2, 9, 3, 1, 1, false, true),
            (2, 3, 11, 4, 2, 1, false, false),
            (1, 4, 7, 5, 3, 4, false, true),
            (2, 2, 6, 3, 1, 0, false, false),
            (1, 3, 5, 4, 1, 0, true, true),
            (2, 2, 7, 6, 3, 0, true, false),
            (1, 5, 4, 3, 2, 0, true, true),
            (1, 1, 1, 2, 2, 1, false, true),
        ];
        for dt in DTS {
            for (ci_case, &(b, c, l, k, s, pad, transpose, has_bias)) in cases.iter().enumerate() {
                let x = round(&det(ci_case as u64 + 21, b * c * l, 1.0), dt);
                let wt = round(&det(ci_case as u64 + 121, c * k, 0.5), dt);
                let bias = round(&det(ci_case as u64 + 221, c, 1.0), dt);
                let lo = if transpose { (l - 1) * s + k } else { (l + 2 * pad - k) / s + 1 };
                let mut want = vec![0f32; b * c * lo];
                for bi in 0..b {
                    for ch in 0..c {
                        for t in 0..lo {
                            let mut acc = if has_bias { bias[ch] } else { 0.0 };
                            for kk in 0..k {
                                if transpose {
                                    if t < kk || (t - kk) % s != 0 || (t - kk) / s >= l {
                                        continue;
                                    }
                                    acc += x[(bi * c + ch) * l + (t - kk) / s] * wt[ch * k + kk];
                                } else {
                                    let xi = (t * s + kk) as isize - pad as isize;
                                    if xi < 0 || xi as usize >= l {
                                        continue;
                                    }
                                    acc += x[(bi * c + ch) * l + xi as usize] * wt[ch * k + kk];
                                }
                            }
                            want[(bi * c + ch) * lo + t] = acc;
                        }
                    }
                }
                let (xs, xl) = tensor(&x, vec![b, c, l], dt);
                let (ws, wl) = tensor(&wt, vec![c, 1, k], dt);
                let (bs, bl) = tensor(&bias, vec![c], dt);
                let (mut os, ol) = empty(vec![b, c, lo], dt);
                dwconv1d((&xs, &xl), (&ws, &wl), has_bias.then_some((&bs, &bl)), (&mut os, &ol), s, pad, transpose, &Stream::CpuNoop).unwrap();
                check(&to_vec(&os, &ol), &want, dt, &format!("dwconv1d {dt:?} case{ci_case}"));
            }
        }
    }

    #[test]
    fn im2col_rows_matches_naive() {
        for &(c, h, w, kh, kw, stride, pad) in &[(2usize, 5usize, 6usize, 3usize, 3usize, (1usize, 1usize), (1usize, 1usize)), (3, 7, 4, 2, 5, (2, 1), (0, 3)), (1, 4, 4, 4, 4, (1, 1), (2, 2))] {
            let b = 2;
            let h_out = (h + 2 * pad.0 - kh) / stride.0 + 1;
            let w_out = (w + 2 * pad.1 - kw) / stride.1 + 1;
            let x: Vec<f32> = (0..b * c * h * w).map(|i| i as f32 + 1.0).collect();
            let k = c * kh * kw;
            let m_total = b * h_out * w_out;
            let (m_offset, m_count) = (1usize, m_total - 2);
            let mut dst = vec![0u8; m_count * k * 4];
            let p = Im2ColRows { c, h, w, kh, kw, h_out, w_out, stride, padding: pad, m_offset, m_count };
            im2col_rows::<f32>(bytemuck::cast_slice(&x), 0, &mut dst, 0, p);
            let got: &[f32] = bytemuck::cast_slice(&dst);
            for r in 0..m_count {
                let m = m_offset + r;
                let (bi, rem) = (m / (h_out * w_out), m % (h_out * w_out));
                let (oy, ox) = (rem / w_out, rem % w_out);
                for ci in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * stride.0 + ky) as isize - pad.0 as isize;
                            let ix = (ox * stride.1 + kx) as isize - pad.1 as isize;
                            let want = if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                x[((bi * c + ci) * h + iy as usize) * w + ix as usize]
                            } else {
                                0.0
                            };
                            assert_eq!(got[r * k + (ci * kh + ky) * kw + kx], want, "r{r} c{ci} ky{ky} kx{kx}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn im2col_2d_identity_patch1() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 2,
            in_w: 2,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 1);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn im2col_2d_patch2_stride2() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 4,
            in_w: 4,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 4 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 4);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(&out[0..4], &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(&out[4..8], &[3.0, 4.0, 7.0, 8.0]);
        assert_eq!(&out[8..12], &[9.0, 10.0, 13.0, 14.0]);
        assert_eq!(&out[12..16], &[11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn im2col_2d_padding() {
        let p = Im2ColParams2d {
            batch: 1,
            channels: 1,
            in_h: 2,
            in_w: 2,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
        };
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src_bytes = bytemuck::cast_slice(&src).to_vec();
        let mut dst = vec![0u8; 4 * 9 * 4];
        let (rows, cols) = im2col_2d_dispatch(DType::F32, &src_bytes, &mut dst, p).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 9);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert_eq!(&out[0..9], &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
    }
}
