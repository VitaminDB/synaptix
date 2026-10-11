use rayon::prelude::*;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;

use crate::common::{check_contig, cpu, cpu_mut, elem_bytes, row_from_f32, row_to_f32};
use crate::quant::{decode_e4m3, e8m0_decode, MXFP8_BLOCK};

fn plain_elem_bytes(dt: DType) -> Result<usize> {
    if dt.is_sub_byte() || dt.is_quantized() {
        return Err(SynaptixError::Unsupported("cpu gather: sub-byte/quant dtype"));
    }
    Ok(elem_bytes(dt).max(1))
}

fn float_dtype(dt: DType, what: &'static str) -> Result<()> {
    if matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        Ok(())
    } else {
        Err(SynaptixError::Unsupported(what))
    }
}

fn u32_slice<'a>(st: &'a Storage, lo: &Layout, what: &'static str) -> Result<&'a [u32]> {
    if lo.dtype() != DType::U32 {
        return Err(SynaptixError::Unsupported(what));
    }
    check_contig(lo)?;
    let bytes = cpu(st, what)?.as_bytes();
    let n = lo.numel();
    Ok(bytemuck::cast_slice(&bytes[lo.offset() * 4..(lo.offset() + n) * 4]))
}

pub fn embed_gather(
    table: (&Storage, &Layout),
    ids: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (t_st, t_lo) = table;
    let (i_st, i_lo) = ids;
    let (o_st, o_lo) = out;
    if t_lo.dims().len() != 2 {
        return Err(SynaptixError::Unsupported("cpu embed_gather: table must be [vocab,dim]"));
    }
    check_contig(t_lo)?;
    check_contig(o_lo)?;
    let dt = t_lo.dtype();
    if o_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, o_lo.dtype()));
    }
    let esz = plain_elem_bytes(dt)?;
    let (vocab, dim) = (t_lo.dims()[0], t_lo.dims()[1]);
    let ids_s = u32_slice(i_st, i_lo, "cpu embed_gather: ids must be U32")?;
    let n = ids_s.len();
    if o_lo.dims() != [n, dim] {
        return Err(SynaptixError::shape_mismatch(&[n, dim], o_lo.dims()));
    }
    let tb = cpu(t_st, "cpu embed_gather: table non-cpu")?.as_bytes();
    let tb = &tb[t_lo.offset() * esz..(t_lo.offset() + vocab * dim) * esz];
    let ob = cpu_mut(o_st, "cpu embed_gather: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + n * dim) * esz];
    let row = dim * esz;
    if row == 0 {
        return Ok(());
    }
    ob.par_chunks_mut(row).zip(ids_s.par_iter()).for_each(|(dst, &id)| {
        let id = id as usize;
        if id < vocab {
            dst.copy_from_slice(&tb[id * row..(id + 1) * row]);
        } else {
            dst.fill(0);
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn embed_gather_mxfp8(
    table: &Storage,
    scales: &Storage,
    ids: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    vocab: usize,
    dim: usize,
    _stream: &Stream,
) -> Result<()> {
    let (i_st, i_lo) = ids;
    let (o_st, o_lo) = out;
    let nb = dim / MXFP8_BLOCK;
    if nb * MXFP8_BLOCK != dim {
        return Err(SynaptixError::Unsupported("cpu embed_gather_mxfp8: dim % 32 != 0"));
    }
    let dt = o_lo.dtype();
    float_dtype(dt, "cpu embed_gather_mxfp8: out dtype")?;
    check_contig(o_lo)?;
    let ids_s = u32_slice(i_st, i_lo, "cpu embed_gather_mxfp8: ids must be U32")?;
    let n = ids_s.len();
    if o_lo.dims() != [n, dim] {
        return Err(SynaptixError::shape_mismatch(&[n, dim], o_lo.dims()));
    }
    let tb = cpu(table, "cpu embed_gather_mxfp8: table non-cpu")?.as_bytes();
    let sb = cpu(scales, "cpu embed_gather_mxfp8: scales non-cpu")?.as_bytes();
    if tb.len() < vocab * dim || sb.len() < vocab * nb {
        return Err(SynaptixError::Unsupported("cpu embed_gather_mxfp8: table/scales too small"));
    }
    let esz = elem_bytes(dt);
    let ob = cpu_mut(o_st, "cpu embed_gather_mxfp8: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + n * dim) * esz];
    ob.par_chunks_mut(dim * esz).zip(ids_s.par_iter()).for_each(|(dst, &id)| {
        let id = id as usize;
        if id >= vocab {
            dst.fill(0);
            return;
        }
        let mut row = vec![0f32; dim];
        let src = &tb[id * dim..(id + 1) * dim];
        let sc = &sb[id * nb..(id + 1) * nb];
        for (blk, &sbyte) in sc.iter().enumerate() {
            let sv = e8m0_decode(sbyte);
            let base = blk * MXFP8_BLOCK;
            for i in 0..MXFP8_BLOCK {
                row[base + i] = decode_e4m3(src[base + i]) * sv;
            }
        }
        row_from_f32(dt, &row, dst, 0);
    });
    Ok(())
}

#[inline]
fn before(a: (f32, u32), b: (f32, u32)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

fn cmp_desc(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    if before(*a, *b) {
        std::cmp::Ordering::Less
    } else if before(*b, *a) {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

fn select_topk(row: &[f32], k: usize, best: &mut Vec<(f32, u32)>) {
    best.clear();
    if k <= 16 {
        for (i, &v) in row.iter().enumerate() {
            let v = if v.is_nan() { f32::NEG_INFINITY } else { v };
            if best.len() == k && best[k - 1].0 >= v {
                continue;
            }
            let pos = best.partition_point(|e| e.0 >= v);
            best.insert(pos, (v, i as u32));
            best.truncate(k);
        }
    } else {
        best.extend(row.iter().enumerate().map(|(i, &v)| (if v.is_nan() { f32::NEG_INFINITY } else { v }, i as u32)));
        if k < best.len() {
            best.select_nth_unstable_by(k - 1, cmp_desc);
            best.truncate(k);
        }
        best.sort_unstable_by(cmp_desc);
    }
}

pub fn topk_rows(
    scores: (&Storage, &Layout),
    indices: (&mut Storage, &Layout),
    values: (&mut Storage, &Layout),
    k: usize,
    _stream: &Stream,
) -> Result<()> {
    let (s_st, s_lo) = scores;
    let (i_st, i_lo) = indices;
    let (v_st, v_lo) = values;
    let dt = s_lo.dtype();
    float_dtype(dt, "cpu topk_rows: dtype")?;
    if s_lo.dims().len() != 2 {
        return Err(SynaptixError::Unsupported("cpu topk_rows: expect [rows, cols]"));
    }
    check_contig(s_lo)?;
    check_contig(i_lo)?;
    check_contig(v_lo)?;
    let (rows, cols) = (s_lo.dims()[0], s_lo.dims()[1]);
    if k == 0 || k > cols {
        return Err(SynaptixError::Unsupported("cpu topk_rows: k out of range"));
    }
    if i_lo.dtype() != DType::U32 {
        return Err(SynaptixError::Unsupported("cpu topk_rows: indices must be U32"));
    }
    if v_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, v_lo.dtype()));
    }
    if i_lo.dims() != [rows, k] || v_lo.dims() != [rows, k] {
        return Err(SynaptixError::shape_mismatch(&[rows, k], i_lo.dims()));
    }
    let esz = elem_bytes(dt);
    let sb = cpu(s_st, "cpu topk_rows: scores non-cpu")?.as_bytes();
    let sb = &sb[s_lo.offset() * esz..(s_lo.offset() + rows * cols) * esz];
    let ib = cpu_mut(i_st, "cpu topk_rows: indices non-cpu")?.as_bytes_mut();
    let ib = &mut ib[i_lo.offset() * 4..(i_lo.offset() + rows * k) * 4];
    let vb = cpu_mut(v_st, "cpu topk_rows: values non-cpu")?.as_bytes_mut();
    let vb = &mut vb[v_lo.offset() * esz..(v_lo.offset() + rows * k) * esz];
    ib.par_chunks_mut(k * 4).zip(vb.par_chunks_mut(k * esz)).enumerate().for_each(|(r, (ib, vb))| {
        let mut row = vec![0f32; cols];
        row_to_f32(dt, sb, r * cols, &mut row);
        let mut best = Vec::with_capacity(k.min(cols));
        select_topk(&row, k, &mut best);
        let idx: &mut [u32] = bytemuck::cast_slice_mut(ib);
        let mut vals = vec![0f32; k];
        for (s, &(v, i)) in best.iter().enumerate() {
            idx[s] = i;
            vals[s] = v;
        }
        row_from_f32(dt, &vals, vb, 0);
    });
    Ok(())
}

pub fn topk_wide(
    scores: (&Storage, &Layout),
    valid: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    k: usize,
    _stream: &Stream,
) -> Result<()> {
    let (s_st, s_lo) = scores;
    let (v_st, v_lo) = valid;
    let (o_st, o_lo) = out;
    let dt = s_lo.dtype();
    float_dtype(dt, "cpu topk_wide: dtype")?;
    if s_lo.dims().len() != 2 {
        return Err(SynaptixError::Unsupported("cpu topk_wide: expect [rows, cols]"));
    }
    check_contig(s_lo)?;
    check_contig(o_lo)?;
    let (rows, cols) = (s_lo.dims()[0], s_lo.dims()[1]);
    if k == 0 || k > cols {
        return Err(SynaptixError::Unsupported("cpu topk_wide: k out of range"));
    }
    if o_lo.dtype() != DType::U32 || o_lo.dims() != [rows, k] {
        return Err(SynaptixError::Unsupported("cpu topk_wide: out must be U32 [rows, k]"));
    }
    let valid_s = u32_slice(v_st, v_lo, "cpu topk_wide: valid must be U32")?;
    if valid_s.len() != rows && valid_s.len() != 1 {
        return Err(SynaptixError::shape_mismatch(&[rows], v_lo.dims()));
    }
    let esz = elem_bytes(dt);
    let sb = cpu(s_st, "cpu topk_wide: scores non-cpu")?.as_bytes();
    let sb = &sb[s_lo.offset() * esz..(s_lo.offset() + rows * cols) * esz];
    let ob = cpu_mut(o_st, "cpu topk_wide: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * 4..(o_lo.offset() + rows * k) * 4];
    ob.par_chunks_mut(k * 4).enumerate().for_each(|(r, ob)| {
        let dst: &mut [u32] = bytemuck::cast_slice_mut(ob);
        let n = (valid_s[if valid_s.len() == 1 { 0 } else { r }] as usize).min(cols);
        if n <= k {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = if i < n { i as u32 } else { u32::MAX };
            }
            return;
        }
        let mut row = vec![0f32; n];
        row_to_f32(dt, sb, r * cols, &mut row);
        let mut best = Vec::with_capacity(if k <= 16 { k } else { n });
        select_topk(&row, k, &mut best);
        for (d, &(_, i)) in dst.iter_mut().zip(best.iter()) {
            *d = i;
        }
    });
    Ok(())
}

pub fn kv_append_dev(
    dst: (&mut Storage, &Layout),
    src: (&Storage, &Layout),
    seq_pos: (&Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (dst_st, dst_lo) = dst;
    let (src_st, src_lo) = src;
    let (sp_st, sp_lo) = seq_pos;
    if src_lo.dtype() != dst_lo.dtype() {
        return Err(SynaptixError::dtype_mismatch(src_lo.dtype(), dst_lo.dtype()));
    }
    let sp = u32_slice(sp_st, sp_lo, "cpu kv_append_dev: seq_pos must be U32")?;
    if sp.is_empty() {
        return Err(SynaptixError::Unsupported("cpu kv_append_dev: seq_pos empty"));
    }
    let seq_pos = sp[0] as usize;
    if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported("cpu kv_append_dev: expect rank-4 [B,nkv,T,hd]"));
    }
    check_contig(dst_lo)?;
    check_contig(src_lo)?;
    let dtype = dst_lo.dtype();
    let (b, nkv, max_seq, hd) = (dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3]);
    let (sb, snkv, t_new, shd) = (src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3]);
    if sb != b || snkv != nkv || shd != hd {
        return Err(SynaptixError::shape_mismatch(src_lo.dims(), dst_lo.dims()));
    }
    if seq_pos + t_new > max_seq {
        return Err(SynaptixError::Other(format!(
            "cpu kv_append_dev: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
        )));
    }
    let src_buf = cpu(src_st, "cpu kv_append_dev: src non-cpu")?;
    let dst_buf = cpu_mut(dst_st, "cpu kv_append_dev: dst non-cpu")?;
    let block = t_new * hd;
    let n_bytes = dtype.bytes_for_numel(block);
    for bh in 0..(b * nkv) {
        let src_o = dtype.bytes_for_numel(src_lo.offset() + bh * block);
        let dst_o = dtype.bytes_for_numel(dst_lo.offset() + (bh * max_seq + seq_pos) * hd);
        dst_buf.as_bytes_mut()[dst_o..dst_o + n_bytes].copy_from_slice(&src_buf.as_bytes()[src_o..src_o + n_bytes]);
    }
    Ok(())
}

pub fn upsample_nearest2x(
    input: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let (i_st, i_lo) = input;
    let (o_st, o_lo) = out;
    if i_lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported("cpu upsample_nearest2x: rank != 4"));
    }
    check_contig(i_lo)?;
    check_contig(o_lo)?;
    let dt = i_lo.dtype();
    if o_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, o_lo.dtype()));
    }
    let esz = plain_elem_bytes(dt)?;
    let (b, c, h, w) = (i_lo.dims()[0], i_lo.dims()[1], i_lo.dims()[2], i_lo.dims()[3]);
    if o_lo.dims() != [b, c, 2 * h, 2 * w] {
        return Err(SynaptixError::shape_mismatch(&[b, c, 2 * h, 2 * w], o_lo.dims()));
    }
    let ib = cpu(i_st, "cpu upsample_nearest2x: input non-cpu")?.as_bytes();
    let ib = &ib[i_lo.offset() * esz..(i_lo.offset() + b * c * h * w) * esz];
    let ob = cpu_mut(o_st, "cpu upsample_nearest2x: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + b * c * 4 * h * w) * esz];
    if h == 0 || w == 0 {
        return Ok(());
    }
    ob.par_chunks_mut(2 * w * esz).enumerate().for_each(|(orow, dst)| {
        let plane = orow / (2 * h);
        let iy = (orow % (2 * h)) / 2;
        let src = &ib[(plane * h + iy) * w * esz..(plane * h + iy + 1) * w * esz];
        match esz {
            1 => {
                for (d, &s) in dst.chunks_exact_mut(2).zip(src.iter()) {
                    d[0] = s;
                    d[1] = s;
                }
            }
            2 => {
                let s: &[u16] = bytemuck::cast_slice(src);
                let d: &mut [u16] = bytemuck::cast_slice_mut(dst);
                for (d, &s) in d.chunks_exact_mut(2).zip(s.iter()) {
                    d[0] = s;
                    d[1] = s;
                }
            }
            4 => {
                let s: &[u32] = bytemuck::cast_slice(src);
                let d: &mut [u32] = bytemuck::cast_slice_mut(dst);
                for (d, &s) in d.chunks_exact_mut(2).zip(s.iter()) {
                    d[0] = s;
                    d[1] = s;
                }
            }
            _ => {
                for (d, s) in dst.chunks_exact_mut(2 * esz).zip(src.chunks_exact(esz)) {
                    d[..esz].copy_from_slice(s);
                    d[esz..].copy_from_slice(s);
                }
            }
        }
    });
    Ok(())
}

fn transpose_planes(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, esz: usize) {
    let plane = rows * cols;
    dst.par_chunks_mut(plane * esz).enumerate().for_each(|(b, dp)| {
        let sp = &src[b * plane * esz..(b + 1) * plane * esz];
        const T: usize = 32;
        for r0 in (0..rows).step_by(T) {
            let r1 = (r0 + T).min(rows);
            for c0 in (0..cols).step_by(T) {
                let c1 = (c0 + T).min(cols);
                for r in r0..r1 {
                    for c in c0..c1 {
                        let s = (r * cols + c) * esz;
                        let d = (c * rows + r) * esz;
                        dp[d..d + esz].copy_from_slice(&sp[s..s + esz]);
                    }
                }
            }
        }
    });
}

fn permute_hwc(src: (&Storage, &Layout), dst: (&mut Storage, &Layout), to_nhwc: bool) -> Result<()> {
    let (s_st, s_lo) = src;
    let (d_st, d_lo) = dst;
    let what = if to_nhwc { "cpu nchw_to_nhwc: rank != 4" } else { "cpu nhwc_to_nchw: rank != 4" };
    if s_lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported(what));
    }
    check_contig(s_lo)?;
    check_contig(d_lo)?;
    let dt = s_lo.dtype();
    if d_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, d_lo.dtype()));
    }
    let esz = plain_elem_bytes(dt)?;
    let d = s_lo.dims();
    let (b, rows, cols) = if to_nhwc { (d[0], d[1], d[2] * d[3]) } else { (d[0], d[1] * d[2], d[3]) };
    let want = if to_nhwc { [d[0], d[2], d[3], d[1]] } else { [d[0], d[3], d[1], d[2]] };
    if d_lo.dims() != want {
        return Err(SynaptixError::shape_mismatch(&want, d_lo.dims()));
    }
    let n = b * rows * cols;
    let sb = cpu(s_st, "cpu permute: src non-cpu")?.as_bytes();
    let sb = &sb[s_lo.offset() * esz..(s_lo.offset() + n) * esz];
    let db = cpu_mut(d_st, "cpu permute: dst non-cpu")?.as_bytes_mut();
    let db = &mut db[d_lo.offset() * esz..(d_lo.offset() + n) * esz];
    if n == 0 {
        return Ok(());
    }
    transpose_planes(sb, db, rows, cols, esz);
    Ok(())
}

pub fn nchw_to_nhwc(src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
    permute_hwc(src, dst, true)
}

pub fn nhwc_to_nchw(src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
    permute_hwc(src, dst, false)
}

fn opt_f32(t: Option<(&Storage, &Layout)>, dt: DType, dims: &[usize], what: &'static str) -> Result<Option<Vec<f32>>> {
    let Some((st, lo)) = t else { return Ok(None) };
    if lo.dtype() != dt || lo.dims() != dims {
        return Err(SynaptixError::Unsupported(what));
    }
    check_contig(lo)?;
    let bytes = cpu(st, what)?.as_bytes();
    let mut v = vec![0f32; lo.numel()];
    row_to_f32(dt, bytes, lo.offset(), &mut v);
    Ok(Some(v))
}

#[allow(clippy::too_many_arguments)]
pub fn conv_epilogue(
    out2d: (&Storage, &Layout),
    bias: Option<(&Storage, &Layout)>,
    residual: Option<(&Storage, &Layout)>,
    temb_bc: Option<(&Storage, &Layout)>,
    out: (&mut Storage, &Layout),
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    _stream: &Stream,
) -> Result<()> {
    let (i_st, i_lo) = out2d;
    let (o_st, o_lo) = out;
    let dt = i_lo.dtype();
    float_dtype(dt, "cpu conv_epilogue: dtype")?;
    check_contig(i_lo)?;
    check_contig(o_lo)?;
    let hw = h * w;
    if i_lo.dims() != [b * hw, c] {
        return Err(SynaptixError::shape_mismatch(&[b * hw, c], i_lo.dims()));
    }
    if o_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, o_lo.dtype()));
    }
    if o_lo.dims() != [b, c, h, w] {
        return Err(SynaptixError::shape_mismatch(&[b, c, h, w], o_lo.dims()));
    }
    let bias_v = opt_f32(bias, dt, &[c], "cpu conv_epilogue: bias shape/dtype")?;
    let temb_v = opt_f32(temb_bc, dt, &[b, c], "cpu conv_epilogue: temb shape/dtype")?;
    let res = match residual {
        Some((st, lo)) => {
            if lo.dtype() != dt || lo.dims() != [b, c, h, w] {
                return Err(SynaptixError::Unsupported("cpu conv_epilogue: residual shape/dtype"));
            }
            check_contig(lo)?;
            Some((cpu(st, "cpu conv_epilogue: residual non-cpu")?.as_bytes(), lo.offset()))
        }
        None => None,
    };
    let esz = elem_bytes(dt);
    let n = b * c * hw;
    let ib = cpu(i_st, "cpu conv_epilogue: out2d non-cpu")?.as_bytes();
    let ib = &ib[i_lo.offset() * esz..(i_lo.offset() + n) * esz];
    let ob = cpu_mut(o_st, "cpu conv_epilogue: out non-cpu")?.as_bytes_mut();
    let ob = &mut ob[o_lo.offset() * esz..(o_lo.offset() + n) * esz];
    if n == 0 || hw == 0 {
        return Ok(());
    }
    let load1 = |idx: usize| -> f32 {
        match dt {
            DType::F32 => f32::from_le_bytes([ib[idx * 4], ib[idx * 4 + 1], ib[idx * 4 + 2], ib[idx * 4 + 3]]),
            DType::F16 => half::f16::from_bits(u16::from_le_bytes([ib[idx * 2], ib[idx * 2 + 1]])).to_f32(),
            _ => crate::common::bf16_bits_to_f32(u16::from_le_bytes([ib[idx * 2], ib[idx * 2 + 1]])),
        }
    };
    const CH: usize = 16;
    const PT: usize = 2048;
    ob.par_chunks_mut(c * hw * esz).enumerate().for_each(|(bi, ob_b)| {
        ob_b.par_chunks_mut(CH * hw * esz).enumerate().for_each(|(chunk, planes)| {
            let c0 = chunk * CH;
            let c1 = (c0 + CH).min(c);
            let mut acc = vec![0f32; PT];
            let mut r = vec![0f32; PT];
            for p0 in (0..hw).step_by(PT) {
                let p1 = (p0 + PT).min(hw);
                let len = p1 - p0;
                for ci in c0..c1 {
                    for (j, a) in acc[..len].iter_mut().enumerate() {
                        *a = load1((bi * hw + p0 + j) * c + ci);
                    }
                    let mut add = 0f32;
                    if let Some(bv) = &bias_v {
                        add += bv[ci];
                    }
                    if let Some(tv) = &temb_v {
                        add += tv[bi * c + ci];
                    }
                    if add != 0.0 {
                        for a in acc[..len].iter_mut() {
                            *a += add;
                        }
                    }
                    if let Some((rb, ro)) = res {
                        row_to_f32(dt, rb, ro + (bi * c + ci) * hw + p0, &mut r[..len]);
                        for (a, rv) in acc[..len].iter_mut().zip(r.iter()) {
                            *a += rv;
                        }
                    }
                    row_from_f32(dt, &acc[..len], planes, (ci - c0) * hw + p0);
                }
            }
        });
    });
    Ok(())
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

    fn tensor(v: &[f32], dims: Vec<usize>, dt: DType) -> (Storage, Layout) {
        let mut bytes = vec![0u8; v.len() * elem_bytes(dt)];
        if dt == DType::U8 {
            for (b, &x) in bytes.iter_mut().zip(v) {
                *b = x as u8;
            }
        } else {
            row_from_f32(dt, v, &mut bytes, 0);
        }
        (Storage::Cpu(CpuBuf::from_vec(bytes)), Layout::contiguous(Shape::new(dims), dt))
    }

    fn u32_tensor(v: &[u32], dims: Vec<usize>) -> (Storage, Layout) {
        (
            Storage::Cpu(CpuBuf::from_vec(bytemuck::cast_slice(v).to_vec())),
            Layout::contiguous(Shape::new(dims), DType::U32),
        )
    }

    fn empty(dims: Vec<usize>, dt: DType) -> (Storage, Layout) {
        let lo = Layout::contiguous(Shape::new(dims), dt);
        (Storage::Cpu(CpuBuf::alloc_zeros(dt.bytes_for_numel(lo.numel()))), lo)
    }

    fn to_vec(st: &Storage, lo: &Layout) -> Vec<f32> {
        let bytes = st.as_cpu().unwrap().as_bytes();
        if lo.dtype() == DType::U8 {
            return bytes[..lo.numel()].iter().map(|&b| b as f32).collect();
        }
        let mut v = vec![0f32; lo.numel()];
        row_to_f32(lo.dtype(), bytes, lo.offset(), &mut v);
        v
    }

    fn to_vec_u32(st: &Storage, lo: &Layout) -> Vec<u32> {
        bytemuck::cast_slice(&st.as_cpu().unwrap().as_bytes()[..lo.numel() * 4]).to_vec()
    }

    fn round(v: &[f32], dt: DType) -> Vec<f32> {
        let mut b = vec![0u8; v.len() * elem_bytes(dt)];
        row_from_f32(dt, v, &mut b, 0);
        let mut o = vec![0f32; v.len()];
        row_to_f32(dt, &b, 0, &mut o);
        o
    }

    fn tol(dt: DType) -> f32 {
        if dt == DType::F32 { 1e-4 } else { 2e-2 }
    }

    const DTS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

    #[test]
    fn embed_gather_matches_naive() {
        for dt in DTS {
            let (vocab, dim) = (7usize, 5usize);
            let t = round(&det(1, vocab * dim, 3.0), dt);
            let ids = [3u32, 0, 6, 9, 3, 2];
            let (ts, tl) = tensor(&t, vec![vocab, dim], dt);
            let (is, il) = u32_tensor(&ids, vec![ids.len()]);
            let (mut os, ol) = empty(vec![ids.len(), dim], dt);
            embed_gather((&ts, &tl), (&is, &il), (&mut os, &ol), &Stream::CpuNoop).unwrap();
            let got = to_vec(&os, &ol);
            for (n, &id) in ids.iter().enumerate() {
                for d in 0..dim {
                    let want = if (id as usize) < vocab { t[id as usize * dim + d] } else { 0.0 };
                    assert_eq!(got[n * dim + d], want, "{dt:?} n{n} d{d}");
                }
            }
        }
    }

    #[test]
    fn embed_gather_mxfp8_matches_naive() {
        let (vocab, dim) = (5usize, 64usize);
        let vals = det(2, vocab * dim, 10.0);
        let mut table = vec![0u8; vocab * dim];
        let mut scales = vec![0u8; vocab * dim / 32];
        let mut deq = vec![0f32; vocab * dim];
        for blk in 0..vocab * dim / 32 {
            let amax = vals[blk * 32..(blk + 1) * 32].iter().fold(0f32, |m, v| m.max(v.abs()));
            let sb = crate::quant::e8m0_scale_byte(amax);
            let sv = e8m0_decode(sb);
            scales[blk] = sb;
            for i in 0..32 {
                let q = crate::quant::encode_e4m3(vals[blk * 32 + i] / sv);
                table[blk * 32 + i] = q;
                deq[blk * 32 + i] = decode_e4m3(q) * sv;
            }
        }
        let ids = [4u32, 1, 7, 0];
        for dt in DTS {
            let ts = Storage::Cpu(CpuBuf::from_vec(table.clone()));
            let ss = Storage::Cpu(CpuBuf::from_vec(scales.clone()));
            let (is, il) = u32_tensor(&ids, vec![ids.len()]);
            let (mut os, ol) = empty(vec![ids.len(), dim], dt);
            embed_gather_mxfp8(&ts, &ss, (&is, &il), (&mut os, &ol), vocab, dim, &Stream::CpuNoop).unwrap();
            let got = to_vec(&os, &ol);
            for (n, &id) in ids.iter().enumerate() {
                for d in 0..dim {
                    let want = if (id as usize) < vocab { deq[id as usize * dim + d] } else { 0.0 };
                    assert!((got[n * dim + d] - want).abs() <= tol(dt) * (1.0 + want.abs()), "{dt:?} n{n} d{d}: {} vs {want}", got[n * dim + d]);
                }
            }
        }
    }

    fn naive_topk(row: &[f32], k: usize) -> Vec<(f32, u32)> {
        let mut v: Vec<(f32, u32)> = row.iter().enumerate().map(|(i, &x)| (x, i as u32)).collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        v.truncate(k);
        v
    }

    #[test]
    fn topk_rows_matches_naive() {
        for dt in DTS {
            for &(rows, cols, k) in &[(3usize, 9usize, 1usize), (4, 17, 4), (2, 40, 20), (5, 64, 33), (1, 5, 5)] {
                let mut s = det(rows as u64 * 31 + cols as u64, rows * cols, 2.0);
                for i in (0..s.len()).step_by(5) {
                    s[i] = 0.5;
                }
                let s = round(&s, dt);
                let (ss, sl) = tensor(&s, vec![rows, cols], dt);
                let (mut is, il) = empty(vec![rows, k], DType::U32);
                let (mut vs, vl) = empty(vec![rows, k], dt);
                topk_rows((&ss, &sl), (&mut is, &il), (&mut vs, &vl), k, &Stream::CpuNoop).unwrap();
                let gi = to_vec_u32(&is, &il);
                let gv = to_vec(&vs, &vl);
                for r in 0..rows {
                    let want = naive_topk(&s[r * cols..(r + 1) * cols], k);
                    for j in 0..k {
                        assert_eq!(gi[r * k + j], want[j].1, "{dt:?} {rows}x{cols} k{k} r{r} j{j}");
                        assert_eq!(gv[r * k + j], want[j].0, "{dt:?} {rows}x{cols} k{k} r{r} j{j}");
                    }
                }
            }
        }
    }

    #[test]
    fn topk_wide_matches_naive() {
        let (rows, cols, k) = (4usize, 50usize, 6usize);
        let s: Vec<f32> = det(7, rows * cols, 1.0).iter().map(|v| v.max(0.0)).collect();
        let valid = [50u32, 4, 6, 20];
        let (ss, sl) = tensor(&s, vec![rows, cols], DType::F32);
        let (vs, vl) = u32_tensor(&valid, vec![rows]);
        let (mut os, ol) = empty(vec![rows, k], DType::U32);
        topk_wide((&ss, &sl), (&vs, &vl), (&mut os, &ol), k, &Stream::CpuNoop).unwrap();
        let got = to_vec_u32(&os, &ol);
        for r in 0..rows {
            let n = valid[r] as usize;
            let want: Vec<u32> = if n <= k {
                (0..k).map(|i| if i < n { i as u32 } else { u32::MAX }).collect()
            } else {
                naive_topk(&s[r * cols..r * cols + n], k).iter().map(|e| e.1).collect()
            };
            assert_eq!(&got[r * k..(r + 1) * k], &want[..], "r{r}");
        }
    }

    #[test]
    fn kv_append_dev_writes_slot() {
        for dt in DTS {
            let (b, nkv, max_seq, hd, t_new) = (2usize, 3usize, 7usize, 6usize, 2usize);
            let src = round(&det(3, b * nkv * t_new * hd, 1.0), dt);
            let (ss, sl) = tensor(&src, vec![b, nkv, t_new, hd], dt);
            let (mut ds, dl) = empty(vec![b, nkv, max_seq, hd], dt);
            let (ps, pl) = u32_tensor(&[4], vec![1]);
            kv_append_dev((&mut ds, &dl), (&ss, &sl), (&ps, &pl), &Stream::CpuNoop).unwrap();
            let got = to_vec(&ds, &dl);
            for bh in 0..b * nkv {
                for t in 0..max_seq {
                    for d in 0..hd {
                        let want = if (4..4 + t_new).contains(&t) { src[(bh * t_new + t - 4) * hd + d] } else { 0.0 };
                        assert_eq!(got[(bh * max_seq + t) * hd + d], want, "{dt:?} bh{bh} t{t} d{d}");
                    }
                }
            }
        }
    }

    #[test]
    fn upsample_nearest2x_matches_naive() {
        for dt in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            let (b, c, h, w) = (2usize, 3usize, 3usize, 5usize);
            let x: Vec<f32> = (0..b * c * h * w).map(|i| (i % 97) as f32).collect();
            let (xs, xl) = tensor(&x, vec![b, c, h, w], dt);
            let (mut os, ol) = empty(vec![b, c, 2 * h, 2 * w], dt);
            upsample_nearest2x((&xs, &xl), (&mut os, &ol), &Stream::CpuNoop).unwrap();
            let got = to_vec(&os, &ol);
            for p in 0..b * c {
                for y in 0..2 * h {
                    for xx in 0..2 * w {
                        let want = x[(p * h + y / 2) * w + xx / 2];
                        assert_eq!(got[(p * 2 * h + y) * 2 * w + xx], want, "{dt:?} p{p} y{y} x{xx}");
                    }
                }
            }
        }
    }

    #[test]
    fn nchw_nhwc_roundtrip_matches_naive() {
        for dt in [DType::F32, DType::F16, DType::U8] {
            let (b, c, h, w) = (2usize, 5usize, 3usize, 7usize);
            let x: Vec<f32> = (0..b * c * h * w).map(|i| (i % 101) as f32).collect();
            let (xs, xl) = tensor(&x, vec![b, c, h, w], dt);
            let (mut hs, hl) = empty(vec![b, h, w, c], dt);
            nchw_to_nhwc((&xs, &xl), (&mut hs, &hl), &Stream::CpuNoop).unwrap();
            let got = to_vec(&hs, &hl);
            for bi in 0..b {
                for ci in 0..c {
                    for y in 0..h {
                        for xx in 0..w {
                            assert_eq!(got[((bi * h + y) * w + xx) * c + ci], x[((bi * c + ci) * h + y) * w + xx], "{dt:?}");
                        }
                    }
                }
            }
            let (mut back, bl) = empty(vec![b, c, h, w], dt);
            nhwc_to_nchw((&hs, &hl), (&mut back, &bl), &Stream::CpuNoop).unwrap();
            assert_eq!(to_vec(&back, &bl), x, "{dt:?}");
        }
    }

    #[test]
    fn conv_epilogue_matches_naive() {
        for dt in DTS {
            let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
            let hw = h * w;
            let x = round(&det(11, b * hw * c, 1.0), dt);
            let bias = round(&det(12, c, 1.0), dt);
            let res = round(&det(13, b * c * hw, 1.0), dt);
            let temb = round(&det(14, b * c, 1.0), dt);
            let (xs, xl) = tensor(&x, vec![b * hw, c], dt);
            let (bs, bl) = tensor(&bias, vec![c], dt);
            let (rs, rl) = tensor(&res, vec![b, c, h, w], dt);
            let (ts, tl) = tensor(&temb, vec![b, c], dt);
            for mask in 0..8u32 {
                let (mut os, ol) = empty(vec![b, c, h, w], dt);
                conv_epilogue(
                    (&xs, &xl),
                    (mask & 1 != 0).then_some((&bs, &bl)),
                    (mask & 2 != 0).then_some((&rs, &rl)),
                    (mask & 4 != 0).then_some((&ts, &tl)),
                    (&mut os, &ol),
                    b, c, h, w,
                    &Stream::CpuNoop,
                )
                .unwrap();
                let got = to_vec(&os, &ol);
                for bi in 0..b {
                    for ci in 0..c {
                        for p in 0..hw {
                            let mut want = x[(bi * hw + p) * c + ci];
                            if mask & 1 != 0 {
                                want += bias[ci];
                            }
                            if mask & 2 != 0 {
                                want += res[(bi * c + ci) * hw + p];
                            }
                            if mask & 4 != 0 {
                                want += temb[bi * c + ci];
                            }
                            let g = got[(bi * c + ci) * hw + p];
                            assert!((g - want).abs() <= tol(dt) * (1.0 + want.abs()), "{dt:?} mask{mask} b{bi} c{ci} p{p}: {g} vs {want}");
                        }
                    }
                }
            }
        }
    }
}
