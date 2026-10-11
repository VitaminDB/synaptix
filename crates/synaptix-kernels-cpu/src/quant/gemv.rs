use rayon::prelude::*;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CpuBuf, Storage};

use super::dequant::WeightView;
use crate::common::{cpu, cpu_mut, from_f32, is_half, par_chunks_cost, to_f32};
use crate::gemm::blocked;
use crate::gemm::simd_dot::dot_f32;

const PREFILL_M: usize = 8;
const NB: usize = 64;

pub fn weight_view<'a>(w: &QuantWeight, packed: &'a Storage, scales: Option<&'a Storage>) -> Result<WeightView<'a>> {
    let p = cpu(packed, "quant: packed не на CPU")?.as_bytes();
    let s = match scales {
        Some(s) => Some(cpu(s, "quant: scales не на CPU")?.as_bytes()),
        None => None,
    };
    WeightView::new(w.dtype(), w.n(), w.k(), p, s)
}

pub fn x_rows_f32(dt: DType, bytes: &[u8], off: usize, m: usize, k: usize) -> Result<Vec<f32>> {
    let mut xf = vec![0f32; m * k];
    if m * k >= 1 << 16 {
        crate::threads::pool();
        xf.par_chunks_mut(k).enumerate().try_for_each(|(r, row)| to_f32(dt, bytes, off + r * k, row))?;
    } else {
        to_f32(dt, bytes, off, &mut xf)?;
    }
    Ok(xf)
}

pub fn gemm_quant_f32(view: &WeightView, xf: &[f32], m: usize, out: &mut [f32]) {
    let (n, k) = (view.n, view.k);
    debug_assert_eq!(out.len(), m * n);
    if m < PREFILL_M && view.dtype == DType::Q8Row {
        use super::q8dot;
        let op = blocked::SendPtr::new(out.as_mut_ptr());
        if k % q8dot::QK == 0 {
            let mut xq = vec![0i8; m * k];
            let mut xs = vec![0f32; m * k / q8dot::QK];
            for i in 0..m {
                q8dot::quantize_act(&xf[i * k..(i + 1) * k], &mut xq[i * k..(i + 1) * k], &mut xs[i * k / q8dot::QK..(i + 1) * k / q8dot::QK]);
            }
            let chunk = (n / (crate::common::threads() * 8)).max(16);
            (0..n.div_ceil(chunk)).into_par_iter().for_each(|ci| {
                let (n0, n1) = (ci * chunk, ((ci + 1) * chunk).min(n));
                for j in n0..n1 {
                    let q = view.row_bytes_of(j);
                    let sc = view.q8_scale(j);
                    for i in 0..m {
                        let nb = k / q8dot::QK;
                        let v = q8dot::dot(&xq[i * k..(i + 1) * k], &xs[i * nb..(i + 1) * nb], q) * sc;
                        unsafe { *op.ptr().add(i * n + j) = v };
                    }
                }
            });
            return;
        }
        par_chunks_cost(n, k * (m + 1) / 8, |n0, n1| {
            for j in n0..n1 {
                let q = view.row_bytes_of(j);
                let sc = view.q8_scale(j);
                for i in 0..m {
                    let v = dot_f32_i8(&xf[i * k..(i + 1) * k], q) * sc;
                    unsafe { *op.ptr().add(i * n + j) = v };
                }
            }
        });
        return;
    }
    if m < PREFILL_M {
        let op = blocked::SendPtr::new(out.as_mut_ptr());
        par_chunks_cost(n, k * (m + 2) / 4, |n0, n1| {
            let mut row = vec![0f32; k];
            for j in n0..n1 {
                view.row(j, &mut row);
                for i in 0..m {
                    let v = dot_f32(&xf[i * k..(i + 1) * k], &row);
                    unsafe { *op.ptr().add(i * n + j) = v };
                }
            }
        });
        return;
    }
    let op = blocked::SendPtr::new(out.as_mut_ptr());
    (0..n.div_ceil(NB)).into_par_iter().for_each(|jb| {
        let c0 = jb * NB;
        let cols = NB.min(n - c0);
        let mut wrows = vec![0f32; cols * k];
        for (jj, r) in wrows.chunks_exact_mut(k).enumerate() {
            view.row(c0 + jj, r);
        }
        blocked::linear_panel(m, cols, k, xf, &wrows, op.at(c0), n);
    });
}

#[inline]
pub fn dot_f32_i8(x: &[f32], q: &[u8]) -> f32 {
    let mut acc = [0f32; 16];
    let n = x.len().min(q.len());
    let full = n / 16 * 16;
    for (xc, qc) in x[..full].chunks_exact(16).zip(q[..full].chunks_exact(16)) {
        for l in 0..16 {
            acc[l] += xc[l] * (qc[l] as i8) as f32;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in full..n {
        s += x[i] * (q[i] as i8) as f32;
    }
    s
}

fn out_dims_check(x_lo: &Layout, w: &QuantWeight, out_lo: &Layout) -> Result<usize> {
    if !x_lo.is_contiguous() || !out_lo.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let k = w.k();
    if x_lo.dims().last().copied() != Some(k) {
        return Err(SynaptixError::ShapeMismatch { expected: vec![w.n(), k], got: x_lo.dims().to_vec() });
    }
    let m = x_lo.numel() / k;
    if out_lo.numel() != m * w.n() {
        return Err(SynaptixError::ShapeMismatch { expected: vec![m, w.n()], got: out_lo.dims().to_vec() });
    }
    Ok(m)
}

pub fn linear_quant(x: (&Storage, &Layout), w: &QuantWeight, out: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
    let (x_st, x_lo) = x;
    let (o_st, o_lo) = out;
    let dt = x_lo.dtype();
    if !(is_half(dt) || dt == DType::F32) || o_lo.dtype() != dt {
        return Err(SynaptixError::Unsupported("cpu linear_quant: активация F16/BF16/F32, выход того же типа"));
    }
    let m = out_dims_check(x_lo, w, o_lo)?;
    let packed = w.packed_arc().ok_or(SynaptixError::Unsupported("cpu linear_quant: packed освобождён"))?;
    let view = weight_view(w, &packed, w.scales_opt())?;
    let xb = cpu(x_st, "cpu linear_quant: x не на CPU")?.as_bytes();
    let xf = x_rows_f32(dt, xb, x_lo.offset(), m, w.k())?;
    let mut of = vec![0f32; m * w.n()];
    gemm_quant_f32(&view, &xf, m, &mut of);
    let ob = cpu_mut(o_st, "cpu linear_quant: out не на CPU")?.as_bytes_mut();
    from_f32(dt, &of, ob, o_lo.offset())
}

#[allow(clippy::too_many_arguments)]
pub fn quant_gemv_batched(
    w_packed: &[&Storage],
    w_scales: &[Option<&Storage>],
    dtype: DType,
    x: &Storage,
    x_rows: &[usize],
    out: (&mut Storage, &Layout),
    n: usize,
    k: usize,
    _stream: &Stream,
) -> Result<()> {
    let (o_st, o_lo) = out;
    let dt = o_lo.dtype();
    if !(is_half(dt) || dt == DType::F32) {
        return Err(SynaptixError::Unsupported("cpu quant_gemv_batched: выход F16/BF16/F32"));
    }
    let e = w_packed.len();
    if w_scales.len() != e || x_rows.len() != e || o_lo.numel() != e * n {
        return Err(SynaptixError::Unsupported("cpu quant_gemv_batched: размеры батча"));
    }
    let xb = cpu(x, "cpu quant_gemv_batched: x не на CPU")?.as_bytes();
    let rows = x_rows.iter().copied().max().map_or(0, |r| r + 1);
    let xf = x_rows_f32(dt, xb, 0, rows, k)?;
    let views: Vec<WeightView> = w_packed
        .iter()
        .zip(w_scales)
        .map(|(p, s)| {
            let pb = cpu(p, "cpu quant_gemv_batched: packed не на CPU")?.as_bytes();
            let sb = match s {
                Some(s) => Some(cpu(s, "cpu quant_gemv_batched: scales не на CPU")?.as_bytes()),
                None => None,
            };
            WeightView::new(dtype, n, k, pb, sb)
        })
        .collect::<Result<_>>()?;
    let mut of = vec![0f32; e * n];
    let op = blocked::SendPtr::new(of.as_mut_ptr());
    par_chunks_cost(e * n, k / 2, |a, b| {
        let mut row = vec![0f32; k];
        for t in a..b {
            let ei = t / n;
            let j = t % n;
            let xr = &xf[x_rows[ei] * k..(x_rows[ei] + 1) * k];
            views[ei].row(j, &mut row);
            unsafe { *op.ptr().add(t) = dot_f32(xr, &row) };
        }
    });
    let ob = cpu_mut(o_st, "cpu quant_gemv_batched: out не на CPU")?.as_bytes_mut();
    from_f32(dt, &of, ob, o_lo.offset())
}

pub fn dequant_rows_to(view: &WeightView, rows: impl Fn(usize) -> Option<usize> + Sync, out_dt: DType, out: &mut [u8], out_off: usize, count: usize) -> Result<()> {
    let k = view.k;
    if !(is_half(out_dt) || out_dt == DType::F32) {
        return Err(SynaptixError::Unsupported("cpu dequant: выход F16/BF16/F32"));
    }
    let eb = crate::common::elem_bytes(out_dt);
    let base = &mut out[out_off * eb..(out_off + count * k) * eb];
    crate::threads::pool();
    base.par_chunks_mut(k * eb).enumerate().try_for_each(|(i, dst)| {
        let mut row = vec![0f32; k];
        match rows(i) {
            Some(r) if r < view.n => view.row(r, &mut row),
            _ => {}
        }
        from_f32(out_dt, &row, dst, 0)
    })
}

pub fn block_dequant(packed: &Storage, dtype: DType, out: (&mut Storage, &Layout), n: usize, k: usize, _stream: &Stream) -> Result<()> {
    let (o_st, o_lo) = out;
    let pb = cpu(packed, "cpu block_dequant: packed не на CPU")?.as_bytes();
    let view = WeightView::new(dtype, n, k, pb, None)?;
    if !o_lo.is_contiguous() || o_lo.numel() != n * k {
        return Err(SynaptixError::Unsupported("cpu block_dequant: выход [n, k] contiguous"));
    }
    let ob = cpu_mut(o_st, "cpu block_dequant: out не на CPU")?.as_bytes_mut();
    dequant_rows_to(&view, Some, o_lo.dtype(), ob, o_lo.offset(), n)
}

#[allow(clippy::too_many_arguments)]
pub fn block_gather_dequant(
    table: &Storage,
    dtype: DType,
    ids: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    vocab: usize,
    k: usize,
    _stream: &Stream,
) -> Result<()> {
    let (i_st, i_lo) = ids;
    let (o_st, o_lo) = out;
    if i_lo.dtype() != DType::U32 || !i_lo.is_contiguous() {
        return Err(SynaptixError::Unsupported("cpu block_gather_dequant: ids U32 contiguous"));
    }
    let tb = cpu(table, "cpu block_gather_dequant: таблица не на CPU")?.as_bytes();
    let view = WeightView::new(dtype, vocab, k, tb, None)?;
    let ib = cpu(i_st, "cpu block_gather_dequant: ids не на CPU")?.as_bytes();
    let idv: &[u32] = bytemuck::cast_slice(ib);
    let n = i_lo.numel();
    let idv = &idv[i_lo.offset()..i_lo.offset() + n];
    if !o_lo.is_contiguous() || o_lo.numel() != n * k {
        return Err(SynaptixError::Unsupported("cpu block_gather_dequant: выход [n, k]"));
    }
    let ob = cpu_mut(o_st, "cpu block_gather_dequant: out не на CPU")?.as_bytes_mut();
    dequant_rows_to(&view, |i| Some(idv[i] as usize), o_lo.dtype(), ob, o_lo.offset(), n)
}

pub fn mxfp8_dequant(packed: (&Storage, &Layout), scales: (&Storage, &Layout), out: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
    let (p_st, p_lo) = packed;
    let (s_st, _s_lo) = scales;
    let (o_st, o_lo) = out;
    let d = p_lo.dims();
    if d.len() != 2 || !o_lo.is_contiguous() || o_lo.numel() != p_lo.numel() {
        return Err(SynaptixError::Unsupported("cpu mxfp8_dequant: packed [n, k], выход той же формы"));
    }
    let (n, k) = (d[0], d[1]);
    let pb = cpu(p_st, "cpu mxfp8_dequant: packed не на CPU")?.as_bytes();
    let sb = cpu(s_st, "cpu mxfp8_dequant: scales не на CPU")?.as_bytes();
    let view = WeightView::new(DType::MXFP8, n, k, &pb[p_lo.offset()..], Some(sb))?;
    let ob = cpu_mut(o_st, "cpu mxfp8_dequant: out не на CPU")?.as_bytes_mut();
    dequant_rows_to(&view, Some, o_lo.dtype(), ob, o_lo.offset(), n)
}

pub fn nvfp4_dequant(packed: &Storage, scales: &Storage, out: (&mut Storage, &Layout), n: usize, k: usize, _stream: &Stream) -> Result<()> {
    let (o_st, o_lo) = out;
    let pb = cpu(packed, "cpu nvfp4_dequant: packed не на CPU")?.as_bytes();
    let sb = cpu(scales, "cpu nvfp4_dequant: scales не на CPU")?.as_bytes();
    let view = WeightView::new(DType::NVFP4, n, k, pb, Some(sb))?;
    if !o_lo.is_contiguous() || o_lo.numel() != n * k {
        return Err(SynaptixError::Unsupported("cpu nvfp4_dequant: выход [n, k]"));
    }
    let ob = cpu_mut(o_st, "cpu nvfp4_dequant: out не на CPU")?.as_bytes_mut();
    dequant_rows_to(&view, Some, o_lo.dtype(), ob, o_lo.offset(), n)
}

fn weight_f32(w: (&Storage, &Layout), n: usize, k: usize) -> Result<Vec<f32>> {
    let (w_st, w_lo) = w;
    if !w_lo.is_contiguous() || w_lo.numel() != n * k {
        return Err(SynaptixError::Unsupported("cpu quantize: вес [n, k] contiguous"));
    }
    let wb = cpu(w_st, "cpu quantize: вес не на CPU")?.as_bytes();
    x_rows_f32(w_lo.dtype(), wb, w_lo.offset(), n, k)
}

pub fn quantize_sq(w: (&Storage, &Layout), bits: u8, n: usize, k: usize, _stream: &Stream) -> Result<Storage> {
    let wf = weight_f32(w, n, k)?;
    let blob = super::encode::quantize_sq_rows(bits, &wf, n, k)?;
    Ok(Storage::Cpu(CpuBuf::from_vec(blob)))
}

pub fn quantize_mxfp8(w: (&Storage, &Layout), n: usize, k: usize, _stream: &Stream) -> Result<(Storage, Storage)> {
    let wf = weight_f32(w, n, k)?;
    let (p, s) = super::encode::quantize_mxfp8_rows(&wf, n, k)?;
    Ok((Storage::Cpu(CpuBuf::from_vec(p)), Storage::Cpu(CpuBuf::from_vec(s))))
}

pub fn quantize_nvfp4(w: (&Storage, &Layout), n: usize, k: usize, _stream: &Stream) -> Result<(Storage, Storage)> {
    let wf = weight_f32(w, n, k)?;
    let (p, s) = super::encode::quantize_nvfp4_rows(&wf, n, k)?;
    Ok((Storage::Cpu(CpuBuf::from_vec(p)), Storage::Cpu(CpuBuf::from_vec(s))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use synaptix_core::tensor::shape::Shape;

    fn tensor_f32(v: &[f32], dims: Vec<usize>, dt: DType) -> (Storage, Layout) {
        let lo = Layout::contiguous(Shape::new(dims), dt);
        let mut bytes = vec![0u8; dt.bytes_for_numel(v.len())];
        from_f32(dt, v, &mut bytes, 0).unwrap();
        (Storage::Cpu(CpuBuf::from_vec(bytes)), lo)
    }

    fn read_f32(st: &Storage, lo: &Layout) -> Vec<f32> {
        let mut v = vec![0f32; lo.numel()];
        to_f32(lo.dtype(), st.as_cpu().unwrap().as_bytes(), lo.offset(), &mut v).unwrap();
        v
    }

    fn check_linear(bits_or_fmt: DType, m: usize, n: usize, k: usize, dt: DType, tol: f32) {
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.173).sin()).collect();
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.071).cos()).collect();
        let (ws, wl) = tensor_f32(&w, vec![n, k], DType::F16);
        let stream = Stream::default_for(synaptix_core::device::Device::Cpu).unwrap();
        let qw = match bits_or_fmt {
            DType::Sq { bits } => {
                let p = quantize_sq((&ws, &wl), bits, n, k, &stream).unwrap();
                QuantWeight::new_block(Arc::new(p), bits_or_fmt, n, k).unwrap()
            }
            DType::MXFP8 => {
                let (p, s) = quantize_mxfp8((&ws, &wl), n, k, &stream).unwrap();
                QuantWeight::new(Arc::new(p), Arc::new(s), DType::MXFP8, n, k).unwrap()
            }
            DType::NVFP4 => {
                let (p, s) = quantize_nvfp4((&ws, &wl), n, k, &stream).unwrap();
                QuantWeight::new(Arc::new(p), Arc::new(s), DType::NVFP4, n, k).unwrap()
            }
            _ => unreachable!(),
        };
        let packed = qw.packed_arc().unwrap();
        let view = weight_view(&qw, &packed, qw.scales_opt()).unwrap();
        let mut wd = vec![0f32; n * k];
        for r in 0..n {
            view.row(r, &mut wd[r * k..(r + 1) * k]);
        }
        let (xs, xl) = tensor_f32(&x, vec![m, k], dt);
        let ol = Layout::contiguous(Shape::new(vec![m, n]), dt);
        let mut os = Storage::Cpu(CpuBuf::alloc_zeros(dt.bytes_for_numel(m * n)));
        linear_quant((&xs, &xl), &qw, (&mut os, &ol), &stream).unwrap();
        let got = read_f32(&os, &ol);
        let xr = read_f32(&xs, &xl);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| xr[i * k + p] * wd[j * k + p]).sum();
                let g = got[i * n + j];
                assert!((g - want).abs() <= tol * (1.0 + want.abs()), "{bits_or_fmt:?} m{m} n{n} k{k} [{i},{j}] {g} vs {want}");
            }
        }
    }

    #[test]
    fn linear_quant_all_formats_decode_and_prefill() {
        for fmt in [DType::Sq { bits: 4 }, DType::Sq { bits: 8 }, DType::Sq { bits: 3 }, DType::MXFP8, DType::NVFP4] {
            check_linear(fmt, 1, 37, 256, DType::F16, 2e-2);
            check_linear(fmt, 3, 70, 320, DType::BF16, 3e-2);
            check_linear(fmt, 33, 130, 192, DType::F32, 1e-3);
            check_linear(fmt, 9, 64, 64, DType::F16, 2e-2);
        }
    }

    #[test]
    fn block_dequant_and_gather() {
        let (n, k) = (12, 96);
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.37).sin()).collect();
        let (ws, wl) = tensor_f32(&w, vec![n, k], DType::BF16);
        let stream = Stream::default_for(synaptix_core::device::Device::Cpu).unwrap();
        let p = quantize_sq((&ws, &wl), 6, n, k, &stream).unwrap();
        let ol = Layout::contiguous(Shape::new(vec![n, k]), DType::F16);
        let mut os = Storage::Cpu(CpuBuf::alloc_zeros(2 * n * k));
        block_dequant(&p, DType::Sq { bits: 6 }, (&mut os, &ol), n, k, &stream).unwrap();
        let d = read_f32(&os, &ol);
        for i in 0..n * k {
            assert!((d[i] - w[i]).abs() < 0.05, "{i}: {} vs {}", d[i], w[i]);
        }
        let ids: Vec<u32> = vec![3, 11, 999, 0];
        let il = Layout::contiguous(Shape::new(vec![4]), DType::U32);
        let is = Storage::Cpu(CpuBuf::from_vec(bytemuck::cast_slice(&ids).to_vec()));
        let gl = Layout::contiguous(Shape::new(vec![4, k]), DType::F32);
        let mut gs = Storage::Cpu(CpuBuf::alloc_zeros(4 * 4 * k));
        block_gather_dequant(&p, DType::Sq { bits: 6 }, (&is, &il), (&mut gs, &gl), n, k, &stream).unwrap();
        let g = read_f32(&gs, &gl);
        for i in 0..k {
            assert!((g[i] - d[3 * k + i]).abs() < 1e-3);
            assert!((g[k + i] - d[11 * k + i]).abs() < 1e-3);
            assert_eq!(g[2 * k + i], 0.0);
            assert!((g[3 * k + i] - d[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn gemv_batched_matches_linear() {
        let (e, n, k) = (3, 40, 128);
        let stream = Stream::default_for(synaptix_core::device::Device::Cpu).unwrap();
        let mut packed = Vec::new();
        let mut dense = Vec::new();
        for ei in 0..e {
            let w: Vec<f32> = (0..n * k).map(|i| ((i + ei * 17) as f32 * 0.11).sin()).collect();
            let (ws, wl) = tensor_f32(&w, vec![n, k], DType::F16);
            let p = quantize_sq((&ws, &wl), 4, n, k, &stream).unwrap();
            let view = WeightView::new(DType::Sq { bits: 4 }, n, k, p.as_cpu().unwrap().as_bytes(), None).unwrap();
            let mut wd = vec![0f32; n * k];
            for r in 0..n {
                view.row(r, &mut wd[r * k..(r + 1) * k]);
            }
            dense.push(wd);
            packed.push(p);
        }
        let x: Vec<f32> = (0..2 * k).map(|i| (i as f32 * 0.3).cos()).collect();
        let (xs, xl) = tensor_f32(&x, vec![2, k], DType::F16);
        let xr = read_f32(&xs, &xl);
        let refs: Vec<&Storage> = packed.iter().collect();
        let scales: Vec<Option<&Storage>> = vec![None; e];
        let ol = Layout::contiguous(Shape::new(vec![e, n]), DType::F16);
        let mut os = Storage::Cpu(CpuBuf::alloc_zeros(2 * e * n));
        let rows = [1usize, 0, 1];
        quant_gemv_batched(&refs, &scales, DType::Sq { bits: 4 }, &xs, &rows, (&mut os, &ol), n, k, &stream).unwrap();
        let got = read_f32(&os, &ol);
        for ei in 0..e {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| xr[rows[ei] * k + p] * dense[ei][j * k + p]).sum();
                assert!((got[ei * n + j] - want).abs() < 2e-2 * (1.0 + want.abs()));
            }
        }
    }
}
