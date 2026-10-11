use synaptix_core::backend::ReduceOp;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

use crate::common::{elem_bytes, from_f32, par_chunks, threads, to_f32};
use crate::elementwise::{check_bounds, walk, SyncPtr, BLK};

const ROW_PAR_MIN: usize = 1 << 12;
const COL_MIN_CHUNK: usize = 64;

pub fn reduce_dispatch(
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
    let rank = src_lo.dims().len();
    let mut red: Vec<usize> = dims.to_vec();
    red.sort_unstable();
    red.dedup();
    for &d in &red {
        if d >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d, rank });
        }
    }
    if matches!(op, ReduceOp::ArgMax) {
        return reduce_argmax(src, src_lo, dst, dst_lo, &red);
    }
    let dt = src_lo.dtype();
    if dst_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, dst_lo.dtype()));
    }
    let plan = plan(src_lo.dims(), &red);
    let out_numel = plan.rows * plan.tail_keep;
    if dst_lo.numel() != out_numel {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), src_lo.dims()));
    }
    if out_numel == 0 {
        return Ok(());
    }
    if src_lo.numel() == 0 {
        return Err(SynaptixError::Unsupported("reduce over empty axis"));
    }
    let esz = elem_bytes(dt);
    check_bounds(src_lo, esz, src.byte_len(), "reduce src")?;
    if (dst_lo.offset() + out_numel) * esz > dst.byte_len() {
        return Err(SynaptixError::Other("reduce dst: layout out of buffer bounds".into()));
    }
    let sb = src.as_bytes();
    let d_off = dst_lo.offset();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    match dt {
        DType::F32 | DType::F16 | DType::BF16 => run::<f32>(op, dt, sb, &plan, dp, d_off),
        DType::F64 => run::<f64>(op, dt, sb, &plan, dp, d_off),
        _ => return Err(SynaptixError::Unsupported("reduce dtype")),
    }
    Ok(())
}

struct Plan {
    keep: Vec<(usize, isize)>,
    red: Vec<(usize, isize)>,
    tail_keep: usize,
    tail_red: usize,
    rows: usize,
    count: usize,
}

fn plan(src_dims: &[usize], red_dims: &[usize]) -> Plan {
    let rank = src_dims.len();
    let mut strides = vec![1isize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * src_dims[i + 1] as isize;
    }
    let mut groups: Vec<(bool, usize, isize)> = Vec::new();
    let mut count = 1usize;
    for k in 0..rank {
        let r = red_dims.contains(&k);
        if r {
            count *= src_dims[k];
        }
        if src_dims[k] == 1 {
            continue;
        }
        if let Some(last) = groups.last_mut() {
            if last.0 == r {
                last.1 *= src_dims[k];
                last.2 = strides[k];
                continue;
            }
        }
        groups.push((r, src_dims[k], strides[k]));
    }
    let mut tail_keep = 1;
    let mut tail_red = 1;
    if let Some(&(r, size, _)) = groups.last() {
        if r {
            tail_red = size;
        } else {
            tail_keep = size;
        }
        groups.pop();
    }
    let keep: Vec<(usize, isize)> = groups.iter().filter(|g| !g.0).map(|g| (g.1, g.2)).collect();
    let red: Vec<(usize, isize)> = groups.iter().filter(|g| g.0).map(|g| (g.1, g.2)).collect();
    let rows = keep.iter().map(|k| k.0).product();
    Plan { keep, red, tail_keep, tail_red, rows, count }
}

impl Plan {
    fn row_base(&self, row: usize) -> usize {
        let mut rem = row;
        let mut off = 0isize;
        for &(size, stride) in self.keep.iter().rev() {
            off += (rem % size) as isize * stride;
            rem /= size;
        }
        off as usize
    }
}

trait Acc: bytemuck::Pod + Copy + Send + Sync + PartialOrd + 'static {
    const DIRECT: DType;
    fn zero() -> Self;
    fn neg_inf() -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn convert(dt: DType, bytes: &[u8], off: usize, buf: &mut [Self]);
    fn row_sum(row: &[Self]) -> Self;
    fn row_max(row: &[Self]) -> Self;
    fn mean(v: Self, inv: f32) -> Self;
    unsafe fn store(dt: DType, row: &[Self], dp: *mut u8, idx: usize);
}

impl Acc for f32 {
    const DIRECT: DType = DType::F32;
    fn zero() -> f32 {
        0.0
    }
    fn neg_inf() -> f32 {
        f32::NEG_INFINITY
    }
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    fn convert(dt: DType, bytes: &[u8], off: usize, buf: &mut [f32]) {
        to_f32(dt, bytes, off, buf).expect("reduce: convert")
    }
    fn row_sum(row: &[f32]) -> f32 {
        row_sum_f32(row)
    }
    fn row_max(row: &[f32]) -> f32 {
        row_max_f32(row)
    }
    fn mean(v: f32, inv: f32) -> f32 {
        v * inv
    }
    unsafe fn store(dt: DType, row: &[f32], dp: *mut u8, idx: usize) {
        let esz = elem_bytes(dt);
        let d = std::slice::from_raw_parts_mut(dp.add(idx * esz), row.len() * esz);
        from_f32(dt, row, d, 0).expect("reduce: store");
    }
}

impl Acc for f64 {
    const DIRECT: DType = DType::F64;
    fn zero() -> f64 {
        0.0
    }
    fn neg_inf() -> f64 {
        f64::NEG_INFINITY
    }
    fn add(a: f64, b: f64) -> f64 {
        a + b
    }
    fn convert(_dt: DType, _bytes: &[u8], _off: usize, _buf: &mut [f64]) {
        unreachable!("reduce: f64 accumulator only for F64 input")
    }
    fn row_sum(row: &[f64]) -> f64 {
        row.iter().fold(0.0, |a, &v| a + v)
    }
    fn row_max(row: &[f64]) -> f64 {
        row.iter().fold(f64::NEG_INFINITY, |a, &v| if v > a { v } else { a })
    }
    fn mean(v: f64, inv: f32) -> f64 {
        ((v as f32) * inv) as f64
    }
    unsafe fn store(_dt: DType, row: &[f64], dp: *mut u8, idx: usize) {
        std::ptr::copy_nonoverlapping(row.as_ptr(), (dp as *mut f64).add(idx), row.len());
    }
}

#[inline]
fn init<A: Acc>(op: ReduceOp) -> A {
    match op {
        ReduceOp::Max => A::neg_inf(),
        _ => A::zero(),
    }
}

#[inline]
fn comb<A: Acc>(op: ReduceOp, acc: A, v: A) -> A {
    match op {
        ReduceOp::Max => if v > acc { v } else { acc },
        _ => A::add(acc, v),
    }
}

#[inline]
fn row_reduce<A: Acc>(op: ReduceOp, row: &[A]) -> A {
    match op {
        ReduceOp::Max => A::row_max(row),
        _ => A::row_sum(row),
    }
}

#[inline]
fn vcomb<A: Acc>(op: ReduceOp, acc: &mut [A], row: &[A]) {
    match op {
        ReduceOp::Max => {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a = if v > *a { v } else { *a };
            }
        }
        _ => {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a = A::add(*a, v);
            }
        }
    }
}

fn row_chunks<A: Acc>(dt: DType, sb: &[u8], off: usize, len: usize, buf: &mut [A], mut f: impl FnMut(usize, &[A])) {
    if dt == A::DIRECT {
        let esz = std::mem::size_of::<A>();
        f(0, bytemuck::cast_slice(&sb[off * esz..(off + len) * esz]));
    } else {
        let mut i = 0;
        while i < len {
            let n = (len - i).min(buf.len());
            A::convert(dt, sb, off + i, &mut buf[..n]);
            f(i, &buf[..n]);
            i += n;
        }
    }
}

fn par_map<A: Send>(n: usize, min_chunk: usize, f: impl Fn(usize, usize) -> A + Sync) -> Vec<A> {
    let t = threads();
    let chunk = n.div_ceil(t).max(min_chunk).max(1);
    if n <= chunk {
        return vec![f(0, n)];
    }
    use rayon::prelude::*;
    (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|i| {
            let a = i * chunk;
            f(a, (a + chunk).min(n))
        })
        .collect()
}

fn run<A: Acc>(op: ReduceOp, dt: DType, sb: &[u8], plan: &Plan, dp: SyncPtr, d_off: usize) {
    let inv = 1.0_f32 / (plan.count as f32);
    let finish = |v: A| if matches!(op, ReduceOp::Mean) { A::mean(v, inv) } else { v };
    let t = threads();
    if plan.tail_keep == 1 {
        let mut rdims: Vec<usize> = plan.red.iter().map(|r| r.0).collect();
        let mut rstrides: Vec<isize> = plan.red.iter().map(|r| r.1).collect();
        rdims.push(plan.tail_red);
        rstrides.push(1);
        let n_red: usize = rdims.iter().product();
        let partial = |base: usize, a: usize, b: usize, buf: &mut Vec<A>| -> A {
            let mut acc = init::<A>(op);
            walk(&rdims, &[&rstrides], [base as isize], a, b, |offs, seg| {
                row_chunks(dt, sb, offs[0] as usize, seg, buf, |_, chunk| {
                    acc = comb(op, acc, row_reduce(op, chunk));
                });
            });
            acc
        };
        if plan.rows >= 2 * t || n_red < ROW_PAR_MIN {
            par_chunks(plan.rows, 1, |r0, r1| {
                let mut buf = vec![A::zero(); BLK];
                for row in r0..r1 {
                    let v = partial(plan.row_base(row), 0, n_red, &mut buf);
                    unsafe { A::store(dt, &[finish(v)], dp.get(), d_off + row) };
                }
            });
        } else {
            for row in 0..plan.rows {
                let base = plan.row_base(row);
                let parts = par_map(n_red, ROW_PAR_MIN, |a, b| {
                    let mut buf = vec![A::zero(); BLK];
                    partial(base, a, b, &mut buf)
                });
                let v = parts.into_iter().fold(init::<A>(op), |acc, p| comb(op, acc, p));
                unsafe { A::store(dt, &[finish(v)], dp.get(), d_off + row) };
            }
        }
    } else {
        let w = plan.tail_keep;
        let rdims: Vec<usize> = plan.red.iter().map(|r| r.0).collect();
        let rstrides: Vec<isize> = plan.red.iter().map(|r| r.1).collect();
        let n_red: usize = rdims.iter().product();
        let s_inner = rstrides.last().copied().unwrap_or(0);
        let row_vec = |base: usize, c0: usize, c1: usize, acc: &mut [A], buf: &mut Vec<A>| {
            acc.fill(init::<A>(op));
            let mut do_row = |off: usize| {
                row_chunks(dt, sb, off + c0, c1 - c0, buf, |i, chunk| {
                    vcomb(op, &mut acc[i..i + chunk.len()], chunk);
                });
            };
            if rdims.is_empty() {
                do_row(base);
            } else {
                walk(&rdims, &[&rstrides], [base as isize], 0, n_red, |offs, seg| {
                    for j in 0..seg as isize {
                        do_row((offs[0] + j * s_inner) as usize);
                    }
                });
            }
            if matches!(op, ReduceOp::Mean) {
                for a in acc.iter_mut() {
                    *a = A::mean(*a, inv);
                }
            }
        };
        if plan.rows >= 2 * t || w < 2 * COL_MIN_CHUNK {
            par_chunks(plan.rows, 1, |r0, r1| {
                let mut buf = vec![A::zero(); BLK];
                let mut acc = vec![A::zero(); w];
                for row in r0..r1 {
                    row_vec(plan.row_base(row), 0, w, &mut acc, &mut buf);
                    unsafe { A::store(dt, &acc, dp.get(), d_off + row * w) };
                }
            });
        } else {
            par_chunks(w, COL_MIN_CHUNK, |c0, c1| {
                let mut buf = vec![A::zero(); BLK];
                let mut acc = vec![A::zero(); c1 - c0];
                for row in 0..plan.rows {
                    row_vec(plan.row_base(row), c0, c1, &mut acc, &mut buf);
                    unsafe { A::store(dt, &acc, dp.get(), d_off + row * w + c0) };
                }
            });
        }
    }
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
    let dt = src_lo.dtype();
    if !matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("argmax dtype"));
    }
    let dim = dims[0];
    let sdims = src_lo.dims();
    let outer: usize = sdims[..dim].iter().product();
    let axis = sdims[dim];
    let inner: usize = sdims[dim + 1..].iter().product();
    let out_numel = outer * inner;
    if dst_lo.numel() != out_numel {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), sdims));
    }
    if out_numel == 0 {
        return Ok(());
    }
    if axis == 0 {
        return Err(SynaptixError::Unsupported("argmax over empty axis"));
    }
    check_bounds(src_lo, elem_bytes(dt), src.byte_len(), "argmax src")?;
    if (dst_lo.offset() + out_numel) * 4 > dst.byte_len() {
        return Err(SynaptixError::Other("argmax dst: layout out of buffer bounds".into()));
    }
    let sb = src.as_bytes();
    let d_off = dst_lo.offset();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    let t = threads();
    if inner == 1 {
        par_chunks(outer, 1, |o0, o1| {
            let mut buf = vec![0f32; BLK];
            for o in o0..o1 {
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0u32;
                row_chunks::<f32>(dt, sb, o * axis, axis, &mut buf, |i, chunk| {
                    if i == 0 {
                        best = chunk[0];
                        best_idx = 0;
                    }
                    let (b, bi) = argmax_row_f32(chunk, i as u32, best, best_idx);
                    best = b;
                    best_idx = bi;
                });
                unsafe { (dp.get() as *mut u32).add(d_off + o).write_unaligned(best_idx) };
            }
        });
        return Ok(());
    }
    let cols = |o0: usize, o1: usize, c0: usize, c1: usize| {
        let n = c1 - c0;
        let mut buf = vec![0f32; BLK];
        let mut best = vec![0f32; n];
        let mut idx = vec![0u32; n];
        for o in o0..o1 {
            let base = o * axis * inner;
            row_chunks::<f32>(dt, sb, base + c0, n, &mut buf, |i, chunk| {
                best[i..i + chunk.len()].copy_from_slice(chunk);
            });
            idx.fill(0);
            for a in 1..axis {
                row_chunks::<f32>(dt, sb, base + a * inner + c0, n, &mut buf, |i, chunk| {
                    for (j, &v) in chunk.iter().enumerate() {
                        if v > best[i + j] {
                            best[i + j] = v;
                            idx[i + j] = a as u32;
                        }
                    }
                });
            }
            unsafe {
                std::ptr::copy_nonoverlapping(idx.as_ptr(), (dp.get() as *mut u32).add(d_off + o * inner + c0), n);
            }
        }
    };
    if outer >= 2 * t || inner < 2 * COL_MIN_CHUNK {
        par_chunks(outer, 1, |o0, o1| cols(o0, o1, 0, inner));
    } else {
        par_chunks(inner, COL_MIN_CHUNK, |c0, c1| cols(0, outer, c0, c1));
    }
    Ok(())
}

fn argmax_row_f32(chunk: &[f32], start: u32, best: f32, best_idx: u32) -> (f32, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            return unsafe { x86::argmax_row(chunk, start, best, best_idx) };
        }
    }
    argmax_row_scalar(chunk, start, best, best_idx)
}

fn argmax_row_scalar(chunk: &[f32], start: u32, mut best: f32, mut best_idx: u32) -> (f32, u32) {
    for (i, &v) in chunk.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = start + i as u32;
        }
    }
    (best, best_idx)
}

pub(crate) fn row_sum_f32(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            return unsafe { x86::row_sum(row) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::row_sum(row) };
    }
    #[allow(unreachable_code)]
    row.iter().fold(0.0, |a, &v| a + v)
}

pub(crate) fn row_max_f32(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            return unsafe { x86::row_max(row) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::row_max(row) };
    }
    #[allow(unreachable_code)]
    row.iter().fold(f32::NEG_INFINITY, |a, &v| if v > a { v } else { a })
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_sum(row: &[f32]) -> f32 {
        let n = row.len();
        let p = row.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(p.add(i)));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(p.add(i + 8)));
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(p.add(i)));
            i += 8;
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut s = lanes.iter().sum::<f32>();
        while i < n {
            s += *p.add(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_max(row: &[f32]) -> f32 {
        let n = row.len();
        let p = row.as_ptr();
        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_max_ps(_mm256_loadu_ps(p.add(i)), acc);
            i += 8;
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut m = f32::NEG_INFINITY;
        for &v in &lanes {
            if v > m {
                m = v;
            }
        }
        while i < n {
            let v = *p.add(i);
            if v > m {
                m = v;
            }
            i += 1;
        }
        m
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn argmax_row(chunk: &[f32], start: u32, best: f32, best_idx: u32) -> (f32, u32) {
        let n = chunk.len();
        let p = chunk.as_ptr();
        let mut bv = _mm256_set1_ps(best);
        let mut bi = _mm256_set1_epi32(best_idx as i32);
        let mut cur = _mm256_add_epi32(_mm256_set1_epi32(start as i32), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        let eight = _mm256_set1_epi32(8);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let m = _mm256_cmp_ps::<_CMP_GT_OQ>(v, bv);
            bv = _mm256_blendv_ps(bv, v, m);
            bi = _mm256_blendv_epi8(bi, cur, _mm256_castps_si256(m));
            cur = _mm256_add_epi32(cur, eight);
            i += 8;
        }
        let mut vals = [0f32; 8];
        let mut idxs = [0i32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), bv);
        _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, bi);
        let mut b = best;
        let mut bidx = best_idx;
        for k in 0..8 {
            let (v, ix) = (vals[k], idxs[k] as u32);
            if v > b || (v == b && ix < bidx) {
                b = v;
                bidx = ix;
            }
        }
        while i < n {
            let v = *p.add(i);
            if v > b {
                b = v;
                bidx = start + i as u32;
            }
            i += 1;
        }
        (b, bidx)
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn row_sum(row: &[f32]) -> f32 {
        let n = row.len();
        let p = row.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= n {
            acc0 = vaddq_f32(acc0, vld1q_f32(p.add(i)));
            acc1 = vaddq_f32(acc1, vld1q_f32(p.add(i + 4)));
            i += 8;
        }
        while i + 4 <= n {
            acc0 = vaddq_f32(acc0, vld1q_f32(p.add(i)));
            i += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            s += *p.add(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn row_max(row: &[f32]) -> f32 {
        let n = row.len();
        let p = row.as_ptr();
        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 4 <= n {
            let v = vld1q_f32(p.add(i));
            acc = vbslq_f32(vcgtq_f32(v, acc), v, acc);
            i += 4;
        }
        let mut lanes = [0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), acc);
        let mut m = f32::NEG_INFINITY;
        for &v in &lanes {
            if v > m {
                m = v;
            }
        }
        while i < n {
            let v = *p.add(i);
            if v > m {
                m = v;
            }
            i += 1;
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::tensor::shape::Shape;

    fn gen(dt: DType, n: usize, seed: u32, ties: bool) -> (Vec<u8>, Vec<f64>) {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(777);
        let mut vals: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let u = (state as f32) / (u32::MAX as f32) * 12.0 - 6.0;
                if ties { (u * 2.0).round() / 2.0 } else { u }
            })
            .collect();
        let (bytes, back): (Vec<u8>, Vec<f64>) = match dt {
            DType::F32 => (bytemuck::cast_slice(&vals).to_vec(), vals.iter().map(|&v| v as f64).collect()),
            DType::F64 => {
                let v64: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
                (bytemuck::cast_slice(&v64).to_vec(), v64)
            }
            DType::F16 => {
                let h: Vec<half::f16> = vals.iter().map(|&v| half::f16::from_f32(v)).collect();
                for (v, h) in vals.iter_mut().zip(&h) {
                    *v = h.to_f32();
                }
                (bytemuck::cast_slice(&h).to_vec(), vals.iter().map(|&v| v as f64).collect())
            }
            DType::BF16 => {
                let h: Vec<half::bf16> = vals.iter().map(|&v| half::bf16::from_f32(v)).collect();
                for (v, h) in vals.iter_mut().zip(&h) {
                    *v = h.to_f32();
                }
                (bytemuck::cast_slice(&h).to_vec(), vals.iter().map(|&v| v as f64).collect())
            }
            _ => unreachable!(),
        };
        (bytes, back)
    }

    fn naive(op: ReduceOp, src: &[f64], dims: &[usize], red: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
        let rank = dims.len();
        let mut strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        let keep: Vec<usize> = (0..rank).filter(|k| !red.contains(k)).collect();
        let out_numel: usize = keep.iter().map(|&k| dims[k]).product();
        let count: usize = red.iter().map(|&k| dims[k]).product();
        let mut out = vec![0f64; out_numel];
        let mut abs = vec![0f64; out_numel];
        let mut arg = vec![0u32; out_numel];
        for o in 0..out_numel {
            let mut rem = o;
            let mut base = 0;
            for &k in keep.iter().rev() {
                base += (rem % dims[k]) * strides[k];
                rem /= dims[k];
            }
            let mut acc = match op {
                ReduceOp::Max | ReduceOp::ArgMax => f64::NEG_INFINITY,
                _ => 0.0,
            };
            let mut sabs = 0.0;
            let mut best_idx = 0u32;
            let mut first = true;
            let mut ridx = vec![0usize; red.len()];
            for _ in 0..count {
                let mut pos = base;
                for (k, &rd) in red.iter().enumerate() {
                    pos += ridx[k] * strides[rd];
                }
                let v = src[pos];
                match op {
                    ReduceOp::Sum | ReduceOp::Mean => {
                        acc += v;
                        sabs += v.abs();
                    }
                    ReduceOp::Max => acc = if v > acc { v } else { acc },
                    ReduceOp::ArgMax => {
                        if first || v > acc {
                            acc = v;
                            best_idx = ridx[0] as u32;
                        }
                    }
                }
                first = false;
                for k in (0..red.len()).rev() {
                    ridx[k] += 1;
                    if ridx[k] < dims[red[k]] {
                        break;
                    }
                    ridx[k] = 0;
                }
            }
            if matches!(op, ReduceOp::Mean) {
                acc = (acc as f32 * (1.0_f32 / count as f32)) as f64;
            }
            out[o] = acc;
            abs[o] = sabs;
            arg[o] = best_idx;
        }
        (out, abs, arg)
    }

    fn read_out(dt: DType, bytes: &[u8], n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match dt {
                DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes)[i] as f64,
                DType::F64 => bytemuck::cast_slice::<u8, f64>(bytes)[i],
                DType::F16 => bytemuck::cast_slice::<u8, half::f16>(bytes)[i].to_f32() as f64,
                DType::BF16 => bytemuck::cast_slice::<u8, half::bf16>(bytes)[i].to_f32() as f64,
                _ => unreachable!(),
            })
            .collect()
    }

    fn check(dt: DType, dims: &[usize], red: &[usize], ties: bool) {
        let n: usize = dims.iter().product();
        let (bytes, vals) = gen(dt, n, 42 + n as u32, ties);
        let src = CpuBuf::from_vec(bytes);
        let src_lo = Layout::contiguous(Shape::new(dims.to_vec()), dt);
        let out_dims: Vec<usize> = (0..dims.len()).filter(|k| !red.contains(k)).map(|k| dims[k]).collect();
        let out_numel: usize = out_dims.iter().product();
        let rel = match dt {
            DType::F32 => 1e-5,
            DType::F64 => 1e-12,
            DType::F16 => 1.0 / 1024.0,
            DType::BF16 => 1.0 / 128.0,
            _ => unreachable!(),
        };
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max] {
            let dst_lo = Layout::contiguous(Shape::new(out_dims.clone()), dt);
            let mut dst = CpuBuf::alloc_zeros(out_numel * elem_bytes(dt));
            reduce_dispatch(op, &src, &src_lo, &mut dst, &dst_lo, red).unwrap();
            let got = read_out(dt, dst.as_bytes(), out_numel);
            let (want, abs, _) = naive(op, &vals, dims, red);
            for i in 0..out_numel {
                if matches!(op, ReduceOp::Max) {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "{dt:?} {op:?} dims={dims:?} red={red:?} idx {i}");
                } else {
                    let count: f64 = red.iter().map(|&k| dims[k] as f64).product();
                    let scale = if matches!(op, ReduceOp::Mean) { abs[i] / count } else { abs[i] };
                    let tol = rel * (scale + want[i].abs()) + 1e-6;
                    assert!(
                        (got[i] - want[i]).abs() <= tol,
                        "{dt:?} {op:?} dims={dims:?} red={red:?} idx {i}: got {} want {} tol {tol}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
        if red.len() == 1 && dt != DType::F64 {
            let dst_lo = Layout::contiguous(Shape::new(out_dims.clone()), DType::U32);
            let mut dst = CpuBuf::alloc_zeros(out_numel * 4);
            reduce_dispatch(ReduceOp::ArgMax, &src, &src_lo, &mut dst, &dst_lo, red).unwrap();
            let got: &[u32] = bytemuck::cast_slice(dst.as_bytes());
            let (_, _, want) = naive(ReduceOp::ArgMax, &vals, dims, red);
            assert_eq!(got, &want[..], "{dt:?} argmax dims={dims:?} red={red:?}");
        }
    }

    const FLOATS: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];

    #[test]
    fn full_reduction_various_sizes() {
        for dt in FLOATS {
            for n in [1usize, 7, 1000, 4097, 100_003] {
                check(dt, &[n], &[0], false);
                check(dt, &[n], &[0], true);
            }
            check(dt, &[64, 1000], &[0, 1], false);
            check(dt, &[], &[], false);
            check(dt, &[1, 1, 1], &[1], false);
        }
    }

    #[test]
    fn last_axis_rows() {
        for dt in FLOATS {
            check(dt, &[64, 1000], &[1], false);
            check(dt, &[64, 1000], &[1], true);
            check(dt, &[3, 5000], &[1], true);
            check(dt, &[5, 6, 7, 8], &[3], true);
            check(dt, &[5, 6, 7, 8], &[2, 3], false);
            check(dt, &[5, 6, 7, 8], &[1, 2, 3], false);
            check(dt, &[200, 9], &[1], true);
        }
    }

    #[test]
    fn leading_and_middle_axes() {
        for dt in FLOATS {
            check(dt, &[64, 1000], &[0], true);
            check(dt, &[1000, 64], &[0], true);
            check(dt, &[5, 6, 7, 8], &[0], true);
            check(dt, &[5, 6, 7, 8], &[1], true);
            check(dt, &[5, 6, 7, 8], &[2], true);
            check(dt, &[5, 6, 7, 8], &[0, 2], false);
            check(dt, &[5, 6, 7, 8], &[1, 3], false);
            check(dt, &[5, 6, 7, 8], &[0, 1, 2], false);
            check(dt, &[5, 6, 7, 8], &[0, 3], false);
            check(dt, &[3, 300, 300], &[1], true);
            check(dt, &[100, 7, 300], &[1], true);
        }
    }

    #[test]
    fn argmax_nan_and_ties_first_occurrence() {
        let vals: Vec<f32> = vec![1.0, 3.0, 3.0, f32::NAN, 7.0, 7.0, -0.0, 0.0];
        let src = CpuBuf::from_vec(bytemuck::cast_slice(&vals).to_vec());
        let lo = Layout::contiguous(Shape::new(vec![8]), DType::F32);
        let dlo = Layout::contiguous(Shape::new(vec![]), DType::U32);
        let mut dst = CpuBuf::alloc_zeros(4);
        reduce_dispatch(ReduceOp::ArgMax, &src, &lo, &mut dst, &dlo, &[0]).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(dst.as_bytes())[0], 4);
        let vals: Vec<f32> = vec![f32::NAN, 3.0, 9.0];
        let src = CpuBuf::from_vec(bytemuck::cast_slice(&vals).to_vec());
        let lo = Layout::contiguous(Shape::new(vec![3]), DType::F32);
        reduce_dispatch(ReduceOp::ArgMax, &src, &lo, &mut dst, &dlo, &[0]).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(dst.as_bytes())[0], 0);
        let mut big: Vec<f32> = (0..5000).map(|i| ((i * 7919) % 101) as f32).collect();
        big[1234] = 200.0;
        big[4321] = 200.0;
        let src = CpuBuf::from_vec(bytemuck::cast_slice(&big).to_vec());
        let lo = Layout::contiguous(Shape::new(vec![5000]), DType::F32);
        reduce_dispatch(ReduceOp::ArgMax, &src, &lo, &mut dst, &dlo, &[0]).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(dst.as_bytes())[0], 1234);
    }

    #[test]
    fn non_contiguous_src_rejected() {
        let lo = Layout::contiguous(Shape::new(vec![4, 5]), DType::F32).transpose(0, 1).unwrap();
        let src = CpuBuf::alloc_zeros(80);
        let mut dst = CpuBuf::alloc_zeros(20);
        let dlo = Layout::contiguous(Shape::new(vec![5]), DType::F32);
        assert!(matches!(reduce_dispatch(ReduceOp::Sum, &src, &lo, &mut dst, &dlo, &[1]), Err(SynaptixError::NonContiguous)));
    }

    #[test]
    #[ignore]
    fn bench_reduce() {
        let dims = [4096usize, 4096];
        let (bytes, _) = gen(DType::F32, dims[0] * dims[1], 1, false);
        let src = CpuBuf::from_vec(bytes);
        let lo = Layout::contiguous(Shape::new(dims.to_vec()), DType::F32);
        for red in [vec![1usize], vec![0], vec![0, 1]] {
            let out: usize = dims.iter().enumerate().filter(|(k, _)| !red.contains(k)).map(|(_, d)| d).product();
            let dlo = Layout::contiguous(Shape::new(vec![out]), DType::F32);
            let mut dst = CpuBuf::alloc_zeros(out * 4);
            reduce_dispatch(ReduceOp::Sum, &src, &lo, &mut dst, &dlo, &red).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..10 {
                reduce_dispatch(ReduceOp::Sum, &src, &lo, &mut dst, &dlo, &red).unwrap();
            }
            println!("reduce sum [4096,4096] red={red:?}: {:?}/iter", t.elapsed() / 10);
        }
        let dlo = Layout::contiguous(Shape::new(vec![4096]), DType::U32);
        let mut dst = CpuBuf::alloc_zeros(4096 * 4);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            reduce_dispatch(ReduceOp::ArgMax, &src, &lo, &mut dst, &dlo, &[1]).unwrap();
        }
        println!("argmax [4096,4096] dim 1: {:?}/iter", t.elapsed() / 10);
    }
}
