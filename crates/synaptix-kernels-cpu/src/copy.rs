use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::CpuBuf;

use crate::common::{elem_bytes, par_chunks};
use crate::elementwise::{check_bounds, coalesce, walk, SyncPtr};

const COPY_MIN_CHUNK: usize = 1 << 18;

pub fn copy_strided(src: &CpuBuf, src_lo: &Layout, dst: &mut CpuBuf, dst_lo: &Layout) -> Result<()> {
    let dims = src_lo.dims();
    if dims != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), dims));
    }
    let dt = src_lo.dtype();
    if dst_lo.dtype() != dt {
        return Err(SynaptixError::dtype_mismatch(dt, dst_lo.dtype()));
    }
    let numel = src_lo.numel();
    if numel == 0 {
        return Ok(());
    }
    if dt.is_quantized() {
        if !src_lo.is_contiguous() || !dst_lo.is_contiguous() {
            return Err(SynaptixError::Unsupported("copy on quantized dtype"));
        }
        let n = dt.bytes_for_numel(numel);
        if src.byte_len() < n || dst.byte_len() < n {
            return Err(SynaptixError::Other("copy: quantized buffer too small".into()));
        }
        dst.as_bytes_mut()[..n].copy_from_slice(&src.as_bytes()[..n]);
        return Ok(());
    }
    match elem_bytes(dt) {
        1 => copy_t::<u8>(src, src_lo, dst, dst_lo),
        2 => copy_t::<u16>(src, src_lo, dst, dst_lo),
        4 => copy_t::<u32>(src, src_lo, dst, dst_lo),
        8 => copy_t::<u64>(src, src_lo, dst, dst_lo),
        _ => Err(SynaptixError::Unsupported("copy: element size")),
    }
}

fn copy_t<T: Copy>(src: &CpuBuf, src_lo: &Layout, dst: &mut CpuBuf, dst_lo: &Layout) -> Result<()> {
    let esz = std::mem::size_of::<T>();
    check_bounds(src_lo, esz, src.byte_len(), "copy src")?;
    check_bounds(dst_lo, esz, dst.byte_len(), "copy dst")?;
    let numel = src_lo.numel();
    let mut st = vec![src_lo.strides().as_slice().to_vec(), dst_lo.strides().as_slice().to_vec()];
    let dims = coalesce(src_lo.dims(), &mut st);
    let base = [src_lo.offset() as isize, dst_lo.offset() as isize];
    let s_st = *st[0].last().unwrap();
    let d_st = *st[1].last().unwrap();
    let sb = src.as_bytes();
    let dp = SyncPtr::new(dst.as_bytes_mut().as_mut_ptr());
    par_chunks(numel, COPY_MIN_CHUNK, |lo, hi| {
        let d = dp.get();
        let sp = sb.as_ptr();
        walk(&dims, &[&st[0], &st[1]], base, lo, hi, |offs, len| unsafe {
            if s_st == 1 && d_st == 1 {
                std::ptr::copy_nonoverlapping(
                    sp.offset(offs[0] * esz as isize),
                    d.offset(offs[1] * esz as isize),
                    len * esz,
                );
            } else {
                let s = sp as *const T;
                let dd = d as *mut T;
                for i in 0..len as isize {
                    let v = s.offset(offs[0] + i * s_st).read_unaligned();
                    dd.offset(offs[1] + i * d_st).write_unaligned(v);
                }
            }
        });
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::dtype::DType;
    use synaptix_core::tensor::shape::Shape;

    fn naive(src: &[u8], src_lo: &Layout, dst: &mut [u8], dst_lo: &Layout, esz: usize) {
        let dims = src_lo.dims();
        let rank = dims.len();
        let numel = src_lo.numel();
        let ss = src_lo.strides().as_slice();
        let ds = dst_lo.strides().as_slice();
        let mut idx = vec![0usize; rank];
        for _ in 0..numel {
            let mut so = src_lo.offset() as isize;
            let mut doff = dst_lo.offset() as isize;
            for k in 0..rank {
                so += idx[k] as isize * ss[k];
                doff += idx[k] as isize * ds[k];
            }
            let (so, doff) = (so as usize * esz, doff as usize * esz);
            dst[doff..doff + esz].copy_from_slice(&src[so..so + esz]);
            for k in (0..rank).rev() {
                idx[k] += 1;
                if idx[k] < dims[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    fn run(src_lo: Layout, dst_lo: Layout, src_elems: usize, dst_elems: usize) {
        let dt = src_lo.dtype();
        let esz = elem_bytes(dt);
        let src_bytes: Vec<u8> = (0..src_elems * esz).map(|i| (i * 7 + 3) as u8).collect();
        let src = CpuBuf::from_vec(src_bytes.clone());
        let mut dst = CpuBuf::alloc_zeros(dst_elems * esz);
        let mut want = vec![0u8; dst_elems * esz];
        naive(&src_bytes, &src_lo, &mut want, &dst_lo, esz);
        copy_strided(&src, &src_lo, &mut dst, &dst_lo).unwrap();
        assert_eq!(dst.as_bytes(), &want[..], "{:?} -> {:?}", src_lo, dst_lo);
    }

    #[test]
    fn contiguous_all_sizes() {
        for dt in [DType::U8, DType::F16, DType::F32, DType::F64] {
            for n in [1usize, 7, 1023, (1 << 18) + 5] {
                let lo = Layout::contiguous(Shape::new(vec![n]), dt);
                run(lo.clone(), lo, n, n);
            }
        }
    }

    #[test]
    fn transposed_source_rows() {
        for dt in [DType::U8, DType::BF16, DType::F32, DType::I64] {
            let base = Layout::contiguous(Shape::new(vec![37, 53]), dt);
            let src = base.transpose(0, 1).unwrap();
            let dst = Layout::contiguous(Shape::new(vec![53, 37]), dt);
            run(src, dst, 37 * 53, 37 * 53);
        }
    }

    #[test]
    fn permuted_3d_and_narrow_offsets() {
        let dt = DType::F32;
        let base = Layout::contiguous(Shape::new(vec![6, 7, 9]), dt);
        let src = base.permute(&[2, 0, 1]).unwrap();
        let dst = Layout::contiguous(Shape::new(vec![9, 6, 7]), dt);
        run(src, dst, 6 * 7 * 9, 6 * 7 * 9);
        let src = base.narrow(1, 2, 3).unwrap().narrow(2, 1, 5).unwrap();
        let dst = Layout::contiguous(Shape::new(vec![6, 3, 5]), dt);
        run(src, dst, 6 * 7 * 9, 6 * 3 * 5);
        let dst_big = Layout::contiguous(Shape::new(vec![6, 7, 9]), dt).narrow(1, 2, 3).unwrap().narrow(2, 1, 5).unwrap();
        run(Layout::contiguous(Shape::new(vec![6, 3, 5]), dt), dst_big, 6 * 3 * 5, 6 * 7 * 9);
    }

    #[test]
    fn broadcast_source_and_large_strided() {
        let dt = DType::F32;
        let src = Layout::contiguous(Shape::new(vec![1, 300]), dt).expand(&Shape::new(vec![1500, 300])).unwrap();
        let dst = Layout::contiguous(Shape::new(vec![1500, 300]), dt);
        run(src, dst, 300, 1500 * 300);
        let src = Layout::contiguous(Shape::new(vec![700, 900]), dt).transpose(0, 1).unwrap();
        let dst = Layout::contiguous(Shape::new(vec![900, 700]), dt);
        run(src, dst, 700 * 900, 700 * 900);
    }

    #[test]
    fn scalar_and_empty() {
        let dt = DType::U32;
        let lo = Layout::contiguous(Shape::scalar(), dt);
        run(lo.clone(), lo, 1, 1);
        let lo = Layout::contiguous(Shape::new(vec![0, 4]), dt);
        let src = CpuBuf::alloc_zeros(0);
        let mut dst = CpuBuf::alloc_zeros(0);
        copy_strided(&src, &lo, &mut dst, &lo).unwrap();
    }

    #[test]
    fn out_of_bounds_is_error() {
        let dt = DType::F32;
        let lo = Layout::contiguous(Shape::new(vec![10]), dt);
        let src = CpuBuf::alloc_zeros(40);
        let mut dst = CpuBuf::alloc_zeros(36);
        assert!(copy_strided(&src, &lo, &mut dst, &lo).is_err());
    }
}
