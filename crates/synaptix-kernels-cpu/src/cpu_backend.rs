use once_cell::sync::OnceCell;
use synaptix_core::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CpuBuf, Storage};

use crate::{elementwise, gemm, reduction};

pub struct CpuBackend;

static CPU_BACKEND: OnceCell<CpuBackend> = OnceCell::new();

pub fn cpu_backend() -> &'static dyn Backend {
    CPU_BACKEND.get_or_init(|| CpuBackend)
}

pub fn ensure_registered() {
    synaptix_core::backend::registry::register_backend(
        synaptix_core::device::DeviceKind::Cpu,
        cpu_backend(),
    );
}

impl Backend for CpuBackend {
    fn device_kind(&self) -> Device { Device::Cpu }

    fn alloc_zeros(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        if !device.is_cpu() {
            return Err(SynaptixError::Unsupported("CpuBackend::alloc_zeros on non-CPU"));
        }
        Ok(Storage::Cpu(CpuBuf::alloc_zeros(n_bytes)))
    }

    fn copy(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        if src_lo.dtype() != dst_lo.dtype() {
            return Err(SynaptixError::dtype_mismatch(src_lo.dtype(), dst_lo.dtype()));
        }
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("copy: src non-cpu"))?;
        let dst_buf = dst_st
            .as_cpu_mut()
            .ok_or(SynaptixError::Unsupported("copy: dst non-cpu"))?;
        let dtype = src_lo.dtype();
        match dtype {
            DType::F32 => copy_strided::<f32>(src_buf, src_lo, dst_buf, dst_lo),
            DType::F64 => copy_strided::<f64>(src_buf, src_lo, dst_buf, dst_lo),
            DType::F16 => copy_strided::<half::f16>(src_buf, src_lo, dst_buf, dst_lo),
            DType::BF16 => copy_strided::<half::bf16>(src_buf, src_lo, dst_buf, dst_lo),
            DType::U8 => copy_strided::<u8>(src_buf, src_lo, dst_buf, dst_lo),
            DType::U32 => copy_strided::<u32>(src_buf, src_lo, dst_buf, dst_lo),
            DType::I32 => copy_strided::<i32>(src_buf, src_lo, dst_buf, dst_lo),
            DType::I64 => copy_strided::<i64>(src_buf, src_lo, dst_buf, dst_lo),
            _ => Err(SynaptixError::Unsupported("copy on quantized dtype")),
        }
    }

    fn cast(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("cast: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("cast: dst non-cpu"))?;
        crate::elementwise::cast_dispatch(src_buf, src_lo, dst_buf, dst_lo)
    }

    fn unary(
        &self,
        op: UnaryOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("unary: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("unary: dst non-cpu"))?;
        elementwise::unary_dispatch(op, src_buf, src_lo, dst_buf, dst_lo)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (a_st, a_lo) = a;
        let (b_st, b_lo) = b;
        let (dst_st, dst_lo) = dst;
        let a_buf = a_st.as_cpu().ok_or(SynaptixError::Unsupported("binary: a non-cpu"))?;
        let b_buf = b_st.as_cpu().ok_or(SynaptixError::Unsupported("binary: b non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("binary: dst non-cpu"))?;
        elementwise::binary_dispatch(op, a_buf, a_lo, b_buf, b_lo, dst_buf, dst_lo)
    }

    fn matmul(
        &self,
        lhs: (&Storage, &Layout),
        rhs: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (a_st, a_lo) = lhs;
        let (b_st, b_lo) = rhs;
        let (dst_st, dst_lo) = dst;
        let a_buf = a_st.as_cpu().ok_or(SynaptixError::Unsupported("matmul: a non-cpu"))?;
        let b_buf = b_st.as_cpu().ok_or(SynaptixError::Unsupported("matmul: b non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("matmul: dst non-cpu"))?;
        gemm::matmul_dispatch(a_buf, a_lo, b_buf, b_lo, dst_buf, dst_lo)
    }

    fn linear(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (w_st, w_lo) = w;
        let (out_st, out_lo) = out;
        let x_buf = x_st.as_cpu().ok_or(SynaptixError::Unsupported("linear: x non-cpu"))?;
        let w_buf = w_st.as_cpu().ok_or(SynaptixError::Unsupported("linear: w non-cpu"))?;
        let out_buf = out_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("linear: out non-cpu"))?;
        gemm::linear_dispatch(x_buf, x_lo, w_buf, w_lo, out_buf, out_lo)
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        dims: &[usize],
        _stream: &Stream,
    ) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("reduce: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("reduce: dst non-cpu"))?;
        reduction::reduce_dispatch(op, src_buf, src_lo, dst_buf, dst_lo, dims)
    }

    fn kv_append(
        &self,
        dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (dst_st, dst_lo) = dst;
        let (src_st, src_lo) = src;
        if src_lo.dtype() != dst_lo.dtype() {
            return Err(SynaptixError::dtype_mismatch(src_lo.dtype(), dst_lo.dtype()));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("kv_append: expect rank-4 [B,nkv,T,hd]"));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let dtype = dst_lo.dtype();
        let (b, nkv, max_seq, hd) =
            (dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3]);
        let (sb, snkv, t_new, shd) =
            (src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3]);
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::shape_mismatch(src_lo.dims(), dst_lo.dims()));
        }
        if seq_pos + t_new > max_seq {
            return Err(SynaptixError::Other(format!(
                "kv_append: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
            )));
        }
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("kv_append: src non-cpu"))?;
        let dst_buf =
            dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("kv_append: dst non-cpu"))?;
        let block = t_new * hd;
        let n_bytes = dtype.bytes_for_numel(block);
        for bh in 0..(b * nkv) {
            let src_o = dtype.bytes_for_numel(src_lo.offset() + bh * block);
            let dst_o = dtype.bytes_for_numel(dst_lo.offset() + (bh * max_seq + seq_pos) * hd);
            dst_buf.as_bytes_mut()[dst_o..dst_o + n_bytes]
                .copy_from_slice(&src_buf.as_bytes()[src_o..src_o + n_bytes]);
        }
        Ok(())
    }

    fn kv_append_quant_mxfp8(
        &self,
        dst: (&mut Storage, &Layout),
        scale_dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (dst_st, dst_lo) = dst;
        let (sc_st, sc_lo) = scale_dst;
        let (src_st, src_lo) = src;
        if dst_lo.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: dst must be MXFP8"));
        }
        if sc_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: scale must be U8"));
        }
        if src_lo.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: src must be BF16"));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 || sc_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: ranks [4,4,4]"));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() || !sc_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, nkv, max_seq, hd) =
            (dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3]);
        let (sb, snkv, t_new, shd) =
            (src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3]);
        if hd % crate::quant::MXFP8_BLOCK != 0 {
            return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: hd % 32 != 0"));
        }
        let nb = hd / crate::quant::MXFP8_BLOCK;
        if sb != b || snkv != nkv || shd != hd || sc_lo.dims() != [b, nkv, max_seq, nb] {
            return Err(SynaptixError::shape_mismatch(src_lo.dims(), dst_lo.dims()));
        }
        if seq_pos + t_new > max_seq {
            return Err(SynaptixError::Other(format!(
                "cpu kv_quant mxfp8: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
            )));
        }
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu kv_quant mxfp8: src non-cpu"))?;
        let src_bf16: &[half::bf16] = bytemuck::cast_slice(src_buf.as_bytes());
        let so = src_lo.offset();
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("cpu kv_quant mxfp8: dst non-cpu"))?;
        let dst_bytes = dst_buf.as_bytes_mut();
        let dst_o0 = dst_lo.offset();
        let sc_buf = sc_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("cpu kv_quant mxfp8: scale non-cpu"))?;
        let sc_bytes = sc_buf.as_bytes_mut();
        let sc_o0 = sc_lo.offset();
        for bh in 0..(b * nkv) {
            for t in 0..t_new {
                let src_row = so + (bh * t_new + t) * hd;
                let dst_row = dst_o0 + (bh * max_seq + seq_pos + t) * hd;
                let sc_row = sc_o0 + (bh * max_seq + seq_pos + t) * nb;
                for blk in 0..nb {
                    let base = blk * crate::quant::MXFP8_BLOCK;
                    let mut amax = 0.0f32;
                    for i in 0..crate::quant::MXFP8_BLOCK {
                        amax = amax.max(src_bf16[src_row + base + i].to_f32().abs());
                    }
                    let sbyte = crate::quant::e8m0_scale_byte(amax);
                    let sv = crate::quant::e8m0_decode(sbyte);
                    for i in 0..crate::quant::MXFP8_BLOCK {
                        let x = src_bf16[src_row + base + i].to_f32() / sv;
                        dst_bytes[dst_row + base + i] = crate::quant::encode_e4m3(x);
                    }
                    sc_bytes[sc_row + blk] = sbyte;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn flash_attention_mxfp8kv(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        k_scale: (&Storage, &Layout),
        v_scale: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (ks_st, ks_lo) = k_scale;
        let (vs_st, vs_lo) = v_scale;
        let (out_st, out_lo) = out;
        let q_dtype = q_lo.dtype();
        if !q_dtype.is_float() {
            return Err(SynaptixError::Unsupported("cpu flash mxfp8: q must be float"));
        }
        if k_lo.dtype() != DType::MXFP8 || v_lo.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("cpu flash mxfp8: k/v must be MXFP8"));
        }
        if ks_lo.dtype() != DType::U8 || vs_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cpu flash mxfp8: scales must be U8"));
        }
        let (b, nh, tq, d) =
            (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2], q_lo.dims()[3]);
        let (nkv, tkv) = (k_lo.dims()[1], k_lo.dims()[2]);
        if nkv == 0 || nh % nkv != 0 {
            return Err(SynaptixError::Unsupported("cpu flash mxfp8: GQA"));
        }
        if d % crate::quant::MXFP8_BLOCK != 0 {
            return Err(SynaptixError::Unsupported("cpu flash mxfp8: d % 32 != 0"));
        }
        let nb = d / crate::quant::MXFP8_BLOCK;
        let group = nh / nkv;

        let q_buf = q_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: q non-cpu"))?;
        let k_buf = k_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: k non-cpu"))?;
        let v_buf = v_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: v non-cpu"))?;
        let ks_buf = ks_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: ks non-cpu"))?;
        let vs_buf = vs_st.as_cpu().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: vs non-cpu"))?;

        let qb = q_buf.as_bytes();
        let kb = k_buf.as_bytes();
        let vb = v_buf.as_bytes();
        let ksb = ks_buf.as_bytes();
        let vsb = vs_buf.as_bytes();

        let qs = q_lo.strides().as_slice().to_vec();
        let ks = k_lo.strides().as_slice().to_vec();
        let vs = v_lo.strides().as_slice().to_vec();
        let kss = ks_lo.strides().as_slice().to_vec();
        let vss = vs_lo.strides().as_slice().to_vec();
        let (qo, ko, vo, kso, vso) =
            (q_lo.offset(), k_lo.offset(), v_lo.offset(), ks_lo.offset(), vs_lo.offset());

        let read_qf = |idx: usize| -> f32 {
            match q_dtype {
                DType::F32 => f32::from_le_bytes(qb[idx * 4..idx * 4 + 4].try_into().unwrap()),
                DType::F16 => {
                    half::f16::from_bits(u16::from_le_bytes(qb[idx * 2..idx * 2 + 2].try_into().unwrap())).to_f32()
                }
                DType::BF16 => {
                    half::bf16::from_bits(u16::from_le_bytes(qb[idx * 2..idx * 2 + 2].try_into().unwrap())).to_f32()
                }
                _ => 0.0,
            }
        };

        let mut out_f32 = vec![0.0f32; b * nh * tq * d];
        for bi in 0..b {
            for h in 0..nh {
                let h_kv = h / group;
                for ti in 0..tq {
                    let q_pos = if tkv >= tq { tkv - tq + ti } else { ti };
                    let mut m = f32::NEG_INFINITY;
                    let mut l = 0.0f32;
                    let mut acc = vec![0.0f32; d];
                    for t in 0..tkv {
                        if causal && t > q_pos {
                            break;
                        }
                        // K-dot блочный: scale зависит от blk=d/32 → внутри суммы.
                        let mut dot = 0.0f32;
                        for blk in 0..nb {
                            let ks_idx = (kso as isize
                                + bi as isize * kss[0] + h_kv as isize * kss[1]
                                + t as isize * kss[2] + blk as isize * kss[3]) as usize;
                            let ksc = crate::quant::e8m0_decode(ksb[ks_idx]);
                            let mut bsum = 0.0f32;
                            for i in 0..crate::quant::MXFP8_BLOCK {
                                let dd = blk * crate::quant::MXFP8_BLOCK + i;
                                let qi = (qo as isize
                                    + bi as isize * qs[0] + h as isize * qs[1] + ti as isize * qs[2] + dd as isize * qs[3]) as usize;
                                let ki = (ko as isize
                                    + bi as isize * ks[0] + h_kv as isize * ks[1] + t as isize * ks[2] + dd as isize * ks[3]) as usize;
                                bsum += read_qf(qi) * crate::quant::decode_e4m3(kb[ki]);
                            }
                            dot += bsum * ksc;
                        }
                        let s = dot * scale;
                        let m_new = m.max(s);
                        let alpha = if m == f32::NEG_INFINITY { 0.0 } else { (m - m_new).exp() };
                        let p = (s - m_new).exp();
                        l = l * alpha + p;
                        for dd in 0..d {
                            let blk = dd / crate::quant::MXFP8_BLOCK;
                            let vs_idx = (vso as isize
                                + bi as isize * vss[0] + h_kv as isize * vss[1]
                                + t as isize * vss[2] + blk as isize * vss[3]) as usize;
                            let vsc = crate::quant::e8m0_decode(vsb[vs_idx]);
                            let vi = (vo as isize
                                + bi as isize * vs[0] + h_kv as isize * vs[1] + t as isize * vs[2] + dd as isize * vs[3]) as usize;
                            acc[dd] = acc[dd] * alpha + p * crate::quant::decode_e4m3(vb[vi]) * vsc;
                        }
                        m = m_new;
                    }
                    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                    let obase = ((bi * nh + h) * tq + ti) * d;
                    for dd in 0..d {
                        out_f32[obase + dd] = acc[dd] * inv;
                    }
                }
            }
        }

        let out_buf = out_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("cpu flash mxfp8: out non-cpu"))?;
        let ob = out_buf.as_bytes_mut();
        let oo = out_lo.offset();
        for (i, &val) in out_f32.iter().enumerate() {
            let idx = oo + i;
            match q_dtype {
                DType::F32 => ob[idx * 4..idx * 4 + 4].copy_from_slice(&val.to_le_bytes()),
                DType::F16 => ob[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(val).to_bits().to_le_bytes()),
                DType::BF16 => ob[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(val).to_bits().to_le_bytes()),
                _ => {}
            }
        }
        Ok(())
    }
}

fn copy_strided<T: bytemuck::Pod>(
    src: &CpuBuf,
    src_lo: &Layout,
    dst: &mut CpuBuf,
    dst_lo: &Layout,
) -> Result<()> {
    let dims = src_lo.dims();
    if dims != dst_lo.dims() {
        return Err(SynaptixError::shape_mismatch(dst_lo.dims(), dims));
    }
    let src_slice: &[T] = bytemuck::cast_slice(src.as_bytes());
    let dst_slice: &mut [T] = bytemuck::cast_slice_mut(dst.as_bytes_mut());
    let src_strides: Vec<isize> = src_lo.strides().as_slice().to_vec();
    let src_offset = src_lo.offset();
    let dst_strides: Vec<isize> = dst_lo.strides().as_slice().to_vec();
    let dst_offset = dst_lo.offset();
    let numel = src_lo.numel();
    if numel == 0 { return Ok(()); }
    let rank = dims.len();
    let mut idx = vec![0usize; rank];
    for _ in 0..numel {
        let mut s_off = src_offset as isize;
        let mut d_off = dst_offset as isize;
        for k in 0..rank {
            s_off += idx[k] as isize * src_strides[k];
            d_off += idx[k] as isize * dst_strides[k];
        }
        dst_slice[d_off as usize] = src_slice[s_off as usize];
        for k in (0..rank).rev() {
            idx[k] += 1;
            if idx[k] < dims[k] { break; }
            idx[k] = 0;
        }
    }
    Ok(())
}
