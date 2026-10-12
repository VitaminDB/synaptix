use once_cell::sync::OnceCell;
use synaptix_core::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CpuBuf, Storage};

use crate::{act, attention, conv, elementwise, gather, gemm, norms, reduction, rope};

pub struct CpuBackend;

static CPU_BACKEND: OnceCell<CpuBackend> = OnceCell::new();

pub fn cpu_backend() -> &'static dyn Backend {
    CPU_BACKEND.get_or_init(|| {
        crate::threads::pool();
        CpuBackend
    })
}

pub fn ensure_registered() {
    synaptix_core::backend::registry::register_backend(synaptix_core::device::DeviceKind::Cpu, cpu_backend());
}

impl Backend for CpuBackend {
    fn device_kind(&self) -> Device {
        Device::Cpu
    }

    fn alloc_zeros(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        if !device.is_cpu() {
            return Err(SynaptixError::Unsupported("CpuBackend::alloc_zeros on non-CPU"));
        }
        Ok(Storage::Cpu(CpuBuf::alloc_zeros(n_bytes)))
    }

    /// Выход операции ядро пишет целиком, обнулять его незачем: `memset` больших
    /// буферов шёл в одном потоке перед каждой операцией (у OmniVoice — больше
    /// половины времени главного потока, пока остальные ждали).
    /// `SYNAPTIX_CPU_ZERO_ALLOC=1` — как раньше, для поиска ядра, читающего выход.
    fn alloc_uninit(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        if !matches!(device, Device::Cpu) {
            return Err(SynaptixError::Unsupported("CpuBackend::alloc_uninit on non-CPU"));
        }
        static ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ZERO.get_or_init(|| std::env::var("SYNAPTIX_CPU_ZERO_ALLOC").is_ok_and(|v| v == "1")) {
            return Ok(Storage::Cpu(CpuBuf::alloc_zeros(n_bytes)));
        }
        Ok(Storage::Cpu(synaptix_core::memory::arena::alloc_uninit_cpu(n_bytes)))
    }

    fn copy(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        if src_lo.dtype() != dst_lo.dtype() {
            return Err(SynaptixError::dtype_mismatch(src_lo.dtype(), dst_lo.dtype()));
        }
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("copy: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("copy: dst non-cpu"))?;
        crate::copy::copy_strided(src_buf, src_lo, dst_buf, dst_lo)
    }

    fn cast(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("cast: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("cast: dst non-cpu"))?;
        elementwise::cast_dispatch(src_buf, src_lo, dst_buf, dst_lo)
    }

    fn unary(&self, op: UnaryOp, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
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

    fn matmul(&self, lhs: (&Storage, &Layout), rhs: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let (a_st, a_lo) = lhs;
        let (b_st, b_lo) = rhs;
        let (dst_st, dst_lo) = dst;
        let a_buf = a_st.as_cpu().ok_or(SynaptixError::Unsupported("matmul: a non-cpu"))?;
        let b_buf = b_st.as_cpu().ok_or(SynaptixError::Unsupported("matmul: b non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("matmul: dst non-cpu"))?;
        gemm::matmul_dispatch(a_buf, a_lo, b_buf, b_lo, dst_buf, dst_lo)
    }

    #[allow(clippy::too_many_arguments)]
    fn im2col(
        &self,
        input: (&Storage, &Layout),
        col: (&mut Storage, &Layout),
        kh: usize,
        kw: usize,
        h_out: usize,
        w_out: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        m_offset: u64,
        m_count: u64,
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = input;
        let (col_st, col_lo) = col;
        if !in_lo.is_contiguous() || in_lo.dims().len() != 4 {
            return Err(SynaptixError::NonContiguous);
        }
        let src = in_st.as_cpu().ok_or(SynaptixError::Unsupported("im2col: input non-cpu"))?;
        let dst = col_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("im2col: col non-cpu"))?;
        let d = in_lo.dims();
        let p = conv::Im2ColRows {
            c: d[1],
            h: d[2],
            w: d[3],
            kh,
            kw,
            h_out,
            w_out,
            stride,
            padding,
            m_offset: m_offset as usize,
            m_count: m_count as usize,
        };
        match in_lo.dtype() {
            DType::F32 => conv::im2col_rows::<f32>(src.as_bytes(), in_lo.offset(), dst.as_bytes_mut(), col_lo.offset(), p),
            DType::F16 => conv::im2col_rows::<half::f16>(src.as_bytes(), in_lo.offset(), dst.as_bytes_mut(), col_lo.offset(), p),
            DType::BF16 => conv::im2col_rows::<half::bf16>(src.as_bytes(), in_lo.offset(), dst.as_bytes_mut(), col_lo.offset(), p),
            _ => return Err(SynaptixError::Unsupported("im2col: dtype")),
        }
        Ok(())
    }

    fn linear(&self, x: (&Storage, &Layout), w: (&Storage, &Layout), out: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let (x_st, x_lo) = x;
        let (w_st, w_lo) = w;
        let (out_st, out_lo) = out;
        let x_buf = x_st.as_cpu().ok_or(SynaptixError::Unsupported("linear: x non-cpu"))?;
        let w_buf = w_st.as_cpu().ok_or(SynaptixError::Unsupported("linear: w non-cpu"))?;
        let out_buf = out_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("linear: out non-cpu"))?;
        gemm::linear_dispatch(x_buf, x_lo, w_buf, w_lo, out_buf, out_lo)
    }

    fn linear_epilogue(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        residual: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()> {
        act::linear_epilogue(x, w, bias, residual, out, stream)
    }

    fn linear_quant(&self, x: (&Storage, &Layout), w: &QuantWeight, out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        crate::quant::gemv::linear_quant(x, w, out, stream)
    }

    fn quantize_nvfp4(&self, w: (&Storage, &Layout), n: usize, k: usize, stream: &Stream) -> Result<(Storage, Storage)> {
        crate::quant::gemv::quantize_nvfp4(w, n, k, stream)
    }

    fn quantize_mxfp8(&self, w: (&Storage, &Layout), n: usize, k: usize, stream: &Stream) -> Result<(Storage, Storage)> {
        crate::quant::gemv::quantize_mxfp8(w, n, k, stream)
    }

    fn quantize_sq(&self, w: (&Storage, &Layout), bits: u8, n: usize, k: usize, stream: &Stream) -> Result<Storage> {
        crate::quant::gemv::quantize_sq(w, bits, n, k, stream)
    }

    fn nvfp4_dequant(&self, packed: &Storage, scales: &Storage, out: (&mut Storage, &Layout), n: usize, k: usize, stream: &Stream) -> Result<()> {
        crate::quant::gemv::nvfp4_dequant(packed, scales, out, n, k, stream)
    }

    fn mxfp8_dequant(&self, packed: (&Storage, &Layout), scales: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        crate::quant::gemv::mxfp8_dequant(packed, scales, out, stream)
    }

    fn block_dequant(&self, packed: &Storage, dtype: DType, out: (&mut Storage, &Layout), n: usize, k: usize, stream: &Stream) -> Result<()> {
        crate::quant::gemv::block_dequant(packed, dtype, out, n, k, stream)
    }

    fn block_gather_dequant(
        &self,
        table: &Storage,
        dtype: DType,
        ids: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        vocab: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        crate::quant::gemv::block_gather_dequant(table, dtype, ids, out, vocab, k, stream)
    }

    fn quant_gemv_batched(
        &self,
        w_packed: &[&Storage],
        w_scales: &[Option<&Storage>],
        dtype: DType,
        x: &Storage,
        x_rows: &[usize],
        out: (&mut Storage, &Layout),
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        crate::quant::gemv::quant_gemv_batched(w_packed, w_scales, dtype, x, x_rows, out, n, k, stream)
    }

    fn rms_norm(&self, x: (&Storage, &Layout), w: (&Storage, &Layout), out: (&mut Storage, &Layout), eps: f32, qwen_gain: bool, stream: &Stream) -> Result<()> {
        norms::rms_norm(x, w, out, eps, qwen_gain, stream)
    }

    fn rms_norm_residual(
        &self,
        x: (&Storage, &Layout),
        residual: (&Storage, &Layout),
        w: (&Storage, &Layout),
        hidden_out: (&mut Storage, &Layout),
        y: (&mut Storage, &Layout),
        eps: f32,
        qwen_gain: bool,
        stream: &Stream,
    ) -> Result<()> {
        norms::rms_norm_residual(x, residual, w, hidden_out, y, eps, qwen_gain, stream)
    }

    fn layer_norm(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        norms::layer_norm(x, w, bias, out, eps, stream)
    }

    fn group_rms(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        rows: usize,
        groups: usize,
        group: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        norms::group_rms(x, w, out, rows, groups, group, eps, stream)
    }

    fn group_norm(
        &self,
        x: (&Storage, &Layout),
        weight: Option<(&Storage, &Layout)>,
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        num_groups: usize,
        eps: f32,
        silu: bool,
        nhwc: bool,
        stream: &Stream,
    ) -> Result<()> {
        norms::group_norm(x, weight, bias, out, num_groups, eps, silu, nhwc, stream)
    }

    fn pixel_norm(&self, x: (&Storage, &Layout), out: (&mut Storage, &Layout), c: usize, eps: f32, silu: bool, stream: &Stream) -> Result<()> {
        norms::pixel_norm(x, out, c, eps, silu, stream)
    }

    fn silu_and_mul(&self, gate: (&Storage, &Layout), up: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        act::silu_and_mul(gate, up, out, stream)
    }

    fn geglu_split(&self, inp: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        act::geglu_split(inp, out, stream)
    }

    fn scale_act(&self, x: (&Storage, &Layout), out: (&mut Storage, &Layout), n: usize, scale: f32, a: u32, stream: &Stream) -> Result<()> {
        act::scale_act(x, out, n, scale, a, stream)
    }

    fn ternary_fused(
        &self,
        kind: u8,
        x: (&Storage, &Layout),
        b: (&Storage, &Layout),
        c: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()> {
        act::ternary_fused(kind, x, b, c, dst, stream)
    }

    fn snake(
        &self,
        x: (&Storage, &Layout),
        a: (&Storage, &Layout),
        binv: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        c: usize,
        t_len: usize,
        stream: &Stream,
    ) -> Result<()> {
        act::snake(x, a, binv, out, c, t_len, stream)
    }

    fn hc_mix(
        &self,
        up: (&Storage, &Layout),
        normed: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        rows: usize,
        hc: usize,
        h: usize,
        stream: &Stream,
    ) -> Result<()> {
        act::hc_mix(up, normed, out, rows, hc, h, stream)
    }

    fn hc_inject(
        &self,
        hyper: (&Storage, &Layout),
        block: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        rows: usize,
        hc: usize,
        h: usize,
        stream: &Stream,
    ) -> Result<()> {
        act::hc_inject(hyper, block, w, out, rows, hc, h, stream)
    }

    fn weighted_rows_sum(
        &self,
        parts: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        t: usize,
        k: usize,
        h: usize,
        stream: &Stream,
    ) -> Result<()> {
        act::weighted_rows_sum(parts, w, out, t, k, h, stream)
    }

    fn moe_combine(
        &self,
        y: (&Storage, &Layout),
        inv: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        t: usize,
        k: usize,
        d: usize,
        stream: &Stream,
    ) -> Result<()> {
        act::moe_combine(y, inv, w, out, t, k, d, stream)
    }

    fn rope_split(&self, x: (&Storage, &Layout), cos: (&Storage, &Layout), sin: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        rope::rope_split(x, cos, sin, out, stream)
    }

    fn rope_split_partial(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        rot_dim: usize,
        out: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()> {
        rope::rope_split_partial(x, cos, sin, rot_dim, out, stream)
    }

    fn rope_interleaved(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        h: usize,
        out: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()> {
        rope::rope_interleaved(x, cos, sin, h, out, stream)
    }

    fn rope_apply_dev(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        start_pos: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        rotary_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        rope::rope_apply_dev(x, cos, sin, start_pos, out, rotary_dim, stream)
    }

    fn flash_attention(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention(q, k, v, out, scale, causal, stream)
    }

    fn flash_attention_window(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        window: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_window(q, k, v, out, scale, window, causal, stream)
    }

    fn flash_attention_window_dev(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        t_cache: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        window: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_window_dev(q, k, v, t_cache, out, scale, window, causal, stream)
    }

    fn flash_attention_bshd(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_bshd(q, k, v, out, scale, causal, stream)
    }

    fn flash_attention_dev(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        t_cache: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_dev(q, k, v, t_cache, out, scale, causal, stream)
    }

    fn flash_attention_prefill_dev(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        t_cache: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_dev(q, k, v, t_cache, out, scale, causal, stream)
    }

    fn flash_attention_blocks(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        table: (&Storage, &Layout),
        tail_from: (&Storage, &Layout),
        tail_len: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        ratio: usize,
        scale: f32,
        row_offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_blocks(q, k, v, table, tail_from, tail_len, out, ratio, scale, row_offset, stream)
    }

    fn flash_attention_blocks_mxfp8(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        k_scale: (&Storage, &Layout),
        v_scale: (&Storage, &Layout),
        table: (&Storage, &Layout),
        tail_from: (&Storage, &Layout),
        tail_len: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        ratio: usize,
        scale: f32,
        row_offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_blocks_mxfp8(q, k, v, k_scale, v_scale, table, tail_from, tail_len, out, ratio, scale, row_offset, stream)
    }

    fn kv_append(&self, dst: (&mut Storage, &Layout), src: (&Storage, &Layout), seq_pos: usize, _stream: &Stream) -> Result<()> {
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
        let (b, nkv, max_seq, hd) = (dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3]);
        let (sb, snkv, t_new, shd) = (src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3]);
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::shape_mismatch(src_lo.dims(), dst_lo.dims()));
        }
        if seq_pos + t_new > max_seq {
            return Err(SynaptixError::Other(format!("kv_append: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}")));
        }
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("kv_append: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("kv_append: dst non-cpu"))?;
        let block = t_new * hd;
        let n_bytes = dtype.bytes_for_numel(block);
        for bh in 0..(b * nkv) {
            let src_o = dtype.bytes_for_numel(src_lo.offset() + bh * block);
            let dst_o = dtype.bytes_for_numel(dst_lo.offset() + (bh * max_seq + seq_pos) * hd);
            dst_buf.as_bytes_mut()[dst_o..dst_o + n_bytes].copy_from_slice(&src_buf.as_bytes()[src_o..src_o + n_bytes]);
        }
        Ok(())
    }

    fn kv_append_dev(&self, dst: (&mut Storage, &Layout), src: (&Storage, &Layout), seq_pos: (&Storage, &Layout), stream: &Stream) -> Result<()> {
        gather::kv_append_dev(dst, src, seq_pos, stream)
    }

    fn kv_append_quant_mxfp8(
        &self,
        dst: (&mut Storage, &Layout),
        scale_dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: usize,
        stream: &Stream,
    ) -> Result<()> {
        attention::kv_append_quant_mxfp8(dst, scale_dst, src, seq_pos, stream)
    }

    fn kv_append_quant_mxfp8_dev(
        &self,
        dst: (&mut Storage, &Layout),
        scale_dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: (&Storage, &Layout),
        stream: &Stream,
    ) -> Result<()> {
        attention::kv_append_quant_mxfp8_dev(dst, scale_dst, src, seq_pos, stream)
    }

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
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_mxfp8kv(q, k, v, k_scale, v_scale, out, scale, causal, stream)
    }

    fn flash_attention_mxfp8kv_dev(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        k_scale: (&Storage, &Layout),
        v_scale: (&Storage, &Layout),
        t_cache: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<()> {
        attention::flash_attention_mxfp8kv_dev(q, k, v, k_scale, v_scale, t_cache, out, scale, causal, stream)
    }

    fn embed_gather(&self, table: (&Storage, &Layout), ids: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        gather::embed_gather(table, ids, out, stream)
    }

    fn embed_gather_mxfp8(
        &self,
        table: &Storage,
        scales: &Storage,
        ids: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        vocab: usize,
        dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        gather::embed_gather_mxfp8(table, scales, ids, out, vocab, dim, stream)
    }

    fn topk_rows(&self, scores: (&Storage, &Layout), indices: (&mut Storage, &Layout), values: (&mut Storage, &Layout), k: usize, stream: &Stream) -> Result<()> {
        gather::topk_rows(scores, indices, values, k, stream)
    }

    fn topk_wide(&self, scores: (&Storage, &Layout), valid: (&Storage, &Layout), out: (&mut Storage, &Layout), k: usize, stream: &Stream) -> Result<()> {
        gather::topk_wide(scores, valid, out, k, stream)
    }

    fn upsample_nearest2x(&self, input: (&Storage, &Layout), out: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        gather::upsample_nearest2x(input, out, stream)
    }

    fn nchw_to_nhwc(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        gather::nchw_to_nhwc(src, dst, stream)
    }

    fn nhwc_to_nchw(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), stream: &Stream) -> Result<()> {
        gather::nhwc_to_nchw(src, dst, stream)
    }

    fn conv_epilogue(
        &self,
        out2d: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        residual: Option<(&Storage, &Layout)>,
        temb_bc: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        b: usize,
        c: usize,
        h: usize,
        w: usize,
        stream: &Stream,
    ) -> Result<()> {
        gather::conv_epilogue(out2d, bias, residual, temb_bc, out, b, c, h, w, stream)
    }

    fn conv2d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: (usize, usize),
        padding: (usize, usize),
        stream: &Stream,
    ) -> Result<()> {
        conv::conv2d(input, weight, bias, out, stride, padding, stream)
    }

    fn conv3d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        stream: &Stream,
    ) -> Result<()> {
        conv::conv3d(input, weight, bias, out, stride, padding, stream)
    }

    fn dwconv1d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: usize,
        padding: usize,
        transpose: bool,
        stream: &Stream,
    ) -> Result<()> {
        conv::dwconv1d(input, weight, bias, out, stride, padding, transpose, stream)
    }

    fn reduce(&self, op: ReduceOp, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), dims: &[usize], _stream: &Stream) -> Result<()> {
        let (src_st, src_lo) = src;
        let (dst_st, dst_lo) = dst;
        let src_buf = src_st.as_cpu().ok_or(SynaptixError::Unsupported("reduce: src non-cpu"))?;
        let dst_buf = dst_st.as_cpu_mut().ok_or(SynaptixError::Unsupported("reduce: dst non-cpu"))?;
        reduction::reduce_dispatch(op, src_buf, src_lo, dst_buf, dst_lo, dims)
    }
}
