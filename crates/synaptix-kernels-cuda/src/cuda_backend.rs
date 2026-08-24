use once_cell::sync::OnceCell;
use synaptix_core::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use synaptix_core::device::{Device, DeviceKind};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CudaBuf, Storage};

pub struct CudaBackend;

static CUDA_BACKEND: OnceCell<CudaBackend> = OnceCell::new();

pub fn cuda_backend() -> &'static dyn Backend {
    CUDA_BACKEND.get_or_init(|| CudaBackend)
}

pub fn ensure_registered() {
    synaptix_core::backend::registry::register_backend(DeviceKind::Cuda, cuda_backend());
}


fn nvfp4_weight_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("SYNAPTIX_NVFP4_WO").as_deref(), Ok("1")))
}

fn stream_is_capturing(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> bool {
    matches!(
        stream.capture_status(),
        Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    )
}

impl Backend for CudaBackend {
    fn device_kind(&self) -> Device {
        Device::Cuda(0)
    }

    fn alloc_zeros(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        let ord = match device {
            Device::Cuda(i) => i,
            _ => {
                return Err(SynaptixError::Unsupported(
                    "CudaBackend::alloc_zeros on non-Cuda",
                ))
            }
        };
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let buf = match synaptix_core::device::cuda::alloc_act_zeros::<u8>(&stream, n_bytes) {
            Ok(b) => b,
            Err(_) => {
                // OOM: пул async-аллокатора держит освобождённые блоки (фрагментация)
                // — вернуть их драйверу (cuMemPoolTrimTo→0) и повторить. Спасает
                // загрузку/prefill 27B на грани 24GB. ВАЖНО: сперва sync ВСЕХ
                // стримов (default+alloc+loader) — cuMemFreeAsync исполняется в
                // порядке СВОЕГО стрима, до sync trim pending-frees не видит.
                let _ = synaptix_core::device::cuda::synchronize_all(ord);
                let _ = synaptix_core::memory::cuda_pool::trim_pools_on_oom(ord);
                {
                    // Ретраи с эскалацией: фрагментация транзиентна (соседние
                    // frees на стримах подходят с задержкой) — sync+trim+пауза
                    // до 5 заходов пробивают почти любую "дырку".
                    let mut got = None;
                    for attempt in 0..5u32 {
                        std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
                        let _ = synaptix_core::device::cuda::synchronize_all(ord);
                        let _ = synaptix_core::memory::cuda_pool::trim_pools_on_oom(ord);
                        if let Ok(b) = synaptix_core::device::cuda::alloc_act_zeros::<u8>(&stream, n_bytes) {
                            got = Some(b);
                            break;
                        }
                    }
                    got.ok_or_else(|| {
                        for (bytes, count) in
                            synaptix_core::memory::cuda_pool::live_alloc_top(12)
                        {
                            eprintln!("[OOM_TOP] {bytes:>12} B × {count} = {:.2}GB",
                                bytes as f64 * count as f64 / 1e9);
                        }
                        eprintln!("[OOM_BT] alloc_zeros({n_bytes}):\n{}",
                            std::backtrace::Backtrace::force_capture());
                        SynaptixError::Cuda(format!("alloc_zeros({n_bytes}) after trim+retries: OOM"))
                    })?
                }
            }
        };
        let ctx = synaptix_core::device::cuda::get(ord)?;
        Ok(Storage::Cuda(CudaBuf::new(ctx, stream, buf, ord)))
    }

    fn alloc_uninit(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        let ord = match device {
            Device::Cuda(i) => i,
            _ => {
                return Err(SynaptixError::Unsupported(
                    "CudaBackend::alloc_uninit on non-Cuda",
                ))
            }
        };
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let buf = match unsafe { synaptix_core::device::cuda::alloc_act_uninit::<u8>(&stream, n_bytes) } {
            Ok(b) => b,
            Err(_) => {
                // см. alloc_zeros: sync ВСЕХ стримов до trim, иначе pending-frees не видны
                let _ = synaptix_core::device::cuda::synchronize_all(ord);
                let _ = synaptix_core::memory::cuda_pool::trim_pools_on_oom(ord);
                {
                    // ретраи с эскалацией — см. alloc_zeros
                    let mut got = None;
                    for attempt in 0..5u32 {
                        std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
                        let _ = synaptix_core::device::cuda::synchronize_all(ord);
                        let _ = synaptix_core::memory::cuda_pool::trim_pools_on_oom(ord);
                        if let Ok(b) = unsafe { synaptix_core::device::cuda::alloc_act_uninit::<u8>(&stream, n_bytes) } {
                            got = Some(b);
                            break;
                        }
                    }
                    got.ok_or_else(|| {
                        for (bytes, count) in
                            synaptix_core::memory::cuda_pool::live_alloc_top(12)
                        {
                            eprintln!("[OOM_TOP] {bytes:>12} B × {count} = {:.2}GB",
                                bytes as f64 * count as f64 / 1e9);
                        }
                        let (free, total) =
                            synaptix_core::device::cuda::mem_info(ord).unwrap_or((0, 0));
                        let (rsv, used) =
                            synaptix_core::memory::cuda_pool::cuda_mempool_stats(ord)
                                .unwrap_or((0, 0));
                        eprintln!(
                            "[OOM_SUM] live={:.2}GB free={:.2}GB total={:.2}GB pool_rsv={:.2}GB pool_used={:.2}GB",
                            synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9,
                            free as f64 / 1e9, total as f64 / 1e9,
                            rsv as f64 / 1e9, used as f64 / 1e9
                        );
                        eprintln!("[OOM_BT] alloc_uninit({n_bytes}):\n{}",
                            std::backtrace::Backtrace::force_capture());
                        SynaptixError::Cuda(format!("alloc_uninit({n_bytes}) after trim+retries: OOM"))
                    })?
                }
            }
        };
        let ctx = synaptix_core::device::cuda::get(ord)?;
        Ok(Storage::Cuda(CudaBuf::new(ctx, stream, buf, ord)))
    }

    fn copy(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let src_lo = src.1;
        let dst_lo = dst.1;
        // Быстрый путь: contiguous src без offset + contiguous dst → raw dtod memcpy.
        if src_lo.is_contiguous() && src_lo.offset() == 0 && dst_lo.is_contiguous() {
            let src_buf = src
                .0
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("cuda copy: src non-cuda"))?;
            let dst_buf = dst
                .0
                .as_cuda_mut()
                .ok_or(SynaptixError::Unsupported("cuda copy: dst non-cuda"))?;
            let stream = synaptix_core::device::cuda::compute_stream_for(src_buf.stream(), src_buf.ordinal())?;
            stream
                .memcpy_dtod(src_buf.slice(), dst_buf.slice_mut())
                .map_err(|e| SynaptixError::Cuda(format!("memcpy_dtod: {e:?}")))?;
            return Ok(());
        }
        // Общий путь (используется `Tensor::contiguous()` на permuted/narrowed
        // тензорах): strided gather из src → contiguous write в dst. dst обязан
        // быть contiguous. Реализован через identity-affine unary kernel
        // (x*1+0), который читает по strides и пишет линейно.
        if !dst_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let ctx = src
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda copy: src non-cuda"))?
            .device()
            .clone();
        let kernels = crate::kernels::elementwise::ElementwiseKernels::for_context(&ctx)?;
        crate::kernels::elementwise::run_unary(&kernels, UnaryOp::Affine(1.0, 0.0), src, dst)
    }

    fn cast(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let ctx = src
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda cast: src non-cuda"))?
            .device()
            .clone();
        let kernels = crate::kernels::elementwise::ElementwiseKernels::for_context(&ctx)?;
        crate::kernels::elementwise::run_cast(&kernels, src, dst)
    }

    fn unary(
        &self,
        op: UnaryOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let ctx = src
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda unary: src non-cuda"))?
            .device()
            .clone();
        let kernels = crate::kernels::elementwise::ElementwiseKernels::for_context(&ctx)?;
        crate::kernels::elementwise::run_unary(&kernels, op, src, dst)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let ctx =
            a.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("cuda binary: a non-cuda"))?
                .device()
                .clone();
        let kernels = crate::kernels::elementwise::ElementwiseKernels::for_context(&ctx)?;
        crate::kernels::elementwise::run_binary(&kernels, op, a, b, dst)
    }

    fn ternary_fused(
        &self,
        kind: u8,
        x: (&Storage, &Layout),
        b: (&Storage, &Layout),
        c: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        use crate::kernels::elementwise::TernaryFusedKind;
        let ctx =
            x.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("ternary fused: x non-cuda"))?
                .device()
                .clone();
        let kernels = crate::kernels::elementwise::ElementwiseKernels::for_context(&ctx)?;
        let k = match kind {
            0 => TernaryFusedKind::FmaFlat,
            1 => TernaryFusedKind::FmaRowb,
            2 => TernaryFusedKind::ModRowb,
            3 => TernaryFusedKind::ModFlat,
            _ => return Err(SynaptixError::Unsupported("ternary fused: kind")),
        };
        crate::kernels::elementwise::run_ternary_fused(&kernels, k, x, b, c, dst)
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
        let (dst_st, _dst_lo) = dst;
        if !a_lo.is_contiguous() || !b_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let dtype = a_lo.dtype();
        let a_dims = a_lo.dims();
        let b_dims = b_lo.dims();
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let n = b_dims[b_dims.len() - 1];
        let batch = a_lo.numel() / (m * k);
        let batch_b = b_lo.numel() / (k * n);
        let b_broadcast = batch_b == 1 && batch > 1;

        let a_buf = a_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("matmul: a non-cuda"))?;
        let b_buf = b_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("matmul: b non-cuda"))?;
        let dst_buf = dst_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("matmul: dst non-cuda"))?;

        // best_cu NN GEMM (float-acc WMMA): F16/BF16, любые M/N/K (K-tail + N-pad),
        // batched + broadcast-B. F32 → отдельная SIMT-ветка ниже.
        if matches!(dtype, DType::F16 | DType::BF16) {
            let ord = a_buf.ordinal();
            let ctx = a_buf.device().clone();
            let stream = synaptix_core::device::cuda::default_stream(ord)?;
            let kernels = crate::best_cu::gemm::gemm_f16::GemmF16Kernels::for_context(&ctx)?;
            return crate::best_cu::gemm::gemm_f16::gemm_nn_u8(
                &kernels,
                dtype,
                &stream,
                a_buf.slice(),
                b_buf.slice(),
                dst_buf.slice_mut(),
                m as u32,
                n as u32,
                k as u32,
                batch as u32,
                b_broadcast,
            );
        }
        // best_cu F32 NN GEMM (истинный f32 SIMT, любые M/N/K, batched + broadcast).
        if dtype == DType::F32 {
            let ord = a_buf.ordinal();
            let ctx = a_buf.device().clone();
            let stream = synaptix_core::device::cuda::default_stream(ord)?;
            let kernels = crate::best_cu::gemm::gemm_f32::GemmF32Kernels::for_context(&ctx)?;
            return crate::best_cu::gemm::gemm_f32::gemm_f32_nn_u8(
                &kernels,
                &stream,
                a_buf.slice(),
                b_buf.slice(),
                dst_buf.slice_mut(),
                m as u32,
                n as u32,
                k as u32,
                batch as u32,
                b_broadcast,
            );
        }

        // Dense matmul полностью покрыт best_cu выше (F16/BF16/F32, любые формы,
        // batched + broadcast). Остаток — неподдержанный dtype.
        let _ = (a_buf, b_buf, dst_buf, b_broadcast);
        Err(SynaptixError::Unsupported("dense matmul: dtype не поддержан (best_cu = F16/BF16/F32)"))
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
        _stream: &Stream,
    ) -> Result<()> {
        let ctx = input
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("dwconv1d: input non-cuda"))?
            .device()
            .clone();
        let kernels = crate::kernels::dwconv1d::Dwconv1dKernels::for_context(&ctx)?;
        crate::kernels::dwconv1d::run_dwconv1d(&kernels, input, weight, bias, out, stride, padding, transpose)
    }

    fn linear_quant(
        &self,
        x: (&Storage, &Layout),
        w: &QuantWeight,
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (out_st, _out_lo) = out;
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_quant: x non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let k = w.k();
        if k == 0 {
            return Err(SynaptixError::Unsupported("linear_quant: K=0"));
        }
        let m = (x_lo.numel() / k) as u32;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("linear_quant: out non-cuda"))?;

        let bf16 = x_lo.dtype() == DType::BF16;
        match w.dtype() {
            DType::NVFP4 => {
                if !bf16 && nvfp4_weight_only() {
                    use half::f16;
                    let nn = w.n();
                    if k % 16 != 0 {
                        return Err(SynaptixError::Cuda(format!(
                            "linear_quant NVFP4 WO: K={k} должно быть кратно 16"
                        )));
                    }
                    let qk = crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?;
                    let bk =
                        crate::best_cu::gemm::gemm_bf16::BestGemmBf16Kernels::for_context(&ctx)?;
                    let w_packed_arc = w.packed_arc().ok_or_else(|| {
                        SynaptixError::Cuda("linear_quant NVFP4 WO: packed W освобождён".into())
                    })?;
                    let w_packed = w_packed_arc
                        .as_cuda()
                        .ok_or(SynaptixError::Unsupported(
                            "linear_quant NVFP4 WO: packed W non-cuda",
                        ))?
                        .slice();
                    let w_scales = w
                        .scales()
                        .as_cuda()
                        .ok_or(SynaptixError::Unsupported(
                            "linear_quant NVFP4 WO: scales W non-cuda",
                        ))?
                        .slice();
                    let nk = nn * k;
                    let mut w_f16_u8 = stream.alloc_zeros::<u8>(nk * 2).map_err(|e| {
                        SynaptixError::Cuda(format!("linear_quant NVFP4 WO: alloc W f16: {e:?}"))
                    })?;
                    {
                        let mut w_view = unsafe { w_f16_u8.transmute_mut::<f16>(nk) }
                            .ok_or_else(|| {
                                SynaptixError::Cuda(
                                    "linear_quant NVFP4 WO: transmute W→f16".into(),
                                )
                            })?;
                        crate::elementwise::quant::nvfp4_dequant_f16(
                            &qk,
                            &stream,
                            w_packed,
                            w_scales,
                            &mut w_view,
                            nn as u32,
                            k as u32,
                        )?;
                    }
                    crate::best_cu::gemm::gemm_bf16::best_gemm_f16tn_linear_u8(
                        &bk,
                        &stream,
                        &w_f16_u8,
                        x_buf.slice(),
                        out_buf.slice_mut(),
                        nn as u32,
                        k as u32,
                        m,
                        None,
                        None,
                    )?;
                    return Ok(());
                }
                let (gemm_k, gemv_k, quant_k) = if bf16 {
                    (
                        crate::best_cu::gemm::gemm_nvfp4::Nvfp4MmaGemmShufKernels::for_context_bf16(&ctx)?,
                        crate::best_cu::gemv::gemv_nvfp4::Nvfp4MmaGemvShufKernels::for_context_bf16(&ctx)?,
                        crate::elementwise::quant::Nvfp4QuantKernels::for_context_bf16(&ctx)?,
                    )
                } else {
                    (
                        crate::best_cu::gemm::gemm_nvfp4::Nvfp4MmaGemmShufKernels::for_context(&ctx)?,
                        crate::best_cu::gemv::gemv_nvfp4::Nvfp4MmaGemvShufKernels::for_context(&ctx)?,
                        crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?,
                    )
                };
                crate::gemm::dispatch::nvfp4_linear_f16(
                    &gemm_k,
                    &gemv_k,
                    &quant_k,
                    &ctx,
                    &stream,
                    ord,
                    x_buf.slice(),
                    out_buf.slice_mut(),
                    w,
                    m,
                    bf16,
                    None,
                )?;
                Ok(())
            }
            // MXFP8 (Blackwell-нативный block-scale FP8) — заменил per-tensor FP8.
            // W: e4m3 [N,K] (natural) + E8M0 natural scales [N,K/32].
            //   decode (M=1): квант x natural → gemv_mxfp8.
            //   prefill (M>1): деквант W→f16 → f16 TN-linear (out = x @ Wᵀ).
            // TODO(perf): для prefill подключить tiled gemm_mxfp8 (TMA+WS, бьёт large-M);
            // требует permuted (bm/bk) scales хранить рядом с natural.
            DType::MXFP8 => {
                use half::f16;
                if bf16 {
                    return Err(SynaptixError::Unsupported(
                        "linear_quant MXFP8: активация BF16 не поддержана",
                    ));
                }
                let nn = w.n();
                if k % 32 != 0 {
                    return Err(SynaptixError::Cuda(format!(
                        "linear_quant MXFP8: K={k} должно быть кратно 32"
                    )));
                }
                let qk = crate::elementwise::quant::Mxfp8QuantKernels::for_context(&ctx)?;
                let w_packed_arc = w.packed_arc().ok_or_else(|| {
                    SynaptixError::Cuda("linear_quant MXFP8: packed W освобождён".into())
                })?;
                let w_packed = w_packed_arc
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported(
                        "linear_quant MXFP8: packed W non-cuda",
                    ))?
                    .slice();
                let w_scales = w
                    .scales()
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported(
                        "linear_quant MXFP8: scales W non-cuda",
                    ))?
                    .slice();

                let mk = (m as usize) * k;
                let mn = (m as usize) * nn;
                if m == 1 {
                    let gv =
                        crate::best_cu::gemv::gemv_mxfp8::GemvMxfp8Kernels::for_context(&ctx)?;
                    let x_view = unsafe { x_buf.slice().transmute::<f16>(mk) }.ok_or_else(|| {
                        SynaptixError::Cuda("linear_quant MXFP8: transmute x→f16".into())
                    })?;
                    let mut x_fp8 = stream.alloc_zeros::<u8>(mk).map_err(|e| {
                        SynaptixError::Cuda(format!("linear_quant MXFP8: alloc x_fp8: {e:?}"))
                    })?;
                    let mut x_sc = stream.alloc_zeros::<u8>(mk / 32).map_err(|e| {
                        SynaptixError::Cuda(format!("linear_quant MXFP8: alloc x_sc: {e:?}"))
                    })?;
                    crate::elementwise::quant::mxfp8_quant_natural(
                        &qk, &stream, &x_view, &mut x_fp8, &mut x_sc, 1, k as u32,
                    )?;
                    let mut out_view =
                        unsafe { out_buf.slice_mut().transmute_mut::<f16>(mn) }.ok_or_else(|| {
                            SynaptixError::Cuda("linear_quant MXFP8: transmute out→f16".into())
                        })?;
                    crate::best_cu::gemv::gemv_mxfp8::gemv_mxfp8(
                        &gv,
                        &stream,
                        w_packed,
                        w_scales,
                        &x_fp8,
                        &x_sc,
                        &mut out_view,
                        nn as u32,
                        k as u32,
                    )?;
                } else {
                    // prefill: нативное MXFP8×MXFP8 v1-ядро (cp.async, порт gau-nernst sm120,
                    // cos=0.999999 на outlier, needle связно). Обоих операнда MXFP8 — быстрее
                    // dequant→BF16 (×1.3-1.95). Возвращает false, если форма не 128-кратная
                    // (N%128 / K%128) → тогда фолбэк dequant→BF16 ниже.
                    let handled = crate::gemm::dispatch::mxfp8_linear_tiled(
                        &qk,
                        &ctx,
                        &stream,
                        x_buf.slice(),
                        out_buf.slice_mut(),
                        w,
                        m,
                    )?;
                    if !handled {
                        // Голова N (кратная 128) — тем же tiled-ядром, хвост —
                        // деквант полосами. Прежний путь дековантовал W целиком
                        // (N*K*2 = 2.5 ГиБ на lm_head 202048×6656) и на полной
                        // карте падал в OOM на каждом шаге спекуляции.
                        crate::gemm::dispatch::mxfp8_linear_dequant_fallback(
                            &qk,
                            &ctx,
                            &stream,
                            x_buf.slice(),
                            out_buf.slice_mut(),
                            w,
                            m,
                        )?;
                    }
                }
                Ok(())
            }
            _ => Err(SynaptixError::Unsupported("linear_quant: dtype веса")),
        }
    }

    fn quantize_nvfp4(
        &self,
        w: (&Storage, &Layout),
        n: usize,
        k: usize,
        _stream: &Stream,
    ) -> Result<(Storage, Storage)> {
        use half::f16;
        let (w_st, w_lo) = w;
        if !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if k % 64 != 0 || n % 16 != 0 {
            return Err(SynaptixError::Unsupported(
                "quantize_nvfp4: требуется N%16==0 и K%64==0",
            ));
        }
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("quantize_nvfp4: w non-cuda"))?;
        let ord = w_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let quant_k = crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?;

        let nk = n * k;
        let w_view = unsafe { w_buf.slice().transmute::<f16>(nk) }
            .ok_or_else(|| SynaptixError::Cuda("quantize_nvfp4: transmute w→f16".into()))?;

        let packed_bytes = nk / 2;
        let scales_bytes = crate::elementwise::quant::nvfp4_scale_buffer_size(n, k);
        let mut packed = stream
            .alloc_zeros::<u8>(packed_bytes)
            .map_err(|e| SynaptixError::Cuda(format!("quantize_nvfp4: alloc packed: {e:?}")))?;
        let mut scales = stream
            .alloc_zeros::<u8>(scales_bytes)
            .map_err(|e| SynaptixError::Cuda(format!("quantize_nvfp4: alloc scales: {e:?}")))?;
        crate::elementwise::quant::quantize_f16_to_nvfp4_view(
            &quant_k,
            &stream,
            &w_view,
            &mut packed,
            &mut scales,
            n as u32,
            k as u32,
        )?;

        let packed_st = Storage::Cuda(CudaBuf::new(ctx.clone(), stream.clone(), packed, ord));
        let scales_st = Storage::Cuda(CudaBuf::new(ctx, stream, scales, ord));
        Ok((packed_st, scales_st))
    }

    fn quantize_mxfp8(
        &self,
        w: (&Storage, &Layout),
        n: usize,
        k: usize,
        _stream: &Stream,
    ) -> Result<(Storage, Storage)> {
        use half::f16;
        let (w_st, w_lo) = w;
        if !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if k % 32 != 0 {
            return Err(SynaptixError::Unsupported(
                "quantize_mxfp8: требуется K%32==0",
            ));
        }
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("quantize_mxfp8: w non-cuda"))?;
        let ord = w_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let quant_k = crate::elementwise::quant::Mxfp8QuantKernels::for_context(&ctx)?;

        let nk = n * k;
        let w_view = unsafe { w_buf.slice().transmute::<f16>(nk) }
            .ok_or_else(|| SynaptixError::Cuda("quantize_mxfp8: transmute w→f16".into()))?;

        let mut packed = stream
            .alloc_zeros::<u8>(nk)
            .map_err(|e| SynaptixError::Cuda(format!("quantize_mxfp8: alloc packed: {e:?}")))?;
        let mut scales = stream
            .alloc_zeros::<u8>(n * (k / 32))
            .map_err(|e| SynaptixError::Cuda(format!("quantize_mxfp8: alloc scales: {e:?}")))?;
        crate::elementwise::quant::mxfp8_quant_natural(
            &quant_k,
            &stream,
            &w_view,
            &mut packed,
            &mut scales,
            n as u32,
            k as u32,
        )?;

        let packed_st = Storage::Cuda(CudaBuf::new(ctx.clone(), stream.clone(), packed, ord));
        let scales_st = Storage::Cuda(CudaBuf::new(ctx, stream, scales, ord));
        Ok((packed_st, scales_st))
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
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda linear: dtype mismatch x/w",
            ));
        }
        // x: [.., K] → M = numel/K; w: [N, K]; out: [.., N].
        let k = *x_lo
            .dims()
            .last()
            .ok_or(SynaptixError::Unsupported("cuda linear: x scalar"))?;
        let m = x_lo.numel() / k.max(1);
        if w_lo.dims().len() != 2 || w_lo.dims()[1] != k {
            return Err(SynaptixError::Unsupported("cuda linear: weight shape"));
        }
        if !x_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let n = w_lo.dims()[0];

        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda linear: x non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda linear: w non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda linear: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;

        if m == 1 {
            // Decode (M=1) → warp-reduction GEMV (учитывает byte-offset x).
            let kernels = crate::best_cu::gemv::mma_gemv::MmaGemvKernels::for_context(&ctx)?;
            return crate::best_cu::gemv::mma_gemv::gemv_linear_u8(
                &kernels,
                &stream,
                w_buf.slice(),
                w_lo.byte_offset(),
                x_buf.slice(),
                x_lo.byte_offset(),
                out_buf.slice_mut(),
                0,
                n as u32,
                k as u32,
                dtype,
            );
        }

        // Prefill (M>1) → CUTLASS Linear GEMM (ColumnMajor-B, без транспонирования
        // веса). device_ptr берёт базу буфера → требуем offset 0 (иначе fallback).
        if x_lo.offset() != 0 || w_lo.offset() != 0 {
            return Err(SynaptixError::Unsupported(
                "cuda linear M>1: nonzero offset → matmul path",
            ));
        }
        // best_cu TN GEMM (свой NVRTC, cutlass-рецепт, float-acc, bit-exact). Y=X@Wᵀ
        // совпадает по layout. Покрывает ЛЮБЫЕ M/N/K: part-ядро (bounds-checked M/N) +
        // K-tail zero-pad. Один gemm_bf16_impl<T> → BF16 и F16 (entry per-dtype).
        if dtype == DType::BF16 || dtype == DType::F16 {
            let bk = crate::best_cu::gemm::gemm_bf16::BestGemmBf16Kernels::for_context(&ctx)?;
            let (w_s, x_s, out_s) = (w_buf.slice(), x_buf.slice(), out_buf.slice_mut());
            return if dtype == DType::BF16 {
                crate::best_cu::gemm::gemm_bf16::best_gemm_bf16_linear_u8(
                    &bk, &stream, w_s, x_s, out_s, n as u32, k as u32, m as u32, None, None,
                )
            } else {
                crate::best_cu::gemm::gemm_bf16::best_gemm_f16tn_linear_u8(
                    &bk, &stream, w_s, x_s, out_s, n as u32, k as u32, m as u32, None, None,
                )
            };
        }
        // best_cu покрыл BF16/F16 TN выше. Остаток (F32) →
        // Unsupported → run_linear падает в matmul-fallback (transpose + NN).
        Err(SynaptixError::Unsupported(
            "cuda linear M>1: не BF16/F16 best_cu (F32 → matmul-fallback)",
        ))
    }

    fn linear_epilogue(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        residual: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (w_st, w_lo) = w;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if !matches!(dtype, DType::BF16 | DType::F16) || w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("linear_epilogue: dtype"));
        }
        let k = *x_lo
            .dims()
            .last()
            .ok_or(SynaptixError::Unsupported("linear_epilogue: x scalar"))?;
        let m = x_lo.numel() / k.max(1);
        if m <= 1 || w_lo.dims().len() != 2 || w_lo.dims()[1] != k {
            return Err(SynaptixError::Unsupported("linear_epilogue: shape/M"));
        }
        if !x_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if x_lo.offset() != 0 || w_lo.offset() != 0 {
            return Err(SynaptixError::Unsupported("linear_epilogue: nonzero offset"));
        }
        let n = w_lo.dims()[0];
        let bias_v = match bias {
            Some((b_st, b_lo)) => {
                if b_lo.dtype() != dtype || !b_lo.is_contiguous() || b_lo.numel() != n {
                    return Err(SynaptixError::Unsupported("linear_epilogue: bias shape"));
                }
                Some(
                    b_st.as_cuda()
                        .ok_or(SynaptixError::Unsupported("linear_epilogue: bias non-cuda"))?,
                )
            }
            None => None,
        };
        let res_v = match residual {
            Some((r_st, r_lo)) => {
                if r_lo.dtype() != dtype || !r_lo.is_contiguous() || r_lo.numel() != m * n
                    || r_lo.offset() != 0
                {
                    return Err(SynaptixError::Unsupported("linear_epilogue: residual shape"));
                }
                Some(
                    r_st.as_cuda()
                        .ok_or(SynaptixError::Unsupported("linear_epilogue: residual non-cuda"))?,
                )
            }
            None => None,
        };
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_epilogue: x non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_epilogue: w non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("linear_epilogue: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let bk = crate::best_cu::gemm::gemm_bf16::BestGemmBf16Kernels::for_context(&ctx)?;
        let (w_s, x_s, out_s) = (w_buf.slice(), x_buf.slice(), out_buf.slice_mut());
        let b_s = bias_v.map(|b| b.slice());
        let r_s = res_v.map(|r| r.slice());
        if dtype == DType::BF16 {
            crate::best_cu::gemm::gemm_bf16::best_gemm_bf16_linear_u8(
                &bk, &stream, w_s, x_s, out_s, n as u32, k as u32, m as u32, b_s, r_s,
            )
        } else {
            crate::best_cu::gemm::gemm_bf16::best_gemm_f16tn_linear_u8(
                &bk, &stream, w_s, x_s, out_s, n as u32, k as u32, m as u32, b_s, r_s,
            )
        }
    }

    fn silu_and_mul(
        &self,
        gate: (&Storage, &Layout),
        up: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        use crate::fused::silu_and_mul::{
            silu_and_mul_bf16, silu_and_mul_f16, silu_and_mul_f32, SiluAndMulKernels,
        };
        let (g_st, g_lo) = gate;
        let (u_st, u_lo) = up;
        let (o_st, o_lo) = out;
        let dtype = g_lo.dtype();
        if u_lo.dtype() != dtype || o_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda silu_and_mul: dtype mismatch",
            ));
        }
        if !g_lo.is_contiguous() || !u_lo.is_contiguous() || !o_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if g_lo.numel() != u_lo.numel() || g_lo.numel() != o_lo.numel() {
            return Err(SynaptixError::Unsupported(
                "cuda silu_and_mul: shape mismatch",
            ));
        }
        let total = g_lo.numel();
        if total == 0 {
            return Ok(());
        }
        let g_buf = g_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda silu_and_mul: gate non-cuda",
        ))?;
        let u_buf = u_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda silu_and_mul: up non-cuda"))?;
        let o_buf = o_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda silu_and_mul: out non-cuda",
        ))?;
        let ctx = g_buf.device().clone();
        let ord = g_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = SiluAndMulKernels::for_context(&ctx)?;
        let total_u32 = u32::try_from(total)
            .map_err(|_| SynaptixError::Cuda(format!("silu_and_mul: total {total} > u32::MAX")))?;
        let esz = (dtype.size_in_bits() / 8) as usize;
        let g_byte_off = g_lo.byte_offset();
        let u_byte_off = u_lo.byte_offset();
        let total_bytes = total * esz;
        match dtype {
            DType::F32 => {
                let g_view = unsafe {
                    g_buf
                        .slice()
                        .slice(g_byte_off..g_byte_off + total_bytes)
                        .transmute::<f32>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute gate".into()))?;
                let u_view = unsafe {
                    u_buf
                        .slice()
                        .slice(u_byte_off..u_byte_off + total_bytes)
                        .transmute::<f32>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute up".into()))?;
                let mut o_re = o_buf.slice_mut().slice_mut(..total_bytes);
                let mut o_view = unsafe { o_re.transmute_mut::<f32>(total) }
                    .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute out".into()))?;
                silu_and_mul_f32(&kernels, &stream, &g_view, &u_view, &mut o_view, total_u32)
            }
            DType::F16 => {
                let g_view = unsafe {
                    g_buf
                        .slice()
                        .slice(g_byte_off..g_byte_off + total_bytes)
                        .transmute::<half::f16>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute gate".into()))?;
                let u_view = unsafe {
                    u_buf
                        .slice()
                        .slice(u_byte_off..u_byte_off + total_bytes)
                        .transmute::<half::f16>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute up".into()))?;
                let mut o_re = o_buf.slice_mut().slice_mut(..total_bytes);
                let mut o_view = unsafe { o_re.transmute_mut::<half::f16>(total) }
                    .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute out".into()))?;
                silu_and_mul_f16(&kernels, &stream, &g_view, &u_view, &mut o_view, total_u32)
            }
            DType::BF16 => {
                let g_view = unsafe {
                    g_buf
                        .slice()
                        .slice(g_byte_off..g_byte_off + total_bytes)
                        .transmute::<half::bf16>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute gate".into()))?;
                let u_view = unsafe {
                    u_buf
                        .slice()
                        .slice(u_byte_off..u_byte_off + total_bytes)
                        .transmute::<half::bf16>(total)
                }
                .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute up".into()))?;
                let mut o_re = o_buf.slice_mut().slice_mut(..total_bytes);
                let mut o_view = unsafe { o_re.transmute_mut::<half::bf16>(total) }
                    .ok_or_else(|| SynaptixError::Cuda("silu_and_mul: transmute out".into()))?;
                silu_and_mul_bf16(&kernels, &stream, &g_view, &u_view, &mut o_view, total_u32)
            }
            other => Err(SynaptixError::Unsupported(Box::leak(
                format!("cuda silu_and_mul: dtype {other:?}").into_boxed_str(),
            ))),
        }
    }

    fn nvfp4_quantize_act(
        &self,
        x: (&Storage, &Layout),
        packed_out: (&mut Storage, &Layout),
        scales_out: (&mut Storage, &Layout),
        m: usize,
        k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        if !x_lo.is_contiguous() || x_lo.byte_offset() != 0 {
            return Err(SynaptixError::NonContiguous);
        }
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("nvfp4_quantize_act: x non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let quant_k = crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?;
        let p_buf = packed_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("nvfp4_quantize_act: packed non-cuda"))?;
        let p_slice = p_buf.slice_mut();
        let s_buf = scales_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("nvfp4_quantize_act: scales non-cuda"))?;
        crate::gemm::dispatch::nvfp4_quantize_act(
            &quant_k,
            &stream,
            x_buf.slice(),
            p_slice,
            s_buf.slice_mut(),
            m as u32,
            k as u32,
        )
    }

    fn silu_mul_quant_nvfp4(
        &self,
        x: (&Storage, &Layout),
        packed_out: (&mut Storage, &Layout),
        scales_out: (&mut Storage, &Layout),
        m: usize,
        k: usize,
        inv_pre: f32,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("silu_mul_quant_nvfp4: x non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let quant_k = match x_lo.dtype() {
            DType::F16 => crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?,
            DType::BF16 => crate::elementwise::quant::Nvfp4QuantKernels::for_context_bf16(&ctx)?,
            _ => return Err(SynaptixError::Unsupported("silu_mul_quant_nvfp4: dtype")),
        };
        let p_buf = packed_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("silu_mul_quant_nvfp4: packed non-cuda"))?;
        let p_slice = p_buf.slice_mut();
        let s_buf = scales_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("silu_mul_quant_nvfp4: scales non-cuda"))?;
        crate::elementwise::quant::silu_mul_quantize_nvfp4_u8(
            &quant_k,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            p_slice,
            s_buf.slice_mut(),
            m as u32,
            k as u32,
            inv_pre,
        )
    }

    fn rms_mod_quant_nvfp4(
        &self,
        x: (&Storage, &Layout),
        scale: (&Storage, &Layout),
        shift: (&Storage, &Layout),
        y: (&mut Storage, &Layout),
        packed_out: &mut Storage,
        scales_out: &mut Storage,
        m: usize,
        k: usize,
        eps: f32,
        kind: u8,
        mod_div: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let kind = match kind {
            0 => crate::fused::rms_mod_quant::NormQuantKind::RmsMod,
            1 => crate::fused::rms_mod_quant::NormQuantKind::LnMod,
            2 => crate::fused::rms_mod_quant::NormQuantKind::RmsW { qwen: false },
            3 => crate::fused::rms_mod_quant::NormQuantKind::RmsW { qwen: true },
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "rms_mod_quant: неизвестный kind {other}"
                )))
            }
        };
        let (x_st, x_lo) = x;
        let (s_st, s_lo) = scale;
        let (b_st, b_lo) = shift;
        if !x_lo.is_contiguous() || !s_lo.is_contiguous() || !b_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let dt = x_lo.dtype();
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: x non-cuda"))?;
        let s_buf = s_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: scale non-cuda"))?;
        let b_buf = b_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: shift non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::fused::rms_mod_quant::RmsModQuantKernels::for_context(&ctx)?;
        let x_off = x_lo.byte_offset();
        let s_off = s_lo.byte_offset();
        let b_off = b_lo.byte_offset();
        let (x_sl, s_sl, b_sl) = (x_buf.slice(), s_buf.slice(), b_buf.slice());
        let y_buf = y
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: y non-cuda"))?;
        let y_sl = y_buf.slice_mut();
        let p_buf = packed_out
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: packed non-cuda"))?;
        let p_sl = p_buf.slice_mut();
        let sc_buf = scales_out
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant: scales non-cuda"))?;
        crate::fused::rms_mod_quant::run_rms_mod_quant_u8(
            &kernels,
            &stream,
            x_sl,
            x_off,
            s_sl,
            s_off,
            b_sl,
            b_off,
            y_sl,
            p_sl,
            sc_buf.slice_mut(),
            m as u32,
            k as u32,
            eps,
            dt,
            kind,
            mod_div as u32,
            false,
        )
    }

    fn rms_mod_quant_mxfp8(
        &self,
        x: (&Storage, &Layout),
        scale: (&Storage, &Layout),
        shift: (&Storage, &Layout),
        y: (&mut Storage, &Layout),
        packed_out: &mut Storage,
        scales_out: &mut Storage,
        m: usize,
        k: usize,
        eps: f32,
        kind: u8,
        mod_div: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let kind = match kind {
            0 => crate::fused::rms_mod_quant::NormQuantKind::RmsMod,
            1 => crate::fused::rms_mod_quant::NormQuantKind::LnMod,
            2 => crate::fused::rms_mod_quant::NormQuantKind::RmsW { qwen: false },
            3 => crate::fused::rms_mod_quant::NormQuantKind::RmsW { qwen: true },
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "rms_mod_quant_mxfp8: неизвестный kind {other}"
                )))
            }
        };
        let (x_st, x_lo) = x;
        let (s_st, s_lo) = scale;
        let (b_st, b_lo) = shift;
        if !x_lo.is_contiguous() || !s_lo.is_contiguous() || !b_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let dt = x_lo.dtype();
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: x non-cuda"))?;
        let s_buf = s_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: scale non-cuda"))?;
        let b_buf = b_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: shift non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::fused::rms_mod_quant::RmsModQuantKernels::for_context(&ctx)?;
        let x_off = x_lo.byte_offset();
        let s_off = s_lo.byte_offset();
        let b_off = b_lo.byte_offset();
        let (x_sl, s_sl, b_sl) = (x_buf.slice(), s_buf.slice(), b_buf.slice());
        let y_buf = y
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: y non-cuda"))?;
        let y_sl = y_buf.slice_mut();
        let p_buf = packed_out
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: packed non-cuda"))?;
        let p_sl = p_buf.slice_mut();
        let sc_buf = scales_out
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("rms_mod_quant_mxfp8: scales non-cuda"))?;
        crate::fused::rms_mod_quant::run_rms_mod_quant_u8(
            &kernels,
            &stream,
            x_sl,
            x_off,
            s_sl,
            s_off,
            b_sl,
            b_off,
            y_sl,
            p_sl,
            sc_buf.slice_mut(),
            m as u32,
            k as u32,
            eps,
            dt,
            kind,
            mod_div as u32,
            true,
        )
    }

    fn mxfp8_quantize_act(
        &self,
        x: (&Storage, &Layout),
        packed_out: (&mut Storage, &Layout),
        scales_out: (&mut Storage, &Layout),
        m: usize,
        k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if x_lo.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported("mxfp8_quantize_act: x не F16"));
        }
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("mxfp8_quantize_act: x non-cuda"))?;
        let ord = x_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let qk = crate::elementwise::quant::Mxfp8QuantKernels::for_context(&ctx)?;
        let n_el = m * k;
        let x_off = x_lo.byte_offset();
        let x_view = unsafe {
            x_buf
                .slice()
                .slice(x_off..x_off + n_el * 2)
                .transmute::<half::f16>(n_el)
                .ok_or_else(|| SynaptixError::Cuda("mxfp8_quantize_act: transmute x".into()))?
        };
        let p_buf = packed_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("mxfp8_quantize_act: packed non-cuda"))?;
        let s_buf = scales_out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("mxfp8_quantize_act: scales non-cuda"))?;
        crate::elementwise::quant::mxfp8_quant_natural(
            &qk,
            &stream,
            &x_view,
            p_buf.slice_mut(),
            s_buf.slice_mut(),
            m as u32,
            k as u32,
        )
    }

    fn linear_quant_prequant(
        &self,
        packed_x: &Storage,
        scales_x: &Storage,
        w: &QuantWeight,
        out: (&mut Storage, &Layout),
        m: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let p_buf = packed_x
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_quant_prequant: packed non-cuda"))?;
        let s_buf = scales_x
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_quant_prequant: scales non-cuda"))?;
        let ord = p_buf.ordinal();
        let ctx = synaptix_core::device::cuda::get(ord)?;
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let out_buf = out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("linear_quant_prequant: out non-cuda"))?;
        let out_bf16 = out.1.dtype() == DType::BF16;
        match w.dtype() {
            DType::NVFP4 => {
                let (gemm_k, gemv_k, quant_k) = if out_bf16 {
                    (
                        crate::best_cu::gemm::gemm_nvfp4::Nvfp4MmaGemmShufKernels::for_context_bf16(&ctx)?,
                        crate::best_cu::gemv::gemv_nvfp4::Nvfp4MmaGemvShufKernels::for_context_bf16(&ctx)?,
                        crate::elementwise::quant::Nvfp4QuantKernels::for_context_bf16(&ctx)?,
                    )
                } else {
                    (
                        crate::best_cu::gemm::gemm_nvfp4::Nvfp4MmaGemmShufKernels::for_context(&ctx)?,
                        crate::best_cu::gemv::gemv_nvfp4::Nvfp4MmaGemvShufKernels::for_context(&ctx)?,
                        crate::elementwise::quant::Nvfp4QuantKernels::for_context(&ctx)?,
                    )
                };
                crate::gemm::dispatch::nvfp4_linear_f16(
                    &gemm_k,
                    &gemv_k,
                    &quant_k,
                    &ctx,
                    &stream,
                    ord,
                    p_buf.slice(), // x_u8 не используется при prequant=Some
                    out_buf.slice_mut(),
                    w,
                    m as u32,
                    out_bf16,
                    Some((p_buf.slice(), s_buf.slice())),
                )?;
                Ok(())
            }
            DType::MXFP8 if out_bf16 => {
                Err(SynaptixError::Unsupported("linear_quant_prequant MXFP8: BF16-выход не поддержан"))
            }
            DType::MXFP8 => crate::gemm::dispatch::mxfp8_linear_prequant(
                &ctx,
                &stream,
                p_buf.slice(),
                s_buf.slice(),
                out_buf.slice_mut(),
                w,
                m as u32,
            ),
            _ => Err(SynaptixError::Unsupported("linear_quant_prequant: вес не NVFP4/MXFP8")),
        }
    }

    fn rms_norm(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        eps: f32,
        qwen_gain: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (w_st, w_lo) = w;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda rms_norm: dtype mismatch x/w",
            ));
        }
        let h = *x_lo
            .dims()
            .last()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm: x scalar"))?;
        if w_lo.dims().len() != 1 || w_lo.dims()[0] != h {
            return Err(SynaptixError::Unsupported("cuda rms_norm: weight shape"));
        }
        if !x_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let batch = x_lo.numel() / h.max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm: x non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm: w non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::reduction::rmsnorm::RmsNormKernels::for_context(&ctx)?;
        let variant = if qwen_gain {
            crate::reduction::rmsnorm::RmsVariant::Qwen
        } else {
            crate::reduction::rmsnorm::RmsVariant::Plain
        };
        crate::reduction::rmsnorm::run_rms_norm_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            w_buf.slice(),
            w_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            batch as u32,
            h as u32,
            eps,
            variant,
            dtype,
        )
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
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (r_st, r_lo) = residual;
        let (w_st, w_lo) = w;
        let (h_st, _h_lo) = hidden_out;
        let (y_st, _y_lo) = y;
        let dtype = x_lo.dtype();
        if r_lo.dtype() != dtype || w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda rms_norm_residual: dtype mismatch",
            ));
        }
        let hd = *x_lo
            .dims()
            .last()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: x scalar"))?;
        if w_lo.dims().len() != 1 || w_lo.dims()[0] != hd {
            return Err(SynaptixError::Unsupported("cuda rms_norm_residual: weight shape"));
        }
        if !x_lo.is_contiguous()
            || !r_lo.is_contiguous()
            || !w_lo.is_contiguous()
            || x_lo.byte_offset() != 0
            || r_lo.byte_offset() != 0
            || w_lo.byte_offset() != 0
        {
            return Err(SynaptixError::NonContiguous);
        }
        let batch = x_lo.numel() / hd.max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: x non-cuda"))?;
        let r_buf = r_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: r non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: w non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::fused::rmsnorm_residual::RmsNormResidualKernels::for_context(&ctx)?;
        let variant = if qwen_gain {
            crate::reduction::rmsnorm::RmsVariant::Qwen
        } else {
            crate::reduction::rmsnorm::RmsVariant::Plain
        };
        let h_buf = h_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: hidden_out non-cuda"))?;
        let h_slice = h_buf.slice_mut();
        // SAFETY: hidden_out и y — разные тензоры (отдельные аллокации).
        let y_buf = y_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda rms_norm_residual: y non-cuda"))?;
        crate::fused::rmsnorm_residual::run_split_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            r_buf.slice(),
            w_buf.slice(),
            h_slice,
            y_buf.slice_mut(),
            batch as u32,
            hd as u32,
            eps,
            variant,
            dtype,
        )
    }

    fn layer_norm(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        eps: f32,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (w_st, w_lo) = w;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda layer_norm: dtype mismatch x/w",
            ));
        }
        let h = *x_lo
            .dims()
            .last()
            .ok_or(SynaptixError::Unsupported("cuda layer_norm: x scalar"))?;
        if w_lo.dims().len() != 1 || w_lo.dims()[0] != h {
            return Err(SynaptixError::Unsupported("cuda layer_norm: weight shape"));
        }
        if !x_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if let Some((_, b_lo)) = bias {
            if b_lo.dtype() != dtype || b_lo.dims().len() != 1 || b_lo.dims()[0] != h {
                return Err(SynaptixError::Unsupported(
                    "cuda layer_norm: bias shape/dtype",
                ));
            }
            if !b_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        let batch = x_lo.numel() / h.max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda layer_norm: x non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda layer_norm: w non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda layer_norm: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let beta = match bias {
            Some((b_st, b_lo)) => {
                let b_buf = b_st
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported("cuda layer_norm: bias non-cuda"))?;
                Some((b_buf.slice(), b_lo.byte_offset()))
            }
            None => None,
        };
        let kernels = crate::reduction::layernorm::LayerNormKernels::for_context(&ctx)?;
        crate::reduction::layernorm::run_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            w_buf.slice(),
            w_lo.byte_offset(),
            beta.as_ref().map(|(s, o)| (*s, *o)),
            out_buf.slice_mut(),
            0,
            batch as u32,
            h as u32,
            eps,
            dtype,
        )
    }

    fn rope_split(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (cos_st, cos_lo) = cos;
        let (sin_st, sin_lo) = sin;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if cos_lo.dtype() != DType::F32 || sin_lo.dtype() != DType::F32 {
            return Err(SynaptixError::Unsupported(
                "cuda rope_split: cos/sin must be F32",
            ));
        }
        let xr = x_lo.dims().len();
        if xr < 2 {
            return Err(SynaptixError::Unsupported("cuda rope_split: x rank < 2"));
        }
        let d = x_lo.dims()[xr - 1];
        let s_len = x_lo.dims()[xr - 2];
        if cos_lo.dims().len() != 2 || cos_lo.dims()[0] != s_len || cos_lo.dims()[1] * 2 != d {
            return Err(SynaptixError::Unsupported("cuda rope_split: cos shape"));
        }
        if sin_lo.dims() != cos_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda rope_split: sin shape"));
        }
        if !x_lo.is_contiguous() || !cos_lo.is_contiguous() || !sin_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let rows = x_lo.numel() / d.max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split: x non-cuda"))?;
        let cos_buf = cos_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split: cos non-cuda"))?;
        let sin_buf = sin_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split: sin non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda rope_split: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::rope::RopeKernels::for_context(&ctx)?;
        crate::elementwise::rope::run_rope_split_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            cos_buf.slice(),
            cos_lo.byte_offset(),
            sin_buf.slice(),
            sin_lo.byte_offset(),
            rows as u32,
            s_len as u32,
            d as u32,
            dtype,
        )
    }

    fn rope_split_partial(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        rot_dim: usize,
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (cos_st, cos_lo) = cos;
        let (sin_st, sin_lo) = sin;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if cos_lo.dtype() != DType::F32 || sin_lo.dtype() != DType::F32 {
            return Err(SynaptixError::Unsupported(
                "cuda rope_split_partial: cos/sin must be F32",
            ));
        }
        let xr = x_lo.dims().len();
        if xr < 2 {
            return Err(SynaptixError::Unsupported("cuda rope_split_partial: x rank < 2"));
        }
        let d = x_lo.dims()[xr - 1];
        let mut s_len = x_lo.dims()[xr - 2];
        let mut pos_div = 1usize;
        if rot_dim == 0 || rot_dim % 2 != 0 || rot_dim > d {
            return Err(SynaptixError::Unsupported("cuda rope_split_partial: rot_dim"));
        }
        if cos_lo.dims().len() != 2 || cos_lo.dims()[1] * 2 != rot_dim {
            return Err(SynaptixError::Unsupported("cuda rope_split_partial: cos shape"));
        }
        if cos_lo.dims()[0] != s_len {
            if xr >= 3 && cos_lo.dims()[0] == x_lo.dims()[xr - 3] {
                pos_div = s_len;
                s_len = x_lo.dims()[xr - 3];
            } else {
                return Err(SynaptixError::Unsupported("cuda rope_split_partial: cos shape"));
            }
        }
        if sin_lo.dims() != cos_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda rope_split_partial: sin shape"));
        }
        if !x_lo.is_contiguous() || !cos_lo.is_contiguous() || !sin_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let rows = x_lo.numel() / d.max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split_partial: x non-cuda"))?;
        let cos_buf = cos_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split_partial: cos non-cuda"))?;
        let sin_buf = sin_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda rope_split_partial: sin non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda rope_split_partial: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::rope::RopeKernels::for_context(&ctx)?;
        crate::elementwise::rope::run_rope_split_partial_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            cos_buf.slice(),
            cos_lo.byte_offset(),
            sin_buf.slice(),
            sin_lo.byte_offset(),
            rows as u32,
            s_len as u32,
            d as u32,
            rot_dim as u32,
            pos_div as u32,
            dtype,
        )
    }

    fn rope_interleaved(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        h: usize,
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (cos_st, cos_lo) = cos;
        let (sin_st, sin_lo) = sin;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if cos_lo.dtype() != DType::F32 || sin_lo.dtype() != DType::F32 {
            return Err(SynaptixError::Unsupported("cuda rope_interleaved: cos/sin must be F32"));
        }
        // x: [B,S,H,D]; cos/sin: ПОЛНАЯ таблица [S,D] (любой layout с numel=S*D).
        let xr = x_lo.dims().len();
        if xr != 4 {
            return Err(SynaptixError::Unsupported("cuda rope_interleaved: x rank != 4"));
        }
        let d = x_lo.dims()[3];
        let s_len = x_lo.dims()[1];
        if x_lo.dims()[2] != h {
            return Err(SynaptixError::Unsupported("cuda rope_interleaved: h mismatch"));
        }
        if cos_lo.numel() != s_len * d || sin_lo.numel() != s_len * d {
            return Err(SynaptixError::Unsupported("cuda rope_interleaved: cos/sin numel"));
        }
        if !x_lo.is_contiguous() || !cos_lo.is_contiguous() || !sin_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let rows = x_lo.numel() / d.max(1);
        let x_buf = x_st.as_cuda().ok_or(SynaptixError::Unsupported("cuda rope_interleaved: x non-cuda"))?;
        let cos_buf = cos_st.as_cuda().ok_or(SynaptixError::Unsupported("cuda rope_interleaved: cos non-cuda"))?;
        let sin_buf = sin_st.as_cuda().ok_or(SynaptixError::Unsupported("cuda rope_interleaved: sin non-cuda"))?;
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported("cuda rope_interleaved: out non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::rope::RopeKernels::for_context(&ctx)?;
        crate::elementwise::rope::run_rope_interleaved_u8(
            &kernels, &stream,
            x_buf.slice(), x_lo.byte_offset(),
            out_buf.slice_mut(), 0,
            cos_buf.slice(), cos_lo.byte_offset(),
            sin_buf.slice(), sin_lo.byte_offset(),
            rows as u32, h as u32, s_len as u32, d as u32, dtype,
        )
    }

    fn flash_attention(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda flash: dtype mismatch q/k/v",
            ));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda flash: expect rank-4 [B,H,T,D]",
            ));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        let t_kv = k_lo.dims()[2];
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda flash: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 || d == 0 || d > 1024 {
            return Err(SynaptixError::Unsupported("cuda flash: GQA/D constraints"));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        // K/V: contiguous ИЛИ strided view preallocated буфера [B,nkv,max_seq,hd].
        // Выводим physical t_stride (число элементов на один T-row) = stride[1]/hd.
        // Требуем row-major инкремент по (T,hd): stride[3]==1, stride[2]==hd,
        // регулярный per-batch шаг stride[0]==nkv*stride[1]. Активная длина = t_kv.
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported(
                "cuda flash: k/v t_stride mismatch",
            ));
        }
        let t_stride = t_stride_k;

        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;

        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda flash: out non-cuda"))?;

        // Выбор attention-backend.
        // Prefill (Tq>1) → flash_splitq (FA-2-схема: split-Q BM=64, softmax в
        // регистрах, mma.sync m16n8k16), hd∈{64,128,256}, F16/BF16.
        // Decode (Tq=1) → flash-decode (split-K по KV; Q-тайл prefill-ядра
        // тратил бы 63/64 строк). SYN_ATTN=fa2 — скалярный FA-2 (диагностика).
        use synaptix_core::backend::attn::{mode, AttnMode};
        enum Choice {
            Fa4,
            Fa2,
            Decode,
        }
        let fa4_ok =
            (dtype == DType::BF16 || dtype == DType::F16) && (d == 64 || d == 128 || d == 256);
        let fa2_ok = dtype == DType::BF16 && (d == 128 || d == 256);
        let choice = match mode() {
            AttnMode::Auto => {
                if t_q > 1 && fa4_ok {
                    Choice::Fa4
                } else {
                    Choice::Decode
                }
            }
            AttnMode::Fa4 => {
                if fa4_ok {
                    Choice::Fa4
                } else {
                    Choice::Decode
                }
            }
            AttnMode::Fa2 => {
                if fa2_ok {
                    Choice::Fa2
                } else {
                    Choice::Decode
                }
            }
            AttnMode::FlashDecode => Choice::Decode,
        };
        if let Choice::Fa4 = choice {
            // FA-5 (split-Q, softmax в регистрах, FA-2-схема) — единственное
            // tensor-core prefill-ядро (flash_v4 удалён: серийный softmax был
            // полуфабрикатом). Tq любой (BM=64-тайл с ceil-гридом).
            let kernels = crate::attention::flash_splitq::FlashSplitQKernels::for_context(&ctx)?;
            return crate::attention::flash_splitq::flash_splitq_u8(
                &kernels,
                &stream,
                q_buf.slice(),
                q_lo.byte_offset(),
                k_buf.slice(),
                k_lo.byte_offset(),
                v_buf.slice(),
                v_lo.byte_offset(),
                out_buf.slice_mut(),
                0,
                b as u32,
                nh as u32,
                nkv as u32,
                t_q as u32,
                t_kv as u32,
                d as u32,
                scale,
                causal,
                t_stride,
                dtype,
                false,
            );
        } else if let Choice::Fa2 = choice {
            let kernels = crate::attention::flash_bf16::FlashAttnBf16Kernels::for_context(&ctx)?;
            let n_rep = (nh / nkv) as u32;
            let q_pos_base = (t_kv - t_q) as u32;
            // combined dispatcher: single-row при Tq=1, tiled(m64) при Tq>1.
            kernels.flash_attn2_fwd_u8(
                &stream,
                q_buf.slice(),
                q_lo.byte_offset(),
                k_buf.slice(),
                k_lo.byte_offset(),
                v_buf.slice(),
                v_lo.byte_offset(),
                out_buf.slice_mut(),
                0,
                scale,
                b as u32,
                nh as u32,
                nkv as u32,
                t_q as u32,
                t_kv as u32,
                d as u32,
                n_rep,
                q_pos_base,
                if causal { 1 } else { 0 },
                t_stride,
            )
        } else {
            // split_k: на decode (Tq=1) — max(occupancy, длина-KV). Для длинного KV
            // больше сегментов (до SPLIT_K_MAX=32) → параллельный скан. Prefill: 1.
            let rows = (b * nh * t_q) as u32;
            let split_k = if t_q == 1 {
                let occ = (128 / rows.max(1)).max(1);
                let long = (t_kv as u32).div_ceil(2048);
                occ.max(long).clamp(1, 32)
            } else {
                1
            };
            let kernels = crate::attention::flash_decode::FlashDecodeKernels::for_context(&ctx)?;
            crate::attention::flash_decode::flash_decode_u8(
                &kernels,
                &stream,
                q_buf.slice(),
                q_lo.byte_offset(),
                k_buf.slice(),
                k_lo.byte_offset(),
                v_buf.slice(),
                v_lo.byte_offset(),
                out_buf.slice_mut(),
                0,
                b as u32,
                nh as u32,
                nkv as u32,
                t_q as u32,
                t_kv as u32,
                d as u32,
                scale,
                causal,
                split_k,
                t_stride,
                dtype,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn flash_attention_window(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        window: i32,
        causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if !matches!(dtype, DType::BF16 | DType::F16) || k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("flash_window: f16/bf16 only"));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("flash_window: rank-4"));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        let t_kv = k_lo.dims()[2];
        if d != 128 {
            return Err(SynaptixError::Unsupported("flash_window: HD=128 only"));
        }
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("flash_window: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 {
            return Err(SynaptixError::Unsupported("flash_window: GQA"));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported("flash_window: k/v t_stride"));
        }
        let t_stride = t_stride_k;
        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("flash_window: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("flash_window: out non-cuda"))?;
        let kernels = crate::attention::flash_splitq::FlashSplitQKernels::for_context(&ctx)?;
        crate::attention::flash_splitq::flash_splitq_window_u8(
            &kernels,
            &stream,
            dtype,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            t_kv as u32,
            scale,
            causal,
            t_stride,
            window,
        )
    }

    fn flash_attention_bshd(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16) || k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("flash_bshd: dtype"));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("flash_bshd: rank"));
        }
        // [B, S, H, D].
        let (b, s_q, h, d) = (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2], q_lo.dims()[3]);
        let (s_kv, hkv) = (k_lo.dims()[1], k_lo.dims()[2]);
        if !(d == 64 || d == 128) || k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("flash_bshd: shape/HD"));
        }
        if hkv == 0 || h % hkv != 0 {
            return Err(SynaptixError::Unsupported("flash_bshd: GQA"));
        }
        if !q_lo.is_contiguous() || !k_lo.is_contiguous() || !v_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if q_lo.offset() != 0 || k_lo.offset() != 0 || v_lo.offset() != 0 {
            return Err(SynaptixError::Unsupported("flash_bshd: nonzero offset"));
        }
        let q_buf = q_st.as_cuda().ok_or(SynaptixError::Unsupported("flash_bshd: q non-cuda"))?;
        let k_buf = k_st.as_cuda().ok_or(SynaptixError::Unsupported("flash_bshd: k non-cuda"))?;
        let v_buf = v_st.as_cuda().ok_or(SynaptixError::Unsupported("flash_bshd: v non-cuda"))?;
        let ctx = q_buf.device().clone();
        let ord = q_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported("flash_bshd: out non-cuda"))?;
        let kernels = crate::attention::flash_splitq::FlashSplitQKernels::for_context(&ctx)?;
        crate::attention::flash_splitq::flash_splitq_u8(
            &kernels,
            &stream,
            q_buf.slice(),
            0,
            k_buf.slice(),
            0,
            v_buf.slice(),
            0,
            out_buf.slice_mut(),
            0,
            b as u32,
            h as u32,
            hkv as u32,
            s_q as u32,
            s_kv as u32,
            d as u32,
            scale,
            causal,
            0,
            dtype,
            true,
        )
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
        let dtype = dst_lo.dtype();
        if src_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append: dtype mismatch src/dst",
            ));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append: expect rank-4 [B,nkv,T,hd]",
            ));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, nkv, max_seq, hd) = (
            dst_lo.dims()[0],
            dst_lo.dims()[1],
            dst_lo.dims()[2],
            dst_lo.dims()[3],
        );
        let (sb, snkv, t_new, shd) = (
            src_lo.dims()[0],
            src_lo.dims()[1],
            src_lo.dims()[2],
            src_lo.dims()[3],
        );
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append: src/dst shape mismatch",
            ));
        }
        if seq_pos + t_new > max_seq {
            return Err(SynaptixError::Other(format!(
                "cuda kv_append: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
            )));
        }
        let cuda = dst_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_append: dst non-cuda"))?;
        let ctx = cuda.device().clone();
        let ord = cuda.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::kv_append::KvAppendKernels::for_context(&ctx)?;
        let src_buf = src_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_append: src non-cuda"))?;
        let dst_buf = dst_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda kv_append: dst non-cuda"))?;
        crate::elementwise::kv_append::append_u8(
            &kernels,
            &stream,
            src_buf.slice(),
            src_lo.byte_offset(),
            dst_buf.slice_mut(),
            dst_lo.byte_offset(),
            b as u32,
            nkv as u32,
            t_new as u32,
            hd as u32,
            max_seq as u32,
            seq_pos as u32,
            dtype,
        )
    }

    fn rope_apply_dev(
        &self,
        x: (&Storage, &Layout),
        cos: (&Storage, &Layout),
        sin: (&Storage, &Layout),
        start_pos: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        rotary_dim: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (cos_st, cos_lo) = cos;
        let (sin_st, sin_lo) = sin;
        let (sp_st, sp_lo) = start_pos;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if cos_lo.dtype() != dtype || sin_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda rope_apply_dev: cos/sin dtype must match x",
            ));
        }
        if sp_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda rope_apply_dev: start_pos must be U32",
            ));
        }
        if x_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda rope_apply_dev: x must be rank-4 [b,h,t,hd]",
            ));
        }
        if !x_lo.is_contiguous() || !cos_lo.is_contiguous() || !sin_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, h, t, head_dim) = (
            x_lo.dims()[0],
            x_lo.dims()[1],
            x_lo.dims()[2],
            x_lo.dims()[3],
        );
        let cos_n = cos_lo.numel();
        if sin_lo.numel() != cos_n {
            return Err(SynaptixError::Unsupported(
                "cuda rope_apply_dev: cos/sin numel mismatch",
            ));
        }
        let x_buf = x_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda rope_apply_dev: x non-cuda",
        ))?;
        let cos_buf = cos_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda rope_apply_dev: cos non-cuda",
        ))?;
        let sin_buf = sin_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda rope_apply_dev: sin non-cuda",
        ))?;
        let sp_buf = sp_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda rope_apply_dev: start_pos non-cuda",
        ))?;
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda rope_apply_dev: out non-cuda",
        ))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::rope::RopeKernels::for_context(&ctx)?;
        let sp_off = sp_lo.byte_offset();
        let sp_view = unsafe {
            sp_buf
                .slice()
                .slice(sp_off..sp_off + 4)
                .transmute::<u32>(1)
                .ok_or_else(|| SynaptixError::Cuda("rope_apply_dev: transmute start_pos".into()))?
        };
        crate::elementwise::rope::apply_partial_u8_dev(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            cos_buf.slice(),
            cos_lo.byte_offset(),
            sin_buf.slice(),
            sin_lo.byte_offset(),
            cos_n,
            &sp_view,
            b as u32,
            h as u32,
            t as u32,
            head_dim as u32,
            rotary_dim as u32,
            dtype,
        )
    }

    fn kv_append_dev(
        &self,
        dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: (&Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (dst_st, dst_lo) = dst;
        let (src_st, src_lo) = src;
        let (sp_st, sp_lo) = seq_pos;
        let dtype = dst_lo.dtype();
        if src_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append_dev: dtype mismatch src/dst",
            ));
        }
        if sp_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append_dev: seq_pos must be U32",
            ));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append_dev: expect rank-4 [B,nkv,T,hd]",
            ));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, nkv, max_seq, hd) = (
            dst_lo.dims()[0],
            dst_lo.dims()[1],
            dst_lo.dims()[2],
            dst_lo.dims()[3],
        );
        let (sb, snkv, t_new, shd) = (
            src_lo.dims()[0],
            src_lo.dims()[1],
            src_lo.dims()[2],
            src_lo.dims()[3],
        );
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::Unsupported(
                "cuda kv_append_dev: src/dst shape mismatch",
            ));
        }
        let cuda = dst_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda kv_append_dev: dst non-cuda",
        ))?;
        let ctx = cuda.device().clone();
        let ord = cuda.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::kv_append::KvAppendKernels::for_context(&ctx)?;
        let sp_buf = sp_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda kv_append_dev: seq_pos non-cuda",
        ))?;
        let sp_off = sp_lo.byte_offset();
        let sp_view = unsafe {
            sp_buf
                .slice()
                .slice(sp_off..sp_off + 4)
                .transmute::<u32>(1)
                .ok_or_else(|| SynaptixError::Cuda("kv_append_dev: transmute seq_pos".into()))?
        };
        let src_buf = src_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda kv_append_dev: src non-cuda",
        ))?;
        let dst_buf = dst_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda kv_append_dev: dst non-cuda",
        ))?;
        crate::elementwise::kv_append::append_u8_dev(
            &kernels,
            &stream,
            src_buf.slice(),
            src_lo.byte_offset(),
            dst_buf.slice_mut(),
            dst_lo.byte_offset(),
            b as u32,
            nkv as u32,
            t_new as u32,
            hd as u32,
            max_seq as u32,
            &sp_view,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (tc_st, tc_lo) = t_cache;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if !matches!(dtype, DType::BF16 | DType::F16) || k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("flash_window_dev: f16/bf16 only"));
        }
        if tc_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("flash_window_dev: t_cache must be U32"));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("flash_window_dev: rank-4"));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        if d != 128 {
            return Err(SynaptixError::Unsupported("flash_window_dev: HD=128 only"));
        }
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("flash_window_dev: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 {
            return Err(SynaptixError::Unsupported("flash_window_dev: GQA"));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported("flash_window_dev: k/v t_stride"));
        }
        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("flash_window_dev: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let tc_buf = tc_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("flash_window_dev: t_cache non-cuda"))?;
        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("flash_window_dev: out non-cuda"))?;
        let kernels = crate::attention::flash_splitq::FlashSplitQKernels::for_context(&ctx)?;
        crate::attention::flash_splitq::flash_splitq_window_u8_dev(
            &kernels,
            &stream,
            dtype,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            tc_buf.slice(),
            tc_lo.byte_offset(),
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            scale,
            causal,
            t_stride_k,
            window,
        )
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
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (tc_st, tc_lo) = t_cache;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda flash_dev: dtype mismatch q/k/v",
            ));
        }
        if tc_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_dev: t_cache must be U32",
            ));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_dev: expect rank-4 [B,H,T,D]",
            ));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda flash_dev: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 || d == 0 || d > 1024 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_dev: GQA/D constraints",
            ));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported(
                "cuda flash_dev: k/v t_stride mismatch",
            ));
        }
        let t_stride = t_stride_k;
        // split_k фиксирован на граф (launch config не должен зависеть от активной
        // длины KV — иначе capture невалиден): берём по физической ёмкости.
        let rows = (b * nh * t_q) as u32;
        let split_k = {
            let occ = (128 / rows.max(1)).max(1);
            let long = t_stride.div_ceil(2048);
            occ.max(long).clamp(1, 32)
        };
        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash_dev: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let tc_buf = tc_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda flash_dev: t_cache non-cuda",
        ))?;
        let tc_off = tc_lo.byte_offset();
        let tc_view = unsafe {
            tc_buf
                .slice()
                .slice(tc_off..tc_off + 4)
                .transmute::<u32>(1)
                .ok_or_else(|| SynaptixError::Cuda("flash_dev: transmute t_cache".into()))?
        };
        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda flash_dev: out non-cuda"))?;
        let kernels = crate::attention::flash_decode::FlashDecodeKernels::for_context(&ctx)?;
        crate::attention::flash_decode::flash_decode_u8_dev(
            &kernels,
            &stream,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            &tc_view,
            d as u32,
            scale,
            causal,
            split_k,
            t_stride,
            dtype,
        )
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
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (tc_st, tc_lo) = t_cache;
        let (out_st, _out_lo) = out;
        let dtype = q_lo.dtype();
        if k_lo.dtype() != dtype || v_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: dtype mismatch q/k/v",
            ));
        }
        if tc_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: t_cache must be U32",
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: tensor-core path требует F16/BF16",
            ));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: expect rank-4 [B,H,T,D]",
            ));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: k/v shape",
            ));
        }
        if nkv == 0 || nh % nkv != 0 {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: GQA constraint NH % NKV == 0",
            ));
        }
        if !matches!(d, 64 | 128 | 256) {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: HD must be 64, 128 or 256",
            ));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        // t_stride через тот же derive что в flash_attention_dev — physical T-row stride
        // strided view'а на preallocated [B,nkv,max_seq,hd]; контролируем что k/v идут
        // через KV-кэш, а не свежий contiguous (последний случай для prefill — наоборот,
        // запрещён: device-resident-Tkv нужен именно с preallocated max_seq).
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: k/v t_stride mismatch",
            ));
        }
        let t_stride = t_stride_k;
        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported(
                "cuda flash_prefill_dev: q non-cuda",
            ))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let tc_buf = tc_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda flash_prefill_dev: t_cache non-cuda",
        ))?;
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda flash_prefill_dev: out non-cuda",
        ))?;
        let kernels = crate::attention::flash_splitq::FlashSplitQKernels::for_context(&ctx)?;
        crate::attention::flash_splitq::flash_splitq_u8_dev(
            &kernels,
            &stream,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            tc_buf.slice(),
            tc_lo.byte_offset(),
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            d as u32,
            scale,
            causal,
            t_stride,
            dtype,
        )
    }

    fn kv_append_quant_mxfp8_dev(
        &self,
        dst: (&mut Storage, &Layout),
        scale_dst: (&mut Storage, &Layout),
        src: (&Storage, &Layout),
        seq_pos: (&Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (dst_st, dst_lo) = dst;
        let (sc_st, sc_lo) = scale_dst;
        let (src_st, src_lo) = src;
        let (sp_st, sp_lo) = seq_pos;
        if dst_lo.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: dst must be MXFP8"));
        }
        if sc_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: scale must be U8"));
        }
        if src_lo.dtype() != DType::BF16 && src_lo.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: src must be BF16/F16"));
        }
        if sp_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: seq_pos must be U32"));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 || sc_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: ranks [4,4,4]"));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() || !sc_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, nkv, max_seq, hd) = (
            dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3],
        );
        let (sb, snkv, t_new, shd) = (
            src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3],
        );
        if hd % 32 != 0 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: hd % 32 != 0"));
        }
        let nb = hd / 32;
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: src/dst shape"));
        }
        if sc_lo.dims() != [b, nkv, max_seq, nb] {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: scale shape"));
        }
        let cuda = dst_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: dst non-cuda"))?;
        let ctx = cuda.device().clone();
        let ord = cuda.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::mxfp8_kv::MxFp8KvKernels::for_context(&ctx)?;
        let sp_buf = sp_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: seq_pos non-cuda"))?;
        let sp_off = sp_lo.byte_offset();
        let sp_view = unsafe {
            sp_buf
                .slice()
                .slice(sp_off..sp_off + 4)
                .transmute::<u32>(1)
                .ok_or_else(|| SynaptixError::Cuda("kv_quant mxfp8 dev: transmute seq_pos".into()))?
        };
        let src_buf = src_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: src non-cuda"))?;
        let dst_buf = dst_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: dst non-cuda"))?;
        let sc_buf = sc_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8 dev: scale non-cuda"))?;
        crate::elementwise::mxfp8_kv::quant_append_mxfp8_u8_dev(
            &kernels,
            &stream,
            src_buf.slice(),
            src_lo.byte_offset(),
            dst_buf.slice_mut(),
            dst_lo.byte_offset(),
            sc_buf.slice_mut(),
            sc_lo.byte_offset(),
            b as u32,
            nkv as u32,
            t_new as u32,
            hd as u32,
            max_seq as u32,
            &sp_view,
            src_lo.dtype(),
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        _stream: &Stream,
    ) -> Result<()> {
        let (q_st, q_lo) = q;
        let (k_st, k_lo) = k;
        let (v_st, v_lo) = v;
        let (ks_st, ks_lo) = k_scale;
        let (vs_st, vs_lo) = v_scale;
        let (tc_st, tc_lo) = t_cache;
        let (out_st, _out_lo) = out;
        let q_dtype = q_lo.dtype();
        if !q_dtype.is_float() {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: q must be float"));
        }
        if k_lo.dtype() != DType::MXFP8 || v_lo.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: k/v must be MXFP8"));
        }
        if ks_lo.dtype() != DType::U8 || vs_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: scales must be U8"));
        }
        if tc_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: t_cache must be U32"));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: rank-4 [B,H,T,D]"));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 || d == 0 || d > 1024 || d % 32 != 0 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: GQA/D/%32"));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let nb = d / 32;
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0 && s[3] == 1 && s[2] == hd_i && s[1] > 0 && s[1] % hd_i == 0 && s[0] == nkv_i * s[1] {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        if derive_tstride(v_lo) != Some(t_stride) {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: k/v t_stride mismatch"));
        }
        let scale_tstride_ok = |lo: &Layout| -> bool {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            dd.len() == 4
                && dd[3] == nb
                && s[3] == 1
                && s[2] == nb as isize
                && s[1] == (t_stride as isize) * (nb as isize)
                && s[0] == (dd[1] as isize) * s[1]
        };
        if !scale_tstride_ok(ks_lo) || !scale_tstride_ok(vs_lo) {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8 dev: scale layout/t_stride"));
        }
        let rows = (b * nh * t_q) as u32;
        let split_k = {
            let occ = (128 / rows.max(1)).max(1);
            let long = t_stride.div_ceil(2048);
            occ.max(long).clamp(1, 32)
        };
        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8 dev: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let tc_buf = tc_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8 dev: t_cache non-cuda"))?;
        let tc_off = tc_lo.byte_offset();
        let tc_view = unsafe {
            tc_buf
                .slice()
                .slice(tc_off..tc_off + 4)
                .transmute::<u32>(1)
                .ok_or_else(|| SynaptixError::Cuda("flash mxfp8 dev: transmute t_cache".into()))?
        };
        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let ks_buf = ks_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8 dev: k_scale non-cuda"))?;
        let vs_buf = vs_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8 dev: v_scale non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8 dev: out non-cuda"))?;
        // GQA-групповое v2-ядро (sm_120a): KV читается один раз на группу голов,
        // деквант E4M3 аппаратным cvt. Недоступно/чужая форма → скалярный путь.
        if crate::attention::flash_decode_mxfp8_v2::v2_shape_ok(
            d as u32,
            t_q as u32,
            q_dtype,
            k_lo.byte_offset(),
            v_lo.byte_offset(),
        ) {
            if let Some(k2) =
                crate::attention::flash_decode_mxfp8_v2::FlashDecodeMxfp8V2Kernels::try_for_context(&ctx)
            {
                return crate::attention::flash_decode_mxfp8_v2::flash_decode_mxfp8_v2_u8_dev(
                    &k2,
                    &stream,
                    q_buf.slice(),
                    q_lo.byte_offset(),
                    k_buf.slice(),
                    k_lo.byte_offset(),
                    v_buf.slice(),
                    v_lo.byte_offset(),
                    ks_buf.slice(),
                    ks_lo.byte_offset(),
                    vs_buf.slice(),
                    vs_lo.byte_offset(),
                    out_buf.slice_mut(),
                    0,
                    b as u32,
                    nh as u32,
                    nkv as u32,
                    t_q as u32,
                    &tc_view,
                    d as u32,
                    scale,
                    causal,
                    t_stride,
                    q_dtype,
                );
            }
        }
        let kernels = crate::attention::flash_decode::FlashDecodeKernels::for_context(&ctx)?;
        crate::attention::flash_decode::flash_decode_mxfp8_u8_dev(
            &kernels,
            &stream,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            ks_buf.slice(),
            ks_lo.byte_offset(),
            vs_buf.slice(),
            vs_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            &tc_view,
            d as u32,
            scale,
            causal,
            split_k,
            t_stride,
            q_dtype,
        )
    }

    fn embed_gather(
        &self,
        table: (&Storage, &Layout),
        ids: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (table_st, table_lo) = table;
        let (ids_st, ids_lo) = ids;
        let (out_st, _out_lo) = out;
        let dtype = table_lo.dtype();
        if ids_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda embed_gather: ids must be U32",
            ));
        }
        if table_lo.dims().len() != 2 {
            return Err(SynaptixError::Unsupported(
                "cuda embed_gather: table must be [vocab,dim]",
            ));
        }
        if !table_lo.is_contiguous() || !ids_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let vocab = table_lo.dims()[0];
        let dim = table_lo.dims()[1];
        let n_ids = ids_lo.numel();
        let table_buf = table_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather: table non-cuda",
        ))?;
        let ctx = table_buf.device().clone();
        let ord = table_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::embed::EmbedKernels::for_context(&ctx)?;
        let ids_buf = ids_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather: ids non-cuda",
        ))?;
        let ids_off = ids_lo.byte_offset();
        let ids_view = unsafe {
            ids_buf
                .slice()
                .slice(ids_off..ids_off + n_ids * 4)
                .transmute::<u32>(n_ids)
                .ok_or_else(|| SynaptixError::Cuda("embed_gather: transmute ids".into()))?
        };
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather: out non-cuda",
        ))?;
        crate::embed::embed_gather_u8(
            &kernels,
            &stream,
            table_buf.slice(),
            table_lo.byte_offset(),
            &ids_view,
            out_buf.slice_mut(),
            0,
            n_ids as u32,
            dim as u32,
            vocab as u32,
            dtype,
        )
    }

    fn embed_gather_mxfp8(
        &self,
        table: &Storage,
        scales: &Storage,
        ids: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        vocab: usize,
        dim: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (ids_st, ids_lo) = ids;
        let (out_st, out_lo) = out;
        if ids_lo.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported(
                "cuda embed_gather_mxfp8: ids must be U32",
            ));
        }
        if out_lo.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported(
                "cuda embed_gather_mxfp8: out must be F16",
            ));
        }
        if !ids_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let n_ids = ids_lo.numel();
        let table_buf = table.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather_mxfp8: table non-cuda",
        ))?;
        let scales_buf = scales.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather_mxfp8: scales non-cuda",
        ))?;
        let ctx = table_buf.device().clone();
        let ord = table_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::embed::EmbedKernels::for_context(&ctx)?;
        let ids_buf = ids_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather_mxfp8: ids non-cuda",
        ))?;
        let ids_off = ids_lo.byte_offset();
        let ids_view = unsafe {
            ids_buf
                .slice()
                .slice(ids_off..ids_off + n_ids * 4)
                .transmute::<u32>(n_ids)
                .ok_or_else(|| SynaptixError::Cuda("embed_gather_mxfp8: transmute ids".into()))?
        };
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda embed_gather_mxfp8: out non-cuda",
        ))?;
        crate::embed::embed_gather_mxfp8(
            &kernels,
            &stream,
            table_buf.slice(),
            0,
            scales_buf.slice(),
            0,
            &ids_view,
            out_buf.slice_mut(),
            0,
            n_ids as u32,
            dim as u32,
            vocab as u32,
        )
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
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: dst must be MXFP8"));
        }
        if sc_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: scale must be U8"));
        }
        if src_lo.dtype() != DType::BF16 && src_lo.dtype() != DType::F16 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: src must be BF16/F16"));
        }
        if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 || sc_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: ranks [4,4,4]"));
        }
        if !dst_lo.is_contiguous() || !src_lo.is_contiguous() || !sc_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let (b, nkv, max_seq, hd) = (
            dst_lo.dims()[0],
            dst_lo.dims()[1],
            dst_lo.dims()[2],
            dst_lo.dims()[3],
        );
        let (sb, snkv, t_new, shd) = (
            src_lo.dims()[0],
            src_lo.dims()[1],
            src_lo.dims()[2],
            src_lo.dims()[3],
        );
        if hd % 32 != 0 {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: hd % 32 != 0"));
        }
        let nb = hd / 32;
        if sb != b || snkv != nkv || shd != hd {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: src/dst shape mismatch"));
        }
        if sc_lo.dims() != [b, nkv, max_seq, nb] {
            return Err(SynaptixError::Unsupported("cuda kv_quant mxfp8: scale shape"));
        }
        if seq_pos + t_new > max_seq {
            return Err(SynaptixError::Other(format!(
                "cuda kv_quant mxfp8: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
            )));
        }
        let cuda = dst_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8: dst non-cuda"))?;
        let ctx = cuda.device().clone();
        let ord = cuda.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::mxfp8_kv::MxFp8KvKernels::for_context(&ctx)?;
        let src_buf = src_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8: src non-cuda"))?;
        let dst_buf = dst_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8: dst non-cuda"))?;
        let sc_buf = sc_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda kv_quant mxfp8: scale non-cuda"))?;
        crate::elementwise::mxfp8_kv::quant_append_mxfp8_u8(
            &kernels,
            &stream,
            src_buf.slice(),
            src_lo.byte_offset(),
            dst_buf.slice_mut(),
            dst_lo.byte_offset(),
            sc_buf.slice_mut(),
            sc_lo.byte_offset(),
            b as u32,
            nkv as u32,
            t_new as u32,
            hd as u32,
            max_seq as u32,
            seq_pos as u32,
            src_lo.dtype(),
        )
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
        let (out_st, _out_lo) = out;
        let q_dtype = q_lo.dtype();
        if !q_dtype.is_float() {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: q must be float"));
        }
        if k_lo.dtype() != DType::MXFP8 || v_lo.dtype() != DType::MXFP8 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: k/v must be MXFP8"));
        }
        if ks_lo.dtype() != DType::U8 || vs_lo.dtype() != DType::U8 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: scales must be U8"));
        }
        if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: expect rank-4 [B,H,T,D]"));
        }
        let b = q_lo.dims()[0];
        let nh = q_lo.dims()[1];
        let t_q = q_lo.dims()[2];
        let d = q_lo.dims()[3];
        let nkv = k_lo.dims()[1];
        let t_kv = k_lo.dims()[2];
        if k_lo.dims()[0] != b || k_lo.dims()[3] != d || v_lo.dims() != k_lo.dims() {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: k/v shape"));
        }
        if nkv == 0 || nh % nkv != 0 || d == 0 || d > 1024 || d % 32 != 0 {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: GQA/D/%32 constraints"));
        }
        if !q_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let nb = d / 32;
        // t_stride из K-layout (physical T-row stride preallocated MXFP8-буфера).
        let derive_tstride = |lo: &Layout| -> Option<u32> {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            let hd_i = dd[3] as isize;
            let nkv_i = dd[1] as isize;
            if hd_i > 0
                && s[3] == 1
                && s[2] == hd_i
                && s[1] > 0
                && s[1] % hd_i == 0
                && s[0] == nkv_i * s[1]
            {
                Some((s[1] / hd_i) as u32)
            } else {
                None
            }
        };
        let t_stride_k = derive_tstride(k_lo).ok_or(SynaptixError::NonContiguous)?;
        let t_stride_v = derive_tstride(v_lo).ok_or(SynaptixError::NonContiguous)?;
        if t_stride_k != t_stride_v {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: k/v t_stride mismatch"));
        }
        let t_stride = t_stride_k;
        // Scale-layout [B,NKV,T,D/32]: physical T-stride = t_stride·nb (тот же max_seq),
        // блочная ось nb — contiguous (s[3]==1).
        let scale_tstride_ok = |lo: &Layout| -> bool {
            let s = lo.strides().as_slice();
            let dd = lo.dims();
            dd.len() == 4
                && dd[3] == nb
                && s[3] == 1
                && s[2] == nb as isize
                && s[1] == (t_stride as isize) * (nb as isize)
                && s[0] == (dd[1] as isize) * s[1]
        };
        if !scale_tstride_ok(ks_lo) || !scale_tstride_ok(vs_lo) {
            return Err(SynaptixError::Unsupported("cuda flash mxfp8: scale layout/t_stride"));
        }

        let ctx = q_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8: q non-cuda"))?
            .device()
            .clone();
        let ord = q_st.as_cuda().unwrap().ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;

        let q_buf = q_st.as_cuda().unwrap();
        let k_buf = k_st.as_cuda().unwrap();
        let v_buf = v_st.as_cuda().unwrap();
        let ks_buf = ks_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8: k_scale non-cuda"))?;
        let vs_buf = vs_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8: v_scale non-cuda"))?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda flash mxfp8: out non-cuda"))?;

        // Малые Tq (decode Tq=1, MTP-verify 2..8) + hd∈{128,256} + F16/BF16 q:
        // GQA-групповое v2-ядро (sm_120a; недоступно → фолбэк ниже). WMMA-путь
        // для Tq=2..8 на длинном контексте не годится: grid = B·NH блоков, каждый
        // сканирует весь Tkv (замер 47k: 45 мс/слой против 0,2 мс у v2).
        if crate::attention::flash_decode_mxfp8_v2::v2_shape_ok(
            d as u32,
            t_q as u32,
            q_dtype,
            k_lo.byte_offset(),
            v_lo.byte_offset(),
        ) {
            if let Some(k2) =
                crate::attention::flash_decode_mxfp8_v2::FlashDecodeMxfp8V2Kernels::try_for_context(&ctx)
            {
                return crate::attention::flash_decode_mxfp8_v2::flash_decode_mxfp8_v2_u8(
                    &k2,
                    &stream,
                    q_buf.slice(),
                    q_lo.byte_offset(),
                    k_buf.slice(),
                    k_lo.byte_offset(),
                    v_buf.slice(),
                    v_lo.byte_offset(),
                    ks_buf.slice(),
                    ks_lo.byte_offset(),
                    vs_buf.slice(),
                    vs_lo.byte_offset(),
                    out_buf.slice_mut(),
                    0,
                    b as u32,
                    nh as u32,
                    nkv as u32,
                    t_q as u32,
                    t_kv as u32,
                    d as u32,
                    scale,
                    causal,
                    t_stride,
                    q_dtype,
                );
            }
        }
        // Prefill (Tq>8) + hd∈{128,256}: сперва split-Q v2 (схема flash_splitq
        // с деквант-fill, sm_120a), фолбэк — старый WMMA-путь (BM=16, серийный
        // softmax; на 47k давал 135 мс/слой против ~0,5 мс у split-Q).
        if t_q > 1
            && crate::attention::flash_mxfp8_splitq::splitq_shape_ok(
                d as u32,
                q_dtype,
                k_lo.byte_offset(),
                v_lo.byte_offset(),
            )
        {
            if let Some(k2) =
                crate::attention::flash_mxfp8_splitq::FlashMxfp8SplitqKernels::try_for_context(&ctx)
            {
                return crate::attention::flash_mxfp8_splitq::flash_mxfp8_splitq_u8(
                    &k2,
                    &stream,
                    q_buf.slice(),
                    q_lo.byte_offset(),
                    k_buf.slice(),
                    k_lo.byte_offset(),
                    v_buf.slice(),
                    v_lo.byte_offset(),
                    ks_buf.slice(),
                    ks_lo.byte_offset(),
                    vs_buf.slice(),
                    vs_lo.byte_offset(),
                    out_buf.slice_mut(),
                    0,
                    b as u32,
                    nh as u32,
                    nkv as u32,
                    t_q as u32,
                    t_kv as u32,
                    d as u32,
                    scale,
                    causal,
                    t_stride,
                    q_dtype,
                );
            }
        }
        // Prefill (Tq>1) + hd∈{128,256}: MXFP8 tensor-core (block-dequant в smem
        // → WMMA). Decode (Tq=1) и прочие формы: scalar split-K flash_decode_mxfp8.
        if t_q > 1 && (d == 128 || d == 256) {
            let kernels =
                crate::attention::flash_mxfp8_prefill::FlashMxfp8PrefillKernels::for_context(&ctx)?;
            return crate::attention::flash_mxfp8_prefill::flash_mxfp8_prefill_u8(
                &kernels,
                &stream,
                q_buf.slice(),
                q_lo.byte_offset(),
                k_buf.slice(),
                k_lo.byte_offset(),
                v_buf.slice(),
                v_lo.byte_offset(),
                ks_buf.slice(),
                ks_lo.byte_offset(),
                vs_buf.slice(),
                vs_lo.byte_offset(),
                out_buf.slice_mut(),
                0,
                b as u32,
                nh as u32,
                nkv as u32,
                t_q as u32,
                t_kv as u32,
                d as u32,
                scale,
                causal,
                t_stride,
                q_dtype,
            );
        }
        let rows = (b * nh * t_q) as u32;
        let split_k = if t_q == 1 {
            let occ = (128 / rows.max(1)).max(1);
            let long = (t_kv as u32).div_ceil(2048);
            occ.max(long).clamp(1, 32)
        } else {
            1
        };
        let kernels = crate::attention::flash_decode::FlashDecodeKernels::for_context(&ctx)?;
        crate::attention::flash_decode::flash_decode_mxfp8_u8(
            &kernels,
            &stream,
            q_buf.slice(),
            q_lo.byte_offset(),
            k_buf.slice(),
            k_lo.byte_offset(),
            v_buf.slice(),
            v_lo.byte_offset(),
            ks_buf.slice(),
            ks_lo.byte_offset(),
            vs_buf.slice(),
            vs_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            t_kv as u32,
            d as u32,
            scale,
            causal,
            split_k,
            t_stride,
            q_dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attn_decode_step(
        &self,
        qkv: (&Storage, &Layout),
        conv_w: (&Storage, &Layout),
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dt_bias: (&Storage, &Layout),
        a_log: (&Storage, &Layout),
        z: (&Storage, &Layout),
        norm_w: (&Storage, &Layout),
        conv_state: (&mut Storage, &Layout),
        ssm_state: (&mut Storage, &Layout),
        out: (&mut Storage, &Layout),
        num_k: usize,
        num_v: usize,
        dk: usize,
        dv: usize,
        conv_kernel: usize,
        q_scale: f32,
        eps: f32,
        _stream: &Stream,
    ) -> Result<()> {
        for (st, name, want) in [
            (qkv.1, "qkv", DType::F16),
            (conv_w.1, "conv_w", DType::F16),
            (a.1, "a", DType::F16),
            (b.1, "b", DType::F16),
            (dt_bias.1, "dt_bias", DType::F32),
            (a_log.1, "a_log", DType::F32),
            (z.1, "z", DType::F16),
            (norm_w.1, "norm_w", DType::F16),
            (conv_state.1, "conv_state", DType::F16),
            (ssm_state.1, "ssm_state", DType::F32),
            (out.1, "out", DType::F16),
        ] {
            if st.dtype() != want {
                return Err(SynaptixError::Cuda(format!(
                    "linear_attn_decode_step: {name} must be {want:?}, got {:?}",
                    st.dtype()
                )));
            }
        }
        let qkv_buf = qkv
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_decode: qkv non-cuda"))?;
        let conv_w_buf = conv_w
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_decode: conv_w non-cuda"))?;
        let a_buf =
            a.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("linear_decode: a non-cuda"))?;
        let b_buf =
            b.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("linear_decode: b non-cuda"))?;
        let dt_buf = dt_bias.0.as_cuda().ok_or(SynaptixError::Unsupported(
            "linear_decode: dt_bias non-cuda",
        ))?;
        let al_buf = a_log
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_decode: a_log non-cuda"))?;
        let z_buf =
            z.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("linear_decode: z non-cuda"))?;
        let nw_buf = norm_w
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_decode: norm_w non-cuda"))?;
        let ctx = qkv_buf.device().clone();
        let ord = qkv_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let conv_kernels = crate::conv::causal_conv1d::CausalConv1dKernels::for_context(&ctx)?;
        let prep_kernels =
            crate::attention::linear_attn_raw::LinearAttnRawKernels::for_context(&ctx)?;
        let gdr_kernels = crate::ssm::gated_delta_rule::GatedDeltaRuleKernels::for_context(&ctx)?;
        let (qkv_off, cw_off, a_off, b_off, dt_off, al_off, z_off, nw_off) = (
            qkv.1.byte_offset(),
            conv_w.1.byte_offset(),
            a.1.byte_offset(),
            b.1.byte_offset(),
            dt_bias.1.byte_offset(),
            a_log.1.byte_offset(),
            z.1.byte_offset(),
            norm_w.1.byte_offset(),
        );
        let (cs_off, ss_off, out_off) = (
            conv_state.1.byte_offset(),
            ssm_state.1.byte_offset(),
            out.1.byte_offset(),
        );
        let cs_buf = conv_state
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported(
                "linear_decode: conv_state non-cuda",
            ))?;
        let ss_buf = ssm_state.0.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "linear_decode: ssm_state non-cuda",
        ))?;
        let out_buf = out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("linear_decode: out non-cuda"))?;
        crate::attention::linear_decode::linear_attn_decode_step_u8_dev(
            &conv_kernels,
            &prep_kernels,
            &gdr_kernels,
            &stream,
            (qkv_buf.slice(), qkv_off),
            (conv_w_buf.slice(), cw_off),
            (a_buf.slice(), a_off),
            (b_buf.slice(), b_off),
            (dt_buf.slice(), dt_off),
            (al_buf.slice(), al_off),
            (z_buf.slice(), z_off),
            (nw_buf.slice(), nw_off),
            cs_buf.slice_mut(),
            cs_off,
            ss_buf.slice_mut(),
            ss_off,
            out_buf.slice_mut(),
            out_off,
            num_k as u32,
            num_v as u32,
            dk as u32,
            dv as u32,
            conv_kernel as u32,
            q_scale,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attn_chunk_prefill(
        &self,
        qkv: (&Storage, &Layout),
        conv_w: (&Storage, &Layout),
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dt_bias: (&Storage, &Layout),
        a_log: (&Storage, &Layout),
        conv_state: (&mut Storage, &Layout),
        ssm_state: (&mut Storage, &Layout),
        out: (&mut Storage, &Layout),
        num_k: usize,
        num_v: usize,
        hk: usize,
        hv: usize,
        conv_kernel: usize,
        t_in: usize,
        t_pad: usize,
        chunk_size: usize,
        q_scale: f32,
        silu: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let compute = qkv.1.dtype();
        if !matches!(compute, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Cuda(format!(
                "linear_attn_chunk_prefill: qkv dtype {compute:?} not supported (F32/F16/BF16)"
            )));
        }
        for (st, name, want) in [
            (qkv.1, "qkv", compute),
            (conv_w.1, "conv_w", compute),
            (conv_state.1, "conv_state", compute),
            (a.1, "a", DType::F16),
            (b.1, "b", DType::F16),
            (dt_bias.1, "dt_bias", DType::F32),
            (a_log.1, "a_log", DType::F32),
            (ssm_state.1, "ssm_state", DType::F32),
            (out.1, "out", DType::F32),
        ] {
            if st.dtype() != want {
                return Err(SynaptixError::Cuda(format!(
                    "linear_attn_chunk_prefill: {name} must be {want:?}, got {:?}",
                    st.dtype()
                )));
            }
        }
        let qkv_buf = qkv
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_prefill: qkv non-cuda"))?;
        let conv_w_buf = conv_w.0.as_cuda().ok_or(SynaptixError::Unsupported(
            "linear_prefill: conv_w non-cuda",
        ))?;
        let a_buf =
            a.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("linear_prefill: a non-cuda"))?;
        let b_buf =
            b.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("linear_prefill: b non-cuda"))?;
        let dt_buf = dt_bias.0.as_cuda().ok_or(SynaptixError::Unsupported(
            "linear_prefill: dt_bias non-cuda",
        ))?;
        let al_buf = a_log
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("linear_prefill: a_log non-cuda"))?;
        let ctx = qkv_buf.device().clone();
        let ord = qkv_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let conv_kernels = crate::conv::causal_conv1d::CausalConv1dKernels::for_context(&ctx)?;
        let prep_kernels =
            crate::attention::linear_attn_raw::LinearAttnRawKernels::for_context(&ctx)?;
        let cfk = crate::attention::chunk_fla::ChunkFlaKernels::for_context(&ctx)?;
        let csk = crate::scan::chunk_scan::ChunkScanKernels::for_context(&ctx)?;
        let (qkv_off, cw_off, a_off, b_off, dt_off, al_off) = (
            qkv.1.byte_offset(),
            conv_w.1.byte_offset(),
            a.1.byte_offset(),
            b.1.byte_offset(),
            dt_bias.1.byte_offset(),
            a_log.1.byte_offset(),
        );
        let (cs_off, ss_off, out_off) = (
            conv_state.1.byte_offset(),
            ssm_state.1.byte_offset(),
            out.1.byte_offset(),
        );
        let cs_buf = conv_state
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported(
                "linear_prefill: conv_state non-cuda",
            ))?;
        let ss_buf = ssm_state.0.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "linear_prefill: ssm_state non-cuda",
        ))?;
        let out_buf = out
            .0
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("linear_prefill: out non-cuda"))?;
        crate::attention::linear_prefill::linear_attn_chunk_prefill_u8_dev(
            &conv_kernels,
            &prep_kernels,
            &cfk,
            &csk,
            &stream,
            (qkv_buf.slice(), qkv_off),
            (conv_w_buf.slice(), cw_off),
            (a_buf.slice(), a_off),
            (b_buf.slice(), b_off),
            (dt_buf.slice(), dt_off),
            (al_buf.slice(), al_off),
            cs_buf.slice_mut(),
            cs_off,
            ss_buf.slice_mut(),
            ss_off,
            out_buf.slice_mut(),
            out_off,
            compute,
            num_k as u32,
            num_v as u32,
            hk as u32,
            hv as u32,
            conv_kernel as u32,
            t_in as u32,
            t_pad as u32,
            chunk_size as u32,
            q_scale,
            silu,
        )?;
        if !stream_is_capturing(&stream) {
            stream
                .synchronize()
                .map_err(|e| SynaptixError::Cuda(format!("linear_prefill sync: {e:?}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_rule_prefill(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        g: (&Storage, &Layout),
        beta: (&Storage, &Layout),
        ssm_state: (&mut Storage, &Layout),
        out: (&mut Storage, &Layout),
        q_scale: f32,
        bh: usize,
        t: usize,
        hk: usize,
        hv: usize,
        cs: usize,
        _stream: &Stream,
    ) -> Result<()> {
        for (st, name) in [
            (q.1, "q"),
            (k.1, "k"),
            (v.1, "v"),
            (g.1, "g"),
            (beta.1, "beta"),
            (ssm_state.1, "ssm_state"),
            (out.1, "out"),
        ] {
            if st.dtype() != DType::F32 {
                return Err(SynaptixError::Cuda(format!(
                    "gated_delta_rule_prefill: {name} must be F32, got {:?}",
                    st.dtype()
                )));
            }
        }
        let q_buf =
            q.0.as_cuda()
                .ok_or(SynaptixError::Unsupported("gdr_prefill: q non-cuda"))?;
        let ctx = q_buf.device().clone();
        let ord = q_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let cfk = crate::attention::chunk_fla::ChunkFlaKernels::for_context(&ctx)?;
        let csk = crate::scan::chunk_scan::ChunkScanKernels::for_context(&ctx)?;

        let (n_qk, n_v, n_g, n_state) = (bh * t * hk, bh * t * hv, bh * t, bh * hk * hv);

        // storage (u8-байты F32) → owned CudaSlice<f32> (оркестратор берёт владеющие
        // слайсы; copy дёшев dtod относительно скана). transmute u8→f32 — как в reduce.rs.
        let copy_in = |src: &Storage, n: usize| -> Result<cudarc::driver::CudaSlice<f32>> {
            let buf = src
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("gdr_prefill: input non-cuda"))?;
            let src_v = buf.slice().as_view();
            let src_f = unsafe { src_v.transmute::<f32>(n) }.ok_or_else(|| {
                SynaptixError::Cuda("gdr_prefill: transmute f32 (in) fail".into())
            })?;
            let mut owned = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill alloc: {e:?}")))?;
            stream
                .memcpy_dtod(&src_f, &mut owned)
                .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill memcpy in: {e:?}")))?;
            Ok(owned)
        };
        let q_o = copy_in(q.0, n_qk)?;
        let k_o = copy_in(k.0, n_qk)?;
        let v_o = copy_in(v.0, n_v)?;
        let g_o = copy_in(g.0, n_g)?;
        let b_o = copy_in(beta.0, n_g)?;
        let mut state_o = copy_in(&*ssm_state.0, n_state)?;
        let mut out_o = stream
            .alloc_zeros::<f32>(n_v)
            .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill alloc out: {e:?}")))?;

        crate::scan::chunk_scan::chunk_gated_delta_rule(
            &cfk,
            &csk,
            &stream,
            &q_o,
            &k_o,
            &v_o,
            &g_o,
            &b_o,
            &mut state_o,
            &mut out_o,
            q_scale,
            bh as u32,
            t as u32,
            hk as u32,
            hv as u32,
            cs as u32,
        )?;

        // state_o → ssm_state storage; out_o → out storage.
        {
            let buf = ssm_state.0.as_cuda_mut().ok_or(SynaptixError::Unsupported(
                "gdr_prefill: ssm_state non-cuda",
            ))?;
            let mut dv = buf.slice_mut().as_view_mut();
            let mut dst_f = unsafe { dv.transmute_mut::<f32>(n_state) }.ok_or_else(|| {
                SynaptixError::Cuda("gdr_prefill: transmute f32 (state out) fail".into())
            })?;
            stream
                .memcpy_dtod(&state_o, &mut dst_f)
                .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill memcpy state out: {e:?}")))?;
        }
        {
            let buf = out
                .0
                .as_cuda_mut()
                .ok_or(SynaptixError::Unsupported("gdr_prefill: out non-cuda"))?;
            let mut dv = buf.slice_mut().as_view_mut();
            let mut dst_f = unsafe { dv.transmute_mut::<f32>(n_v) }.ok_or_else(|| {
                SynaptixError::Cuda("gdr_prefill: transmute f32 (out) fail".into())
            })?;
            stream
                .memcpy_dtod(&out_o, &mut dst_f)
                .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill memcpy out: {e:?}")))?;
        }
        stream
            .synchronize()
            .map_err(|e| SynaptixError::Cuda(format!("gdr_prefill sync: {e:?}")))?;
        Ok(())
    }

    fn conv2d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: (usize, usize),
        padding: (usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = input;
        let (w_st, w_lo) = weight;
        let (out_st, _out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda conv2d: dtype"));
        }
        if w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported(
                "cuda conv2d: dtype mismatch x/w",
            ));
        }
        if in_lo.dims().len() != 4 || w_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda conv2d: rank != 4"));
        }
        if !in_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let (b, c_in, h, w) = (d[0], d[1], d[2], d[3]);
        let wd = w_lo.dims();
        let (c_out, kh, kw) = (wd[0], wd[2], wd[3]);
        if wd[1] != c_in {
            return Err(SynaptixError::Unsupported("cuda conv2d: c_in mismatch"));
        }
        if let Some((_, b_lo)) = bias {
            if b_lo.dtype() != dtype || b_lo.dims().len() != 1 || b_lo.dims()[0] != c_out {
                return Err(SynaptixError::Unsupported("cuda conv2d: bias shape/dtype"));
            }
            if !b_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda conv2d: in non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda conv2d: w non-cuda"))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::conv2d::Conv2dKernels::for_context(&ctx)?;
        let bias_arg = match bias {
            Some((b_st, b_lo)) => {
                let b_buf = b_st
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported("cuda conv2d: bias non-cuda"))?;
                Some((b_buf.slice(), b_lo.byte_offset()))
            }
            None => None,
        };
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let w_slice = w_buf.slice();
        let w_off = w_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda conv2d: out non-cuda"))?;
        crate::conv::conv2d::run_conv2d_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            w_slice,
            w_off,
            bias_arg,
            out_buf.slice_mut(),
            0,
            b as u32,
            c_in as u32,
            h as u32,
            w as u32,
            c_out as u32,
            kh as u32,
            kw as u32,
            stride.0 as u32,
            stride.1 as u32,
            padding.0 as u32,
            padding.1 as u32,
            dtype,
        )
    }

    fn conv3d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = input;
        let (w_st, w_lo) = weight;
        let (out_st, _out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda conv3d: dtype"));
        }
        if w_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("cuda conv3d: dtype mismatch x/w"));
        }
        if in_lo.dims().len() != 5 || w_lo.dims().len() != 5 {
            return Err(SynaptixError::Unsupported("cuda conv3d: rank != 5"));
        }
        if !in_lo.is_contiguous() || !w_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let (b, c_in, dz, h, w) = (d[0], d[1], d[2], d[3], d[4]);
        let wd = w_lo.dims();
        let (c_out, kd, kh, kw) = (wd[0], wd[2], wd[3], wd[4]);
        if wd[1] != c_in {
            return Err(SynaptixError::Unsupported("cuda conv3d: c_in mismatch"));
        }
        if let Some((_, b_lo)) = bias {
            if b_lo.dtype() != dtype || b_lo.dims().len() != 1 || b_lo.dims()[0] != c_out {
                return Err(SynaptixError::Unsupported("cuda conv3d: bias shape/dtype"));
            }
            if !b_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda conv3d: in non-cuda"))?;
        let w_buf = w_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda conv3d: w non-cuda"))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::conv3d::Conv3dKernels::for_context(&ctx)?;
        let bias_arg = match bias {
            Some((b_st, b_lo)) => {
                let b_buf = b_st
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported("cuda conv3d: bias non-cuda"))?;
                Some((b_buf.slice(), b_lo.byte_offset()))
            }
            None => None,
        };
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let w_slice = w_buf.slice();
        let w_off = w_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda conv3d: out non-cuda"))?;
        crate::conv::conv3d::run_conv3d_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            w_slice,
            w_off,
            bias_arg,
            out_buf.slice_mut(),
            0,
            b as u32,
            c_in as u32,
            dz as u32,
            h as u32,
            w as u32,
            c_out as u32,
            kd as u32,
            kh as u32,
            kw as u32,
            stride.0 as u32,
            stride.1 as u32,
            stride.2 as u32,
            padding.0 as u32,
            padding.1 as u32,
            padding.2 as u32,
            dtype,
        )
    }

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
        let (col_st, _col_lo) = col;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda im2col: dtype"));
        }
        if in_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda im2col: rank != 4"));
        }
        if !in_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let (b, c_in, h, w) = (d[0], d[1], d[2], d[3]);
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda im2col: in non-cuda"))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::im2col::Im2colKernels::for_context(&ctx)?;
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let col_buf = col_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda im2col: col non-cuda"))?;
        crate::conv::im2col::run_im2col_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            col_buf.slice_mut(),
            0,
            b as u32,
            c_in as u32,
            h as u32,
            w as u32,
            kh as u32,
            kw as u32,
            h_out as u32,
            w_out as u32,
            stride.0 as u32,
            stride.1 as u32,
            padding.0 as u32,
            padding.1 as u32,
            m_offset,
            m_count,
            dtype,
        )
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
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda group_norm: dtype"));
        }
        if x_lo.dims().len() < 2 {
            return Err(SynaptixError::Unsupported("cuda group_norm: rank < 2"));
        }
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = x_lo.dims();
        let b = d[0];
        // NHWC: x=[B,H,W,C] → C=last, HW=средние. NCHW: [B,C,...] → C=d[1], HW=rest.
        let (c, hw) = if nhwc {
            (d[d.len() - 1], d[1..d.len() - 1].iter().product::<usize>().max(1))
        } else {
            (d[1], d[2..].iter().product::<usize>().max(1))
        };
        if num_groups == 0 || c % num_groups != 0 {
            return Err(SynaptixError::Unsupported(
                "cuda group_norm: c % num_groups",
            ));
        }
        // bias без weight не поддержан ядром (gamma/beta вместе).
        if weight.is_none() && bias.is_some() {
            return Err(SynaptixError::Unsupported(
                "cuda group_norm: bias без weight",
            ));
        }
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda group_norm: x non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::reduction::groupnorm::GroupNormKernels::for_context(&ctx)?;
        let affine = match (weight, bias) {
            (Some((w_st, w_lo)), Some((b_st, b_lo))) => {
                if w_lo.dtype() != dtype || b_lo.dtype() != dtype {
                    return Err(SynaptixError::Unsupported("cuda group_norm: affine dtype"));
                }
                if !w_lo.is_contiguous() || !b_lo.is_contiguous() {
                    return Err(SynaptixError::NonContiguous);
                }
                let w_buf = w_st
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported("cuda group_norm: w non-cuda"))?;
                let b_buf = b_st
                    .as_cuda()
                    .ok_or(SynaptixError::Unsupported("cuda group_norm: b non-cuda"))?;
                Some((
                    w_buf.slice(),
                    w_lo.byte_offset(),
                    b_buf.slice(),
                    b_lo.byte_offset(),
                ))
            }
            (Some(_), None) => {
                return Err(SynaptixError::Unsupported(
                    "cuda group_norm: weight без bias",
                ))
            }
            _ => None,
        };
        let x_slice = x_buf.slice();
        let x_off = x_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda group_norm: out non-cuda"))?;
        crate::reduction::groupnorm::run_u8(
            &kernels,
            &stream,
            x_slice,
            x_off,
            affine,
            out_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            hw as u32,
            num_groups as u32,
            eps,
            silu,
            nhwc,
            dtype,
        )
    }

    fn pixel_norm(
        &self,
        x: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        c: usize,
        eps: f32,
        silu: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda pixel_norm: dtype"));
        }
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = x_lo.dims();
        if d.len() < 2 || d[1] != c {
            return Err(SynaptixError::Unsupported("cuda pixel_norm: shape"));
        }
        let b = d[0];
        let s = d[2..].iter().product::<usize>().max(1);
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda pixel_norm: x non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::reduction::pixelnorm::PixelNormKernels::for_context(&ctx)?;
        let x_slice = x_buf.slice();
        let x_off = x_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda pixel_norm: out non-cuda"))?;
        crate::reduction::pixelnorm::run_u8(
            &kernels,
            &stream,
            x_slice,
            x_off,
            out_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            s as u64,
            eps,
            silu,
            dtype,
        )
    }

    fn nchw_to_nhwc(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (s_st, s_lo) = src;
        let (d_st, _d_lo) = dst;
        let dtype = s_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda nchw_to_nhwc: dtype"));
        }
        if s_lo.dims().len() != 4 || !s_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = s_lo.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let s_buf = s_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda nchw_to_nhwc: src non-cuda",
        ))?;
        let ctx = s_buf.device().clone();
        let ord = s_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::nchw_nhwc::NchwNhwcKernels::for_context(&ctx)?;
        let s_slice = s_buf.slice();
        let s_off = s_lo.byte_offset();
        let d_buf = d_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda nchw_to_nhwc: dst non-cuda",
        ))?;
        crate::conv::nchw_nhwc::run_nchw_to_nhwc_u8(
            &kernels,
            &stream,
            s_slice,
            s_off,
            d_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            h as u32,
            w as u32,
            dtype,
        )
    }

    fn nhwc_to_nchw(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (s_st, s_lo) = src;
        let (d_st, d_lo) = dst;
        let dtype = s_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda nhwc_to_nchw: dtype"));
        }
        if s_lo.dims().len() != 4 || !s_lo.strides().is_contiguous(s_lo.shape()) {
            return Err(SynaptixError::NonContiguous);
        }
        // параметры ядра — NCHW-размеры выхода
        let d = d_lo.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let s_buf = s_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda nhwc_to_nchw: src non-cuda",
        ))?;
        let ctx = s_buf.device().clone();
        let ord = s_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::nchw_nhwc::NchwNhwcKernels::for_context(&ctx)?;
        let s_slice = s_buf.slice();
        let s_off = s_lo.byte_offset();
        let d_buf = d_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda nhwc_to_nchw: dst non-cuda",
        ))?;
        crate::conv::nchw_nhwc::run_nhwc_to_nchw_u8(
            &kernels,
            &stream,
            s_slice,
            s_off,
            d_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            h as u32,
            w as u32,
            dtype,
        )
    }

    fn conv2d_implicit_nhwc(
        &self,
        input_nhwc: (&Storage, &Layout),
        filter_krsc: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        residual: Option<(&Storage, &Layout)>,
        temb: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        out_nhwc: bool,
        stride: (usize, usize),
        padding: (usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = input_nhwc;
        let (f_st, f_lo) = filter_krsc;
        let (out_st, out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("conv2d_implicit: dtype F16/BF16"));
        }
        if f_lo.dtype() != dtype {
            return Err(SynaptixError::Unsupported("conv2d_implicit: filter dtype"));
        }
        if in_lo.dims().len() != 4 || f_lo.dims().len() != 4 || out_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("conv2d_implicit: rank != 4"));
        }
        // вход: contiguous-strides достаточно — offset-слайс (conv3d kd-окно)
        // легален, byte_offset уходит в ядро; фильтр — строго contiguous.
        if !in_lo.strides().is_contiguous(in_lo.shape()) || !f_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let (n, h, w, c) = (d[0], d[1], d[2], d[3]);
        let fd = f_lo.dims();
        let (k, r, s, c_chk) = (fd[0], fd[1], fd[2], fd[3]);
        if c_chk != c {
            return Err(SynaptixError::Unsupported("conv2d_implicit: C mismatch"));
        }
        let od = out_lo.dims();
        let (p, q) = if out_nhwc {
            if od[0] != n || od[3] != k {
                return Err(SynaptixError::Unsupported("conv2d_implicit: out shape nhwc"));
            }
            (od[1], od[2])
        } else {
            if od[0] != n || od[1] != k {
                return Err(SynaptixError::Unsupported("conv2d_implicit: out shape"));
            }
            (od[2], od[3])
        };
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("conv2d_implicit: in non-cuda"))?;
        let f_buf = f_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("conv2d_implicit: f non-cuda"))?;
        let bias_buf = match bias {
            Some((st, lo)) => {
                if lo.dtype() != dtype || !lo.is_contiguous() || lo.numel() != k {
                    return Err(SynaptixError::Unsupported("conv2d_implicit: bias shape"));
                }
                Some(st.as_cuda().ok_or(SynaptixError::Unsupported("conv2d_implicit: bias non-cuda"))?)
            }
            None => None,
        };
        let res_buf = match residual {
            Some((st, lo)) => {
                if lo.dtype() != dtype || !lo.is_contiguous() || lo.numel() != n * k * p * q || lo.offset() != 0 {
                    return Err(SynaptixError::Unsupported("conv2d_implicit: residual shape"));
                }
                Some(st.as_cuda().ok_or(SynaptixError::Unsupported("conv2d_implicit: residual non-cuda"))?)
            }
            None => None,
        };
        let temb_buf = match temb {
            Some((st, lo)) => {
                if lo.dtype() != dtype || !lo.is_contiguous() || lo.numel() != n * k {
                    return Err(SynaptixError::Unsupported("conv2d_implicit: temb shape"));
                }
                Some(st.as_cuda().ok_or(SynaptixError::Unsupported("conv2d_implicit: temb non-cuda"))?)
            }
            None => None,
        };
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::implicit_conv::ImplicitConvKernels::for_context(&ctx)?;
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let f_slice = f_buf.slice();
        let f_off = f_lo.byte_offset();
        let bias_s = bias_buf.map(|b| b.slice());
        let res_s = res_buf.map(|b| b.slice());
        let temb_s = temb_buf.map(|b| b.slice());
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("conv2d_implicit: out non-cuda"))?;
        crate::conv::implicit_conv::run_conv2d_implicit_nhwc_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            f_slice,
            f_off,
            out_buf.slice_mut(),
            0,
            n as u32,
            h as u32,
            w as u32,
            c as u32,
            k as u32,
            r as u32,
            s as u32,
            p as u32,
            q as u32,
            padding.0 as u32,
            padding.1 as u32,
            stride.0 as u32,
            stride.1 as u32,
            bias_s,
            res_s,
            temb_s,
            out_nhwc,
            dtype,
        )
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
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = out2d;
        let (out_st, _out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda conv_epilogue: dtype"));
        }
        if !in_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if let Some((_, b_lo)) = bias {
            if b_lo.dtype() != dtype || b_lo.dims().len() != 1 || b_lo.dims()[0] != c {
                return Err(SynaptixError::Unsupported(
                    "cuda conv_epilogue: bias shape/dtype",
                ));
            }
            if !b_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        if let Some((_, r_lo)) = residual {
            if r_lo.dtype() != dtype || r_lo.dims().len() != 4 || r_lo.dims() != [b, c, h, w] {
                return Err(SynaptixError::Unsupported(
                    "cuda conv_epilogue: residual shape/dtype",
                ));
            }
            if !r_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        if let Some((_, t_lo)) = temb_bc {
            if t_lo.dtype() != dtype || t_lo.dims().len() != 2 || t_lo.dims() != [b, c] {
                return Err(SynaptixError::Unsupported(
                    "cuda conv_epilogue: temb shape/dtype",
                ));
            }
            if !t_lo.is_contiguous() {
                return Err(SynaptixError::NonContiguous);
            }
        }
        let in_buf = in_st.as_cuda().ok_or(SynaptixError::Unsupported(
            "cuda conv_epilogue: in non-cuda",
        ))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::epilogue::ConvEpilogueKernels::for_context(&ctx)?;
        let bias_arg = match bias {
            Some((b_st, b_lo)) => {
                let b_buf = b_st.as_cuda().ok_or(SynaptixError::Unsupported(
                    "cuda conv_epilogue: bias non-cuda",
                ))?;
                Some((b_buf.slice(), b_lo.byte_offset()))
            }
            None => None,
        };
        let res_arg = match residual {
            Some((r_st, r_lo)) => {
                let r_buf = r_st.as_cuda().ok_or(SynaptixError::Unsupported(
                    "cuda conv_epilogue: residual non-cuda",
                ))?;
                Some((r_buf.slice(), r_lo.byte_offset()))
            }
            None => None,
        };
        let temb_arg = match temb_bc {
            Some((t_st, t_lo)) => {
                let t_buf = t_st.as_cuda().ok_or(SynaptixError::Unsupported(
                    "cuda conv_epilogue: temb non-cuda",
                ))?;
                Some((t_buf.slice(), t_lo.byte_offset()))
            }
            None => None,
        };
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let out_buf = out_st.as_cuda_mut().ok_or(SynaptixError::Unsupported(
            "cuda conv_epilogue: out non-cuda",
        ))?;
        crate::conv::epilogue::run_conv_epilogue_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            bias_arg,
            res_arg,
            temb_arg,
            out_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            h as u32,
            w as u32,
            dtype,
        )
    }

    fn geglu_split(
        &self,
        inp: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = inp;
        let (out_st, _out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda geglu_split: dtype"));
        }
        if !in_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let last = *d
            .last()
            .ok_or(SynaptixError::Unsupported("cuda geglu_split: scalar"))?;
        if last % 2 != 0 {
            return Err(SynaptixError::Unsupported("cuda geglu_split: last dim odd"));
        }
        let inner = last / 2;
        let t = in_lo.numel() / last.max(1);
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda geglu_split: in non-cuda"))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::fused::geglu_split::GegluSplitKernels::for_context(&ctx)?;
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda geglu_split: out non-cuda"))?;
        crate::fused::geglu_split::run_geglu_split_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            out_buf.slice_mut(),
            0,
            t as u64,
            inner as u32,
            dtype,
        )
    }

    fn snake(
        &self,
        x: (&Storage, &Layout),
        a: (&Storage, &Layout),
        binv: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        c: usize,
        t_len: usize,
        _stream: &Stream,
    ) -> Result<()> {
        let (x_st, x_lo) = x;
        let (a_st, a_lo) = a;
        let (binv_st, binv_lo) = binv;
        let (out_st, _out_lo) = out;
        let dtype = x_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda snake: dtype"));
        }
        if !x_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if a_lo.dtype() != DType::F32 || binv_lo.dtype() != DType::F32 {
            return Err(SynaptixError::Unsupported("cuda snake: a/binv must be f32"));
        }
        let n = x_lo.numel();
        let x_buf = x_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda snake: x non-cuda"))?;
        let a_buf = a_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda snake: a non-cuda"))?;
        let binv_buf = binv_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda snake: binv non-cuda"))?;
        let ctx = x_buf.device().clone();
        let ord = x_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::elementwise::activations::ActivationsKernels::for_context(&ctx)?;
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda snake: out non-cuda"))?;
        crate::elementwise::activations::run_snake_u8(
            &kernels,
            &stream,
            x_buf.slice(),
            x_lo.byte_offset(),
            a_buf.slice(),
            a_lo.byte_offset(),
            binv_buf.slice(),
            binv_lo.byte_offset(),
            out_buf.slice_mut(),
            0,
            n,
            c as u32,
            t_len as u32,
            dtype,
        )
    }

    fn upsample_nearest2x(
        &self,
        input: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let (in_st, in_lo) = input;
        let (out_st, _out_lo) = out;
        let dtype = in_lo.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("cuda upsample2x: dtype"));
        }
        if in_lo.dims().len() != 4 {
            return Err(SynaptixError::Unsupported("cuda upsample2x: rank != 4"));
        }
        if !in_lo.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let d = in_lo.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let in_buf = in_st
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda upsample2x: in non-cuda"))?;
        let ctx = in_buf.device().clone();
        let ord = in_buf.ordinal();
        let stream = synaptix_core::device::cuda::default_stream(ord)?;
        let kernels = crate::conv::upsample::Upsample2xKernels::for_context(&ctx)?;
        let in_slice = in_buf.slice();
        let in_off = in_lo.byte_offset();
        let out_buf = out_st
            .as_cuda_mut()
            .ok_or(SynaptixError::Unsupported("cuda upsample2x: out non-cuda"))?;
        crate::conv::upsample::run_upsample2x_u8(
            &kernels,
            &stream,
            in_slice,
            in_off,
            out_buf.slice_mut(),
            0,
            b as u32,
            c as u32,
            h as u32,
            w as u32,
            dtype,
        )
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        dims: &[usize],
        _stream: &Stream,
    ) -> Result<()> {
        let ctx = src
            .0
            .as_cuda()
            .ok_or(SynaptixError::Unsupported("cuda reduce: src non-cuda"))?
            .device()
            .clone();
        let kernels = crate::kernels::reduce::ReduceKernels::for_context(&ctx)?;
        crate::kernels::reduce::run_reduce(&kernels, op, src, dst, dims)
    }
}
