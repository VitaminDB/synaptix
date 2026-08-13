
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use rayon::prelude::*;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CudaBuf, Storage};
use synaptix_core::tensor::Tensor;

use synaptix_kernels_cuda::best_cu::gemm::gemm_nvfp4::Nvfp4FullCfg;
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_dequant_f16, nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};
use synaptix_kernels_cuda::gemm::dispatch::{pick_nvfp4, Nvfp4Plan};

// ─────────────────────────── pick_nvfp4 (без GPU) ───────────────────────────

#[test]
fn pick_qwen_shapes() {
    // M≥128 (prefill): best_cu Full (0.81–0.86× cuBLASLt, first-party). Выбор swizzle vs
    // persistent по тому, влезает ли FP4-вес N*K/2 в L2 (порог 24MB).
    let swz = Nvfp4Plan::Full(Nvfp4FullCfg::C_128_128_C256_S4_SWZ);
    let persist = Nvfp4Plan::Full(Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ);

    // attn_qkv N=5120 K=5120 — вес 13MB ≤ L2 → swizzle.
    assert_eq!(pick_nvfp4(1, 5120, 5120), Nvfp4Plan::Gemv);
    assert_eq!(pick_nvfp4(16, 5120, 5120), Nvfp4Plan::N8);
    // m 32-96: Full b64 (TMA-OOB; events-свип 2026-06-05 бьёт их pure и N8).
    assert_eq!(pick_nvfp4(64, 5120, 5120), Nvfp4Plan::Full(Nvfp4FullCfg::C_128_64_S4_SWZ));
    // m 129-320 узкий N → c256_s4_drot (events-свип 2026-06-05).
    let swz_drot = Nvfp4Plan::Full(Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT);
    assert_eq!(pick_nvfp4(256, 5120, 5120), swz_drot);

    // ffn_gate N=27648 K=5120 — вес 70MB > L2 → persistent L2-raster.
    assert_eq!(pick_nvfp4(1, 27648, 5120), Nvfp4Plan::Gemv);
    assert_eq!(pick_nvfp4(256, 27648, 5120), persist);

    // ffn_down N=5120 K=27648 — k>n (ff_down-класс): 128×128 вдвое быстрее persist
    // (A/B 2026-06-04, weight_fits_l2 |= k>n).
    assert_eq!(pick_nvfp4(1, 5120, 27648), Nvfp4Plan::Gemv);
    assert_eq!(pick_nvfp4(16, 5120, 27648), Nvfp4Plan::N8);
    assert_eq!(pick_nvfp4(64, 5120, 27648), Nvfp4Plan::Full(Nvfp4FullCfg::C_128_64_S4_SWZ));
    assert_eq!(pick_nvfp4(256, 5120, 27648), swz_drot);
    // m≥512 кратно 256 → ROT (порт CUTLASS sm120, default с 2026-06-04).
    assert_eq!(
        pick_nvfp4(512, 5120, 5120),
        Nvfp4Plan::Full(Nvfp4FullCfg::C_128_256_S3_SWZ_ROT)
    );

    // lm_head N=248320 K=5120 — вес 635MB > L2 → persistent.
    assert_eq!(pick_nvfp4(1, 248320, 5120), Nvfp4Plan::Gemv);
    assert_eq!(pick_nvfp4(256, 248320, 5120), persist);
}

#[test]
fn pick_fallbacks() {
    // n%64==0, batch не подходит 2dr/n8 → Broadcast (4-A любой batch).
    assert_eq!(pick_nvfp4(3, 64, 64), Nvfp4Plan::Broadcast);
    // n%64!=0, n%32==0 → 2D cooperative.
    assert_eq!(pick_nvfp4(16, 96, 64), Nvfp4Plan::Coop);
    // huge-K, M=64 → Full b64 (порог Full = 32 с 2026-06-05).
    assert_eq!(pick_nvfp4(64, 5120, 27648), Nvfp4Plan::Full(Nvfp4FullCfg::C_128_64_S4_SWZ));
}

// ─────────────────────────── числовой путь (GPU) ───────────────────────────

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    synaptix_kernels_cuda::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f16::from_f32(f * scale)
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

/// Phase 2: квант веса через публичный `Tensor::quantize_to_nvfp4` (путь loader-а
/// модели) + `linear_quant`, сверка с плотным F16 @ Wᵀ (CPU f32 reference).
#[test]
fn tensor_quantize_to_nvfp4_roundtrip() {
    let Some((_ctx, stream)) = setup() else {
        return;
    };
    let (n, k, m) = (512usize, 256usize, 4usize);
    let w_host = det_f16(0xBEEF_1234, n * k, 0.5);
    let x_host = det_f16(0x1357_9BDF, m * k, 0.5);

    let w = Tensor::from_vec(w_host.clone(), (n, k), Device::Cuda(0)).unwrap();
    let x = Tensor::from_vec(x_host.clone(), (m, k), Device::Cuda(0)).unwrap();

    // Публичный путь модели: F16-вес → QuantWeight (квант на загрузке).
    let qw = w.quantize_to_nvfp4().expect("quantize_to_nvfp4");
    assert_eq!(qw.dtype(), DType::NVFP4);
    assert_eq!((qw.n(), qw.k()), (n, k));

    let out = x.linear_quant(&qw).expect("linear_quant");
    assert_eq!(out.dims(), &[m, n]);
    let out_f32 = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Плотный reference: out[m,n] = sum_k x[m,k]*w[n,k].
    let mut dense = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x_host[mi * k + ki].to_f32() * w_host[ni * k + ki].to_f32();
            }
            dense[mi * n + ni] = acc;
        }
    }
    let cs = cos_sim(&out_f32, &dense);
    assert!(cs >= 0.99, "NVFP4 roundtrip cos_sim={cs} < 0.99");

    let _ = stream;
}

/// Прогон через публичный `Tensor::linear_quant` + сверка с CPU-dequant reference.
fn run_dispatch(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n: u32,
    k: u32,
    m: u32,
    expected: Nvfp4Plan,
    name: &str,
) {
    assert_eq!(pick_nvfp4(m, n, k), expected, "{name}: план");

    let q = Nvfp4QuantKernels::for_context(ctx).expect("compile nvfp4_quant");

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (m * k) as usize, 0.5);

    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let w_scale_bytes = nvfp4_scale_buffer_size(n as usize, k as usize);
    let x_scale_bytes = nvfp4_scale_buffer_size(m as usize, k as usize);

    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(w_scale_bytes).unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((m * k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> = stream.alloc_zeros(x_scale_bytes).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_packed, &mut x_scales, m, k).unwrap();

    // linear_quant внутри безусловно освобождает packed-вес (release_packed, OOM-fix:
    // shuffled-W — единственный читаемый формат). Поэтому дека́нтим эталонный вес ЗДЕСЬ,
    // пока локальные w_packed/w_scales ещё живы (до перемещения в QuantWeight).
    let mut w_deq: CudaSlice<f16> = stream.alloc_zeros((n * k) as usize).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_packed, &w_scales, &mut w_deq.as_view_mut(), n, k).unwrap();
    stream.synchronize().unwrap();
    let w_deq_host: Vec<f16> = stream.clone_dtoh(&w_deq).unwrap();

    // QuantWeight владеет packed/scales (доступ к ним для эталонов — через qw).
    let qw = QuantWeight::new(
        Arc::new(Storage::Cuda(CudaBuf::new(
            ctx.clone(),
            stream.clone(),
            w_packed,
            0,
        ))),
        Arc::new(Storage::Cuda(CudaBuf::new(
            ctx.clone(),
            stream.clone(),
            w_scales,
            0,
        ))),
        DType::NVFP4,
        n as usize,
        k as usize,
    )
    .unwrap();

    // Публичный путь: Tensor::linear_quant (X квантуется внутри dispatch).
    let x_tensor =
        Tensor::from_vec(x_host.clone(), (m as usize, k as usize), Device::Cuda(0)).unwrap();
    let out = x_tensor.linear_quant(&qw).unwrap();
    assert_eq!(out.dims(), &[m as usize, n as usize], "{name}: out dims");

    // x дека́нтим из локальных буферов (в qw не перемещались, packed-вес уже освобождён).
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros((m * k) as usize).unwrap();
    nvfp4_dequant_f16(&q, stream, &x_packed, &x_scales, &mut x_deq.as_view_mut(), m, k).unwrap();
    stream.synchronize().unwrap();
    let x_deq_host: Vec<f16> = stream.clone_dtoh(&x_deq).unwrap();
    let out_bytes: Vec<u8> = stream
        .clone_dtoh(out.storage().as_cuda().unwrap().slice())
        .unwrap();
    let y_ours_f32: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&out_bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();

    // Наивный O(M·N·K) эталон на Qwen-формах (K=27648, N=5120, M=256 ≈ 36 млрд MAC)
    // в один поток молотит минутами — распараллеливаем по выходным строкам через
    // rayon (24 ядра → ~секунда). Пред-конвертим dequant в f32 один раз, чтобы во
    // внутреннем цикле не было f16→f32 на каждый элемент.
    let k_us = k as usize;
    let n_us = n as usize;
    let w_f32: Vec<f32> = w_deq_host.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x_deq_host.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; (m * n) as usize];
    y_ref.par_chunks_mut(n_us).enumerate().for_each(|(b, row)| {
        let x_row = &x_f32[b * k_us..(b + 1) * k_us];
        for (o, slot) in row.iter_mut().enumerate() {
            let w_row = &w_f32[o * k_us..(o + 1) * k_us];
            let mut acc = 0.0_f32;
            for j in 0..k_us {
                acc += w_row[j] * x_row[j];
            }
            *slot = acc;
        }
    });
    let cos_ref = cos_sim(&y_ours_f32, &y_ref);
    eprintln!(
        "[{name} N={n} K={k} M={m} plan={}] vs CPU cos={cos_ref:.6}",
        expected.name()
    );
    assert!(cos_ref >= 0.99, "{name}: cos vs CPU={cos_ref} < 0.99");
}

#[test]
fn dispatch_attn_qkv_gemv_m1() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(&ctx, &stream, 5120, 5120, 1, Nvfp4Plan::Gemv, "attn_qkv_m1");
}

#[test]
fn dispatch_attn_qkv_n8_m16() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(&ctx, &stream, 5120, 5120, 16, Nvfp4Plan::N8, "attn_qkv_m16");
}

#[test]
fn dispatch_attn_qkv_reg_m64() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(
        &ctx,
        &stream,
        5120,
        5120,
        64,
        Nvfp4Plan::Full(Nvfp4FullCfg::C_128_64_S4_SWZ),
        "attn_qkv_m64",
    );
}

#[test]
fn dispatch_attn_qkv_full_swz_m256() {
    let Some((ctx, stream)) = setup() else { return };
    // attn (вес 13MB ≤ L2) → best_cu Full swizzle.
    run_dispatch(
        &ctx,
        &stream,
        5120,
        5120,
        256,
        Nvfp4Plan::Full(Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT),
        "attn_qkv_m256",
    );
}

#[test]
fn dispatch_ffn_down_reg_m64() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(
        &ctx,
        &stream,
        5120,
        27648,
        64,
        Nvfp4Plan::Full(Nvfp4FullCfg::C_128_64_S4_SWZ),
        "ffn_down_m64",
    );
}

#[test]
fn dispatch_ffn_down_full_persist_m256() {
    let Some((ctx, stream)) = setup() else { return };
    // ffn_down k>n → 128×128 swizzle (вдвое быстрее persist, A/B 2026-06-04).
    run_dispatch(
        &ctx,
        &stream,
        5120,
        27648,
        256,
        Nvfp4Plan::Full(Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT),
        "ffn_down_m256",
    );
}

#[test]
fn dispatch_broadcast_m3() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(
        &ctx,
        &stream,
        128,
        64,
        3,
        Nvfp4Plan::Broadcast,
        "broadcast_m3",
    );
}

#[test]
fn dispatch_coop_m16_n96() {
    let Some((ctx, stream)) = setup() else { return };
    run_dispatch(&ctx, &stream, 96, 64, 16, Nvfp4Plan::Coop, "coop_m16_n96");
}
