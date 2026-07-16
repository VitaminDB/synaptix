#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use rayon::prelude::*;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CudaBuf, Storage};
use synaptix_core::tensor::Tensor;

use synaptix_kernels_cuda::elementwise::quant::{mxfp8_quant_natural, Mxfp8QuantKernels};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    synaptix_kernels_cuda::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
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

/// MXFP8-вес (natural e4m3 [N,K] + natural E8M0 scales [N,K/32]) → QuantWeight →
/// публичный `Tensor::linear_quant`. M=1 идёт в gemv_mxfp8, M>1 — деквант W→f16 +
/// f16 TN-linear. Сверка с плотным F16@Wᵀ (CPU f32, по исходным весам).
fn run_mxfp8(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, n: usize, k: usize, m: usize) {
    assert_eq!(k % 32, 0, "K%32 для MXFP8");
    // prefill (m>1) идёт через корректное v1-ядро (cp.async), m=1 — gemv.
    let q = Mxfp8QuantKernels::for_context(ctx).expect("compile mxfp8_quant");

    let w_host = det_f16(0xA110_C8E1, n * k, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, m * k, 0.5);

    // Квант веса на device: e4m3 [N,K] + natural E8M0 scales [N,K/32].
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let mut w_fp8: CudaSlice<u8> = stream.alloc_zeros(n * k).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(n * (k / 32)).unwrap();
    mxfp8_quant_natural(
        &q,
        stream,
        &dev_w.as_view(),
        &mut w_fp8,
        &mut w_scales,
        n as u32,
        k as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let qw = QuantWeight::new(
        Arc::new(Storage::Cuda(CudaBuf::new(
            ctx.clone(),
            stream.clone(),
            w_fp8,
            0,
        ))),
        Arc::new(Storage::Cuda(CudaBuf::new(
            ctx.clone(),
            stream.clone(),
            w_scales,
            0,
        ))),
        DType::MXFP8,
        n,
        k,
    )
    .unwrap();
    assert_eq!(qw.dtype(), DType::MXFP8);
    assert_eq!((qw.n(), qw.k()), (n, k));

    let x_tensor = Tensor::from_vec(x_host.clone(), (m, k), Device::Cuda(0)).unwrap();
    let out = x_tensor.linear_quant(&qw).expect("linear_quant MXFP8");
    assert_eq!(out.dims(), &[m, n], "out dims");

    let out_f32 = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Плотный reference: out[m,n] = sum_k x[m,k]*w[n,k] (по исходным f16).
    let w_f32: Vec<f32> = w_host.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x_host.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; m * n];
    y_ref.par_chunks_mut(n).enumerate().for_each(|(b, row)| {
        let x_row = &x_f32[b * k..(b + 1) * k];
        for (o, slot) in row.iter_mut().enumerate() {
            let w_row = &w_f32[o * k..(o + 1) * k];
            let mut acc = 0.0_f32;
            for j in 0..k {
                acc += w_row[j] * x_row[j];
            }
            *slot = acc;
        }
    });

    let cs = cos_sim(&out_f32, &y_ref);
    let path = if m == 1 { "gemv" } else { "v1" };
    eprintln!("[mxfp8 linear_quant N={n} K={k} M={m} path={path}] cos={cs:.6}");
    assert!(cs >= 0.99, "MXFP8 linear_quant M={m} cos={cs} < 0.99");
}

#[test]
fn linear_quant_mxfp8_gemv_m1() {
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 512, 256, 1);
}

#[test]
fn linear_quant_mxfp8_prefill_m8() {
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 512, 256, 8);
}

#[test]
fn linear_quant_mxfp8_prefill_m128() {
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 768, 512, 128);
}

/// Публичный путь loader-а модели: `Tensor::quantize_to_mxfp8` (F16-вес →
/// QuantWeight) + `linear_quant`, сверка с плотным F16@Wᵀ.
#[test]
fn tensor_quantize_to_mxfp8_roundtrip() {
    let Some((_ctx, stream)) = setup() else { return };
    let (n, k, m) = (512usize, 256usize, 4usize);
    let w_host = det_f16(0xBEEF_1234, n * k, 0.5);
    let x_host = det_f16(0x1357_9BDF, m * k, 0.5);

    let w = Tensor::from_vec(w_host.clone(), (n, k), Device::Cuda(0)).unwrap();
    let x = Tensor::from_vec(x_host.clone(), (m, k), Device::Cuda(0)).unwrap();

    let qw = w.quantize_to_mxfp8().expect("quantize_to_mxfp8");
    assert_eq!(qw.dtype(), DType::MXFP8);
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
    assert!(cs >= 0.99, "MXFP8 quantize_to roundtrip cos={cs} < 0.99");
    let _ = stream;
}

/// Невыровненный M>1 (200, не кратно bm) — проверка M-паддинга tiled-пути vs dense.
#[test]
fn linear_quant_mxfp8_prefill_unaligned_m200() {
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 768, 512, 200);
}

/// ROW-CONSISTENCY (класс бага NVFP4 «chunked≠single→потеря контекста»): выход строки i
/// НЕ должен зависеть от total-M. Считаем общие 128 строк двумя способами — батч M=128 и
/// первые 128 строк батча M=256 — и требуем BIT-идентичности (одна квант-активация строки,
/// одна scale-позиция вне зависимости от outer-M). v1 MXFP8 prefill.
#[test]
fn linear_quant_mxfp8_row_consistency() {
    let Some((ctx, stream)) = setup() else { return };
    let (n, k) = (768usize, 512usize);
    let q = Mxfp8QuantKernels::for_context(&ctx).expect("compile mxfp8_quant");
    let w_host = det_f16(0xA110_C8E1, n * k, 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let mut w_fp8: CudaSlice<u8> = stream.alloc_zeros(n * k).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(n * (k / 32)).unwrap();
    mxfp8_quant_natural(&q, &stream, &dev_w.as_view(), &mut w_fp8, &mut w_scales, n as u32, k as u32)
        .unwrap();
    stream.synchronize().unwrap();
    let qw = QuantWeight::new(
        Arc::new(Storage::Cuda(CudaBuf::new(ctx.clone(), stream.clone(), w_fp8, 0))),
        Arc::new(Storage::Cuda(CudaBuf::new(ctx.clone(), stream.clone(), w_scales, 0))),
        DType::MXFP8,
        n,
        k,
    )
    .unwrap();

    // Общий пул активаций на 256 строк; первые 128 — общие.
    let x_all = det_f16(0xC0DE_BA5E, 256 * k, 0.5);
    let run = |m: usize| -> Vec<f32> {
        let x = Tensor::from_vec(x_all[..m * k].to_vec(), (m, k), Device::Cuda(0)).unwrap();
        x.linear_quant(&qw)
            .expect("linear_quant")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let out128 = run(128);
    let out256 = run(256);
    let mut worst = 0.0_f32;
    for i in 0..128 * n {
        worst = worst.max((out128[i] - out256[i]).abs());
    }
    eprintln!("[mxfp8 row-consistency] M=128 vs первые 128 строк M=256 | worst |Δ|={worst}");
    assert_eq!(worst, 0.0, "MXFP8 tiled: выход строки зависит от total-M (worst={worst})");
}

#[test]
fn linear_quant_mxfp8_ktail_m1() {
    // K не кратно 32 запрещён квантом; но K=288 (=32*9) кратно 32, проверяем
    // нестандартную K вне степени двойки + gemv путь.
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 320, 288, 1);
}

/// Модельная форма (k=2048 как hidden 1.7B): проверка cos tiled-пути на крупном K
/// (тест выше шёл лишь до k=512). Высокий cos → GEMM механически верен на форме модели.
#[test]
fn linear_quant_mxfp8_model_shape_k2048() {
    let Some((ctx, stream)) = setup() else { return };
    run_mxfp8(&ctx, &stream, 2048, 2048, 256);
}
