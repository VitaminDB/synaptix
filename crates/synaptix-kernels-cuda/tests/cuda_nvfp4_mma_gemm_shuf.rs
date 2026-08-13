
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use rayon::prelude::*;

use synaptix_kernels_cuda::best_cu::gemm::gemm_nvfp4::{
    gemm_nvfp4_full_cfg_view, nvfp4_mma_gemm_shuf_2d_f16, nvfp4_mma_gemm_shuf_2dr_f16,
    nvfp4_mma_gemm_shuf_f16,
    nvfp4_mma_gemm_shuf_n8_f16, GemmNvfp4FullKernels,
    Gemm2drConfig, Nvfp4FullCfg, Nvfp4MmaGemmShufKernels,
};
use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{nvfp4_w_repack, Nvfp4MmaGemvShufKernels};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_dequant_f16, nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
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

#[derive(Clone, Copy)]
enum Variant {
    /// Session 4-A — broadcast B over batch.grid.y.
    A,
    /// Session 4-B — native MMA n=8 throughput.
    N8,
    /// Session 4-C — cooperative 2D tiling.
    D2,
    /// Session 6-R — register-blocked warp-tile (MU×NU MMA, reuse A/B).
    D2R,
}

fn run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n: u32,
    k: u32,
    batch: u32,
    variant: Variant,
    name: &str,
    cos_floor: f32,
) {
    let q = Nvfp4QuantKernels::for_context(ctx).expect("compile nvfp4_quant");
    let gemv = Nvfp4MmaGemvShufKernels::for_context(ctx).expect("compile gemv shuf");
    let gemm = Nvfp4MmaGemmShufKernels::for_context(ctx).expect("compile gemm shuf");

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch * k) as usize, 0.5);

    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let w_scale_bytes = nvfp4_scale_buffer_size(n as usize, k as usize);
    let x_scale_bytes = nvfp4_scale_buffer_size(batch as usize, k as usize);

    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(w_scale_bytes).unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((batch * k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> = stream.alloc_zeros(x_scale_bytes).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_packed, &mut x_scales, batch, k).unwrap();

    let mut w_packed_shuf: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    nvfp4_w_repack(&gemv, stream, &w_packed, &mut w_packed_shuf, n, k).expect("repack");

    let mut y_ours: CudaSlice<f16> = stream.alloc_zeros((batch * n) as usize).unwrap();
    match variant {
        Variant::A => {
            nvfp4_mma_gemm_shuf_f16(
                &gemm,
                stream,
                &w_packed_shuf,
                &w_scales,
                &x_packed,
                &x_scales,
                &mut y_ours,
                n,
                k,
                batch,
            )
            .expect("shuf gemm 4-A");
        }
        Variant::N8 => {
            nvfp4_mma_gemm_shuf_n8_f16(
                &gemm,
                stream,
                &w_packed_shuf,
                &w_scales,
                &x_packed,
                &x_scales,
                &mut y_ours,
                n,
                k,
                batch,
            )
            .expect("shuf gemm 4-B n8");
        }
        Variant::D2 => {
            nvfp4_mma_gemm_shuf_2d_f16(
                &gemm,
                stream,
                &w_packed_shuf,
                &w_scales,
                &x_packed,
                &x_scales,
                &mut y_ours,
                n,
                k,
                batch,
            )
            .expect("shuf gemm 4-C 2d");
        }
        Variant::D2R => {
            nvfp4_mma_gemm_shuf_2dr_f16(
                &gemm,
                stream,
                &w_packed_shuf,
                &w_scales,
                &x_packed,
                &x_scales,
                &mut y_ours,
                n,
                k,
                batch,
            )
            .expect("shuf gemm 6-R 2dr");
        }
    }

    // CPU F32 reference через dequant.
    let mut w_deq: CudaSlice<f16> = stream.alloc_zeros((n * k) as usize).unwrap();
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros((batch * k) as usize).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_packed, &w_scales, &mut w_deq.as_view_mut(), n, k).unwrap();
    nvfp4_dequant_f16(&q, stream, &x_packed, &x_scales, &mut x_deq.as_view_mut(), batch, k).unwrap();
    stream.synchronize().unwrap();

    let w_deq_host: Vec<f16> = stream.clone_dtoh(&w_deq).unwrap();
    let x_deq_host: Vec<f16> = stream.clone_dtoh(&x_deq).unwrap();
    let y_ours_host: Vec<f16> = stream.clone_dtoh(&y_ours).unwrap();

    // Наивный O(M·N·K) на Qwen shapes (K=27648, N=5120, M=256 ≈ 36 млрд MAC)
    // в один поток молотит минутами — распараллеливаем по выходным строкам через
    // rayon (24 ядра → секунды). Пред-конвертим dequant в f32 один раз, чтобы во
    // внутреннем цикле не было f16→f32 на каждый элемент.
    let k_us = k as usize;
    let n_us = n as usize;
    let w_f32: Vec<f32> = w_deq_host.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x_deq_host.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; (batch * n) as usize];
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
    let y_ours_f32: Vec<f32> = y_ours_host.iter().map(|v| v.to_f32()).collect();

    let cos_ref = cos_sim(&y_ours_f32, &y_ref);
    let mut max_abs_ref = 0.0_f32;
    for i in 0..y_ours_f32.len() {
        let d_ref = (y_ours_f32[i] - y_ref[i]).abs();
        if d_ref > max_abs_ref {
            max_abs_ref = d_ref;
        }
    }
    eprintln!(
        "[{name} gemm N={n} K={k} M={batch}] vs CPU: cos={cos_ref:.6} max_abs={max_abs_ref:.4}"
    );
    assert!(
        cos_ref >= cos_floor,
        "{name}: cos vs CPU={cos_ref} < {cos_floor}"
    );
}

// ─── Session 4-A: broadcast over batch ───

#[test]
fn gemm_shuf_64x64_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        64,
        64,
        16,
        Variant::A,
        "small_64x64_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_128x128_b32() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 128, 128, 32, Variant::A, "128x128_b32", 0.99);
}

#[test]
fn gemm_shuf_256x256_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 256, 256, 64, Variant::A, "256x256_b64", 0.99);
}

#[test]
fn gemm_shuf_qwen_attn_qkv_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        16,
        Variant::A,
        "qwen_attn_qkv_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_qwen_attn_qkv_b256() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        256,
        Variant::A,
        "qwen_attn_qkv_b256",
        0.99,
    );
}

#[test]
fn gemm_shuf_qwen_ffn_gate_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        27648,
        5120,
        16,
        Variant::A,
        "qwen_ffn_gate_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_qwen_ffn_down_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        27648,
        16,
        Variant::A,
        "qwen_ffn_down_b16",
        0.99,
    );
}

// ─── Session 4-B: native n=8 throughput ───

#[test]
fn gemm_shuf_n8_64x64_b8() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        64,
        64,
        8,
        Variant::N8,
        "n8_small_64x64_b8",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_128x128_b32() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        128,
        128,
        32,
        Variant::N8,
        "n8_128x128_b32",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_256x256_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        256,
        256,
        64,
        Variant::N8,
        "n8_256x256_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_qwen_attn_qkv_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        16,
        Variant::N8,
        "n8_qwen_attn_qkv_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_qwen_attn_qkv_b256() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        256,
        Variant::N8,
        "n8_qwen_attn_qkv_b256",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_qwen_ffn_gate_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        27648,
        5120,
        16,
        Variant::N8,
        "n8_qwen_ffn_gate_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_n8_qwen_ffn_down_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        27648,
        16,
        Variant::N8,
        "n8_qwen_ffn_down_b16",
        0.99,
    );
}

// ─── Session 4-C: cooperative 2D tiling ───

#[test]
fn gemm_shuf_2d_128x128_b32() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        128,
        128,
        32,
        Variant::D2,
        "2d_128x128_b32",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_256x256_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        256,
        256,
        64,
        Variant::D2,
        "2d_256x256_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_qwen_attn_qkv_b16() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        16,
        Variant::D2,
        "2d_qwen_attn_qkv_b16",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_qwen_attn_qkv_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        64,
        Variant::D2,
        "2d_qwen_attn_qkv_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_qwen_attn_qkv_b256() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        256,
        Variant::D2,
        "2d_qwen_attn_qkv_b256",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_qwen_ffn_gate_b32() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        27648,
        5120,
        32,
        Variant::D2,
        "2d_qwen_ffn_gate_b32",
        0.99,
    );
}

#[test]
fn gemm_shuf_2d_qwen_ffn_down_b32() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        27648,
        32,
        Variant::D2,
        "2d_qwen_ffn_down_b32",
        0.99,
    );
}

// ─── Session 6-R: register-blocked warp-tile (2dr) ───

#[test]
fn gemm_shuf_2dr_128x128_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        128,
        128,
        64,
        Variant::D2R,
        "2dr_128x128_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_2dr_qwen_attn_qkv_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        64,
        Variant::D2R,
        "2dr_qwen_attn_qkv_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_2dr_qwen_attn_qkv_b256() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        5120,
        256,
        Variant::D2R,
        "2dr_qwen_attn_qkv_b256",
        0.99,
    );
}

#[test]
fn gemm_shuf_2dr_qwen_ffn_gate_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        27648,
        5120,
        64,
        Variant::D2R,
        "2dr_qwen_ffn_gate_b64",
        0.99,
    );
}

#[test]
fn gemm_shuf_2dr_qwen_ffn_down_b64() {
    let Some((ctx, stream)) = setup() else { return };
    run(
        &ctx,
        &stream,
        5120,
        27648,
        64,
        Variant::D2R,
        "2dr_qwen_ffn_down_b64",
        0.99,
    );
}







// ─── DIAGNOSTIC: per-row max-abs A vs N8 vs 2dr at identical M=256 ───

fn per_row_maxabs(y: &[f32], r: &[f32], n: usize) -> (f64, f64) {
    let rows = y.len() / n;
    let mut sum = 0.0_f64;
    let mut worst = 0.0_f64;
    for row in 0..rows {
        let mut m = 0.0_f64;
        for c in 0..n {
            let d = (y[row * n + c] - r[row * n + c]).abs() as f64;
            if d > m {
                m = d;
            }
        }
        sum += m;
        if m > worst {
            worst = m;
        }
    }
    (sum / rows as f64, worst)
}

fn run_plan_y(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    w_packed: &CudaSlice<u8>,
    w_packed_shuf: &CudaSlice<u8>,
    w_scales: &CudaSlice<u8>,
    x_packed: &CudaSlice<u8>,
    x_scales: &CudaSlice<u8>,
    gemm: &Nvfp4MmaGemmShufKernels,
    n: u32,
    k: u32,
    batch: u32,
    variant: Variant,
) -> Vec<f32> {
    let mut y: CudaSlice<f16> = stream.alloc_zeros((batch * n) as usize).unwrap();
    match variant {
        Variant::A => nvfp4_mma_gemm_shuf_f16(
            gemm, stream, w_packed_shuf, w_scales, x_packed, x_scales, &mut y, n, k, batch,
        )
        .unwrap(),
        Variant::N8 => nvfp4_mma_gemm_shuf_n8_f16(
            gemm, stream, w_packed_shuf, w_scales, x_packed, x_scales, &mut y, n, k, batch,
        )
        .unwrap(),
        Variant::D2R => nvfp4_mma_gemm_shuf_2dr_f16(
            gemm, stream, w_packed_shuf, w_scales, x_packed, x_scales, &mut y, n, k, batch,
        )
        .unwrap(),
        Variant::D2 => nvfp4_mma_gemm_shuf_2d_f16(
            gemm, stream, w_packed_shuf, w_scales, x_packed, x_scales, &mut y, n, k, batch,
        )
        .unwrap(),
        _ => unreachable!(),
    }
    let _ = w_packed;
    stream.synchronize().unwrap();
    let h: Vec<f16> = stream.clone_dtoh(&y).unwrap();
    h.iter().map(|v| v.to_f32()).collect()
}

#[test]
fn diag_perrow_a_vs_n8_vs_2dr_m256() {
    let Some((ctx, stream)) = setup() else { return };
    let (n, k, batch) = (5120u32, 5120u32, 256u32);

    let q = Nvfp4QuantKernels::for_context(&ctx).expect("compile nvfp4_quant");
    let gemv = Nvfp4MmaGemvShufKernels::for_context(&ctx).expect("compile gemv shuf");
    let gemm = Nvfp4MmaGemmShufKernels::for_context(&ctx).expect("compile gemm shuf");

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch * k) as usize, 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let w_scale_bytes = nvfp4_scale_buffer_size(n as usize, k as usize);
    let x_scale_bytes = nvfp4_scale_buffer_size(batch as usize, k as usize);
    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(w_scale_bytes).unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((batch * k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> = stream.alloc_zeros(x_scale_bytes).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_x, &mut x_packed, &mut x_scales, batch, k).unwrap();
    let mut w_packed_shuf: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    nvfp4_w_repack(&gemv, &stream, &w_packed, &mut w_packed_shuf, n, k).expect("repack");

    // CPU F32 ref from dequant.
    let mut w_deq: CudaSlice<f16> = stream.alloc_zeros((n * k) as usize).unwrap();
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros((batch * k) as usize).unwrap();
    nvfp4_dequant_f16(&q, &stream, &w_packed, &w_scales, &mut w_deq.as_view_mut(), n, k).unwrap();
    nvfp4_dequant_f16(&q, &stream, &x_packed, &x_scales, &mut x_deq.as_view_mut(), batch, k).unwrap();
    stream.synchronize().unwrap();
    let w_f32: Vec<f32> = stream
        .clone_dtoh(&w_deq)
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let x_f32: Vec<f32> = stream
        .clone_dtoh(&x_deq)
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let (k_us, n_us) = (k as usize, n as usize);
    let mut y_ref = vec![0.0_f32; (batch * n) as usize];
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
    let mean_ref_abs: f64 =
        y_ref.iter().map(|v| v.abs() as f64).sum::<f64>() / y_ref.len() as f64;

    let y_a = run_plan_y(
        &ctx, &stream, &w_packed, &w_packed_shuf, &w_scales, &x_packed, &x_scales, &gemm, n, k,
        batch, Variant::A,
    );
    let y_n8 = run_plan_y(
        &ctx, &stream, &w_packed, &w_packed_shuf, &w_scales, &x_packed, &x_scales, &gemm, n, k,
        batch, Variant::N8,
    );
    let y_2dr = run_plan_y(
        &ctx, &stream, &w_packed, &w_packed_shuf, &w_scales, &x_packed, &x_scales, &gemm, n, k,
        batch, Variant::D2R,
    );
    let y_2d = run_plan_y(
        &ctx, &stream, &w_packed, &w_packed_shuf, &w_scales, &x_packed, &x_scales, &gemm, n, k,
        batch, Variant::D2,
    );

    let (a_mean, a_worst) = per_row_maxabs(&y_a, &y_ref, n_us);
    let (n8_mean, n8_worst) = per_row_maxabs(&y_n8, &y_ref, n_us);
    let (r_mean, r_worst) = per_row_maxabs(&y_2dr, &y_ref, n_us);
    let (d2_mean, d2_worst) = per_row_maxabs(&y_2d, &y_ref, n_us);
    let (_, an8_worst) = per_row_maxabs(&y_a, &y_n8, n_us);
    let (_, ar_worst) = per_row_maxabs(&y_a, &y_2dr, n_us);
    let (_, ad2_worst) = per_row_maxabs(&y_a, &y_2d, n_us);

    eprintln!("DIAG M=256 N=5120 K=5120 mean|ref|={mean_ref_abs:.3}");
    eprintln!("  A   (Broadcast) vs ref: mean_row_maxabs={a_mean:.4} worst={a_worst:.4}");
    eprintln!("  N8           vs ref: mean_row_maxabs={n8_mean:.4} worst={n8_worst:.4}");
    eprintln!("  2dr (Reg)    vs ref: mean_row_maxabs={r_mean:.4} worst={r_worst:.4}");
    eprintln!("  2d  (Coop)   vs ref: mean_row_maxabs={d2_mean:.4} worst={d2_worst:.4}");
    eprintln!("  A vs N8={an8_worst:.4} | A vs 2dr={ar_worst:.4} | A vs 2d={ad2_worst:.4}");

    // РЕГРЕСС-ГЕЙТ: per-row корректность ВСЕХ production NVFP4-GEMM ядер (pick_nvfp4
    // m>1: Reg/N8/Broadcast/Coop). Каждое ДОЛЖНО давать worst ≈ квант-шум (mean|ref|≈319
    // → worst<1.0 = <0.3%) И быть бит-идентичным Broadcast (per-row, не cos!). Был баг:
    // N8/2dr/2d брали scale ЧУЖОЙ batch-строки (sfb=lane&7 при B-data=lane>>2) → worst≈7.7,
    // глобальный cos=0.99998 СКРЫВАЛ → ломал chunked-prefill (разные M → разные ядра).
    let tol = (mean_ref_abs * 0.01).max(1.0);
    for (name, worst, vs_a) in [
        ("Broadcast", a_worst, 0.0),
        ("N8", n8_worst, an8_worst),
        ("2dr/Reg", r_worst, ar_worst),
        ("2d/Coop", d2_worst, ad2_worst),
    ] {
        assert!(
            worst < tol,
            "{name} NVFP4-GEMM per-row worst={worst:.4} >= tol={tol:.4} (mean|ref|={mean_ref_abs:.1}) \
             — строко-НЕкорректно, chunked-prefill сломается"
        );
        assert!(
            vs_a < tol,
            "{name} НЕ бит-идентичен Broadcast (per-row worst={vs_a:.4} >= {tol:.4}) — \
             разные M выберут расходящиеся ядра → потеря контекста"
        );
    }
}

// ─── DIAGNOSTIC: Full-ядро (gn_nvfp4_full_*, M%128==0) per-row vs dense + vs Broadcast ───
// Full загейчен в pick_nvfp4 (SYN_NVFP4_FULL) по раннему вердикту «M%128 баг». Этот тест
// устанавливает ИСТИНУ: даёт ли Full per-row-корректный результат (=Broadcast) или реально
// повреждает строки. M=256 N=5120 K=5120 (M%128==0, N%128==0, K%128==0).
#[test]
fn full_perrow_vs_dense() {
    let Some((ctx, stream)) = setup() else { return };
    let (n, k, batch) = (5120u32, 5120u32, 256u32);
    let q = Nvfp4QuantKernels::for_context(&ctx).expect("quant");
    let gemv = Nvfp4MmaGemvShufKernels::for_context(&ctx).expect("gemv");
    let gemm = Nvfp4MmaGemmShufKernels::for_context(&ctx).expect("gemm");
    let full = GemmNvfp4FullKernels::for_context(&ctx).expect("full");

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch * k) as usize, 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();
    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> =
        stream.alloc_zeros(nvfp4_scale_buffer_size(n as usize, k as usize)).unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((batch * k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> =
        stream.alloc_zeros(nvfp4_scale_buffer_size(batch as usize, k as usize)).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_x, &mut x_packed, &mut x_scales, batch, k).unwrap();
    let mut w_shuf: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    nvfp4_w_repack(&gemv, &stream, &w_packed, &mut w_shuf, n, k).unwrap();

    // dense f32-ref из dequant.
    let mut w_deq: CudaSlice<f16> = stream.alloc_zeros((n * k) as usize).unwrap();
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros((batch * k) as usize).unwrap();
    nvfp4_dequant_f16(&q, &stream, &w_packed, &w_scales, &mut w_deq.as_view_mut(), n, k).unwrap();
    nvfp4_dequant_f16(&q, &stream, &x_packed, &x_scales, &mut x_deq.as_view_mut(), batch, k).unwrap();
    stream.synchronize().unwrap();
    let wf: Vec<f32> = stream.clone_dtoh(&w_deq).unwrap().iter().map(|v| v.to_f32()).collect();
    let xf: Vec<f32> = stream.clone_dtoh(&x_deq).unwrap().iter().map(|v| v.to_f32()).collect();
    let (ku, nu) = (k as usize, n as usize);
    let mut y_ref = vec![0.0f32; (batch * n) as usize];
    y_ref.par_chunks_mut(nu).enumerate().for_each(|(b, row)| {
        let xr = &xf[b * ku..(b + 1) * ku];
        for (o, s) in row.iter_mut().enumerate() {
            let wr = &wf[o * ku..(o + 1) * ku];
            *s = (0..ku).map(|j| wr[j] * xr[j]).sum();
        }
    });
    let mean_ref_abs: f64 = y_ref.iter().map(|v| v.abs() as f64).sum::<f64>() / y_ref.len() as f64;

    // Broadcast (эталонно-корректное ядро).
    let y_a = run_plan_y(
        &ctx, &stream, &w_packed, &w_shuf, &w_scales, &x_packed, &x_scales, &gemm, n, k, batch,
        Variant::A,
    );

    // Все Full-конфиги, которые fits(256,5120,5120).
    let cfgs = [
        ("128x128_c256_s4_swz", Nvfp4FullCfg::C_128_128_C256_S4_SWZ),
        ("128x128_c256_s3_swz", Nvfp4FullCfg::C_128_128_C256_S3_SWZ),
        ("persist_c256_s4_swz", Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ),
        ("persist_c256_s3_swz", Nvfp4FullCfg::C_PERSIST_C256_S3_SWZ),
    ];
    eprintln!("FULL M=256 N=5120 K=5120 mean|ref|={mean_ref_abs:.3} (Broadcast worst={:.4})",
        per_row_maxabs(&y_a, &y_ref, nu).1);
    let tol = (mean_ref_abs * 0.02).max(1.0);
    let _ = &w_packed; // raw-W заведомо неверен (Full ждёт shuffled-W) — не тестируем
    for (name, cfg) in cfgs {
        if !cfg.fits(batch, n, k) { eprintln!("  {name}: НЕ fits, скип"); continue; }
        let mut yf: CudaSlice<f16> = stream.alloc_zeros((batch * n) as usize).unwrap();
        {
            let mut view = yf.as_view_mut();
            gemm_nvfp4_full_cfg_view(&full, &stream, &w_shuf, &w_scales, &x_packed, &x_scales,
                &mut view, n, k, batch, cfg).expect("full launch");
        }
        stream.synchronize().unwrap();
        let y: Vec<f32> = stream.clone_dtoh(&yf).unwrap().iter().map(|v| v.to_f32()).collect();
        let (fmean, fworst) = per_row_maxabs(&y, &y_ref, nu);
        let (_, vs_a) = per_row_maxabs(&y, &y_a, nu);
        eprintln!("  Full {name}: per-row mean={fmean:.4} worst={fworst:.4} | vs Broadcast={vs_a:.4}");
        // РЕГРЕСС-ГЕЙТ: Full ПОЧИНЕН (sfb=lane>>2 + swizzle off>>7) → должен быть per-row
        // корректен (=Broadcast). Был баг: sfb=lane&7 (как N8) + swizzle off>>6 → worst 15.
        assert!(fworst < tol,
            "Full {name} per-row worst={fworst:.4} >= {tol:.4} — повреждает строки");
        assert!(vs_a < tol,
            "Full {name} НЕ бит-идентичен Broadcast (worst={vs_a:.4}) — разойдётся с другими M");
    }
}

// ─── Session TMA-WS2: warp-specialized + K-tile=128 (2dtw2, m4n4) ───



