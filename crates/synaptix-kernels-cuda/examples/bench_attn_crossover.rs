//! Attention-backend crossover bench (Session 11 Phase 2).
//!
//! Qwen3-1.7B attn shape (b=1, nh=16, nkv=8, hd=128). Сравнивает FA-2
//! (flash_bf16: tiled prefill / single-row decode) vs split-K flash-decode на
//! растущей длине контекста — ищем реальный кроссовер на sm_120.
//!
//! Запуск: `cargo run -p synaptix-kernels-cuda --example bench_attn_crossover \
//!          --features cuda --profile fast-release`
#![cfg(feature = "cuda")]

use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use std::sync::Arc;

use synaptix_kernels_cuda::attention::flash_bf16::FlashAttnBf16Kernels;
use synaptix_kernels_cuda::attention::flash_decode::{flash_decode_bf16, FlashDecodeKernels};

fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            bf16::from_f32(((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn time_ms<F: FnMut()>(stream: &Arc<CudaStream>, iters: usize, mut f: F) -> f64 {
    f();
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let fa = FlashAttnBf16Kernels::for_context(&ctx).expect("flash_bf16");
    let fd = FlashDecodeKernels::for_context(&ctx).expect("flash_decode");

    let (b, nh, nkv, hd) = (1u32, 16u32, 8u32, 128u32);
    let n_rep = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();

    println!("# Qwen3 attn (b={b} nh={nh} nkv={nkv} hd={hd}) — sm_120, BF16");
    println!("# PREFILL (Tq=Tkv=L): FA-2 tiled(m64) vs flash-decode(split_k=1)");
    println!(
        "{:>8} | {:>14} | {:>16} | {:>8}",
        "L", "FA2 tok/s", "FlashDec tok/s", "winner"
    );
    let prefill_lens = [2048usize, 4096, 8192, 12288, 16384, 24576, 32768];
    for &l in &prefill_lens {
        let lu = l as u32;
        let q = det_bf16(0x11, (b * nh * lu * hd) as usize, 0.3);
        let k = det_bf16(0x22, (b * nkv * lu * hd) as usize, 0.3);
        let v = det_bf16(0x33, (b * nkv * lu * hd) as usize, 0.3);
        let dq: CudaSlice<bf16> = stream.clone_htod(&q).unwrap();
        let dk: CudaSlice<bf16> = stream.clone_htod(&k).unwrap();
        let dv: CudaSlice<bf16> = stream.clone_htod(&v).unwrap();
        let mut dout: CudaSlice<bf16> = stream.alloc_zeros((b * nh * lu * hd) as usize).unwrap();

        let iters = if l >= 16384 { 3 } else { 10 };
        let fa_ms = time_ms(&stream, iters, || {
            fa.flash_attn2_fwd_bf16_tiled(
                &stream, &dq, &dk, &dv, &mut dout, scale, b, nh, nkv, lu, lu, hd, n_rep, 0, 1, 64,
                0,
            )
            .unwrap();
        });
        let fa_tps = l as f64 / (fa_ms / 1000.0);

        // flash-decode prefill (split_k=1) импрактично при больших L (block explosion).
        let (fd_tps, winner) = if l <= 24576 {
            let fd_ms = time_ms(&stream, iters, || {
                flash_decode_bf16(
                    &fd, &stream, &dq, &dk, &dv, &mut dout, b, nh, nkv, lu, lu, hd, scale, true, 1,
                )
                .unwrap();
            });
            let tps = l as f64 / (fd_ms / 1000.0);
            (
                format!("{tps:.0}"),
                if fa_tps > tps { "FA2" } else { "FlashDec" },
            )
        } else {
            ("skip(O(L²))".to_string(), "FA2")
        };
        println!("{l:>8} | {fa_tps:>14.0} | {fd_tps:>16} | {winner:>8}");
    }

    println!();
    println!("# DECODE (Tq=1, Tkv=L): FA-2 single-row vs flash-decode(split_k tuned)");
    println!(
        "{:>8} | {:>14} | {:>16} | {:>8}",
        "L", "FA2 tok/s", "FlashDec tok/s", "winner"
    );
    let decode_lens = [2048usize, 8192, 32768, 65536, 131072];
    let rows = b * nh; // Tq=1
    for &l in &decode_lens {
        let lu = l as u32;
        let q = det_bf16(0x44, (b * nh * hd) as usize, 0.3);
        let k = det_bf16(0x55, (b * nkv * lu * hd) as usize, 0.3);
        let v = det_bf16(0x66, (b * nkv * lu * hd) as usize, 0.3);
        let dq: CudaSlice<bf16> = stream.clone_htod(&q).unwrap();
        let dk: CudaSlice<bf16> = stream.clone_htod(&k).unwrap();
        let dv: CudaSlice<bf16> = stream.clone_htod(&v).unwrap();
        let mut dout: CudaSlice<bf16> = stream.alloc_zeros((b * nh * hd) as usize).unwrap();

        let fa_ms = time_ms(&stream, 30, || {
            // single-row: t_chunk=1, q_pos_base = L-1 (causal), t_stride=0 (contig).
            fa.flash_attn2_fwd_bf16(
                &stream,
                &dq,
                &dk,
                &dv,
                &mut dout,
                scale,
                b,
                nh,
                nkv,
                1,
                lu,
                hd,
                n_rep,
                lu - 1,
                1,
                0,
            )
            .unwrap();
        });
        let fa_tps = 1000.0 / fa_ms;

        let occ = (128 / rows.max(1)).max(1);
        let long = lu.div_ceil(2048);
        let split_k = occ.max(long).clamp(1, 32);
        let fd_ms = time_ms(&stream, 30, || {
            flash_decode_bf16(
                &fd, &stream, &dq, &dk, &dv, &mut dout, b, nh, nkv, 1, lu, hd, scale, true, split_k,
            )
            .unwrap();
        });
        let fd_tps = 1000.0 / fd_ms;
        let winner = if fa_tps > fd_tps { "FA2" } else { "FlashDec" };
        println!("{l:>8} | {fa_tps:>14.0} | {fd_tps:>16.0} | {winner:>8} (sk={split_k})");
    }
}
