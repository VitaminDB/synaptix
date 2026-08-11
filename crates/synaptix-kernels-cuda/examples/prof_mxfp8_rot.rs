//! Baseline/итерации MXFP8 GEMM (рецепт CUTLASS-порта nvfp4).
//! Pure-GEMM: квант 1×, ядра в цикле. Гейт: per-row max|Δ| rot-vs-base == 0 (бит-в-бит).
//! SHAPE=attn|ff_up|ff_down, M (default 26624), ITERS (default 30),
//! DOBENCH=1 (L2-flush + медиана), CHECK=0 (только перф), CFG=base|rot (режим ncu).

use cudarc::driver::CudaSlice;
use half::f16;
use std::time::Instant;

use synaptix_kernels_cuda::best_cu::gemm::gemm_mxfp8::{
    gemm_mxfp8, gemm_mxfp8_rot, GemmMxFp8Kernels,
};
use synaptix_kernels_cuda::elementwise::quant::{mxfp8_quant_natural, Mxfp8QuantKernels};

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

fn per_row_max_abs(a: &[f16], b: &[f16], rows: usize, cols: usize) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut worst_row = 0usize;
    for r in 0..rows {
        let mut mx = 0.0f64;
        for c in 0..cols {
            let d = (a[r * cols + c].to_f64() - b[r * cols + c].to_f64()).abs();
            if d > mx {
                mx = d;
            }
        }
        if mx > worst {
            worst = mx;
            worst_row = r;
        }
    }
    (worst, worst_row)
}

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let qk = Mxfp8QuantKernels::for_context(&ctx).expect("mxfp8_quant");
    let gk = GemmMxFp8Kernels::for_context(&ctx).expect("mxfp8_gemm");

    let shape = std::env::var("SHAPE").unwrap_or_else(|_| "attn".to_string());
    let batch: u32 = std::env::var("M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(26624);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let only_cfg = std::env::var("CFG").ok();
    let check = std::env::var("CHECK").map(|v| v != "0").unwrap_or(true);
    let (n, k) = match shape.as_str() {
        "ff_up" => (16384u32, 4096u32),
        "ff_down" => (4096u32, 16384u32),
        _ => (4096u32, 4096u32),
    };

    let w_host = det_f16(0xA110_C8E1, (n as usize) * (k as usize), 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch as usize) * (k as usize), 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let mut wq: CudaSlice<u8> = stream.alloc_zeros((n as usize) * (k as usize)).unwrap();
    let mut sb: CudaSlice<u8> = stream
        .alloc_zeros((n as usize) * (k as usize) / 32)
        .unwrap();
    let mut xq: CudaSlice<u8> = stream
        .alloc_zeros((batch as usize) * (k as usize))
        .unwrap();
    let mut sa: CudaSlice<u8> = stream
        .alloc_zeros((batch as usize) * (k as usize) / 32)
        .unwrap();
    mxfp8_quant_natural(&qk, &stream, &dev_w.as_view(), &mut wq, &mut sb, n, k).unwrap();
    mxfp8_quant_natural(&qk, &stream, &dev_x.as_view(), &mut xq, &mut sa, batch, k).unwrap();
    drop(dev_w);
    drop(dev_x);

    let mut y: CudaSlice<f16> = stream.alloc_zeros((batch as usize) * (n as usize)).unwrap();
    let flops = 2.0 * batch as f64 * n as f64 * k as f64;

    let y_len = (batch as usize) * (n as usize);
    let run = |mode: u32, y: &mut CudaSlice<f16>| {
        let mut yv = y.slice_mut(0..y_len);
        match mode {
            1 => gemm_mxfp8_rot(&gk, &stream, &xq, &wq, &sa, &sb, &mut yv, batch, n, k, false),
            2 => gemm_mxfp8_rot(&gk, &stream, &xq, &wq, &sa, &sb, &mut yv, batch, n, k, true),
            _ => gemm_mxfp8(&gk, &stream, &xq, &wq, &sa, &sb, &mut yv, batch, n, k),
        }
    };

    if let Some(ref cfgname) = only_cfg {
        let mode = match cfgname.as_str() {
            "rot" => 1,
            "drot" => 2,
            _ => 0,
        };
        for _ in 0..10 {
            run(mode, &mut y).unwrap();
        }
        stream.synchronize().unwrap();
        for _ in 0..3 {
            run(mode, &mut y).unwrap();
        }
        stream.synchronize().unwrap();
        println!("done: mxfp8 {cfgname} shape={shape} {n}x{k} M={batch}");
        return;
    }

    println!("== mxfp8 shape={shape} N={n} K={k} M={batch} check={check} ==");
    let y_ref: Vec<f16> = if check {
        run(0, &mut y).unwrap();
        stream.synchronize().unwrap();
        stream.clone_dtoh(&y).unwrap()
    } else {
        Vec::new()
    };

    for (name, mode) in [
        ("gn_mxfp8_128x128", 0u32),
        ("gn_mxfp8_rot_128x128_s2", 1),
        ("gn_mxfp8_drot_128x128_s2", 2),
    ] {
        let (worst, wrow) = if check {
            stream.memset_zeros(&mut y).unwrap();
            run(mode, &mut y).unwrap();
            stream.synchronize().unwrap();
            let y_t: Vec<f16> = stream.clone_dtoh(&y).unwrap();
            per_row_max_abs(&y_ref, &y_t, batch as usize, n as usize)
        } else {
            (0.0, 0)
        };

        for _ in 0..10 {
            run(mode, &mut y).unwrap();
        }
        stream.synchronize().unwrap();
        let dobench = std::env::var("DOBENCH").map(|v| v == "1").unwrap_or(false);
        let dt = if dobench {
            // triton do_bench 1-в-1: SM-флаш (квант мусорного буфера 256MB —
            // CE-memset ронял SM-клок, см. bf16-сессию), события CUDA,
            // БЕЗ sync между итерациями, медиана.
            let flush_src: CudaSlice<f16> = stream.alloc_zeros(128 * 1024 * 1024).unwrap();
            let mut fq: CudaSlice<u8> = stream.alloc_zeros(128 * 1024 * 1024).unwrap();
            let mut fs: CudaSlice<u8> = stream.alloc_zeros(128 * 1024 * 1024 / 32).unwrap();
            let mk_ev = || {
                stream
                    .context()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap()
            };
            let evs_a: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
            let evs_b: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
            for i in 0..iters {
                mxfp8_quant_natural(&qk, &stream, &flush_src.as_view(), &mut fq, &mut fs, 32768, 4096)
                    .unwrap();
                evs_a[i].record(&stream).unwrap();
                run(mode, &mut y).unwrap();
                evs_b[i].record(&stream).unwrap();
            }
            stream.synchronize().unwrap();
            let mut times: Vec<f64> = (0..iters)
                .map(|i| evs_a[i].elapsed_ms(&evs_b[i]).unwrap() as f64 / 1e3)
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[times.len() / 2]
        } else {
            let t0 = Instant::now();
            for _ in 0..iters {
                run(mode, &mut y).unwrap();
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        println!(
            "{:28} per-row max|Δ|={:8.6} (row {wrow:5})  {:7.1} TF",
            name,
            worst,
            flops / dt / 1e12
        );
    }
}
