//! Свип ROT-ядер (CUTLASS-шедулинг k64) vs базовые Full-конфиги на формах LTX.
//! Гейт: per-row max|Δ| == 0 (бит-в-бит) vs дефолтного C_128_128_C256_S4_SWZ.
//! SHAPE=attn|ff_up|ff_down (default attn), M (default 26624, кратно 256),
//! CFG=<fname> — прогнать только один конфиг (режим ncu: 10 warmup + 3 замера),
//! ITERS (default 30).
#![cfg(feature = "cuda")]

use cudarc::driver::CudaSlice;
use half::f16;
use std::time::Instant;

use synaptix_kernels_cuda::best_cu::gemm::gemm_nvfp4::{
    gemm_nvfp4_full_cfg_view, GemmNvfp4FullKernels, Nvfp4FullCfg,
};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};

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
        let mut m = 0.0f64;
        for c in 0..cols {
            let d = (a[r * cols + c].to_f64() - b[r * cols + c].to_f64()).abs();
            if d > m {
                m = d;
            }
        }
        if m > worst {
            worst = m;
            worst_row = r;
        }
    }
    (worst, worst_row)
}

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let q = Nvfp4QuantKernels::for_context(&ctx).expect("nvfp4_quant");
    let full = GemmNvfp4FullKernels::for_context(&ctx).expect("nvfp4_full");

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
    let (n, mut k) = match shape.as_str() {
        "ff_up" => (16384u32, 4096u32),
        "ff_down" => (4096u32, 16384u32),
        _ => (4096u32, 4096u32),
    };
    if let Some(kk) = std::env::var("K").ok().and_then(|s| s.parse::<u32>().ok()) {
        k = kk;
    }

    let w_host = det_f16(0xA110_C8E1, (n as usize) * (k as usize), 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch as usize) * (k as usize), 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n as usize) * (k as usize) / 2).unwrap();
    let mut w_scales: CudaSlice<u8> = stream
        .alloc_zeros(nvfp4_scale_buffer_size(n as usize, k as usize))
        .unwrap();
    let mut x_packed: CudaSlice<u8> = stream
        .alloc_zeros((batch as usize) * (k as usize) / 2)
        .unwrap();
    let mut x_scales: CudaSlice<u8> = stream
        .alloc_zeros(nvfp4_scale_buffer_size(batch as usize, k as usize))
        .unwrap();

    quantize_f16_to_nvfp4(&q, &stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_x, &mut x_packed, &mut x_scales, batch, k).unwrap();
    drop(dev_w);
    drop(dev_x);

    let out_len = (batch as usize) * (n as usize);
    let mut y: CudaSlice<f16> = stream.alloc_zeros(out_len).unwrap();

    let flops = 2.0 * batch as f64 * n as f64 * k as f64;

    let cfgs: Vec<Nvfp4FullCfg> = vec![
        Nvfp4FullCfg::C_128_64_S3_SWZ,
        Nvfp4FullCfg::C_128_64_S4_SWZ,
        Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
        Nvfp4FullCfg::C_128_128_C256_S3_SWZ,
        Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ,
        Nvfp4FullCfg::C_128_256_S3_SWZ,
        Nvfp4FullCfg::C_128_256_S3_SWZ_ROT,
        Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT,
        Nvfp4FullCfg::C_128_256_S3_SWZ_DROT,
    ];

    let run = |cfg: Nvfp4FullCfg, y: &mut CudaSlice<f16>| {
        let mut yv = y.as_view_mut();
        gemm_nvfp4_full_cfg_view(
            &full, &stream, &w_packed, &w_scales, &x_packed, &x_scales, &mut yv, n, k, batch, cfg,
        )
    };

    if let Some(ref name) = only_cfg {
        let cfg = *cfgs
            .iter()
            .find(|c| c.fname() == name)
            .unwrap_or_else(|| panic!("CFG {name} не найден"));
        if !cfg.fits(batch, n, k) {
            panic!("CFG {name}: не подходит под M={batch} N={n} K={k}");
        }
        for _ in 0..10 {
            run(cfg, &mut y).unwrap();
        }
        stream.synchronize().unwrap();
        let sus_iters = if iters > 30 { iters } else { 3 };
        let t0 = Instant::now();
        for i in 0..sus_iters {
            run(cfg, &mut y).unwrap();
            if i % 8 == 7 {
                stream.synchronize().unwrap();
            }
        }
        stream.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64() / sus_iters as f64;
        println!(
            "done: {name} shape={shape} {n}x{k} M={batch} iters={sus_iters} {:.1} TF",
            flops / dt / 1e12
        );
        return;
    }

    let check = std::env::var("CHECK").map(|v| v != "0").unwrap_or(true);
    println!("== shape={shape} N={n} K={k} M={batch} check={check} ==");
    let ref_cfg = Nvfp4FullCfg::C_128_128_C256_S4_SWZ;
    run(ref_cfg, &mut y).unwrap();
    stream.synchronize().unwrap();
    let y_ref: Vec<f16> = if check {
        stream.clone_dtoh(&y).unwrap()
    } else {
        Vec::new()
    };

    let skip_env = std::env::var("SKIP_CFG").unwrap_or_default();
    for cfg in &cfgs {
        if !cfg.fits(batch, n, k) {
            println!("{:42} SKIP (shape/smem)", cfg.fname());
            continue;
        }
        if skip_env.split(',').any(|s| s == cfg.fname()) {
            println!("{:42} SKIP (env: висяк, см. ретест 2026-06-05)", cfg.fname());
            continue;
        }
        let (worst, wrow) = if check {
            stream.memset_zeros(&mut y).unwrap();
            run(*cfg, &mut y).unwrap();
            stream.synchronize().unwrap();
            let y_t: Vec<f16> = stream.clone_dtoh(&y).unwrap();
            per_row_max_abs(&y_ref, &y_t, batch as usize, n as usize)
        } else {
            (0.0, 0)
        };

        for _ in 0..10 {
            run(*cfg, &mut y).unwrap();
        }
        stream.synchronize().unwrap();
        let dobench = std::env::var("DOBENCH").map(|v| v == "1").unwrap_or(false);
        let dt = if dobench {
            // Протокол triton.do_bench 1-в-1 (как bench_ltx_gemm::time_loop):
            // SM-флаш (mul_scalar, НЕ CE-memset — тот ронял SM-клок) + тайминг
            // СОБЫТИЯМИ без per-iter sync (wall+sync штрафовал ~5мкс/ячейку).
            use synaptix_core::device::Device;
            use synaptix_core::dtype::DType;
            use synaptix_core::tensor::Tensor;
            let mk_ev = || {
                stream
                    .context()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap()
            };
            let evs_a: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
            let evs_b: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
            let flush_src =
                Tensor::ones(vec![64 * 1024 * 1024], DType::F32, Device::Cuda(0)).unwrap();
            for i in 0..iters {
                let fz = flush_src.mul_scalar(0.0).unwrap();
                std::hint::black_box(&fz);
                evs_a[i].record(&stream).unwrap();
                run(*cfg, &mut y).unwrap();
                evs_b[i].record(&stream).unwrap();
            }
            stream.synchronize().unwrap();
            let total: f64 = (0..iters)
                .map(|i| evs_a[i].elapsed_ms(&evs_b[i]).unwrap() as f64 / 1e3)
                .sum();
            total / iters as f64
        } else {
            let t0 = Instant::now();
            for _ in 0..iters {
                run(*cfg, &mut y).unwrap();
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        println!(
            "{:42} per-row max|Δ|={:8.6} (row {wrow:5})  {:7.1} TF",
            cfg.fname(),
            worst,
            flops / dt / 1e12
        );
    }
}
