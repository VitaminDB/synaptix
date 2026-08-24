use cudarc::driver::CudaSlice;
use half::bf16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::{
    best_gemm_bf16_linear_u8, BestGemmBf16Kernels,
};
use synaptix_kernels_cuda::best_cu::gemv::mma_gemv::{gemv_bf16, MmaGemvKernels};

fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(f * scale)
        })
        .collect()
}

fn gemv_bf16_view(
    kernels: &MmaGemvKernels,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    w: &cudarc::driver::CudaView<bf16>,
    x: &cudarc::driver::CudaView<bf16>,
    y: &mut cudarc::driver::CudaViewMut<bf16>,
    n: u32,
    k: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let cfg = LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kernels.bf16_fn());
    b.arg(w).arg(x).arg(&mut *y).arg(&n).arg(&k);
    unsafe { b.launch(cfg)? };
    Ok(())
}

fn bytes_of(v: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for b in v {
        out.extend_from_slice(&b.to_bits().to_le_bytes());
    }
    out
}

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let gemv = MmaGemvKernels::for_context(&ctx).expect("gemv kernels");
    let gemm = BestGemmBf16Kernels::for_context(&ctx).expect("gemm kernels");

    let peak_gbps = std::env::var("PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(896.0);
    let iters = 300u32;
    let warmup = 50u32;

    let shapes: &[(&str, u32, u32)] = &[
        ("tiny        ", 64, 64),
        ("head_adaln  ", 4608, 1536),
        ("head_ffn    ", 4608, 1536),
        ("qwen2_qo    ", 1536, 1536),
        ("qwen2_kv    ", 256, 1536),
        ("qwen2_gate  ", 8960, 1536),
        ("qwen2_down  ", 1536, 8960),
        ("qwen7_qo    ", 3584, 3584),
        ("qwen7_gate  ", 18944, 3584),
        ("lm_head     ", 151936, 1536),
        ("vae_ffn1_2048", 8192, 2048),
        ("vae_ffn2_2048", 2048, 8192),
        ("vae_ffn1_1024", 4096, 1024),
        ("vae_ffn2_1024", 1024, 4096),
        ("vae_pixshuf1 ", 8192, 4096),
    ];

    {
        use synaptix_core::device::Device;
        use synaptix_core::dtype::DType;
        use synaptix_core::tensor::Tensor;
        synaptix_kernels_cuda::ensure_registered();
        let dev = Device::Cuda(0);
        println!("== Tensor::linear (обвязка) vs raw kernel ==");
        println!("{:<14}{:>8}{:>8}{:>12}{:>12}", "shape", "N", "K", "tensor us", "raw us");
        for &(name, n, k) in shapes {
            let wt = Tensor::zeros(vec![n as usize, k as usize], DType::BF16, dev).unwrap();
            let xt = Tensor::zeros(vec![1usize, 1, k as usize], DType::BF16, dev).unwrap();
            for _ in 0..warmup {
                let _ = xt.linear(&wt).unwrap();
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = xt.linear(&wt).unwrap();
            }
            stream.synchronize().unwrap();
            let tensor_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let w_host = det_bf16(0xA110_C8E1, (n as usize) * (k as usize), 0.5);
            let x_host = det_bf16(0xC0DE_BA5E, k as usize, 0.5);
            let w: CudaSlice<bf16> = stream.clone_htod(&w_host).unwrap();
            let x: CudaSlice<bf16> = stream.clone_htod(&x_host).unwrap();
            let mut y: CudaSlice<bf16> = stream.alloc_zeros(n as usize).unwrap();
            for _ in 0..warmup {
                gemv_bf16(&gemv, &stream, &w, &x, &mut y, n, k).unwrap();
            }
            stream.synchronize().unwrap();
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                gemv_bf16(&gemv, &stream, &w, &x, &mut y, n, k).unwrap();
            }
            stream.synchronize().unwrap();
            let raw_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("{:<14}{:>8}{:>8}{:>12.2}{:>12.2}", name, n, k, tensor_us, raw_us);
        }
        println!();
    }

    {
        println!("== холодные веса (ротация копий > L2) ==");
        println!("{:<14}{:>8}{:>8}{:>10}{:>10}{:>9}", "shape", "N", "K", "M", "us/call", "GB/s");
        for &(name, n, k) in &[("qwen2_gate  ", 8960u32, 1536u32), ("head_adaln  ", 4608, 1536)] {
            let wbytes = (n as usize) * (k as usize) * 2;
            let copies = (768 * 1024 * 1024 / wbytes).clamp(2, 40);
            let mut ws: Vec<CudaSlice<u8>> = Vec::with_capacity(copies);
            for _ in 0..copies {
                ws.push(stream.alloc_zeros(wbytes).unwrap());
            }
            for m in [1u32, 2, 4] {
                let x_host = bytes_of(&det_bf16(0xC0DE, (m as usize) * (k as usize), 0.5));
                let x: CudaSlice<u8> = stream.clone_htod(&x_host).unwrap();
                let mut y: CudaSlice<u8> =
                    stream.alloc_zeros((m as usize) * (n as usize) * 2).unwrap();
                let run = |i: usize, y: &mut CudaSlice<u8>| {
                    if m == 1 {
                        let wv = unsafe { ws[i].transmute::<bf16>(wbytes / 2).unwrap() };
                        let xv = unsafe { x.transmute::<bf16>(k as usize).unwrap() };
                        let mut yv = unsafe { y.transmute_mut::<bf16>(n as usize).unwrap() };
                        gemv_bf16_view(&gemv, &stream, &wv, &xv, &mut yv, n, k).unwrap();
                    } else {
                        best_gemm_bf16_linear_u8(&gemm, &stream, &ws[i], &x, y, n, k, m, None, None)
                            .unwrap();
                    }
                };
                for i in 0..warmup as usize {
                    run(i % copies, &mut y);
                }
                stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for i in 0..iters as usize {
                    run(i % copies, &mut y);
                }
                stream.synchronize().unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let gbps = wbytes as f64 / (us * 1e3);
                println!("{:<14}{:>8}{:>8}{:>10}{:>10.2}{:>10.1}", name, n, k, m, us, gbps);
            }
        }
        println!();
    }

    println!("== M=1 (GEMV) ==");
    println!(
        "{:<14}{:>8}{:>8}{:>10}{:>10}{:>9}",
        "shape", "N", "K", "us/call", "GB/s", "%peak"
    );
    for &(name, n, k) in shapes {
        let w_host = det_bf16(0xA110_C8E1, (n as usize) * (k as usize), 0.5);
        let x_host = det_bf16(0xC0DE_BA5E, k as usize, 0.5);
        let w: CudaSlice<bf16> = stream.clone_htod(&w_host).unwrap();
        let x: CudaSlice<bf16> = stream.clone_htod(&x_host).unwrap();
        let mut y: CudaSlice<bf16> = stream.alloc_zeros(n as usize).unwrap();

        for _ in 0..warmup {
            gemv_bf16(&gemv, &stream, &w, &x, &mut y, n, k).unwrap();
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gemv_bf16(&gemv, &stream, &w, &x, &mut y, n, k).unwrap();
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes = (n as f64) * (k as f64) * 2.0 + (k as f64) * 2.0 + (n as f64) * 2.0;
        let gbps = bytes / (us * 1e3);
        println!(
            "{:<14}{:>8}{:>8}{:>10.2}{:>10.1}{:>8.1}%",
            name,
            n,
            k,
            us,
            gbps,
            100.0 * gbps / peak_gbps
        );
    }

    for m in [2u32, 8, 64] {
        println!("\n== M={m} (GEMM) ==");
        println!(
            "{:<14}{:>8}{:>8}{:>10}{:>10}{:>12}",
            "shape", "N", "K", "us/call", "GB/s", "TFLOP/s"
        );
        for &(name, n, k) in shapes {
            let w_host = bytes_of(&det_bf16(0xA110_C8E1, (n as usize) * (k as usize), 0.5));
            let x_host = bytes_of(&det_bf16(0xC0DE_BA5E, (m as usize) * (k as usize), 0.5));
            let w: CudaSlice<u8> = stream.clone_htod(&w_host).unwrap();
            let x: CudaSlice<u8> = stream.clone_htod(&x_host).unwrap();
            let mut y: CudaSlice<u8> =
                stream.alloc_zeros((m as usize) * (n as usize) * 2).unwrap();

            for _ in 0..warmup {
                best_gemm_bf16_linear_u8(&gemm, &stream, &w, &x, &mut y, n, k, m, None, None)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                best_gemm_bf16_linear_u8(&gemm, &stream, &w, &x, &mut y, n, k, m, None, None)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let bytes = (n as f64) * (k as f64) * 2.0
                + (m as f64) * (k as f64) * 2.0
                + (m as f64) * (n as f64) * 2.0;
            let gbps = bytes / (us * 1e3);
            let tflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / (us * 1e-6) / 1e12;
            println!(
                "{:<14}{:>8}{:>8}{:>10.2}{:>10.1}{:>12.2}",
                name, n, k, us, gbps, tflops
            );
        }
    }
}
