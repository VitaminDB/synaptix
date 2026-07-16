//! Перф Full-ядра vs остальных (2dr/Broadcast) на large-M prefill-формах модели.
//! Проверяет, что re-enable Full реально ускоряет (а не просто корректен). Запускать
//! ДВАЖДЫ: дефолт (Full вкл) и SYN_NVFP4_NO_FULL=1 (Full выкл) — сравнить TFLOPS.
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//!   --example nvfp4_full_perf
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cuda::gemm::dispatch::pick_nvfp4;

fn det(seed: u64, n: usize, s: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0) * s }).collect()
}

fn main() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let full_on = std::env::var("SYN_NVFP4_NO_FULL").as_deref() != Ok("1");
    eprintln!("=== NVFP4 large-M GEMM perf | Full {} ===", if full_on { "ВКЛ" } else { "ВЫКЛ" });
    // prefill-формы 27B-hybrid (N,K) × prefill-batch M.
    let shapes: &[(usize, usize, &str)] = &[
        (10240, 5120, "in_proj_qkv"),
        (17408, 5120, "mlp_gate_up"),
        (5120, 17408, "mlp_down   "),
    ];
    let ms = [512usize, 1024, 2048];
    let iters = 50;
    for &(n, k, lbl) in shapes {
        let wf = det(0xA53F ^ (n as u64) ^ (k as u64), n * k, 0.5);
        let qw = Tensor::from_vec(wf, vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap()
            .quantize_to_nvfp4().unwrap();
        for &m in &ms {
            let plan = format!("{:?}", pick_nvfp4(m as u32, n as u32, k as u32));
            let plan_short = plan.split('(').next().unwrap_or(&plan).replace("Nvfp4Plan::", "");
            let xf = det(0x7777, m * k, 0.4);
            let x = Tensor::from_vec(xf, vec![m, k], dev).unwrap().to_dtype(DType::F16).unwrap();
            // warmup
            for _ in 0..5 { let _ = x.linear_quant(&qw).unwrap(); }
            synaptix_core::device::cuda::default_stream(0).unwrap().synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters { let _ = x.linear_quant(&qw).unwrap(); }
            synaptix_core::device::cuda::default_stream(0).unwrap().synchronize().unwrap();
            let us = t0.elapsed().as_micros() as f64 / iters as f64;
            let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e6);
            eprintln!("  {lbl} N={n} K={k} M={m:5} | {us:7.1} мкс | {tflops:6.1} TF | plan={plan_short}");
        }
    }
}
