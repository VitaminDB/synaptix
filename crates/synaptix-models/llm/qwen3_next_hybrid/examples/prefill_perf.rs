//! Чистый prefill-перф 27B-hybrid: forward(prompt) на разных длинах, tok/s + per-op
//! профиль (SYN_PREFILL_PROF). Сравнение с llama.cpp/pytorch (~1660 tok/s).
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//!   --example prefill_perf -- "models/qwen3.6 27B.syn" [lens...]
use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prefill_perf MODEL.syn [lens...]");
    let lens: Vec<usize> = {
        let v: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() { vec![128, 350, 640, 960, 1400, 1660] } else { v }
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let cap = lens.iter().max().copied().unwrap_or(2048) + 64;
    let t0 = std::time::Instant::now();
    let pipe = HybridPipeline::load_with_precision(&path, dev, PrecisionConfig::nvfp4(), Some(cap))
        .expect("load");
    eprintln!("loaded {:.1}s | NVFP4 F16 | формы 27B-hybrid\n", t0.elapsed().as_secs_f32());

    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let prof = std::env::var("SYN_PREFILL_PROF").as_deref() == Ok("1");
    synaptix_llm_common::model::set_prefill_prof(prof);
    eprintln!("len(ток) | prefill_ms | tok/s | (llama.cpp ~1660)");
    for &n in &lens {
        let ids: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % 200 + 5) as u32).collect();
        let t = Tensor::from_vec(ids.clone(), vec![1usize, n], dev).unwrap();
        // warmup (3 прогона — прогрев кэшей/NVRTC/буст-частоты)
        for _ in 0..3 {
            let mut kv = pipe.model.make_kv_cache(1, cap).unwrap();
            let _ = no_grad(|| pipe.model.forward(&t, &mut kv)).unwrap();
        }
        stream.synchronize().unwrap();
        // замер: МИН из N итераций (буст-частота, без троттла) + медиана.
        let iters = if n <= 512 { 12 } else { 6 };
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            if prof { let _ = synaptix_llm_common::model::prefill_prof_report_and_clear(); }
            let mut kv2 = pipe.model.make_kv_cache(1, cap).unwrap();
            let t1 = std::time::Instant::now();
            let _ = no_grad(|| pipe.model.forward(&t, &mut kv2)).unwrap();
            stream.synchronize().unwrap();
            times.push(t1.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = times[0]; // min = лучший clock
        let med = times[times.len() / 2];
        let tps = n as f64 / (ms / 1000.0);
        eprintln!("{n:8} | min {ms:7.1} med {med:7.1} | {tps:7.0} tok/s | {:.1}x медленнее llama", 1660.0 / tps);
        if prof {
            eprintln!("{}", synaptix_llm_common::model::prefill_prof_report_and_clear());
        }
    }
}
