//! Изолированный DiT denoise-bench (без AR/VAE/load-шума в трейсе).
//! `DEN_T` латент-фреймы (180s ≈ 4500), `DEN_L` enc-seq, `DEN_STEPS`, `DEN_N`.
use std::path::Path;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::config::DitConfig;
use synaptix_music_acestep::dit::Dit;
use synaptix_music_acestep::loader::CompLoader;
use synaptix_music_acestep::pipeline::{denoise, SamplerOptions};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let cfg = DitConfig::xl_base();
    let dit_path = Path::new("storage/syn_models/acestep_v15_xl_base.syn");
    let dit_ck = CompLoader::open(dit_path, None, device).unwrap();
    let dit = Dit::load(&dit_ck, &cfg, DType::BF16, DType::BF16).unwrap();

    let getv = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let t = getv("DEN_T", 4500);
    let l = getv("DEN_L", 128);
    let steps = getv("DEN_STEPS", 32);
    let n = getv("DEN_N", 3);
    let eh = cfg.encoder_hidden_size;

    let mk = |dims: Vec<usize>, seed: u64| {
        Tensor::randn_seeded(dims, seed, Device::Cpu).unwrap().to_device(device).unwrap()
    };
    let x0 = mk(vec![1, t, 64], 42);
    let context = mk(vec![1, t, 128], 1);
    let enc = mk(vec![1, l, eh], 2);
    let null = mk(vec![1, l, eh], 3);
    let opts = SamplerOptions { steps, shift: 3.0, guidance_scale: 7.0, ..Default::default() };

    for i in 0..n {
        let t0 = Instant::now();
        let out = denoise(&dit, &x0, &context, &enc, Some(&null), &opts).unwrap();
        let _ = out.narrow(2, 0, 1).unwrap().contiguous().unwrap()
            .to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        eprintln!("[denoise-bench] run {i}: {:.3}s ({steps} steps, T={t}, L={l})", t0.elapsed().as_secs_f32());
    }
}
