//! FLUX.1 «как на малой карте»: балласт в VRAM оставляет модели
//! `FLUX_VRAM_GB` (по умолчанию 7), фоновый поток следит за минимумом
//! свободной памяти. Прогон: CLIP + T5 → DiT → денойз → VAE.
//!
//! ```sh
//! FLUX_SYN=~/Storage/syn_models/flux.1-dev.syn FLUX_QUANT=nvfp4 FLUX_SIZE=1024 FLUX_OUT=/tmp/out \
//!   cargo test --release -p synaptix-image-flux --test low_vram -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::{FluxModel, OffloadMode, SampleParams};

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn free() -> usize {
    synaptix_core::device::cuda::mem_info(0).map(|(f, _)| f).unwrap_or(0)
}

fn save_ppm(img: &Tensor, path: &str) {
    let d = img.dims().to_vec();
    let (h, w) = (d[1], d[2]);
    let v = img.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out.push((v[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    std::fs::write(path, out).unwrap();
}

#[test]
#[ignore]
fn generate_under_budget() {
    let Some(model) = env("FLUX_SYN") else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let budget = (env("FLUX_VRAM_GB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(7.0) * 1e9) as usize;
    let (compute, quant) = match env("FLUX_QUANT").as_deref() {
        Some("mxfp8") => (DType::F16, DType::MXFP8),
        Some("bf16") | Some("dense") => (DType::BF16, DType::BF16),
        _ => (DType::F16, DType::NVFP4),
    };
    let size = env("FLUX_SIZE").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
    let steps = env("FLUX_STEPS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(20);
    let prompt = env("FLUX_PROMPT").unwrap_or_else(|| {
        "a lighthouse on a rocky coast at sunset, dramatic clouds, photo".into()
    });
    let dev = Device::Cuda(0);
    let start_free = free();
    let ballast_bytes = start_free.saturating_sub(budget);
    let ballast = (ballast_bytes > 0).then(|| Tensor::zeros((ballast_bytes,), DType::U8, dev).unwrap());
    let avail0 = free();
    eprintln!("свободно было {:.1} ГБ → модели {:.1} ГБ", start_free as f64 / 1e9, avail0 as f64 / 1e9);
    let min_free = Arc::new(AtomicUsize::new(avail0));
    let stop = Arc::new(AtomicBool::new(false));
    let mon = {
        let (min_free, stop) = (min_free.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                min_free.fetch_min(free(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
        })
    };
    let overall = std::cell::Cell::new(avail0);
    let phase = |name: &str| {
        let v = min_free.swap(free(), Ordering::Relaxed);
        overall.set(overall.get().min(v));
        eprintln!("  фаза {name}: минимум свободной {:.2} ГБ", v as f64 / 1e9);
    };
    let t0 = Instant::now();
    let m = FluxModel::open(&model, dev, compute, quant).unwrap().with_offload(OffloadMode::Auto);
    let seq = m.default_max_seq_len();
    let cond = m.encode_prompt(&prompt, seq).unwrap();
    let t_enc = t0.elapsed().as_secs_f64();
    phase("энкодеры");
    let t1 = Instant::now();
    let t = m.load_transformer(FluxModel::tokens_for(size, size, seq)).unwrap();
    let t_load = t1.elapsed().as_secs_f64();
    let res = t.resident_blocks();
    phase("загрузка DiT");
    let p = SampleParams { width: size, height: size, steps, guidance: 3.5, seed: 7, denoise: 1.0 };
    let t2 = Instant::now();
    let lat = m.sample(&t, &cond, None, &p, &mut |_, _| true).unwrap();
    let t_sample = t2.elapsed().as_secs_f64();
    phase("денойз");
    drop(t);
    let t3 = Instant::now();
    let img = m.decode(&lat).unwrap();
    let t_dec = t3.elapsed().as_secs_f64();
    phase("VAE");
    stop.store(true, Ordering::Relaxed);
    mon.join().unwrap();
    let mf = overall.get().min(min_free.load(Ordering::Relaxed));
    eprintln!(
        "FLUX.1 {quant:?} {size}² {steps} шагов: энкодеры {t_enc:.1} с, DiT {t_load:.1} с (на карте {}+{} блоков), \
         денойз {t_sample:.1} с, VAE {t_dec:.1} с; минимум свободной VRAM {:.2} ГБ, пик ≈ {:.2} ГБ из {:.1}",
        res.0,
        res.1,
        mf as f64 / 1e9,
        avail0.saturating_sub(mf) as f64 / 1e9,
        budget as f64 / 1e9
    );
    if let Some(out) = env("FLUX_OUT") {
        save_ppm(&img, &format!("{out}/flux1_{quant:?}_{size}.ppm"));
    }
    drop(ballast);
}
