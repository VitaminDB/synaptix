//! Вся цепочка FLUX.2 «как на малой карте»: балласт в VRAM оставляет модели
//! `FLUX2_VRAM_GB` (по умолчанию 7), фоновый поток следит за минимумом
//! свободной памяти. Прогон: промпт → DiT → денойз → VAE (+ правка по
//! референсу, если задан `FLUX2_EDIT`).
//!
//! ```sh
//! FLUX2_MODEL=~/Storage/syn_models/flux.2-klein-4b.syn FLUX2_QUANT=nvfp4 FLUX2_SIZE=1024 \
//! FLUX2_OUT=/tmp/out cargo test --release -p synaptix-image-flux2 --test low_vram -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux2::model::SampleParams;
use synaptix_image_flux2::Flux2Model;

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
    let Some(model) = env("FLUX2_MODEL") else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let budget = (env("FLUX2_VRAM_GB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(7.0) * 1e9) as usize;
    let quant = match env("FLUX2_QUANT").as_deref() {
        Some("mxfp8") => DType::MXFP8,
        Some("bf16") | Some("dense") => DType::BF16,
        _ => DType::NVFP4,
    };
    let size = env("FLUX2_SIZE").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
    let prompt = env("FLUX2_PROMPT").unwrap_or_else(|| {
        "a cozy reading nook by a rainy window, warm lamp light, a ginger cat asleep on a knitted blanket, photo".into()
    });
    let dev = Device::Cuda(0);

    // Балласт: модели остаётся `budget`.
    let start_free = free();
    let ballast_bytes = start_free.saturating_sub(budget);
    let ballast = if ballast_bytes > 0 {
        Some(Tensor::zeros((ballast_bytes,), DType::U8, dev).unwrap())
    } else {
        None
    };
    eprintln!(
        "свободно было {:.1} ГБ, балласт {:.1} ГБ → модели {:.1} ГБ",
        start_free as f64 / 1e9,
        ballast_bytes as f64 / 1e9,
        free() as f64 / 1e9
    );
    let avail0 = free();
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

    let t0 = Instant::now();
    let m = Flux2Model::open(&model, dev, DType::BF16, quant).unwrap();
    let variant = m.variant();
    let steps = env("FLUX2_STEPS").and_then(|v| v.parse().ok()).unwrap_or(variant.default_steps());
    let guidance = variant.default_guidance();
    let cond = m.encode_prompt(&prompt, variant.uses_cfg(guidance)).unwrap();
    let t_enc = t0.elapsed().as_secs_f64();

    let refs = env("FLUX2_EDIT").map(|path| {
        let img = load_ppm(&path);
        m.encode_references(&[img]).unwrap()
    });
    let tokens = Flux2Model::tokens_for(size, size, refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0));
    let t1 = Instant::now();
    let t = m.load_transformer(tokens).unwrap();
    let t_load = t1.elapsed().as_secs_f64();
    let r = t.residency();
    let p = SampleParams { width: size, height: size, steps, guidance, seed: 7, denoise: 1.0 };
    let t2 = Instant::now();
    let lat = m.sample(&t, &cond, refs.as_ref(), None, &p, &mut |_, _| true).unwrap();
    let t_sample = t2.elapsed().as_secs_f64();
    drop(t);
    let t3 = Instant::now();
    let img = m.decode(&lat).unwrap();
    let t_dec = t3.elapsed().as_secs_f64();

    stop.store(true, Ordering::Relaxed);
    mon.join().unwrap();
    let peak = avail0.saturating_sub(min_free.load(Ordering::Relaxed));
    eprintln!(
        "{variant:?} {quant:?} {size}² {steps} шагов: энкодер {t_enc:.1} с, DiT {t_load:.1} с \
         (на карте {} / хост {} / источник {}), денойз {t_sample:.1} с, VAE {t_dec:.1} с; \
         минимум свободной VRAM {:.2} ГБ из бюджета {:.1} ГБ (пик ≈ {:.2} ГБ)",
        r.device,
        r.host,
        r.source,
        min_free.load(Ordering::Relaxed) as f64 / 1e9,
        budget as f64 / 1e9,
        peak as f64 / 1e9,
    );
    if let Some(out) = env("FLUX2_OUT") {
        let name = std::path::Path::new(&model).file_name().unwrap().to_string_lossy().replace(".syn", "");
        let suffix = if env("FLUX2_EDIT").is_some() { "_edit" } else { "" };
        save_ppm(&img, &format!("{out}/{name}_{quant:?}_{size}{suffix}.ppm"));
    }
    drop(ballast);
}

fn load_ppm(path: &str) -> Tensor {
    let bytes = std::fs::read(path).unwrap();
    // P6\nW H\n255\n
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[s..i]).to_string());
    }
    i += 1;
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let px = &bytes[i..i + w * h * 3];
    let mut v = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                v[c * h * w + y * w + x] = px[(y * w + x) * 3 + c] as f32 / 255.0;
            }
        }
    }
    let (h16, w16) = (h / 16 * 16, w / 16 * 16);
    Tensor::from_vec(v, (3, h, w), Device::Cpu).unwrap().narrow(1, 0, h16).unwrap().narrow(2, 0, w16).unwrap().contiguous().unwrap()
}
