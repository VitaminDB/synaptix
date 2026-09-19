//! Стадии SDXL на GPU: txt2img 1024² и img2img из результата, отмена.
//!
//! ```sh
//! SDXL_MODEL=~/Storage/syn_models/sdxl-base-1.0.syn SDXL_OUT=/tmp/sdxl \
//!   cargo test --release -p synaptix-image-sdxl --test stages_e2e -- --ignored --nocapture --test-threads=1
//! ```
//! `SDXL_QUANT=mxfp8|nvfp4` — квант UNet, `SDXL_STEPS` — шагов (30).

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_sdxl::{SdxlCheckpoint, SdxlError, SdxlSampleParams};

fn setup() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var("SDXL_MODEL").ok()?;
    let out = PathBuf::from(std::env::var("SDXL_OUT").unwrap_or_else(|_| "/tmp/sdxl_e2e".into()));
    std::fs::create_dir_all(&out).unwrap();
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    Some((PathBuf::from(model), out))
}

fn stats(img: &Tensor) -> (f32, f32) {
    let v = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    assert!(v.iter().all(|x| x.is_finite()), "NaN/Inf в картинке");
    (mean, var.sqrt())
}

fn quant() -> DType {
    match std::env::var("SDXL_QUANT").as_deref() {
        Ok("mxfp8") => DType::MXFP8,
        Ok("nvfp4") => DType::NVFP4,
        _ => DType::F16,
    }
}

#[test]
#[ignore]
fn txt2img_then_img2img() {
    let Some((model, out)) = setup() else { return };
    let dev = Device::Cuda(0);
    let ck = SdxlCheckpoint::open(&model, dev).unwrap();
    let t0 = std::time::Instant::now();
    let cond = ck
        .encode_prompt(
            "a lighthouse on a rocky cliff at golden hour, crashing waves, dramatic clouds, highly detailed photograph",
            "blurry, low quality, deformed, watermark, text",
        )
        .unwrap();
    eprintln!("encode {:.1} с, {:?}", t0.elapsed().as_secs_f64(), cond);
    let unet = ck.load_unet(quant()).unwrap();
    let steps = std::env::var("SDXL_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let p = SdxlSampleParams { width: 1024, height: 1024, steps, guidance: 5.0, seed: 42, denoise: 1.0 };
    let lat = ck.sample(&unet, &cond, None, &p, &mut |_, _| true).unwrap();
    assert_eq!(lat.dims(), &[1, 4, 128, 128]);
    let img = ck.decode(&lat).unwrap();
    let (m, s) = stats(&img);
    eprintln!("txt2img: mean {m:.3} std {s:.3}");
    assert!(s > 0.08, "картинка почти однотонная (std {s})");
    synaptix_io::image::png::save_image(&img, out.join("txt2img.png")).unwrap();

    // img2img: латент картинки → лёгкая правка; результат близок к исходнику.
    let x0 = ck.encode_image(&img).unwrap();
    let rec = ck.decode(&x0).unwrap();
    let mae = rec.sub(&img).unwrap().abs().unwrap().mean().unwrap().to_scalar::<f32>().unwrap();
    eprintln!("VAE encode→decode MAE {mae:.4}");
    assert!(mae < 0.05, "VAE encode→decode MAE {mae}");
    let cond2 = ck.encode_prompt("the same lighthouse as an oil painting, thick brush strokes", "").unwrap();
    let p2 = SdxlSampleParams { denoise: 0.5, ..p.clone() };
    let lat2 = ck.sample(&unet, &cond2, Some(&x0), &p2, &mut |_, _| true).unwrap();
    let img2 = ck.decode(&lat2).unwrap();
    let diff = img2.sub(&img).unwrap().abs().unwrap().mean().unwrap().to_scalar::<f32>().unwrap();
    eprintln!("img2img denoise 0.5: MAE к исходнику {diff:.4}");
    assert!(diff > 0.01 && diff < 0.3, "img2img: MAE {diff}");
    synaptix_io::image::png::save_image(&img2, out.join("img2img.png")).unwrap();

    // Отмена на втором шаге.
    let mut k = 0;
    let r = ck.sample(&unet, &cond, None, &p, &mut |i, _| {
        k = i;
        i < 2
    });
    assert!(matches!(r, Err(SdxlError::Cancelled)), "отмена не сработала");
    assert_eq!(k, 2);
}

/// Латент VAE Encode против diffusers (`AutoencoderKL.encode(x).latent_dist.mean
/// × scaling_factor`, F32): `SDXL_IMAGE` — картинка, `SDXL_REF_LATENT` —
/// эталон (raw f32 + `.json` с формой).
#[test]
#[ignore]
fn vae_encode_matches_diffusers() {
    let Some((model, _)) = setup() else { return };
    let (Ok(img), Ok(refp)) = (std::env::var("SDXL_IMAGE"), std::env::var("SDXL_REF_LATENT")) else { return };
    let ck = SdxlCheckpoint::open(&model, Device::Cuda(0)).unwrap();
    let img = synaptix_io::image::png::load_image(&img, Device::Cpu).unwrap();
    let img = img.narrow(0, 0, 3).unwrap().contiguous().unwrap();
    let lat = ck.encode_image(&img).unwrap();
    let r: Vec<f32> = std::fs::read(&refp).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let v = lat.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(v.len(), r.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in v.iter().zip(&r) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let (m, s) = stats(&lat);
    eprintln!("латент: mean {m:.4} std {s:.4}, cos к diffusers {cos:.6}");
    assert!(cos > 0.999, "cos {cos}");
}

fn load_ref(dir: &std::path::Path, name: &str) -> Tensor {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).unwrap()).unwrap();
    let shape: Vec<usize> = meta["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let v: Vec<f32> = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::from_vec(v, shape, Device::Cpu).unwrap()
}

fn cos(a: &Tensor, b: &Tensor) -> f64 {
    let (a, b) = (a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

/// Один шаг img2img против diffusers (`SDXL_STEP_REF` — каталог с hidden,
/// pooled, xt, eps, meta.json): ε на зашумлённом латенте картинки.
#[test]
#[ignore]
fn img2img_step_matches_diffusers() {
    let Some((model, _)) = setup() else { return };
    let Ok(dir) = std::env::var("SDXL_STEP_REF") else { return };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let (t, sigma) = (meta["t"].as_f64().unwrap() as f32, meta["sigma"].as_f64().unwrap() as f32);
    let ck = SdxlCheckpoint::open(&model, Device::Cuda(0)).unwrap();
    let unet = ck.load_unet(DType::F16).unwrap();
    let cond = synaptix_image_sdxl::SdxlConditioning { hidden: load_ref(&dir, "hidden"), pooled: load_ref(&dir, "pooled") };
    let eps = ck.unet_eps(&unet, &cond, &load_ref(&dir, "xt"), t, sigma, 1024, 1024).unwrap().to_device(Device::Cpu).unwrap();
    let c = cos(&eps, &load_ref(&dir, "eps"));
    eprintln!("ε img2img-шага (t {t}, σ {sigma}): cos к diffusers {c:.6}");
    // Свой энкодер промпта против эталонного кондиционирования.
    let mine = ck
        .encode_prompt("the same lighthouse in winter, snow on the cliff, overcast", "blurry, low quality, deformed, watermark, text")
        .unwrap();
    eprintln!(
        "hidden cos {:.6}, pooled cos {:.6}",
        cos(&mine.hidden, &load_ref(&dir, "hidden")),
        cos(&mine.pooled, &load_ref(&dir, "pooled"))
    );
    assert!(c > 0.99, "cos {c}");
}

/// Хвост img2img (4 шага из 8 с силой 0.5) из эталонного `xt` против
/// diffusers (`tail_latent`, `sdxl_tail_ref.py`).
#[test]
#[ignore]
fn img2img_tail_matches_diffusers() {
    let Some((model, out)) = setup() else { return };
    let Ok(dir) = std::env::var("SDXL_STEP_REF") else { return };
    let dir = PathBuf::from(dir);
    let ck = SdxlCheckpoint::open(&model, Device::Cuda(0)).unwrap();
    let unet = ck.load_unet(DType::F16).unwrap();
    let cond = synaptix_image_sdxl::SdxlConditioning { hidden: load_ref(&dir, "hidden"), pooled: load_ref(&dir, "pooled") };
    let sched = synaptix_image_sdxl::scheduler::SdxlEuler::new(8, &Default::default());
    let start = sched.start_for_strength(0.5);
    let mut x = load_ref(&dir, "xt").to_device(Device::Cuda(0)).unwrap();
    for i in start..sched.num_steps() {
        let e = ck.unet_eps(&unet, &cond, &x, sched.timestep(i), sched.sigma(i), 1024, 1024).unwrap();
        let (eu, ec) = (e.narrow(0, 0, 1).unwrap(), e.narrow(0, 1, 1).unwrap());
        let g = eu.add(&ec.sub(&eu).unwrap().mul_scalar(5.0).unwrap()).unwrap();
        x = x.add(&g.mul_scalar(sched.dt(i)).unwrap()).unwrap();
    }
    let x = x.to_device(Device::Cpu).unwrap();
    let c = cos(&x, &load_ref(&dir, "tail_latent"));
    let (m, s) = stats(&x);
    eprintln!("хвост img2img: латент mean {m:.4} std {s:.4}, cos к diffusers {c:.6}");
    let img = ck.decode(&x).unwrap();
    synaptix_io::image::png::save_image(&img, out.join("tail_mine.png")).unwrap();
    assert!(c > 0.99, "cos {c}");
}
