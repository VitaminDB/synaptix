//! Сверка с эталоном diffusers (CPU, F32), снятым скриптом
//! `scripts/reference/gen_flux2_klein.py`: токены промпта, эмбеддинги
//! энкодера, один шаг DiT с тем же шумом, VAE и полная генерация.
//!
//! ```sh
//! FLUX2_MODEL=…/FLUX.2-klein-4B FLUX2_REF=…/refout/klein4b_512 \
//!   cargo test --release -p synaptix-image-flux2 --test reference -- --ignored --nocapture --test-threads=1
//! ```
//! `FLUX2_MODEL` может быть и `.syn`-бандлом.

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux2::model::{pack, SampleParams};
use synaptix_image_flux2::text_encoder::MAX_SEQ;
use synaptix_image_flux2::transformer::{build_rope, Placement};
use synaptix_image_flux2::Flux2Model;

fn setup() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var("FLUX2_MODEL").ok()?;
    let refd = std::env::var("FLUX2_REF").ok()?;
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    Some((PathBuf::from(model), PathBuf::from(refd)))
}

fn load_ref(dir: &PathBuf, name: &str) -> (Vec<f32>, Vec<usize>) {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).unwrap()).unwrap();
    let shape: Vec<usize> = meta["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(v.len(), shape.iter().product::<usize>());
    (v, shape)
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// (косинус, относительная L2-ошибка, max |Δ|).
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut d2, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        d2 += (x - y) * (x - y);
        mx = mx.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), (d2 / nb.max(1e-30)).sqrt(), mx)
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (1.0 / mse.max(1e-12)).log10()
}

fn prompt(refd: &PathBuf) -> (String, Vec<u32>) {
    let p: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("prompt.json")).unwrap()).unwrap();
    let ids = p["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    (p["prompt"].as_str().unwrap().to_string(), ids)
}

fn open(model: &PathBuf, quant: DType) -> Flux2Model {
    Flux2Model::open(model, Device::Cuda(0), DType::BF16, quant).unwrap()
}

#[test]
#[ignore]
fn prompt_embeds_match() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let cond = m.encode_prompt(&prompt(&refd).0, false).unwrap();
    let (r, shape) = load_ref(&refd, "prompt_embeds");
    assert_eq!(cond.embeds.dims(), &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&cond.embeds), &r);
    eprintln!("prompt_embeds: cos {cos:.6} rel {rel:.4} max {mx:.3}");
    assert!(cos > 0.999, "cos {cos}");
}

fn dit_step0(quant: DType, min_cos: f64) {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, quant).with_memory(Placement::Auto);
    let (pe, pe_shape) = load_ref(&refd, "prompt_embeds");
    let (noise, ns) = load_ref(&refd, "noise");
    let (sig, _) = load_ref(&refd, "sigmas");
    let (h, w) = (ns[2], ns[3]);
    let dev = Device::Cuda(0);
    let txt = Tensor::from_vec(pe, pe_shape, dev).unwrap().to_dtype(DType::BF16).unwrap();
    let img = pack(&Tensor::from_vec(noise, ns.clone(), dev).unwrap()).unwrap();
    let t = m.load_transformer(MAX_SEQ + h * w).unwrap();
    let cfg = m.config().clone();
    let mut ids: Vec<[f64; 4]> = (0..MAX_SEQ).map(|l| [0.0, 0.0, 0.0, l as f64]).collect();
    for y in 0..h {
        for x in 0..w {
            ids.push([0.0, y as f64, x as f64, 0.0]);
        }
    }
    let (c, s) = build_rope(&ids, &cfg.axes_dims, cfg.rope_theta, dev).unwrap();
    let v = t.forward(&img, &txt, sig[0], None, &c, &s).unwrap();
    let (r, _) = load_ref(&refd, "dit_out0");
    let (cos, rel, mx) = compare(&to_vec(&v), &r);
    eprintln!("dit step0 {quant:?}: cos {cos:.6} rel {rel:.4} max {mx:.3}");
    assert!(cos > min_cos, "cos {cos}");
}

#[test]
#[ignore]
fn dit_step0_dense() {
    dit_step0(DType::BF16, 0.999);
}

#[test]
#[ignore]
fn dit_step0_mxfp8() {
    dit_step0(DType::MXFP8, 0.994);
}

#[test]
#[ignore]
fn dit_step0_nvfp4() {
    dit_step0(DType::NVFP4, 0.96);
}

#[test]
#[ignore]
fn vae_roundtrip_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let (img, is) = load_ref(&refd, "image");
    let image = Tensor::from_vec(img.clone(), vec![is[1], is[2], is[3]], Device::Cpu).unwrap();
    // encode → сравнение с модой эталона (упакованной и нормированной тем же
    // путём, что в пайплайне) через decode: картинки должны совпасть.
    let lat = m.encode_image(&image).unwrap();
    let dec = m.decode(&lat).unwrap();
    let (rdec, _) = load_ref(&refd, "vae_dec");
    // эталон — сырой выход декодера в [-1, 1]
    let rdec01: Vec<f32> = rdec.iter().map(|v| (v * 0.5 + 0.5).clamp(0.0, 1.0)).collect();
    let p = psnr(&to_vec(&dec), &rdec01);
    eprintln!("vae encode→decode vs эталон: PSNR {p:.2} dB");
    assert!(p > 40.0, "psnr {p}");
}

#[test]
#[ignore]
fn full_generation_matches() {
    let Some((model, refd)) = setup() else { return };
    for quant in [DType::BF16, DType::MXFP8, DType::NVFP4] {
        let m = open(&model, quant);
        let (pe, pe_shape) = load_ref(&refd, "prompt_embeds");
        let (noise, ns) = load_ref(&refd, "noise");
        let cond = synaptix_image_flux2::Flux2Conditioning {
            embeds: Tensor::from_vec(pe, pe_shape, Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap(),
            negative: None,
        };
        let (h, w) = (ns[2] * 16, ns[3] * 16);
        let t = m.load_transformer(Flux2Model::tokens_for(w, h, 0)).unwrap();
        let p = SampleParams {
            width: w,
            height: h,
            steps: m.variant().default_steps(),
            guidance: m.variant().default_guidance(),
            seed: 0,
            denoise: 1.0,
        };
        let noise = Tensor::from_vec(noise, ns, Device::Cpu).unwrap();
        let lat = m.sample_from_noise(&t, &cond, None, None, Some(&noise), &p, &mut |_, _| true).unwrap();
        drop(t);
        let img = m.decode(&lat).unwrap();
        let (r, _) = load_ref(&refd, "image");
        let ps = psnr(&to_vec(&img), &r);
        eprintln!("генерация {quant:?} vs эталон: PSNR {ps:.2} dB");
        if let Ok(out) = std::env::var("FLUX2_OUT") {
            save_ppm(&img, &format!("{out}/klein_{quant:?}.ppm"));
        }
        let min = if quant == DType::BF16 { 28.0 } else { 20.0 };
        assert!(ps > min, "{quant:?}: psnr {ps}");
    }
}

/// Все блоки DiT стримятся прямо из источника (путь «RAM не хватает»):
/// квант делается на каждом шаге заново, результат — тот же, что резидентно.
#[test]
#[ignore]
fn full_generation_streamed_from_source() {
    let Some((model, refd)) = setup() else { return };
    std::env::set_var("FLUX2_STREAM_FROM", "source");
    for quant in [DType::MXFP8, DType::BF16] {
        let m = open(&model, quant).with_memory(Placement::Stream);
        let (pe, pe_shape) = load_ref(&refd, "prompt_embeds");
        let (noise, ns) = load_ref(&refd, "noise");
        let cond = synaptix_image_flux2::Flux2Conditioning {
            embeds: Tensor::from_vec(pe, pe_shape, Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap(),
            negative: None,
        };
        let (h, w) = (ns[2] * 16, ns[3] * 16);
        let t = m.load_transformer(Flux2Model::tokens_for(w, h, 0)).unwrap();
        assert_eq!(t.residency().source, m.config().num_blocks());
        let p = SampleParams {
            width: w,
            height: h,
            steps: m.variant().default_steps(),
            guidance: m.variant().default_guidance(),
            seed: 0,
            denoise: 1.0,
        };
        let noise = Tensor::from_vec(noise, ns, Device::Cpu).unwrap();
        let lat = m.sample_from_noise(&t, &cond, None, None, Some(&noise), &p, &mut |_, _| true).unwrap();
        drop(t);
        let img = m.decode(&lat).unwrap();
        let (r, _) = load_ref(&refd, "image");
        let ps = psnr(&to_vec(&img), &r);
        eprintln!("генерация {quant:?}, все блоки из источника: PSNR {ps:.2} дБ");
        let min = if quant == DType::BF16 { 28.0 } else { 20.0 };
        assert!(ps > min, "{quant:?}: psnr {ps}");
    }
    std::env::remove_var("FLUX2_STREAM_FROM");
}

fn save_ppm(img: &Tensor, path: &str) {
    let d = img.dims().to_vec();
    let (h, w) = (d[1], d[2]);
    let v = to_vec(img);
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
fn tokens_match() {
    let Some((model, refd)) = setup() else { return };
    let src = synaptix_image_flux2::Flux2Source::open(&model).unwrap();
    let te = synaptix_image_flux2::TextEncoderConfig::from_json(&src.read("text_encoder/config.json").unwrap()).unwrap();
    let tok = synaptix_image_flux2::text_encoder::open_tokenizer(&src.read("tokenizer/tokenizer.json").unwrap(), te.arch).unwrap();
    let (p, ids) = prompt(&refd);
    let ours = tok.encode(&p).unwrap();
    assert_eq!(ours.ids, ids);
}

/// Урезанный DiT dev (1 double + 1 single блок на настоящих весах, все
/// эмбеддеры, guidance-эмбеддинг, модуляция, финал) против diffusers:
/// `scripts/reference/gen_flux2_dev_mini.py`. 32B целиком на CPU не поднять,
/// а guidance-эмбеддинг есть только у dev — его klein не проверяет.
///
/// ```sh
/// FLUX2_DEV=…/flux.2-dev.syn FLUX2_DEV_MINI_REF=…/refout/dev_mini \
///   cargo test --release -p synaptix-image-flux2 --test reference dev_mini -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn dev_mini_dit_matches() {
    let (Ok(model), Ok(refd)) = (std::env::var("FLUX2_DEV"), std::env::var("FLUX2_DEV_MINI_REF")) else {
        return;
    };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let refd = PathBuf::from(refd);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("mini.json")).unwrap()).unwrap();
    let (sigma, guidance) = (meta["sigma"].as_f64().unwrap() as f32, meta["guidance"].as_f64().unwrap() as f32);
    let (h, w, l) = (
        meta["h"].as_u64().unwrap() as usize,
        meta["w"].as_u64().unwrap() as usize,
        meta["txt"].as_u64().unwrap() as usize,
    );
    let src = synaptix_image_flux2::Flux2Source::open(&model).unwrap();
    let mut cfg = synaptix_image_flux2::Flux2Model::open(&model, Device::Cuda(0), DType::BF16, DType::BF16)
        .unwrap()
        .config()
        .clone();
    assert!(cfg.guidance_embeds, "это не dev: нет guidance-эмбеддинга");
    cfg.num_layers = 1;
    cfg.num_single_layers = 1;
    let dev = Device::Cuda(0);
    // FLUX2_MINI_F32=1 — счёт в F32: вклад guidance мал, и в BF16 его
    // заметно шумит округление.
    let dt = if std::env::var("FLUX2_MINI_F32").is_ok_and(|v| v == "1") { DType::F32 } else { DType::BF16 };
    let weights = src.weights(synaptix_image_flux2::source::TRANSFORMER).unwrap();
    let t = synaptix_image_flux2::Flux2Transformer::load(
        &weights,
        &cfg,
        dev,
        dt,
        dt,
        Placement::Resident,
        l + h * w,
    )
    .unwrap();
    let (img, is) = load_ref(&refd, "mini_img");
    let (txt, ts) = load_ref(&refd, "mini_txt");
    let img = Tensor::from_vec(img, is, dev).unwrap().to_dtype(dt).unwrap();
    let txt = Tensor::from_vec(txt, ts, dev).unwrap().to_dtype(dt).unwrap();
    let mut ids: Vec<[f64; 4]> = (0..l).map(|i| [0.0, 0.0, 0.0, i as f64]).collect();
    for y in 0..h {
        for x in 0..w {
            ids.push([0.0, y as f64, x as f64, 0.0]);
        }
    }
    let (c, s) = build_rope(&ids, &cfg.axes_dims, cfg.rope_theta, dev).unwrap();
    let v = t.forward(&img, &txt, sigma, Some(guidance), &c, &s).unwrap();
    let (r, _) = load_ref(&refd, "mini_out");
    let (cos, rel, mx) = compare(&to_vec(&v), &r);
    eprintln!("dev mini DiT {dt:?} (guidance {guidance}): cos {cos:.6} rel {rel:.4} max {mx:.3}");
    // Вклад guidance в одном блоке мал (косинус выходов g=4 и g=1 ~0,99997),
    // поэтому отдельно сверяется именно он: разность выходов g=4 − g=1.
    let v1 = t.forward(&img, &txt, sigma, Some(1.0), &c, &s).unwrap();
    let (r1, _) = load_ref(&refd, "mini_out_g1");
    let ours: Vec<f32> = to_vec(&v).iter().zip(to_vec(&v1)).map(|(a, b)| a - b).collect();
    let theirs: Vec<f32> = r.iter().zip(&r1).map(|(a, b)| a - b).collect();
    let (dcos, drel, _) = compare(&ours, &theirs);
    eprintln!("  вклад guidance (g=4 − g=1): cos {dcos:.6} rel {drel:.4}");
    assert!(cos > 0.999, "cos {cos}");
    assert!(dcos > 0.99, "вклад guidance: cos {dcos}");
}
