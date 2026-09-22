//! Сверка с эталоном diffusers main (CPU, F32), снятым скриптом
//! `scripts/reference/gen_qwen_image21.py`: VAE RGBA, препроцессинг
//! референса (Lanczos с премультипликацией, композит на белом, процессор
//! Qwen3-VL), башня зрения с deepstack, токены и кондиционирование (t2i, с
//! картинкой, негатив), DiT на NL блоках (prefill с KV-кэшем, decode из
//! кэша, без кэша), sigmas расписания и сквозной прогон полной модели.
//!
//! ```sh
//! QWEN21_MODEL=…/Qwen-Image-2.1 QWEN21_REF=…/refout/q21 \
//!   cargo test --release -p synaptix-image-qwen21 --test reference -- --ignored --nocapture --test-threads=1
//! ```
//! `QWEN21_MODEL` может быть и `.syn`-бандлом. `NL` — сколько блоков DiT снял
//! эталон (2).

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_qwen::scheduler::QwenScheduler;
use synaptix_image_qwen::source;
use synaptix_image_qwen21::image::{vision_patches, RgbaImage, VisionPatches};
use synaptix_image_qwen21::model::{pack, unpack, SampleParams};
use synaptix_image_qwen21::text_encoder::{ImageEmbeds, TextEncoder};
use synaptix_image_qwen21::transformer::{build_rope, Layout, Placement, QwenImage21Transformer};
use synaptix_image_qwen21::vision::VisionTower;
use synaptix_image_qwen21::QwenImage21Model;

const PROMPT: &str = "Replace the background with a sunset beach; keep the red circle and the gradient stripe unchanged";
const T2I_PROMPT: &str = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement";
const RES: usize = 256;

fn setup() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var("QWEN21_MODEL").ok()?;
    let refd = std::env::var("QWEN21_REF").ok()?;
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    Some((PathBuf::from(model), PathBuf::from(refd)))
}

fn nl() -> usize {
    std::env::var("NL").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
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

fn ref_tensor(dir: &PathBuf, name: &str, dev: Device) -> Tensor {
    let (v, s) = load_ref(dir, name);
    Tensor::from_vec(v, s, dev).unwrap()
}

fn ref_ids(dir: &PathBuf, name: &str) -> Vec<u32> {
    load_ref(dir, name).0.iter().map(|v| v.round() as u32).collect()
}

fn ref_mask(dir: &PathBuf, name: &str) -> Vec<bool> {
    load_ref(dir, name).0.iter().map(|v| *v > 0.5).collect()
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

fn open(model: &PathBuf, quant: DType) -> QwenImage21Model {
    QwenImage21Model::open(model, Device::Cuda(0), DType::BF16, quant).unwrap()
}

fn png(dir: &PathBuf, name: &str) -> RgbaImage {
    let t = synaptix_io::image::png::load_image_rgba(dir.join(name), Device::Cpu).unwrap();
    RgbaImage::from_tensor(&t).unwrap()
}

#[test]
#[ignore]
fn vae_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let img = png(&refd, "vae_image.png");
    let (vin, _) = load_ref(&refd, "vae_input");
    let (cos, _, mx) = compare(&to_vec(&img.to_tensor().unwrap()), &vin);
    assert!(cos > 0.999999 && mx < 1e-6, "vae_input: cos {cos} max {mx}");
    let lat = m.encode_image(&img).unwrap();
    let (r, shape) = load_ref(&refd, "vae_latent");
    assert_eq!(lat.dims(), &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&lat), &r);
    eprintln!("vae encode: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9999, "cos {cos}");
    let dec = m.decode(&ref_tensor(&refd, "vae_latent", Device::Cpu)).unwrap();
    let (r, shape) = load_ref(&refd, "vae_decoded");
    assert_eq!(dec.dims(), &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&dec), &r);
    eprintln!("vae decode: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9999, "cos {cos}");
    // Плитки: та же картинка через плитки 128/96 — швы гладкие.
    let tiled = synaptix_image_qwen21::vae::decode(
        &m.source().weights(source::VAE).unwrap(),
        m.vae_config(),
        Device::Cuda(0),
        &ref_tensor(&refd, "vae_latent", Device::Cpu),
        Some(synaptix_image_qwen21::vae::Tiling { tile: 128, stride: 96 }),
    )
    .unwrap();
    let (cos, rel, mx) = compare(&to_vec(&tiled), &r);
    eprintln!("vae decode tiled 128/96: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.995, "tiled cos {cos}");
}

#[test]
#[ignore]
fn preprocess_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let img = png(&refd, "ref_image.png");
    assert!(img.has_alpha());
    let fitted = QwenImage21Model::fit_reference(&img, RES);
    let (sz, _) = load_ref(&refd, "cond_size");
    assert_eq!((fitted.width, fitted.height), (sz[0] as usize, sz[1] as usize));
    let (r, _) = load_ref(&refd, "edit_cond_rgba");
    let (cos, _, mx) = compare(&to_vec(&fitted.to_tensor().unwrap()), &r);
    eprintln!("cond rgba (Lanczos, премультипликация): cos {cos:.7} max {mx:.5}");
    assert!(mx < 1.5 / 255.0, "cond rgba max {mx}");
    let white = fitted.composite_white_chw();
    let (r, _) = load_ref(&refd, "edit_cond_white");
    let mine: Vec<f32> = white.iter().map(|&b| b as f32 / 255.0).collect();
    let (cos, _, mx) = compare(&mine, &r);
    eprintln!("composite white: cos {cos:.7} max {mx:.5}");
    assert!(mx < 1.5 / 255.0, "white max {mx}");
    let vc = &m.text_encoder_config().vision;
    let p = vision_patches(&fitted, vc, m.processor_config()).unwrap();
    let (g, _) = load_ref(&refd, "edit_grid_thw");
    assert_eq!(p.grid, (g[0] as usize, g[1] as usize, g[2] as usize));
    let (pv, _) = load_ref(&refd, "edit_pixel_values");
    let (cos, rel, mx) = compare(&to_vec(&p.patches), &pv);
    eprintln!("pixel_values: cos {cos:.7} rel {rel:.6} max {mx:.5}");
    assert!(cos > 0.9999, "pixel_values cos {cos}");
}

#[test]
#[ignore]
fn vision_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let vc = m.text_encoder_config().vision.clone();
    let (g, _) = load_ref(&refd, "edit_grid_thw");
    let grid = (g[0] as usize, g[1] as usize, g[2] as usize);
    let w = m.source().weights(source::TEXT_ENCODER).unwrap();
    let tower = VisionTower::load(&w, &vc, Device::Cuda(0), DType::F32).unwrap();
    let p = VisionPatches { patches: ref_tensor(&refd, "edit_pixel_values", Device::Cpu), grid };
    let (e, taps) = tower.forward(&p).unwrap();
    let (r, shape) = load_ref(&refd, "edit_vision_embeds");
    assert_eq!(e.dims(), &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&e), &r);
    eprintln!("vision embeds F32: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9999, "cos {cos}");
    assert_eq!(taps.len(), 3);
    for (i, t) in taps.iter().enumerate() {
        let (r, shape) = load_ref(&refd, &format!("edit_deepstack_{i}"));
        assert_eq!(t.dims(), &shape[..]);
        let (cos, rel, mx) = compare(&to_vec(t), &r);
        eprintln!("deepstack {i}: cos {cos:.6} rel {rel:.5} max {mx:.4}");
        assert!(cos > 0.9999, "deepstack {i} cos {cos}");
    }
}

#[test]
#[ignore]
fn tokens_match() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let (d, _) = load_ref(&refd, "drop_idx");
    assert_eq!(m.tokenizer().drop_idx(), d[0] as usize);
    let (g, _) = load_ref(&refd, "edit_grid_thw");
    let slots = (g[1] as usize / 2) * (g[2] as usize / 2);
    let t = m.tokenizer().encode(PROMPT, &[slots]).unwrap();
    assert_eq!(t.ids, ref_ids(&refd, "edit_input_ids"));
    let t = m.tokenizer().encode(T2I_PROMPT, &[]).unwrap();
    assert_eq!(t.ids, ref_ids(&refd, "t2i_input_ids"));
}

#[test]
#[ignore]
fn encoder_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let img = png(&refd, "ref_image.png");
    // Энкодер целиком на эталонных эмбеддингах башни (F32 LLM): устройство
    // слоёв, M-RoPE, deepstack, выход до нормы.
    {
        let w = m.source().weights(source::TEXT_ENCODER).unwrap();
        let (g, _) = load_ref(&refd, "edit_grid_thw");
        let grid = (g[1] as usize / 2, g[2] as usize / 2);
        let embeds = vec![ImageEmbeds {
            tokens: ref_tensor(&refd, "edit_vision_embeds", Device::Cpu),
            deepstack: (0..3).map(|i| ref_tensor(&refd, &format!("edit_deepstack_{i}"), Device::Cpu)).collect(),
            grid,
        }];
        let ids = ref_ids(&refd, "edit_input_ids");
        for (dt, min_cos) in [(DType::F32, 0.9999), (DType::BF16, 0.99)] {
            let enc = TextEncoder::build(w.clone(), m.text_encoder_config().clone(), Device::Cuda(0), dt, 512 << 20).unwrap();
            let h = enc.encode(&ids, &embeds).unwrap();
            let skip = m.tokenizer().drop_idx();
            let h = h.narrow(0, skip, h.dims()[0] - skip).unwrap().contiguous().unwrap();
            let (r, shape) = load_ref(&refd, "edit_prompt_embeds");
            assert_eq!(h.dims(), &shape[..]);
            let (cos, rel, mx) = compare(&to_vec(&h), &r);
            eprintln!("encoder {dt:?} на эталонной башне: cos {cos:.6} rel {rel:.5} max {mx:.3}");
            assert!(cos > min_cos, "cos {cos}");
            drop(enc);
            synaptix_image_qwen21::model::release_pools(Device::Cuda(0));
        }
    }
    // Весь путь: картинка → башня F32 → LLM BF16 → без системной части.
    let cond = m.encode_prompt(PROMPT, Some(" "), std::slice::from_ref(&img), RES).unwrap();
    assert_eq!(cond.image_pad_mask, ref_mask(&refd, "edit_image_pad_mask"));
    let (r, shape) = load_ref(&refd, "edit_prompt_embeds");
    assert_eq!(&cond.embeds.dims()[1..], &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&cond.embeds), &r);
    eprintln!("prompt embeds с картинкой (BF16): cos {cos:.6} rel {rel:.5} max {mx:.3}");
    assert!(cos > 0.99, "cos {cos}");
    // Негатив кодируется с теми же картинками: слоты те же, длина — как у
    // промпта из одного пробела плюс картинка.
    let (neg, neg_mask) = cond.negative.as_ref().unwrap();
    assert_eq!(neg_mask.iter().filter(|&&m| m).count(), cond.image_pad_mask.iter().filter(|&&m| m).count());
    assert!(neg.dims()[1] < cond.embeds.dims()[1]);
    // Эталон негатива снят без картинки.
    let cond_neg = m.encode_prompt(" ", None, &[], RES).unwrap();
    let (r, shape) = load_ref(&refd, "neg_prompt_embeds");
    assert_eq!(&cond_neg.embeds.dims()[1..], &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&cond_neg.embeds), &r);
    eprintln!("negative embeds (BF16): cos {cos:.6} rel {rel:.5} max {mx:.3}");
    assert!(cos > 0.99, "cos {cos}");
    let cond = m.encode_prompt(T2I_PROMPT, None, &[], RES).unwrap();
    assert!(cond.image_pad_mask.iter().all(|m| !m));
    let (r, shape) = load_ref(&refd, "t2i_prompt_embeds");
    assert_eq!(&cond.embeds.dims()[1..], &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&cond.embeds), &r);
    eprintln!("t2i prompt embeds (BF16): cos {cos:.6} rel {rel:.5} max {mx:.3}");
    assert!(cos > 0.99, "cos {cos}");
}

/// DiT на NL блоках: t2i и правка (prefill с кэшем → decode из кэша, и без
/// кэша), в F32 (устройство) и BF16/MXFP8/NVFP4 (рабочие точности).
#[test]
#[ignore]
fn dit_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let dev = Device::Cuda(0);
    let mut cfg = m.config().clone();
    cfg.num_layers = nl();
    let w = m.source().weights(source::TRANSFORMER).unwrap();
    let (sig1, sig2) = (0.9f32, 0.5f32);
    let hw = RES / 16;
    let (sz, _) = load_ref(&refd, "cond_size");
    let (cw, ch) = (sz[0] as usize / 16, sz[1] as usize / 16);

    let t2i_txt = ref_tensor(&refd, "t2i_prompt_embeds", Device::Cpu);
    let t2i_txt = t2i_txt.reshape((1, t2i_txt.dims()[0], t2i_txt.dims()[1])).unwrap();
    let t2i_mask = vec![false; t2i_txt.dims()[1]];
    let edit_txt = ref_tensor(&refd, "edit_prompt_embeds", Device::Cpu);
    let edit_txt = edit_txt.reshape((1, edit_txt.dims()[0], edit_txt.dims()[1])).unwrap();
    let edit_mask = ref_mask(&refd, "edit_image_pad_mask");
    let noise_t2i = ref_tensor(&refd, "dit_t2i_noise", dev);
    let noise_edit = ref_tensor(&refd, "dit_edit_noise", dev);
    let refs = ref_tensor(&refd, "edit_ref_tokens", dev);
    let edit_in = Tensor::cat(&[&refs, &noise_edit], 1).unwrap();

    for (compute, quant, min_cos) in [
        (DType::F32, DType::F32, 0.9999),
        (DType::BF16, DType::BF16, 0.999),
        (DType::BF16, DType::MXFP8, 0.995),
        (DType::BF16, DType::NVFP4, 0.98),
    ] {
        let tr = QwenImage21Transformer::load(&w, &cfg, dev, compute, quant, Placement::Resident, 2048).unwrap();
        let mods = tr.modulations(&[sig1, sig2]).unwrap();
        // t2i.
        let layout = Layout::build(&t2i_mask, &[(hw, hw)]).unwrap();
        let (cos, sin) = build_rope(&layout.positions, &cfg.axes_dims, dev).unwrap();
        let txt = tr.embed_text(&t2i_txt).unwrap();
        let out = tr.forward(&txt, &noise_t2i, &layout, &mods, 0, &cos, &sin, None).unwrap();
        let (r, shape) = load_ref(&refd, "dit_t2i_out");
        assert_eq!(out.dims(), &shape[..]);
        let (c, rel, mx) = compare(&to_vec(&out), &r);
        eprintln!("dit t2i {compute:?}/{quant:?}: cos {c:.6} rel {rel:.5} max {mx:.3}");
        assert!(c > min_cos, "t2i cos {c}");
        // Правка: prefill + кэш, decode из кэша, без кэша.
        let layout = Layout::build(&edit_mask, &[(ch, cw), (ch, cw)]).unwrap();
        let (cos, sin) = build_rope(&layout.positions, &cfg.axes_dims, dev).unwrap();
        let txt = tr.embed_text(&edit_txt).unwrap();
        let mut cache = tr.new_cache();
        let o1 = tr.forward(&txt, &edit_in, &layout, &mods, 0, &cos, &sin, Some(&mut cache)).unwrap();
        assert!(cache.is_filled());
        let (r, _) = load_ref(&refd, "dit_edit_out1");
        let (c, rel, mx) = compare(&to_vec(&o1), &r);
        eprintln!("dit edit prefill {compute:?}/{quant:?}: cos {c:.6} rel {rel:.5} max {mx:.3}");
        assert!(c > min_cos, "edit prefill cos {c}");
        let o2 = tr.forward(&txt, &edit_in, &layout, &mods, 1, &cos, &sin, Some(&mut cache)).unwrap();
        let (r, _) = load_ref(&refd, "dit_edit_out2_cached");
        let (c, rel, mx) = compare(&to_vec(&o2), &r);
        eprintln!("dit edit cached {compute:?}/{quant:?}: cos {c:.6} rel {rel:.5} max {mx:.3}");
        assert!(c > min_cos, "edit cached cos {c}");
        let o2f = tr.forward(&txt, &edit_in, &layout, &mods, 1, &cos, &sin, None).unwrap();
        let (r, _) = load_ref(&refd, "dit_edit_out2_full");
        let (c, rel, mx) = compare(&to_vec(&o2f), &r);
        eprintln!("dit edit no-cache {compute:?}/{quant:?}: cos {c:.6} rel {rel:.5} max {mx:.3}");
        assert!(c > min_cos, "edit full cos {c}");
        drop((tr, mods, cache));
        synaptix_image_qwen21::model::release_pools(dev);
    }
}

#[test]
#[ignore]
fn sigmas_match() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let (sz, _) = load_ref(&refd, "cond_size");
    for (name, n_tok) in [("sigmas_t2i", (RES / 16) * (RES / 16)), ("sigmas_edit", (sz[0] as usize / 16) * (sz[1] as usize / 16))] {
        let s = QwenScheduler::new(8, n_tok, m.scheduler_config());
        let (r, _) = load_ref(&refd, name);
        assert_eq!(s.sigmas().len(), r.len());
        for (i, (a, b)) in s.sigmas().iter().zip(&r).enumerate() {
            assert!((a - b).abs() < 2e-6, "{name}[{i}]: {a} vs {b}");
        }
    }
}

/// Полная модель в BF16 против сквозного прогона эталона (BF16 на CPU): t2i
/// 256² и правка с референсом, 4 шага, тот же шум.
#[test]
#[ignore]
fn e2e_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let dev = Device::Cuda(0);
    let hw = RES / 16;
    let cond = m.encode_prompt(T2I_PROMPT, None, &[], RES).unwrap();
    let tr = m.load_transformer(QwenImage21Model::tokens_for(RES, RES, 0, cond.embeds.dims()[1])).unwrap();
    let noise = unpack(&ref_tensor(&refd, "e2e_t2i_noise", dev), hw, hw).unwrap();
    let p = SampleParams { width: RES, height: RES, steps: 4, cfg: 1.0, seed: 0, kv_cache: true };
    let lat = m.sample_from_noise(&tr, &cond, None, Some(&noise), &p, &mut |_, _| true).unwrap();
    let (r, _) = load_ref(&refd, "e2e_t2i_latent");
    let (c, rel, mx) = compare(&to_vec(&pack(&lat).unwrap()), &r);
    eprintln!("e2e t2i latent: cos {c:.5} rel {rel:.4} max {mx:.3}");
    assert!(c > 0.97, "t2i latent cos {c}");
    let img = m.decode(&lat).unwrap();
    let (r, _) = load_ref(&refd, "e2e_t2i_image");
    let (c, rel, mx) = compare(&to_vec(&img), &r);
    eprintln!("e2e t2i image: cos {c:.5} rel {rel:.4} max {mx:.3}");
    assert!(c > 0.97, "t2i image cos {c}");
    drop(tr);

    let src = png(&refd, "ref_image.png");
    let cond = m.encode_prompt(PROMPT, None, std::slice::from_ref(&src), RES).unwrap();
    let refs = m.encode_references(std::slice::from_ref(&src), RES).unwrap();
    let (r, _) = load_ref(&refd, "edit_ref_tokens");
    let (c, rel, mx) = compare(&to_vec(&refs.tokens), &r);
    eprintln!("reference tokens: cos {c:.6} rel {rel:.5} max {mx:.3}");
    assert!(c > 0.9999, "ref tokens cos {c}");
    let (w, h) = QwenImage21Model::default_size(&[(src.width, src.height)], RES);
    let tr = m.load_transformer(QwenImage21Model::tokens_for(w, h, refs.num_tokens(), cond.embeds.dims()[1])).unwrap();
    let noise = unpack(&ref_tensor(&refd, "e2e_edit_noise", dev), h / 16, w / 16).unwrap();
    let p = SampleParams { width: w, height: h, steps: 4, cfg: 1.0, seed: 0, kv_cache: true };
    let lat = m.sample_from_noise(&tr, &cond, Some(&refs), Some(&noise), &p, &mut |_, _| true).unwrap();
    let (r, _) = load_ref(&refd, "e2e_edit_latent");
    let (c, rel, mx) = compare(&to_vec(&pack(&lat).unwrap()), &r);
    eprintln!("e2e edit latent: cos {c:.5} rel {rel:.4} max {mx:.3}");
    assert!(c > 0.97, "edit latent cos {c}");
    let img = m.decode(&lat).unwrap();
    let (r, _) = load_ref(&refd, "e2e_edit_image");
    let (c, rel, mx) = compare(&to_vec(&img), &r);
    eprintln!("e2e edit image: cos {c:.5} rel {rel:.4} max {mx:.3}");
    assert!(c > 0.97, "edit image cos {c}");
}
