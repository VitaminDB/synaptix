//! Сверка с эталоном diffusers 0.40 (CPU, F32), снятым скриптом
//! `scripts/reference/gen_qwen_image_edit.py`: VAE, препроцессинг и башня
//! зрения, кондиционирование промпта с картинкой, один шаг DiT (урезанного до
//! NL блоков) с `zero_cond_t` и без, sigmas расписания.
//!
//! ```sh
//! QWEN_MODEL=…/Qwen-Image-Edit-2511 QWEN_REF=…/refout/edit2511 \
//!   cargo test --release -p synaptix-image-qwen --test reference -- --ignored --nocapture --test-threads=1
//! ```
//! `QWEN_MODEL` может быть и `.syn`-бандлом.

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_qwen::model::pack;
use synaptix_image_qwen::preprocess::{vision_patches, VisionPatches};
use synaptix_image_qwen::scheduler::QwenScheduler;
use synaptix_image_qwen::source;
use synaptix_image_qwen::text_encoder::TextEncoder;
use synaptix_image_qwen::transformer::{build_rope, rope_positions, Placement, QwenImageTransformer};
use synaptix_image_qwen::vision::VisionTower;
use synaptix_image_qwen::QwenImageModel;

fn setup() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var("QWEN_MODEL").ok()?;
    let refd = std::env::var("QWEN_REF").ok()?;
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

fn ref_tensor(dir: &PathBuf, name: &str, dev: Device) -> Tensor {
    let (v, s) = load_ref(dir, name);
    Tensor::from_vec(v, s, dev).unwrap()
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

fn open(model: &PathBuf, quant: DType) -> QwenImageModel {
    QwenImageModel::open(model, Device::Cuda(0), DType::BF16, quant).unwrap()
}

#[test]
#[ignore]
fn vae_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let x = ref_tensor(&refd, "vae_input", Device::Cpu);
    let lat = m.encode_image(&x).unwrap();
    let (r, shape) = load_ref(&refd, "vae_latent");
    assert_eq!(lat.dims(), &shape[..]);
    let (cos, rel, mx) = compare(&to_vec(&lat), &r);
    eprintln!("vae encode: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9999, "cos {cos}");
    let img = m.decode(&ref_tensor(&refd, "vae_latent", Device::Cpu)).unwrap();
    let (r, _) = load_ref(&refd, "vae_decoded");
    let (cos, rel, mx) = compare(&to_vec(&img), &r);
    eprintln!("vae decode: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9999, "cos {cos}");
}

#[test]
#[ignore]
fn vision_matches() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let vc = m.text_encoder_config().vision.clone();
    let (g, _) = load_ref(&refd, "grid_thw");
    let grid = (g[0] as usize, g[1] as usize, g[2] as usize);
    // Препроцессинг: та же уменьшенная картинка → патчи процессора Qwen2-VL.
    let cond = ref_tensor(&refd, "cond_image", Device::Cpu);
    let pc = synaptix_image_qwen::config::ProcessorConfig::from_json(
        m.source().read_opt("processor/preprocessor_config.json").as_deref(),
    )
    .unwrap();
    let mine = vision_patches(&cond, &vc, &pc).unwrap();
    assert_eq!(mine.grid, grid);
    let (pv, _) = load_ref(&refd, "pixel_values");
    let (cos, rel, mx) = compare(&to_vec(&mine.patches), &pv);
    eprintln!("pixel_values: cos {cos:.6} rel {rel:.5} max {mx:.4}");
    assert!(cos > 0.9995, "cos {cos}");

    let w = m.source().weights(source::TEXT_ENCODER).unwrap();
    // F32 — сверка устройства башни (порядок окон, RoPE, слияние); BF16 —
    // рабочая точность (32 блока копят ошибку: cos ≈ 0,997).
    for (dt, min_cos) in [(DType::F32, 0.9999), (DType::BF16, 0.995)] {
        let tower = VisionTower::load(&w, &vc, Device::Cuda(0), dt).unwrap();
        let p = VisionPatches { patches: ref_tensor(&refd, "pixel_values", Device::Cpu), grid };
        let e = tower.forward(&p).unwrap();
        let (r, shape) = load_ref(&refd, "vision_embeds");
        assert_eq!(e.dims(), &shape[..]);
        let (cos, rel, mx) = compare(&to_vec(&e), &r);
        eprintln!("vision embeds {dt:?}: cos {cos:.6} rel {rel:.5} max {mx:.4}");
        assert!(cos > min_cos, "{dt:?}: cos {cos}");
        // Те же патчи, но из своего ресайза.
        let e2 = tower.forward(&mine).unwrap();
        let (cos2, rel2, _) = compare(&to_vec(&e2), &r);
        eprintln!("vision embeds {dt:?} (свой препроцессинг): cos {cos2:.6} rel {rel2:.5}");
    }
}

#[test]
#[ignore]
fn prompt_embeds_match() {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, DType::BF16);
    let p: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("prompt.json")).unwrap()).unwrap();
    let prompt = p["prompt"].as_str().unwrap().to_string();
    let ids: Vec<u32> = p["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (g, _) = load_ref(&refd, "grid_thw");
    let merge = m.text_encoder_config().vision.merge_size;
    let lgrid = (g[1] as usize / merge, g[2] as usize / merge);

    // Токенизация шаблона с развёрнутыми <|image_pad|>.
    let mine = m.tokenizer().encode(&prompt, &[lgrid.0 * lgrid.1]).unwrap();
    assert_eq!(mine.ids, ids, "ids шаблона");

    // Энкодер на эталонных эмбеддингах картинки.
    let w = m.source().weights(source::TEXT_ENCODER).unwrap();
    // QWEN_TE_F32=1 — энкодер в F32 (сверка устройства; слои, не влезшие в
    // VRAM, стримятся), иначе рабочая BF16.
    let te_dt = if std::env::var("QWEN_TE_F32").is_ok_and(|v| v == "1") { DType::F32 } else { DType::BF16 };
    let enc = TextEncoder::build(w, m.text_encoder_config().clone(), Device::Cuda(0), te_dt, 512 << 20).unwrap();
    let vis = ref_tensor(&refd, "vision_embeds", Device::Cpu);
    let h = enc.encode(&ids, &[(vis, lgrid)]).unwrap();
    let skip = p["drop_idx"].as_u64().unwrap() as usize;
    let s = h.dims()[0];
    let h = h.narrow(0, skip, s - skip).unwrap();
    // Эталон — с M-RoPE (`qwen_te_positions.py`): diffusers 0.40 на
    // transformers 5.x не передаёт `mm_token_type_ids`, и позиции токенов
    // картинки там линейные, а модель обучалась с 3D-позициями (transformers
    // 4.5x считал их по input_ids). Без файла — старый эталон пайплайна.
    let reff = if refd.join("prompt_embeds_mrope.f32").exists() { "prompt_embeds_mrope" } else { "prompt_embeds" };
    let (r, shape) = load_ref(&refd, reff);
    assert_eq!(h.dims()[0], shape[1]);
    let (cos, rel, mx) = compare(&to_vec(&h), &r);
    eprintln!("prompt_embeds {te_dt:?} (эталонная картинка): cos {cos:.6} rel {rel:.4} max {mx:.3}");
    assert!(cos > if te_dt == DType::F32 { 0.9999 } else { 0.99 }, "cos {cos}");
    drop(enc);

    // Весь путь от картинки (своя подготовка + башня зрения) и негатив.
    let cond = ref_tensor(&refd, "cond_image", Device::Cpu);
    let c = m.encode_prompt(&prompt, Some(" "), &[cond]).unwrap();
    let (cos, _, _) = compare(&to_vec(&c.embeds), &r);
    eprintln!("prompt_embeds (encode_prompt): cos {cos:.6}");
    assert!(cos > 0.99, "cos {cos}");
    // Негатив с картинкой: у эталона пайплайна позиции линейные — только
    // форма и порядок величины.
    let (rn, ns) = load_ref(&refd, "negative_embeds");
    assert_eq!(c.negative.as_ref().unwrap().dims(), &ns[..]);
    let (cos, _, _) = compare(&to_vec(c.negative.as_ref().unwrap()), &rn);
    eprintln!("negative_embeds (к эталону пайплайна с линейными позициями): cos {cos:.6}");
}

#[test]
#[ignore]
fn sigmas_match() {
    let Some((model, refd)) = setup() else { return };
    let m = QwenImageModel::open(&model, Device::Cpu, DType::F32, DType::F32).unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("dit.json")).unwrap()).unwrap();
    let side = meta["side"].as_u64().unwrap() as usize;
    let seq = (side / 16) * (side / 16);
    let sc = synaptix_image_qwen::config::SchedulerConfig::from_json(
        m.source().read_opt("scheduler/scheduler_config.json").as_deref(),
    )
    .unwrap();
    let s = QwenScheduler::new(4, seq, &sc);
    let (r, _) = load_ref(&refd, "sigmas");
    for (i, e) in r.iter().enumerate() {
        assert!((s.sigma(i) - e).abs() < 2e-6, "{i}: {} vs {e}", s.sigma(i));
    }
}

fn dit_step(quant: DType, zero_cond: bool, min_cos: f64) {
    let Some((model, refd)) = setup() else { return };
    let m = open(&model, quant);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("dit.json")).unwrap()).unwrap();
    let nl = meta["nl"].as_u64().unwrap() as usize;
    let side = meta["side"].as_u64().unwrap() as usize;
    let dev = Device::Cuda(0);
    let mut cfg = m.config().clone();
    cfg.num_layers = nl;
    cfg.zero_cond_t = zero_cond;
    let w = m.source().weights(source::TRANSFORMER).unwrap();
    let (lh, lw) = (side / 8, side / 8);
    let n_target = (lh / 2) * (lw / 2);
    let (pe, pes) = load_ref(&refd, "prompt_embeds");
    let st = pes[1];
    let t = QwenImageTransformer::load(&w, &cfg, dev, DType::BF16, quant, Placement::Resident, st + 2 * n_target).unwrap();
    let noise = pack(&ref_tensor(&refd, "noise", dev)).unwrap();
    let refl = pack(&ref_tensor(&refd, "vae_latent", dev)).unwrap();
    let img = Tensor::cat(&[&noise, &refl], 1).unwrap();
    let txt = t.embed_text(&Tensor::from_vec(pe, pes, dev).unwrap()).unwrap();
    let (sig, _) = load_ref(&refd, "sigmas");
    let row_sigma = sig[meta["t_index"].as_u64().unwrap() as usize];
    let mods = t.modulations(&[row_sigma]).unwrap();
    let grid = (lh / 2, lw / 2);
    let (c, s) = build_rope(&rope_positions(&[grid, grid], st), &cfg.axes_dims, dev).unwrap();
    let v = t.forward(&img, n_target, &txt, &mods, 0, &c, &s).unwrap();
    let (r, _) = load_ref(&refd, if zero_cond { "dit_out_zc1" } else { "dit_out_zc0" });
    let (cos, rel, mx) = compare(&to_vec(&v), &r);
    eprintln!("dit {nl} блоков {quant:?} zero_cond_t={zero_cond}: cos {cos:.6} rel {rel:.4} max {mx:.3}");
    assert!(cos > min_cos, "cos {cos}");
}

#[test]
#[ignore]
fn dit_step_dense_zero_cond() {
    dit_step(DType::BF16, true, 0.999);
}

#[test]
#[ignore]
fn dit_step_dense_plain() {
    dit_step(DType::BF16, false, 0.999);
}

#[test]
#[ignore]
fn dit_step_mxfp8() {
    dit_step(DType::MXFP8, true, 0.995);
}

#[test]
#[ignore]
fn dit_step_nvfp4() {
    dit_step(DType::NVFP4, true, 0.97);
}

/// Время прохода полного DiT на 1024² + референс 1024² (≈8,4 тыс. токенов):
/// `QWEN_BENCH_QUANT=nvfp4|mxfp8|dense`, `QWEN_BENCH_ITERS` (2).
#[test]
#[ignore]
fn dit_bench() {
    let Some((model, _)) = setup() else { return };
    let quant = match std::env::var("QWEN_BENCH_QUANT").as_deref() {
        Ok("mxfp8") => DType::MXFP8,
        Ok("dense") => DType::BF16,
        _ => DType::NVFP4,
    };
    let iters: usize = std::env::var("QWEN_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let m = open(&model, quant);
    let dev = Device::Cuda(0);
    let (st, grid) = (233usize, (64usize, 64usize));
    let n = grid.0 * grid.1;
    let t = m.load_transformer(st + 2 * n).unwrap();
    let mut rng = synaptix_ops::rng::Philox4x32::new(1);
    let mut randn = |shape: &[usize]| synaptix_diffusion::schedulers::randn_seeded(shape, dev, &mut rng).unwrap();
    let img = randn(&[1, 2 * n, 64]).to_dtype(DType::BF16).unwrap();
    let txt = t.embed_text(&randn(&[1, st, 3584])).unwrap();
    let mods = t.modulations(&[0.9]).unwrap();
    let cfg = m.config().clone();
    let (c, s) = build_rope(&rope_positions(&[grid, grid], st), &cfg.axes_dims, dev).unwrap();
    for i in 0..iters {
        let t0 = std::time::Instant::now();
        let v = t.forward(&img, n, &txt, &mods, 0, &c, &s).unwrap();
        let _ = v.to_device(Device::Cpu).unwrap();
        eprintln!("проход {i}: {:.2} с", t0.elapsed().as_secs_f64());
    }
}
