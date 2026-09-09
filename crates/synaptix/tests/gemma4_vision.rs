//! Сверка башни зрения Gemma-4 с эталоном transformers.
//!
//! Эталон снимается скриптом `gemma4_vision_ref.py` (см. документацию
//! `docs/gemma4_2026.md`): он кладёт в каталог `pixel_values.f32`,
//! `position_ids.i32`, `soft_tokens.f32`, `projected.f32` и `meta.json`.
//!
//! Запуск:
//!   SYN_GEMMA4_BUNDLE=…/gemma-4-26b-a4b-it.syn \
//!   SYN_GEMMA4_VISION_REF=…/vref \
//!   cargo test -p synaptix --release --test gemma4_vision -- --nocapture

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_gemma4::vision::{VisionConfig, VisionTower};
use synaptix_llm_gemma4::Gemma4Weights;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "разная длина: {} и {}", a.len(), b.len());
    let mut max_abs = 0f32;
    let mut sum2 = 0f32;
    let mut ref2 = 0f32;
    for (x, y) in a.iter().zip(b) {
        max_abs = max_abs.max((x - y).abs());
        sum2 += (x - y) * (x - y);
        ref2 += y * y;
    }
    (max_abs, (sum2 / ref2.max(1e-12)).sqrt())
}

#[test]
fn vision_tower_matches_transformers() {
    let (Ok(bundle), Ok(refdir)) = (
        std::env::var("SYN_GEMMA4_BUNDLE"),
        std::env::var("SYN_GEMMA4_VISION_REF"),
    ) else {
        return;
    };
    synaptix::init().expect("init");
    let refdir = PathBuf::from(refdir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(refdir.join("meta.json")).expect("meta.json"))
            .expect("meta json");
    let shape = |name: &str| -> Vec<usize> {
        meta[name]
            .as_array()
            .expect(name)
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };

    let device = Device::Cuda(0);
    // Сверка идёт в F32: эталон посчитан в float32, и разница bf16 (≈1e-2)
    // спрятала бы ошибку в раскладке RoPE или порядке каналов патча.
    let weights = Gemma4Weights::open(&bundle, device, DType::F32).expect("открыть бандл");
    let cfg_bytes = synaptix_llm_gemma4::read_aux(bundle.as_ref(), "config.json").expect("config");
    let vcfg = VisionConfig::from_hf_bytes(&cfg_bytes).expect("vision_config");
    let tower = VisionTower::load(&weights, vcfg.clone(), device, DType::F32).expect("башня");

    let pv_shape = shape("pixel_values");
    let pixel_values = read_f32(&refdir.join("pixel_values.f32"));
    let positions: Vec<u32> = read_i32(&refdir.join("position_ids.i32"))
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let pixel = Tensor::from_vec(pixel_values, pv_shape.clone(), device).expect("pixels");
    let patch_cols = positions.iter().step_by(2).max().copied().unwrap() as usize + 1;

    let out = tower
        .forward_patches(&pixel, &positions, patch_cols)
        .expect("forward");
    let got = out
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("вывод на хост");
    let want = read_f32(&refdir.join("projected.f32"));
    let want_shape = shape("projected");
    assert_eq!(out.dims(), want_shape.as_slice(), "форма мягких токенов");

    let (max_abs, rel) = rel_err(&got, &want);
    eprintln!(
        "[gemma4 vision] патчей {} → {:?}: max|Δ| = {max_abs:.4}, отн. L2 = {rel:.5}",
        pv_shape[0], want_shape
    );
    eprintln!("[gemma4 vision] наши  {:?}", &got[..6]);
    eprintln!("[gemma4 vision] эталон {:?}", &want[..6]);
    assert!(rel < 2e-3, "относительная ошибка {rel} велика: max|Δ| = {max_abs}");
}

/// Сквозной ход с картинкой через фасад: башня зрения → мягкие токены в
/// промпте → ответ модели. Нужны бандл и путь к PNG (`SYN_GEMMA4_IMAGE`).
#[test]
fn describes_image_through_facade() {
    let (Ok(bundle), Ok(image)) = (
        std::env::var("SYN_GEMMA4_BUNDLE"),
        std::env::var("SYN_GEMMA4_IMAGE"),
    ) else {
        return;
    };
    use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration, Message};
    use synaptix_core::precision::PrecisionConfig;

    let (model, tokenizer) = load_llm(
        bundle.as_ref(),
        Device::Cuda(0),
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::BF16,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        Some(2048),
    )
    .expect("load_llm gemma4");
    assert!(model.supports_media(), "бандл без башни зрения");
    assert!(model.ensure_media_tower().expect("башня"), "башня не загрузилась");

    let media = model
        .encode_image(std::path::Path::new(&image), None)
        .expect("encode_image");
    eprintln!(
        "[gemma4 image] мягких токенов {}, блок {:?}",
        media.tokens(),
        &media.prompt_block[..media.prompt_block.len().min(40)]
    );
    assert!(media.tokens() > 0);

    let msgs = [
        Message::system("Отвечай одним словом."),
        Message::user(&format!(
            "{}Какого цвета фигура на картинке?",
            media.prompt_block
        )),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids = tokenizer.encode(&prompt).expect("encode");

    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 24,
            max_seq_len: 2048,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
    );
    generation.set_stop_tokens(tokenizer.eos_ids().to_vec());
    let mut out = String::new();
    generation
        .generate_streaming_media(&ids, &tokenizer, &[&media], |_id, delta| {
            out.push_str(delta);
            true
        })
        .expect("generate media");
    eprintln!("[gemma4 image] промпт {} ток. → {out:?}", ids.len());
    assert!(!out.trim().is_empty(), "модель ничего не сказала");
    // `SYN_GEMMA4_IMAGE_EXPECT` — подстрока, которую обязан содержать ответ
    // (для синего круга это «син»): без неё тест ловит только падения.
    if let Ok(expect) = std::env::var("SYN_GEMMA4_IMAGE_EXPECT") {
        assert!(
            out.to_lowercase().contains(&expect.to_lowercase()),
            "ожидали {expect:?} в ответе {out:?}"
        );
    }
}
