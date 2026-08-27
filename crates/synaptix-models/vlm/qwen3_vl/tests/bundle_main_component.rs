//! Гейтед-проверка на реальном бандле Qwen3.8-27B (HF-упаковка `syn-pack`):
//! башня лежит не отдельным компонентом `vision`, а тензорами
//! `model.visual.*` в `tensors:main`. Загрузчик обязан её найти, поднять на
//! CPU и закодировать картинку в `[tokens, out_hidden_size]`.
//!
//! `SYN_QWEN38_BUNDLE=/path/qwen3.8-27b.syn cargo test -p synaptix-vlm-qwen3 --release --test bundle_main_component`

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_vlm_qwen3::{
    bundle_has_vision, load_from_bundle, prepare_tensor, BundleVisionWeights, PreprocessLimits,
};

fn bundle() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("SYN_QWEN38_BUNDLE")?);
    p.exists().then_some(p)
}

#[test]
fn tower_in_main_component_is_found_and_encodes_on_cpu() {
    let Some(path) = bundle() else {
        eprintln!("SYN_QWEN38_BUNDLE не задан или файла нет — пропуск");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();

    assert!(bundle_has_vision(&path), "башня в tensors:main не распознана");
    let weights = BundleVisionWeights::open(&path).expect("open weights");
    assert!(weights.has("model.visual.patch_embed.proj.weight"));
    assert!(weights.has("model.visual.merger.linear_fc2.weight"));

    let tower = load_from_bundle(&path, Device::Cpu, DType::F32).expect("load tower");
    assert_eq!(tower.config.depth, 27);
    assert_eq!(tower.config.out_hidden_size, 5120);

    // Синтетическая картинка 64×96 (CHW, 0..1) — smart_resize дотянет до
    // min_pixels, число токенов считаем по итоговой сетке.
    let (h, w) = (64usize, 96usize);
    let mut data = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data[c * h * w + y * w + x] = ((x * 2 + y * 3 + c * 41) % 256) as f32 / 255.0;
            }
        }
    }
    let img = Tensor::from_vec(data, vec![3, h, w], Device::Cpu).expect("image tensor");
    let prepared = prepare_tensor(&img, &tower.config, PreprocessLimits::default(), Device::Cpu)
        .expect("prepare");
    let expected_tokens = prepared.grid.patches() / tower.config.merge_unit();
    assert!(expected_tokens > 0);

    let emb = tower.forward(&prepared.patches, prepared.grid).expect("forward");
    assert_eq!(emb.dims(), &[expected_tokens, tower.config.out_hidden_size]);
    let rows = emb.to_vec2::<f32>().expect("to_vec2");
    let finite = rows.iter().flatten().all(|v| v.is_finite());
    assert!(finite, "в эмбеддингах NaN/inf");
    let energy: f32 = rows[0].iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "нулевые эмбеддинги");
    eprintln!("ok: {expected_tokens} vision-токенов, ||row0||² = {energy:.3}");
}
