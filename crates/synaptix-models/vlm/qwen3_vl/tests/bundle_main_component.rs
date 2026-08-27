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
    bundle_has_vision, load_from_bundle, prepare_tensor, prepare_video, BundleVisionWeights,
    PreprocessLimits, VideoLimits,
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

/// Видео: синтетический ролик ffmpeg (lavfi testsrc) → сэмплинг кадров →
/// башня по группам кадров → `[groups·tokens_per_group, out_hidden_size]`.
/// Пропуск без бандла или без ffmpeg.
#[test]
fn tower_encodes_synthetic_video_on_cpu() {
    let Some(path) = bundle() else {
        eprintln!("SYN_QWEN38_BUNDLE не задан или файла нет — пропуск");
        return;
    };
    let dir = std::env::temp_dir().join("synaptix_vlm_qwen3_video_test");
    let _ = std::fs::create_dir_all(&dir);
    let video = dir.join("testsrc.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10"])
        .args(["-pix_fmt", "yuv420p"])
        .arg(&video)
        .status();
    match status {
        Ok(st) if st.success() => {}
        _ => {
            eprintln!("ffmpeg недоступен — пропуск видео-теста");
            return;
        }
    }
    synaptix_kernels_cpu::ensure_registered();
    let tower = load_from_bundle(&path, Device::Cpu, DType::F32).expect("load tower");
    let limits = VideoLimits { target_fps: 2.0, max_frames: 8, max_total_tokens: 512, min_group_tokens: 16 };
    let prepared = prepare_video(&video, &tower.config, limits, Device::Cpu).expect("prepare video");
    // 3 с × 2 fps = 6 кадров → 3 группы.
    assert_eq!(prepared.grid.t, 3, "групп кадров: {:?}", prepared.group_timestamps);
    assert_eq!(prepared.group_timestamps.len(), 3);
    assert!(prepared.group_timestamps.windows(2).all(|p| p[0] < p[1]));
    let per_group = (prepared.grid.h * prepared.grid.w) / tower.config.merge_unit();
    assert!(per_group >= 16 && per_group * 3 <= 512, "токенов на группу: {per_group}");

    let emb = tower.forward(&prepared.patches, prepared.grid).expect("forward");
    assert_eq!(emb.dims(), &[3 * per_group, tower.config.out_hidden_size]);
    let rows = emb.to_vec2::<f32>().expect("to_vec2");
    assert!(rows.iter().flatten().all(|v| v.is_finite()), "NaN/inf в видео-эмбеддингах");
    // Группы независимы: первая группа отдельно даёт те же строки, что в
    // общем прогоне (внимание не перетекает между кадрами).
    let per_patches = prepared.grid.h * prepared.grid.w;
    let first = prepared.patches.narrow(0, 0, per_patches).and_then(|t| t.contiguous()).unwrap();
    let g0 = tower
        .forward(&first, synaptix_vlm_qwen3::ImageGrid { t: 1, h: prepared.grid.h, w: prepared.grid.w })
        .expect("forward group 0");
    let g0 = g0.to_vec2::<f32>().unwrap();
    for (a, b) in g0.iter().zip(rows.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }
    eprintln!("ok: видео {} групп × {per_group} токенов, ts={:?}", prepared.grid.t, prepared.group_timestamps);
}
