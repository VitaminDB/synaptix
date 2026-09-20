//! Сквозной прогон: стиль и лирика → партитура → музыка → латенты → звук.
//!
//! Тем же путём идёт нода приложения: части грузятся отдельно и собираются
//! через `from_parts`, декодер подключается в конце. Требует GPU и бандлы,
//! поэтому `#[ignore]`:
//!
//! ```sh
//! cargo test --release -p synaptix-music-yue2 --test song_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_yue2::ar::Yue2Ar;
use synaptix_music_yue2::nar::Yue2Nar;
use synaptix_music_yue2::pipeline::{Callbacks, Yue2Options, Yue2Pipeline};
use synaptix_music_yue2::protocol::{Cot, GenerationConfig, SongRequest};
use synaptix_music_yue2::tokenizer::Yue2Tokenizer;
use synaptix_music_yue2::vae::Yue2Vae;

fn bundles() -> Option<(PathBuf, PathBuf)> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    let model = std::env::var("SYN_YUE2_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join("Storage/syn_models/yue2-3b.syn"));
    let vae = std::env::var("SYN_YUE2_VAE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join("Storage/syn_models/yue2-vae.syn"));
    (model.exists() && vae.exists()).then_some((model, vae))
}

#[test]
#[ignore = "нужен GPU и бандлы YuE2"]
fn song_from_style_and_lyrics() {
    let Some((model_path, vae_path)) = bundles() else {
        eprintln!("[yue2-e2e] бандлов нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let compute = DType::BF16;

    // Короткий прогон: 8 секунд музыки и 8 шагов решателя — проверяем тракт,
    // а не качество.
    let mut generation = GenerationConfig::default();
    generation.ode_steps = 8;
    generation.semantic.max_tokens = 200;
    generation.semantic.min_tokens = 200;
    generation.abc.max_tokens = 512;
    let options = Yue2Options {
        device,
        compute,
        quant: None,
        vae_dtype: DType::F32,
        vae_core_frames: 256,
        vae_halo_frames: 16,
        generation: generation.clone(),
    };

    let ar = Arc::new(Yue2Ar::open(&model_path, device, compute, None, 4096).expect("AR"));
    let nar = Arc::new(Yue2Nar::open(&model_path, device, compute, None).expect("NAR"));
    let tokenizer = Arc::new(
        Yue2Tokenizer::from_tiktoken_bytes(
            &synaptix_music_yue2::loader::read_bundle_file(&model_path, "qwen.tiktoken")
                .expect("qwen.tiktoken"),
        )
        .expect("токенизатор"),
    );
    let pipe = Yue2Pipeline::from_parts(ar, nar, tokenizer, None, options);

    let request = SongRequest {
        style: "english, gentle piano ballad, female vocal".into(),
        lyrics: "[verse]\nTonight I'm awake\n".into(),
        cot: Cot::Full,
        seed: 7,
        abc: None,
        cfg_scale: None,
    };

    let semantic = pipe.generate(&request, Callbacks::default()).expect("партитура и музыка");
    let abc = semantic.plan.abc.as_deref().unwrap_or_default();
    eprintln!(
        "[yue2-e2e] партитура {} токенов, музыка {} токенов ({:.1} с), {:.1} ток/с",
        semantic.plan.abc_ids.len(),
        semantic.tokens.len(),
        semantic.seconds(),
        semantic.timing.tokens_per_second
    );
    assert!(!abc.trim().is_empty(), "партитура пустая");
    // ABC-партитура релиза начинается с номера мелодии и несёт голоса.
    assert!(abc.starts_with("X:"), "это не ABC: {}", &abc[..abc.len().min(40)]);
    assert!(abc.contains("V:"), "в партитуре нет голосов");
    assert_eq!(semantic.tokens.len(), generation.semantic.max_tokens, "музыка обрезана не по лимиту");
    assert!(semantic.tokens.iter().all(|&t| t < 32768), "семантический токен вне словаря");

    let latents = pipe.synthesize(&semantic, Callbacks::default()).expect("акустика");
    assert_eq!(latents.dims(), &[semantic.tokens.len(), 64]);
    let values: Vec<f32> = latents.flatten_all().unwrap().to_vec1().unwrap();
    assert!(values.iter().all(|v| v.is_finite()), "латенты не конечны");
    let rms = (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt();
    eprintln!("[yue2-e2e] латенты rms={rms:.4}");
    assert!(rms > 1e-3, "латенты вырождены (rms {rms})");

    let vae = Arc::new(Yue2Vae::open(&vae_path, device, DType::F32, true).expect("VAE"));
    let audio = pipe.decode_with(&vae, &latents, Callbacks::default()).expect("декод");
    let expected = vae.output_len(semantic.tokens.len()) * 2;
    assert_eq!(audio.len(), expected, "длина звука не совпала с длиной латентов");
    let peak = audio.iter().fold(0f32, |a, v| a.max(v.abs()));
    let audio_rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
    eprintln!(
        "[yue2-e2e] звук {:.1} с, rms={audio_rms:.4}, peak={peak:.3}",
        audio.len() as f32 / (2.0 * 48000.0)
    );
    assert!(audio_rms > 1e-3 && peak > 1e-2, "тишина на выходе");
    assert!(peak <= 1.0, "выход не ограничен ±1");

    // Латенты можно передекодировать, не повторяя генерацию, — на этом
    // держится нода «YuE2 VAE Decode».
    let again = pipe.decode_with(&vae, &latents, Callbacks::default()).expect("повторный декод");
    assert_eq!(again.len(), audio.len());
}
