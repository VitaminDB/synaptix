//! TTS-фасад. OmniVoice — нативный `synaptix-tts-omnivoice`. `core` несёт
//! конфиг/режим генерации; [`TtsPipeline`] мапит их на нативный greedy-путь.

pub mod core {
    use std::fmt;
    use std::path::PathBuf;

    #[derive(Debug, Clone)]
    pub struct GenerationConfig {
        pub num_step: u32,
        pub guidance_scale: f32,
        pub t_shift: f32,
        pub speed: f32,
        pub seed: u64,
    }

    impl Default for GenerationConfig {
        fn default() -> Self {
            Self { num_step: 32, guidance_scale: 2.0, t_shift: 0.1, speed: 1.0, seed: 0 }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct VoiceClonePrompt {
        pub audio_path: PathBuf,
        pub text: Option<String>,
        pub lang: Option<String>,
    }

    impl VoiceClonePrompt {
        pub fn new(audio_path: PathBuf) -> Self {
            Self { audio_path, text: None, lang: None }
        }

        pub fn with_text(mut self, text: String) -> Self {
            self.text = Some(text);
            self
        }

        pub fn with_lang(mut self, lang: String) -> Self {
            self.lang = Some(lang);
            self
        }
    }

    #[derive(Debug, Clone)]
    pub enum GenerationMode {
        Auto,
        Clone(VoiceClonePrompt),
        Design { instruct: String },
    }

    #[derive(Debug, Clone)]
    pub enum OmniVoiceError {
        Inference(String),
    }

    impl fmt::Display for OmniVoiceError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Inference(msg) => write!(f, "{msg}"),
            }
        }
    }

    impl std::error::Error for OmniVoiceError {}
}

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::dtype::DType as StorageDType;

use synaptix_tts_omnivoice::config::OmniVoiceGenerationConfig as NativeGenConfig;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline as NativePipeline;

use self::core::{GenerationConfig, GenerationMode, OmniVoiceError};

fn map_err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Inference(e.to_string())
}

/// Нативный конфиг из фасадного `GenerationConfig`: температуры как в
/// upstream, шум Гумбеля — от `cfg.seed` (одно зерно — один результат).
fn native_gen(cfg: &GenerationConfig) -> NativeGenConfig {
    NativeGenConfig {
        num_step: cfg.num_step as usize,
        guidance_scale: cfg.guidance_scale,
        t_shift: cfg.t_shift,
        seed: Some(cfg.seed),
        denoise: true,
        postprocess_output: true,
        ..NativeGenConfig::default()
    }
}

pub struct TtsPipeline {
    inner: NativePipeline,
}

impl TtsPipeline {
    pub fn best_device() -> Device {
        {
            Device::Cuda(0)
        }
    }

    /// Грузит `.syn`-бандл: счёт в `compute`, линейные слои LM — в `storage`,
    /// если это квант-формат (NVFP4/MXFP8/Sq): квантуются на лету при
    /// загрузке, как бы ни были сохранены веса. Иначе веса плотные в `compute`.
    pub fn from_syn(
        bundle_path: &Path,
        device: &Device,
        storage: StorageDType,
        compute: DType,
    ) -> Result<Self, OmniVoiceError> {
        let quant = matches!(storage, DType::NVFP4 | DType::MXFP8 | DType::Sq { .. }).then_some(storage);
        let inner = NativePipeline::from_syn_quant(bundle_path, *device, compute, quant).map_err(map_err)?;
        Ok(Self { inner })
    }

    pub fn synthesize(
        &self,
        text: &str,
        mode: &GenerationMode,
        cfg: &GenerationConfig,
    ) -> Result<Vec<f32>, OmniVoiceError> {
        let gen = native_gen(cfg);
        let speed = cfg.speed as f64;
        match mode {
            GenerationMode::Auto => self
                .inner
                .generate_styled(text, None, None, speed, &gen)
                .map_err(map_err),
            GenerationMode::Clone(p) => {
                let Some(ref_text) = p.text.as_deref() else {
                    return Err(OmniVoiceError::Inference(
                        "voice-clone требует ref_text (распознай аудио через \
                         synaptix ASR и передай .with_text(...))"
                            .into(),
                    ));
                };
                let vcp = self
                    .inner
                    .create_voice_clone_prompt(&p.audio_path, ref_text)
                    .map_err(map_err)?;
                let lang = p.lang.as_deref().map(str::trim).filter(|l| !l.is_empty() && *l != "auto");
                self.inner
                    .synthesize(text, Some(&vcp), lang, None, speed, None, &gen)
                    .map_err(map_err)
            }
            GenerationMode::Design { instruct } => self
                .inner
                .generate_styled(text, None, Some(instruct.as_str()), speed, &gen)
                .map_err(map_err),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        24_000
    }
}

/// Темп речи без смены высоты и деление текста на куски для потока.
#[cfg(feature = "engine")]
pub use synaptix_audio::tempo::{speech_chunks, time_stretch};

/// Готовые голоса-персонажи для диалогов: имя → (описание для OmniVoice,
/// описание для VoxCPM, зерно). Зерно закрепляет тембр: один и тот же
/// персонаж звучит одинаково от фразы к фразе и между запусками.
pub const SPEAKERS: &[(&str, &str, &str, u64)] = &[
    ("female_young", "female, young adult, moderate pitch, american accent", "A young woman in her twenties, clear bright friendly voice, natural conversational pace.", 1101),
    ("female_adult", "female, middle-aged, low pitch, british accent", "A middle-aged woman, warm calm low-pitched voice, professional and articulate.", 2203),
    ("female_elderly", "female, elderly, low pitch", "An elderly woman, soft gentle slightly husky voice, kind and unhurried.", 3307),
    ("male_young", "male, young adult, moderate pitch, american accent", "A young man in his twenties, clear energetic voice, casual and friendly.", 4409),
    ("male_adult", "male, middle-aged, low pitch, british accent", "A middle-aged man, deep calm confident voice, like a professional interviewer.", 5501),
    ("male_elderly", "male, elderly, very low pitch", "An elderly man, deep slightly raspy voice, slow and thoughtful.", 6607),
    ("girl", "female, child, high pitch", "A little girl, about 8 years old, high-pitched cheerful childish voice.", 7703),
    ("boy", "male, child, high pitch", "A little boy, about 8 years old, high-pitched playful childish voice.", 8807),
];

/// Персонаж по имени (без учёта регистра).
pub fn speaker(name: &str) -> Option<(&'static str, &'static str, u64)> {
    SPEAKERS.iter().find(|s| s.0.eq_ignore_ascii_case(name.trim())).map(|s| (s.1, s.2, s.3))
}
