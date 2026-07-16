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
            Self { num_step: 32, guidance_scale: 2.0, t_shift: 0.5, speed: 1.0, seed: 0 }
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

/// Нативный greedy-конфиг из фасадного `GenerationConfig`. Сэмплинг отключён
/// (position/class temperature = 0.0) — сверенный детерминированный путь.
fn native_gen(cfg: &GenerationConfig) -> NativeGenConfig {
    NativeGenConfig {
        num_step: cfg.num_step as usize,
        guidance_scale: cfg.guidance_scale,
        t_shift: cfg.t_shift,
        position_temperature: 0.0,
        class_temperature: 0.0,
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
        #[cfg(feature = "cuda")]
        {
            Device::Cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    }

    /// Грузит реальный `.syn`-бандл в `compute`-dtype. `storage` игнорируется —
    /// OmniVoice грузит веса сразу в compute-dtype.
    pub fn from_syn(
        bundle_path: &Path,
        device: &Device,
        _storage: StorageDType,
        compute: DType,
    ) -> Result<Self, OmniVoiceError> {
        let inner = NativePipeline::from_syn(bundle_path, *device, compute).map_err(map_err)?;
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
                self.inner.generate_clone(text, &vcp, &gen).map_err(map_err)
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
