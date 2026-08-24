use std::path::Path;

use synaptix_audio::io::{read_wav_mono_f32, write_wav_mono_f32};
use synaptix_audio::resample::resample_linear;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

use crate::config::GenerationConfig;
use crate::generate::{GenerationOutput, NormalRng, SpeechGenerator};
use crate::loader::VibeVoiceCheckpoint;
use crate::model::VibeVoiceModel;
use crate::processor::VibeVoiceProcessor;
use crate::{Result, VibeVoiceError};

const DEFAULT_ROPE_CAPACITY: usize = 32_768;

#[derive(Debug, Clone)]
pub struct VoiceSample {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl VoiceSample {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    pub fn from_wav(path: impl AsRef<Path>) -> Result<Self> {
        let (samples, sample_rate) = read_wav_mono_f32(path.as_ref())
            .map_err(|e| VibeVoiceError::Audio(format!("{}: {e}", path.as_ref().display())))?;
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "wav" | "wave" => Self::from_wav(path),
            _ => Self::decode_compressed(path, &ext),
        }
    }

    #[cfg(feature = "audio-decode")]
    fn decode_compressed(path: &Path, ext: &str) -> Result<Self> {
        use synaptix_io::audio::{flac::decode_flac, mp3::decode_mp3, ogg::decode_ogg};
        let buf = match ext {
            "mp3" | "m4a" | "aac" | "mp4" => decode_mp3(path),
            "ogg" | "oga" | "opus" => decode_ogg(path),
            "flac" => decode_flac(path),
            other => {
                return Err(VibeVoiceError::Audio(format!(
                    "неподдерживаемый формат аудио: .{other}"
                )))
            }
        }
        .map_err(|e| VibeVoiceError::Audio(format!("{}: {e}", path.display())))?;
        Ok(Self {
            samples: buf.to_mono(),
            sample_rate: buf.sample_rate,
        })
    }

    #[cfg(not(feature = "audio-decode"))]
    fn decode_compressed(path: &Path, ext: &str) -> Result<Self> {
        Err(VibeVoiceError::Audio(format!(
            "{}: .{ext} требует сборки с feature `audio-decode`",
            path.display()
        )))
    }

    pub fn to_rate(&self, target: u32) -> Result<Vec<f32>> {
        if self.sample_rate == target {
            return Ok(self.samples.clone());
        }
        resample_linear(&self.samples, self.sample_rate, target)
            .map_err(|e| VibeVoiceError::Audio(e.to_string()))
    }
}

pub struct VibeVoicePipeline {
    model: VibeVoiceModel,
    processor: VibeVoiceProcessor,
}

impl VibeVoicePipeline {
    pub fn best_device() -> Device {
        Device::Cuda(0)
    }

    pub fn from_syn(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        Self::open(path, device, dtype, DEFAULT_ROPE_CAPACITY)
    }

    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        rope_capacity: usize,
    ) -> Result<Self> {
        let ckpt = VibeVoiceCheckpoint::open(path, device, dtype)?;
        let rope = rope_capacity
            .min(ckpt.config.decoder_config.max_position_embeddings)
            .max(2048);
        let processor = VibeVoiceProcessor::new(&ckpt.tokenizer_json, &ckpt.preprocessor)?;
        let model = VibeVoiceModel::load(&ckpt, rope)?;
        Ok(Self { model, processor })
    }

    pub fn model(&self) -> &VibeVoiceModel {
        &self.model
    }

    pub fn processor(&self) -> &VibeVoiceProcessor {
        &self.processor
    }

    pub fn sample_rate(&self) -> u32 {
        self.processor.sampling_rate
    }

    pub fn synthesize(
        &self,
        script: &str,
        voices: &[VoiceSample],
        cfg: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        self.synthesize_with(script, voices, cfg, None, None)
    }

    pub fn synthesize_with(
        &self,
        script: &str,
        voices: &[VoiceSample],
        cfg: &GenerationConfig,
        on_chunk: Option<&mut dyn FnMut(&[f32])>,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<GenerationOutput> {
        let rate = self.sample_rate();
        let mut wavs: Vec<Vec<f32>> = Vec::with_capacity(voices.len());
        for v in voices {
            wavs.push(v.to_rate(rate)?);
        }
        let prompt = self.processor.build_prompt(script, &wavs)?;
        let rng = if cfg.zero_noise {
            NormalRng::zeros()
        } else {
            NormalRng::new(cfg.seed)
        };
        let mut gen = SpeechGenerator::with_rng(&self.model, &self.processor, rng)?;
        gen.generate(&prompt, cfg, on_chunk, on_step)
    }

    pub fn save_wav(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        write_wav_mono_f32(path.as_ref(), samples, self.sample_rate())
            .map_err(|e| VibeVoiceError::Audio(e.to_string()))
    }
}
