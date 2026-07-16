//! High-level транскрибация GigaAM: PCM → ресэмпл 16 кГц → log-mel → энкодер →
//! CTC-head → greedy-CTC-decode → SentencePiece-текст.

use std::path::Path;

use synaptix_audio::resample_linear;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::GigaAmConfig;
use crate::loader::GigaAmWeights;
use crate::mel::log_mel;
use crate::model::GigaAmModel;
use crate::spm::SpmDecoder;
use crate::{GigaAmError, Result};

pub const SR: u32 = 16000;

pub struct GigaAm {
    model: GigaAmModel,
    spm: SpmDecoder,
    config: GigaAmConfig,
    blank_id: u32,
    device: Device,
    dtype: DType,
}

impl GigaAm {
    /// Загрузить из `.syn`-бандла.
    pub fn from_syn(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let w = GigaAmWeights::from_syn(path, *device, dtype)?;
        Self::build(w)
    }

    /// Загрузить из распакованного HF-снапшота (`model.safetensors` +
    /// `config.json` + `tokenizer.model`).
    pub fn from_unpacked(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let w = GigaAmWeights::from_unpacked(dir, *device, dtype)?;
        Self::build(w)
    }

    fn build(w: GigaAmWeights) -> Result<Self> {
        let config = w.config.clone();
        let blank_id = config.blank_id() as u32;
        let spm = SpmDecoder::from_model_bytes(&w.tokenizer_model)?;
        // blank_id (= len(tokenizer)) и num_classes (= len+1) — согласованность.
        if spm.len() + 1 != config.head.num_classes {
            return Err(GigaAmError::Config(format!(
                "tokenizer size {} + 1 != head.num_classes {}",
                spm.len(),
                config.head.num_classes
            )));
        }
        let device = w.device;
        let dtype = w.dtype;
        let model = GigaAmModel::load(&w)?;
        Ok(Self { model, spm, config, blank_id, device, dtype })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// log-mel `[1, n_mels, T]` на устройстве модели.
    fn mel_tensor(&self, audio: &[f32]) -> Result<Tensor> {
        let (flat, n_mels, n_frames) = log_mel(audio, &self.config.preprocessor);
        let t = Tensor::from_vec(flat, (1, n_mels, n_frames), self.device)?;
        Ok(t.to_dtype(self.dtype)?)
    }

    /// PCM (любой sample_rate) → текст. Ресэмпл до 16 кГц при необходимости.
    pub fn transcribe_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<String> {
        let resampled;
        let audio: &[f32] = if sample_rate == SR {
            pcm
        } else {
            resampled =
                resample_linear(pcm, sample_rate, SR).map_err(|e| GigaAmError::Audio(e.to_string()))?;
            &resampled
        };
        let mel = self.mel_tensor(audio)?;
        let encoded = self.model.encode(&mel)?;
        let logits = self.model.head_logits(&encoded)?; // [1, T', C]
        let ids = self.greedy_ctc(&logits)?;
        Ok(self.spm.decode(&ids))
    }

    /// Промежуточные тензоры (для гейтов): mel `[1,n_mels,T]`, encoder
    /// `[1,d_model,T']`, logits `[1,T',C]`.
    pub fn forward_debug(&self, audio16k: &[f32]) -> Result<(Tensor, Tensor, Tensor)> {
        let mel = self.mel_tensor(audio16k)?;
        let encoded = self.model.encode(&mel)?;
        let logits = self.model.head_logits(&encoded)?;
        Ok((mel, encoded, logits))
    }

    /// greedy-CTC: argmax по классу, схлопывание повторов, удаление blank.
    pub fn greedy_ctc(&self, log_probs: &Tensor) -> Result<Vec<u32>> {
        let labels = log_probs.argmax(2)?; // [1, T']
        let labels: Vec<u32> = labels
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        Ok(collapse_ctc(&labels, self.blank_id))
    }

    pub fn greedy_ctc_decode(&self, log_probs: &Tensor) -> Result<String> {
        let ids = self.greedy_ctc(log_probs)?;
        Ok(self.spm.decode(&ids))
    }
}

/// unique_consecutive + remove blank.
fn collapse_ctc(labels: &[u32], blank_id: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut prev: Option<u32> = None;
    for &l in labels {
        if Some(l) != prev {
            if l != blank_id {
                out.push(l);
            }
            prev = Some(l);
        }
    }
    out
}
