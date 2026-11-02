//! High-level транскрибация GigaAM: PCM → ресэмпл 16 кГц → log-mel → энкодер →
//! CTC-head → greedy-CTC-decode → SentencePiece-текст.

use std::path::Path;

use synaptix_audio::resample;
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

    /// Линейные слои квантуются при загрузке в `quant` (NVFP4/MXFP8/Sq) — на
    /// лету, независимо от формата весов; `dtype` — точность счёта.
    pub fn from_syn_quant(path: impl AsRef<Path>, device: &Device, dtype: DType, quant: Option<DType>) -> Result<Self> {
        let mut w = GigaAmWeights::from_syn(path, *device, dtype)?;
        w.quant = quant;
        Self::build(w)
    }

    pub fn from_unpacked_quant(dir: impl AsRef<Path>, device: &Device, dtype: DType, quant: Option<DType>) -> Result<Self> {
        let mut w = GigaAmWeights::from_unpacked(dir, *device, dtype)?;
        w.quant = quant;
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
                resample(pcm, sample_rate, SR).map_err(|e| GigaAmError::Audio(e.to_string()))?;
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

/// Слово CTC-выхода с таймкодами (секунды) и уверенностью.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// Среднее по токенам слова максимальной вероятности кадров токена.
    pub prob: f32,
}

impl GigaAm {
    /// Секунд на кадр выхода энкодера (hop × subsampling).
    pub fn frame_seconds(&self) -> f32 {
        (self.config.preprocessor.hop_length * self.config.encoder.subsampling_factor) as f32
            / self.config.preprocessor.sample_rate as f32
    }

    /// PCM → текст и слова по выравниванию greedy-CTC: токен занимает кадры своей
    /// серии argmax, слово — от первого кадра первого токена до конца последнего.
    pub fn transcribe_pcm_words(&self, pcm: &[f32], sample_rate: u32) -> Result<(String, Vec<CtcWord>)> {
        let resampled;
        let audio: &[f32] = if sample_rate == SR {
            pcm
        } else {
            resampled = resample(pcm, sample_rate, SR).map_err(|e| GigaAmError::Audio(e.to_string()))?;
            &resampled
        };
        let mel = self.mel_tensor(audio)?;
        let encoded = self.model.encode(&mel)?;
        let lp = self.model.head_logits(&encoded)?; // [1, T', C]
        let c = lp.dims()[2];
        let lp: Vec<f32> = lp.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let fs = self.frame_seconds();
        // серии одинаковых argmax-меток: (метка, первый кадр, конец, макс. вероятность)
        let mut runs: Vec<(u32, usize, usize, f32)> = Vec::new();
        for (t, row) in lp.chunks(c).enumerate() {
            let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    best = i;
                    bv = v;
                }
            }
            let p = bv.exp();
            match runs.last_mut() {
                Some(r) if r.0 == best as u32 => {
                    r.2 = t + 1;
                    r.3 = r.3.max(p);
                }
                _ => runs.push((best as u32, t, t + 1, p)),
            }
        }
        let mut words: Vec<CtcWord> = Vec::new();
        let mut probs: Vec<Vec<f32>> = Vec::new();
        for &(id, s, e, p) in runs.iter().filter(|r| r.0 != self.blank_id) {
            let piece = self.spm.id_to_piece(id as usize).unwrap_or("");
            let new_word = piece.starts_with('\u{2581}') || words.is_empty();
            let text: String = piece.trim_start_matches('\u{2581}').to_string();
            if new_word {
                words.push(CtcWord { text, start: s as f32 * fs, end: e as f32 * fs, prob: 0.0 });
                probs.push(vec![p]);
            } else {
                let w = words.last_mut().unwrap();
                w.text.push_str(&text);
                w.end = e as f32 * fs;
                probs.last_mut().unwrap().push(p);
            }
        }
        for (w, p) in words.iter_mut().zip(&probs) {
            w.prob = p.iter().sum::<f32>() / p.len() as f32;
        }
        words.retain(|w| !w.text.is_empty());
        let text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        Ok((text, words))
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
