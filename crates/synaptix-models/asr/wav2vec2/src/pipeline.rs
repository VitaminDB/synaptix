//! Высокоуровневый CTC-распознаватель: PCM → ресэмпл 16 кГц → нормализация →
//! log-probs `[T, V]` (на хосте) → greedy-декод.

use std::path::Path;

use synaptix_audio::resample;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::Wav2Vec2Config;
use crate::loader::Wav2Vec2Weights;
use crate::model::{ForwardTrace, Wav2Vec2Model};
use crate::vocab::CtcVocab;
use crate::{Result, Wav2Vec2Error};

pub const SR: u32 = 16000;

/// Покадровые log-вероятности CTC на хосте: `data[t * vocab + v]`.
#[derive(Debug, Clone)]
pub struct CtcLogProbs {
    pub frames: usize,
    pub vocab: usize,
    pub data: Vec<f32>,
    /// Длительность кадра в секундах (20 мс у стандартного extractor'а).
    pub frame_sec: f32,
}

impl CtcLogProbs {
    pub fn row(&self, t: usize) -> &[f32] {
        &self.data[t * self.vocab..(t + 1) * self.vocab]
    }

    pub fn at(&self, t: usize, v: usize) -> f32 {
        self.data[t * self.vocab + v]
    }
}

pub struct Wav2Vec2Ctc {
    model: Wav2Vec2Model,
    vocab: CtcVocab,
}

impl Wav2Vec2Ctc {
    pub fn from_syn(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        Self::build(Wav2Vec2Weights::from_syn(path, *device, dtype)?)
    }

    pub fn from_unpacked(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        Self::build(Wav2Vec2Weights::from_unpacked(dir, *device, dtype)?)
    }

    /// Каталог с `config.json` → распакованный снапшот, иначе `.syn`-файл.
    pub fn load(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let p = path.as_ref();
        if p.is_dir() {
            Self::from_unpacked(p, device, dtype)
        } else {
            Self::from_syn(p, device, dtype)
        }
    }

    fn build(w: Wav2Vec2Weights) -> Result<Self> {
        let model = Wav2Vec2Model::load(&w)?;
        if !model.has_ctc_head() {
            return Err(Wav2Vec2Error::Config("чекпойнт без lm_head".into()));
        }
        if w.vocab.len() > w.config.vocab_size {
            return Err(Wav2Vec2Error::Config(format!(
                "vocab.json: {} токенов > vocab_size {}",
                w.vocab.len(),
                w.config.vocab_size
            )));
        }
        Ok(Self { model, vocab: w.vocab })
    }

    pub fn config(&self) -> &Wav2Vec2Config {
        self.model.config()
    }

    pub fn vocab(&self) -> &CtcVocab {
        &self.vocab
    }

    pub fn model(&self) -> &Wav2Vec2Model {
        &self.model
    }

    pub fn blank_id(&self) -> u32 {
        self.model.config().blank_id() as u32
    }

    /// Вход модели `[1, N]` (с дополнением до корзины) и число кадров настоящей записи.
    fn input_tensor(&self, pcm: &[f32], sample_rate: u32) -> Result<(Tensor, usize)> {
        let resampled;
        let audio: &[f32] = if sample_rate == SR {
            pcm
        } else {
            resampled =
                resample(pcm, sample_rate, SR).map_err(|e| Wav2Vec2Error::Audio(e.to_string()))?;
            &resampled
        };
        let min = self.config().conv_kernel.iter().copied().max().unwrap_or(1).max(400);
        if audio.len() < min {
            return Err(Wav2Vec2Error::Audio(format!(
                "слишком короткая запись: {} сэмплов",
                audio.len()
            )));
        }
        let mut x = normalize_audio(audio);
        let real = x.len();
        // Устройства с графом фиксированных форм (NPU): длина записи — из
        // набора «корзин», хвост — тишина; лишние кадры отрезаются после модели.
        let padded = self.bucket_len(real);
        if padded > real {
            x.resize(padded, 0.0);
        }
        let n = x.len();
        let t = Tensor::from_vec(x, (1, n), self.model.device())?;
        Ok((t.to_dtype(self.model.dtype())?, self.config().frames_for(real)))
    }

    /// Корзины длины записи для устройств с графом фиксированных форм (NPU):
    /// каждая новая длина входа — новая сборка графа (десятки секунд), поэтому
    /// запись дополняется тишиной до ближайшей из 4 длин. Каждая корзина — свой
    /// набор контекстов с копией весов (~600 МБ у wav2vec2-large на диске и в PD),
    /// поэтому корзин мало, а резидентными держатся последние (см. кэш графов).
    /// На CPU/CUDA — как есть.
    /// `SYNAPTIX_LEN_BUCKETS=0` выключает, `SYNAPTIX_LEN_BUCKETS=1` включает везде.
    pub fn length_buckets_active(&self) -> bool {
        match std::env::var("SYNAPTIX_LEN_BUCKETS").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => matches!(self.model.device(), Device::Npu(_)),
        }
    }

    /// Длины корзин в сэмплах 16 кГц.
    pub fn length_buckets() -> &'static [usize] {
        const SEC: [f32; 4] = [2.0, 4.0, 8.0, 15.0];
        static B: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
        B.get_or_init(|| SEC.iter().map(|s| (s * SR as f32) as usize).collect())
    }

    fn bucket_len(&self, n: usize) -> usize {
        if !self.length_buckets_active() {
            return n;
        }
        Self::length_buckets().iter().copied().find(|&b| b >= n).unwrap_or(n)
    }

    /// PCM (любой sample_rate, моно f32) → log-probs на хосте.
    pub fn log_probs(&self, pcm: &[f32], sample_rate: u32) -> Result<CtcLogProbs> {
        let (x, real_frames) = self.input_tensor(pcm, sample_rate)?;
        let lp = self.model.log_probs(&x)?; // [1,T,V] f32
        let d = lp.dims().to_vec();
        let mut data = lp.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let frames = real_frames.min(d[1]);
        data.truncate(frames * d[2]);
        Ok(CtcLogProbs {
            frames,
            vocab: d[2],
            data,
            frame_sec: self.config().hop_samples() as f32 / SR as f32,
        })
    }

    /// Log-probs и последний hidden state энкодера (`[T, D]` построчно на хосте).
    pub fn log_probs_hidden(&self, pcm: &[f32], sample_rate: u32) -> Result<(CtcLogProbs, Vec<f32>, usize)> {
        let (x, real_frames) = self.input_tensor(pcm, sample_rate)?;
        let (lp, hidden) = self.model.log_probs_hidden(&x)?;
        let d = lp.dims().to_vec();
        let mut data = lp.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let hd = hidden.dims()[2];
        let mut h = hidden.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let frames = real_frames.min(d[1]);
        data.truncate(frames * d[2]);
        h.truncate(frames * hd);
        Ok((
            CtcLogProbs { frames, vocab: d[2], data, frame_sec: self.config().hop_samples() as f32 / SR as f32 },
            h,
            hd,
        ))
    }

    /// Промежуточные тензоры (для гейтов). Вход — уже 16 кГц.
    pub fn forward_trace(&self, audio16k: &[f32]) -> Result<ForwardTrace> {
        let (x, _) = self.input_tensor(audio16k, SR)?;
        self.model.forward_trace(&x)
    }

    /// Greedy-CTC: argmax по кадрам → схлопывание повторов → без blank.
    pub fn greedy(&self, lp: &CtcLogProbs) -> Vec<u32> {
        let labels: Vec<u32> = (0..lp.frames)
            .map(|t| {
                let row = lp.row(t);
                let mut best = 0usize;
                for (i, &v) in row.iter().enumerate() {
                    if v > row[best] {
                        best = i;
                    }
                }
                best as u32
            })
            .collect();
        collapse_ctc(&labels, self.blank_id())
    }

    /// PCM → последовательность токенов (фонем/символов) greedy-декодом.
    pub fn recognize(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<String>> {
        let lp = self.log_probs(pcm, sample_rate)?;
        Ok(self.greedy(&lp).into_iter().map(|i| self.vocab.token(i).to_string()).collect())
    }
}

/// Нормализация HF `Wav2Vec2FeatureExtractor(do_normalize=True)`:
/// `(x - mean) / sqrt(var + 1e-7)`.
pub fn normalize_audio(audio: &[f32]) -> Vec<f32> {
    let n = audio.len().max(1) as f64;
    let mean = audio.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = audio.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-7).sqrt();
    audio.iter().map(|&x| ((x as f64 - mean) * inv) as f32).collect()
}

/// unique_consecutive + удаление blank.
pub fn collapse_ctc(labels: &[u32], blank_id: u32) -> Vec<u32> {
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
