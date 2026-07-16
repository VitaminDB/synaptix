//! High-level транскрибация: WAV/сэмплы → mel → энкодер → greedy-декод с
//! KV-cache → текст. Поддержка >30 с (chunking), авто-детекции языка,
//! suppress-токенов; temperature fallback и timestamps — следующим слоем.

use std::path::Path;

use synaptix_audio::{read_wav_mono_f32, resample_linear};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::{GenerationConfig, WhisperConfig};
use crate::loader::WhisperWeights;
use crate::mel::whisper_log_mel;
use crate::model::WhisperModel;
#[cfg(feature = "cuda")]
use crate::model::WhisperDecodeState;
use crate::{Result, WhisperError};

pub const SR: u32 = 16000;
pub const N_SAMPLES: usize = 30 * SR as usize; // 480000

/// Спец-токены Whisper, разрешённые через токенизатор (id зависят от vocab).
pub struct SpecialTokens {
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Transcribe,
    Translate,
}

pub struct WhisperPipeline {
    model: WhisperModel,
    tokenizer: HfTokenizer,
    config: WhisperConfig,
    gen_config: GenerationConfig,
    special: SpecialTokens,
    device: Device,
    dtype: DType,
}

impl WhisperPipeline {
    pub fn from_syn(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        let w = WhisperWeights::open(path, device, dtype)?;
        let model = WhisperModel::load(&w)?;
        let tok_bytes = w.read_aux("tokenizer.json")?;
        let tokenizer =
            HfTokenizer::from_bytes(&tok_bytes).map_err(|e| WhisperError::Tokenizer(e.to_string()))?;

        let id = |s: &str| -> Result<u32> {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| WhisperError::Tokenizer(format!("missing token {s}")))
        };
        let special = SpecialTokens {
            sot: id("<|startoftranscript|>")?,
            eot: id("<|endoftext|>")?,
            transcribe: id("<|transcribe|>")?,
            translate: id("<|translate|>")?,
            no_timestamps: id("<|notimestamps|>")?,
            timestamp_begin: id("<|0.00|>")?,
        };

        Ok(Self {
            config: w.config.clone(),
            gen_config: w.gen_config.clone(),
            model,
            tokenizer,
            special,
            device,
            dtype,
        })
    }

    pub fn language_token(&self, lang: &str) -> Option<u32> {
        self.tokenizer.token_to_id(&format!("<|{lang}|>"))
    }

    /// Прочитать WAV, свести в моно (read_wav_mono_f32 уже сводит) и ресэмплить
    /// до 16 кГц.
    pub fn load_audio(path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let (samples, sr) = read_wav_mono_f32(path).map_err(|e| WhisperError::Audio(e.to_string()))?;
        if sr == SR {
            Ok(samples)
        } else {
            resample_linear(&samples, sr, SR).map_err(|e| WhisperError::Audio(e.to_string()))
        }
    }

    fn mel_tensor(&self, segment: &[f32]) -> Result<Tensor> {
        let n_mels = self.config.num_mel_bins;
        let target = self.config.max_source_positions * 2;
        let (flat, n_mels, n_frames) = whisper_log_mel(segment, n_mels, target)?;
        let t = Tensor::from_vec(flat, (1, n_mels, n_frames), self.device)?;
        Ok(t.to_dtype(self.dtype)?)
    }

    /// Авто-детекция языка: один шаг с `sot`, argmax по языковым токенам.
    pub fn detect_language(&self, enc_out: &Tensor) -> Result<u32> {
        let mut cache = self.model.decoder.init_cache(enc_out)?;
        let logits = self.model.decoder.decode_step(self.special.sot, 0, &mut cache)?;
        let mut v = logits_to_f32(&logits)?;
        // Разрешаем только диапазон языковых токенов [sot+1 .. timestamp_begin).
        let lo = self.special.sot as usize + 1;
        let hi = self.special.timestamp_begin as usize;
        for (i, x) in v.iter_mut().enumerate() {
            if i < lo || i >= hi {
                *x = f32::NEG_INFINITY;
            }
        }
        Ok(argmax(&v) as u32)
    }

    /// Токен-id одного 30-с сегмента (для сверки/CLI). `samples` дополняются/
    /// обрезаются до 30 с.
    pub fn segment_token_ids(&self, samples: &[f32], lang: &str, task: Task) -> Result<Vec<u32>> {
        let mut seg = samples.to_vec();
        seg.resize(N_SAMPLES, 0.0);
        let mel = self.mel_tensor(&seg)?;
        let lang_token = self
            .language_token(lang)
            .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {lang}")))?;
        self.decode_segment(&mel, lang_token, task)
    }

    /// Декод одного 30-с сегмента (greedy). `lang_token` — id `<|xx|>`.
    /// На CUDA идёт через CUDA-graph replay (decode_segment_graph), иначе —
    /// host-loop с KV-cache.
    fn decode_segment(&self, mel: &Tensor, lang_token: u32, task: Task) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::Cuda(_)) {
            return self.decode_segment_graph(mel, lang_token, task);
        }
        let enc_out = self.model.encoder.forward(mel)?;
        let mut cache = self.model.decoder.init_cache(&enc_out)?;

        let task_tok = match task {
            Task::Transcribe => self.special.transcribe,
            Task::Translate => self.special.translate,
        };
        let prefix = [self.special.sot, lang_token, task_tok, self.special.no_timestamps];

        let mut pos = 0usize;
        let mut logits = None;
        for &t in &prefix {
            logits = Some(self.model.decoder.decode_step(t, pos, &mut cache)?);
            pos += 1;
        }

        let max_len = self.config.max_target_positions;
        let mut out: Vec<u32> = Vec::new();
        loop {
            let mut v = logits_to_f32(&logits.take().unwrap())?;
            self.suppress(&mut v, out.is_empty());
            let next = argmax(&v) as u32;
            if next == self.special.eot || pos >= max_len {
                break;
            }
            out.push(next);
            logits = Some(self.model.decoder.decode_step(next, pos, &mut cache)?);
            pos += 1;
        }
        Ok(out)
    }

    /// Greedy-декод сегмента через CUDA-graph: префикс прогоняется обычными
    /// device-шагами (populate self-KV), затем шаг захватывается в граф и
    /// реплеится. Логиты копятся в стабильный `state.logits`, argmax+suppress
    /// на host (1 DtoH/шаг).
    #[cfg(feature = "cuda")]
    fn decode_segment_graph(&self, mel: &Tensor, lang_token: u32, task: Task) -> Result<Vec<u32>> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        let ord = match self.device {
            Device::Cuda(o) => o,
            _ => return Err(WhisperError::Audio("graph decode requires CUDA".into())),
        };
        let enc_out = self.model.encoder.forward(mel)?;
        let max_target = self.config.max_target_positions;
        let mut cache =
            self.model
                .decoder
                .make_dev_cache(&enc_out, max_target, self.device, self.dtype)?;
        let mut state = WhisperDecodeState::new(self.device, self.dtype, self.config.vocab_size)?;

        let task_tok = match task {
            Task::Transcribe => self.special.transcribe,
            Task::Translate => self.special.translate,
        };
        let prefix = [self.special.sot, lang_token, task_tok, self.special.no_timestamps];

        // Префикс — обычные device-шаги (без графа): заполняют self-KV 0..len-1,
        // оставляют logits, предсказывающие первый контент-токен.
        for (p, &t) in prefix.iter().enumerate() {
            state.update(t, p as u32)?;
            no_grad(|| self.model.decoder.forward_decode_dev(&mut state, &mut cache))?;
        }

        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| WhisperError::Audio(format!("stream: {e}")))?;
        let mut capturer = GraphCapturer::new(3);
        let mut graph = None;

        let mut out: Vec<u32> = Vec::new();
        let mut cur_logits = logits_to_f32(&state.logits)?;
        let mut pos = prefix.len();
        loop {
            self.suppress(&mut cur_logits, out.is_empty());
            let next = argmax(&cur_logits) as u32;
            if next == self.special.eot || pos >= max_target {
                break;
            }
            out.push(next);

            // Продвигаем шаг: вход = next на позиции pos.
            state.update(next, pos as u32)?;
            match &graph {
                None => {
                    let g = {
                        let model = &self.model;
                        let st = &mut state;
                        let cc = &mut cache;
                        no_grad(|| {
                            capturer.capture_with(&stream, |_s| {
                                model
                                    .decoder
                                    .forward_decode_dev(st, cc)
                                    .map_err(|e| InferError::Other(e.to_string()))
                            })
                        })
                    }
                    .map_err(|e| WhisperError::Audio(format!("graph capture: {e}")))?;
                    let _ = g.upload();
                    graph = Some(g);
                }
                Some(g) => {
                    g.launch().map_err(|e| WhisperError::Audio(format!("graph launch: {e:?}")))?;
                    stream
                        .synchronize()
                        .map_err(|e| WhisperError::Audio(format!("sync: {e:?}")))?;
                }
            }
            cur_logits = logits_to_f32(&state.logits)?;
            pos += 1;
        }
        Ok(out)
    }

    /// Декод одного сегмента С timestamps (порт HF `WhisperTimeStampLogitsProcessor`).
    /// Возвращает сырые сгенерированные токены (с timestamp-токенами, без префикса).
    fn decode_segment_ts(&self, mel: &Tensor, lang_token: u32, task: Task) -> Result<Vec<u32>> {
        let enc_out = self.model.encoder.forward(mel)?;
        let mut cache = self.model.decoder.init_cache(&enc_out)?;
        let task_tok = match task {
            Task::Transcribe => self.special.transcribe,
            Task::Translate => self.special.translate,
        };
        // Префикс БЕЗ notimestamps — модель эмитит timestamp-токены.
        let prefix = [self.special.sot, lang_token, task_tok];

        let mut pos = 0usize;
        let mut logits = None;
        for &t in &prefix {
            logits = Some(self.model.decoder.decode_step(t, pos, &mut cache)?);
            pos += 1;
        }

        let max_len = self.config.max_target_positions;
        let mut out_ids: Vec<u32> = Vec::new();
        loop {
            let mut v = logits_to_f32(&logits.take().unwrap())?;
            self.apply_timestamp_rules(&mut v, &out_ids);
            let next = argmax(&v) as u32;
            if next == self.special.eot || pos >= max_len {
                break;
            }
            out_ids.push(next);
            logits = Some(self.model.decoder.decode_step(next, pos, &mut cache)?);
            pos += 1;
        }
        Ok(out_ids)
    }

    /// Правила timestamp-декода (порядок как в HF: suppress → begin → last/penult
    /// → max_initial/force-first → log-sum-exp). `tokens` — токены после префикса.
    fn apply_timestamp_rules(&self, scores: &mut [f32], tokens: &[u32]) {
        let tb = self.special.timestamp_begin as usize;
        let eot = self.special.eot as usize;
        let max_initial_index = 50; // max_initial_timestamp=1.0s / 0.02

        for &t in &self.gen_config.suppress_tokens {
            if let Some(x) = scores.get_mut(t as usize) {
                *x = f32::NEG_INFINITY;
            }
        }
        if let Some(x) = scores.get_mut(self.special.no_timestamps as usize) {
            *x = f32::NEG_INFINITY;
        }
        if tokens.is_empty() {
            for &t in &self.gen_config.begin_suppress_tokens {
                if let Some(x) = scores.get_mut(t as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }

        let n = tokens.len();
        let last_ts = n >= 1 && tokens[n - 1] as usize >= tb;
        let penult_ts = n < 2 || tokens[n - 2] as usize >= tb;
        if last_ts {
            if penult_ts {
                // два timestamp подряд → следующий обязан быть текстом.
                for x in scores.iter_mut().skip(tb) {
                    *x = f32::NEG_INFINITY;
                }
            } else {
                // одиночный timestamp → форсируем timestamp (текст < eot запрещён).
                for x in scores.iter_mut().take(eot) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }

        if tokens.is_empty() {
            // первый токен — обязательно timestamp, не позже max_initial.
            for x in scores.iter_mut().take(tb) {
                *x = f32::NEG_INFINITY;
            }
            for x in scores.iter_mut().skip(tb + max_initial_index + 1) {
                *x = f32::NEG_INFINITY;
            }
        }

        // log-sum-exp: если масса вероятности timestamp-токенов > макс. лог-проба
        // текстового токена → форсируем timestamp.
        let logprobs = log_softmax(scores);
        let ts_logsumexp = logsumexp(&logprobs[tb..]);
        let max_text = logprobs[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if ts_logsumexp > max_text {
            for x in scores.iter_mut().take(tb) {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    /// Применить suppress-токены к логитам (CPU). `first_step` — первый
    /// генерируемый токен (тогда также begin_suppress).
    fn suppress(&self, logits: &mut [f32], first_step: bool) {
        for &t in &self.gen_config.suppress_tokens {
            if let Some(x) = logits.get_mut(t as usize) {
                *x = f32::NEG_INFINITY;
            }
        }
        // notimestamps-режим: запрещаем timestamp-токены.
        for x in logits.iter_mut().skip(self.special.timestamp_begin as usize) {
            *x = f32::NEG_INFINITY;
        }
        if first_step {
            for &t in &self.gen_config.begin_suppress_tokens {
                if let Some(x) = logits.get_mut(t as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Транскрибировать весь аудиосигнал (16 кГц), при необходимости разбивая
    /// на 30-с сегменты. Язык: `Some("en")`/`None` (авто-детекция).
    pub fn transcribe(&self, audio: &[f32], lang: Option<&str>, task: Task) -> Result<String> {
        let mut text = String::new();
        let n_segments = audio.len().div_ceil(N_SAMPLES).max(1);
        for s in 0..n_segments {
            let start = s * N_SAMPLES;
            let end = (start + N_SAMPLES).min(audio.len());
            let mut seg = audio[start..end].to_vec();
            seg.resize(N_SAMPLES, 0.0);

            let mel = self.mel_tensor(&seg)?;
            let lang_token = match lang {
                Some(l) => self
                    .language_token(l)
                    .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {l}")))?,
                None => {
                    let enc = self.model.encoder.forward(&mel)?;
                    self.detect_language(&enc)?
                }
            };
            let ids = self.decode_segment(&mel, lang_token, task)?;
            let chunk = self
                .tokenizer
                .decode(&ids, true)
                .map_err(|e| WhisperError::Tokenizer(e.to_string()))?;
            if !text.is_empty() && !chunk.is_empty() {
                text.push(' ');
            }
            text.push_str(chunk.trim());
        }
        Ok(text)
    }
}

/// Сегмент с временными метками (секунды от начала всего аудио).
#[derive(Debug, Clone)]
pub struct TsSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

impl WhisperPipeline {
    /// Сырые токены timestamp-декода одного 30-с сегмента (для сверки/CLI).
    pub fn segment_token_ids_timestamps(
        &self,
        samples: &[f32],
        lang: &str,
        task: Task,
    ) -> Result<Vec<u32>> {
        let mut seg = samples.to_vec();
        seg.resize(N_SAMPLES, 0.0);
        let mel = self.mel_tensor(&seg)?;
        let lang_token = self
            .language_token(lang)
            .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {lang}")))?;
        self.decode_segment_ts(&mel, lang_token, task)
    }

    /// Транскрибация с временными метками: список сегментов (start,end,text).
    pub fn transcribe_timestamped(
        &self,
        audio: &[f32],
        lang: Option<&str>,
        task: Task,
    ) -> Result<Vec<TsSegment>> {
        let mut segments = Vec::new();
        let n_segments = audio.len().div_ceil(N_SAMPLES).max(1);
        for s in 0..n_segments {
            let start = s * N_SAMPLES;
            let end = (start + N_SAMPLES).min(audio.len());
            let mut seg = audio[start..end].to_vec();
            seg.resize(N_SAMPLES, 0.0);
            let offset = (s * 30) as f32;

            let mel = self.mel_tensor(&seg)?;
            let lang_token = match lang {
                Some(l) => self
                    .language_token(l)
                    .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {l}")))?,
                None => {
                    let enc = self.model.encoder.forward(&mel)?;
                    self.detect_language(&enc)?
                }
            };
            let seg_ids = self.decode_segment_ts(&mel, lang_token, task)?;
            segments.extend(self.parse_timestamps(&seg_ids, offset)?);
        }
        Ok(segments)
    }

    fn parse_timestamps(&self, tokens: &[u32], offset_s: f32) -> Result<Vec<TsSegment>> {
        let tb = self.special.timestamp_begin;
        let mut segs = Vec::new();
        let mut cur_start: Option<f32> = None;
        let mut text_ids: Vec<u32> = Vec::new();
        for &t in tokens {
            if t >= tb {
                let time = (t - tb) as f32 * 0.02 + offset_s;
                match cur_start {
                    None => cur_start = Some(time),
                    Some(st) => {
                        if !text_ids.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&text_ids, true)
                                .map_err(|e| WhisperError::Tokenizer(e.to_string()))?;
                            segs.push(TsSegment { start: st, end: time, text: text.trim().to_string() });
                        }
                        text_ids.clear();
                        cur_start = Some(time);
                    }
                }
            } else {
                text_ids.push(t);
            }
        }
        Ok(segs)
    }
}

fn logits_to_f32(logits: &Tensor) -> Result<Vec<f32>> {
    Ok(logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sum.ln();
    scores.iter().map(|&x| x - lse).collect()
}

fn logsumexp(v: &[f32]) -> f32 {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    max + v.iter().map(|&x| (x - max).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
