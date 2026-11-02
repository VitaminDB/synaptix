//! High-level транскрибация: WAV/сэмплы → mel → энкодер → greedy-декод с
//! KV-cache → текст. Поддержка >30 с (chunking), авто-детекции языка,
//! suppress-токенов; temperature fallback и timestamps — следующим слоем.

use std::path::Path;

use synaptix_audio::{read_wav_mono_f32, resample};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::{GenerationConfig, WhisperConfig};
use crate::loader::WhisperWeights;
use crate::mel::whisper_log_mel;
use crate::model::{WhisperDecoder, WhisperEncoder, WhisperModel};
use crate::model::WhisperDecodeState;
use crate::words::{
    alignment_matrix, duration_limits, dtw, merge_punctuations, split_to_words, truncate_after_pause,
    truncate_at_sentence_ends, silences, snap_to_silences, RawWord, WordTiming, TOKENS_PER_SECOND,
};
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
    pub sot_prev: u32,
    /// `<|nospeech|>` (v3) / `<|nocaptions|>` (v1/v2).
    pub no_speech: Option<u32>,
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
    alignment_heads: Vec<(usize, usize)>,
    special: SpecialTokens,
    device: Device,
    dtype: DType,
    enc_device: Device,
    enc_dtype: DType,
}

impl WhisperPipeline {
    fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let r = self.encode_inner(mel);
        if std::env::var_os("SYNAPTIX_TIMING").is_some() {
            eprintln!("whisper: энкодер {:.0} мс", t0.elapsed().as_secs_f64() * 1e3);
        }
        r
    }

    fn encode_inner(&self, mel: &Tensor) -> Result<Tensor> {
        if self.enc_device == self.device && self.enc_dtype == self.dtype {
            return self.model.encoder.forward(mel);
        }
        let x = mel.to_device(self.enc_device)?.to_dtype(self.enc_dtype)?;
        let y = self.model.encoder.forward(&x)?;
        Ok(y.to_device(self.device)?.to_dtype(self.dtype)?)
    }

    pub fn from_syn(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        Self::from_syn_split(path, device, dtype, device, dtype)
    }

    /// Линейные слои квантуются при загрузке в `quant` (NVFP4/MXFP8/Sq) — на
    /// лету, независимо от формата весов в бандле; `dtype` — точность счёта.
    pub fn from_syn_quant(path: impl AsRef<Path>, device: Device, dtype: DType, quant: Option<DType>) -> Result<Self> {
        Self::from_syn_split_quant(path, device, dtype, device, dtype, quant)
    }

    /// Энкодер — на `enc_device` (например, NPU), декодер — на `device`.
    pub fn from_syn_split(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        enc_device: Device,
        enc_dtype: DType,
    ) -> Result<Self> {
        Self::from_syn_split_quant(path, device, dtype, enc_device, enc_dtype, None)
    }

    pub fn from_syn_split_quant(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        enc_device: Device,
        enc_dtype: DType,
        quant: Option<DType>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut w = WhisperWeights::open(path, device, dtype)?;
        w.quant = quant;
        let model = if enc_device == device && enc_dtype == dtype {
            WhisperModel::load(&w)?
        } else {
            let mut we = WhisperWeights::open(path, enc_device, enc_dtype)?;
            we.quant = quant;
            WhisperModel { encoder: WhisperEncoder::load(&we)?, decoder: WhisperDecoder::load(&w)? }
        };
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
            sot_prev: id("<|startofprev|>")?,
            no_speech: tokenizer.token_to_id("<|nospeech|>").or_else(|| tokenizer.token_to_id("<|nocaptions|>")),
        };
        let alignment_heads = crate::config::alignment_heads(&w.config, &w.gen_config);

        Ok(Self {
            config: w.config.clone(),
            gen_config: w.gen_config.clone(),
            alignment_heads,
            model,
            tokenizer,
            special,
            device,
            dtype,
            enc_device,
            enc_dtype,
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
            resample(&samples, sr, SR).map_err(|e| WhisperError::Audio(e.to_string()))
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
        let hi = self.special.translate.min(self.special.transcribe) as usize;
        for (i, x) in v.iter_mut().enumerate() {
            if i < lo || i >= hi {
                *x = f32::NEG_INFINITY;
            }
        }
        Ok(argmax(&v) as u32)
    }

    fn lang_code(&self, id: u32) -> Result<String> {
        let token = self
            .tokenizer
            .id_to_token(id)
            .ok_or_else(|| WhisperError::Tokenizer(format!("нет токена языка {id}")))?;
        Ok(token.trim_start_matches("<|").trim_end_matches("|>").to_string())
    }

    pub fn detect_language_code(&self, audio: &[f32]) -> Result<String> {
        let mut seg = audio[..audio.len().min(N_SAMPLES)].to_vec();
        seg.resize(N_SAMPLES, 0.0);
        let mel = self.mel_tensor(&seg)?;
        let enc = self.encode(&mel)?;
        let id = self.detect_language(&enc)?;
        let token = self
            .tokenizer
            .id_to_token(id)
            .ok_or_else(|| WhisperError::Tokenizer(format!("нет токена языка {id}")))?;
        Ok(token.trim_start_matches("<|").trim_end_matches("|>").to_string())
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
        self.decode_segment(&self.encode(&mel)?, lang_token, task)
    }

    /// Декод одного 30-с сегмента (greedy). `lang_token` — id `<|xx|>`.
    /// На CUDA идёт через CUDA-graph replay (decode_segment_graph), иначе —
    /// host-loop с KV-cache.
    fn decode_segment(&self, enc_out: &Tensor, lang_token: u32, task: Task) -> Result<Vec<u32>> {
        let prefix = self.sot_prefix(lang_token, task, &[]);
        self.decode_prefixed(enc_out, &prefix)
    }

    /// Префикс декода: `[<|startofprev|> подсказка…] sot язык задача notimestamps`.
    fn sot_prefix(&self, lang_token: u32, task: Task, prompt: &[u32]) -> Vec<u32> {
        let task_tok = match task {
            Task::Transcribe => self.special.transcribe,
            Task::Translate => self.special.translate,
        };
        let mut p = Vec::with_capacity(prompt.len() + 5);
        if !prompt.is_empty() {
            p.push(self.special.sot_prev);
            p.extend_from_slice(prompt);
        }
        p.extend([self.special.sot, lang_token, task_tok, self.special.no_timestamps]);
        p
    }

    /// Greedy-декод сегмента после заданного префикса (без timestamps).
    fn decode_prefixed(&self, enc_out: &Tensor, prefix: &[u32]) -> Result<Vec<u32>> {
        if matches!(self.device, Device::Cuda(_)) {
            return self.decode_segment_graph(enc_out, prefix);
        }
        if matches!(self.device, Device::Npu(_)) {
            return self.decode_segment_dev(enc_out, prefix);
        }
        let mut cache = self.model.decoder.init_cache(&enc_out)?;

        let mut pos = 0usize;
        let mut logits = None;
        for &t in prefix {
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

    /// Greedy-декод сегмента device-шагами с фиксированными формами (preallocated self-KV,
    /// позиция и длина — тензоры): каждый шаг — один и тот же граф (NPU).
    fn decode_segment_dev(&self, enc_out: &Tensor, prefix: &[u32]) -> Result<Vec<u32>> {
        let max_target = self.config.max_target_positions;
        let mut cache = self.model.decoder.make_dev_cache(enc_out, max_target, self.device, self.dtype)?;
        let mut state = WhisperDecodeState::new(self.device, self.dtype, self.config.vocab_size)?;
        let mut logits = Vec::new();
        for (p, &t) in prefix.iter().enumerate() {
            state.update(t, p as u32)?;
            self.model.decoder.forward_decode_dev(&mut state, &mut cache)?;
            logits = logits_to_f32(&state.logits)?;
        }
        let mut out: Vec<u32> = Vec::new();
        let mut pos = prefix.len();
        loop {
            self.suppress(&mut logits, out.is_empty());
            let next = argmax(&logits) as u32;
            if next == self.special.eot || pos >= max_target {
                break;
            }
            out.push(next);
            state.update(next, pos as u32)?;
            self.model.decoder.forward_decode_dev(&mut state, &mut cache)?;
            logits = logits_to_f32(&state.logits)?;
            pos += 1;
        }
        Ok(out)
    }

    /// Greedy-декод сегмента через CUDA-graph: префикс прогоняется обычными
    /// device-шагами (populate self-KV), затем шаг захватывается в граф и
    /// реплеится. Логиты копятся в стабильный `state.logits`, argmax+suppress
    /// на host (1 DtoH/шаг).
    fn decode_segment_graph(&self, enc_out: &Tensor, prefix: &[u32]) -> Result<Vec<u32>> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        let ord = match self.device {
            Device::Cuda(o) => o,
            _ => return Err(WhisperError::Audio("graph decode requires CUDA".into())),
        };
        let max_target = self.config.max_target_positions;
        let mut cache =
            self.model
                .decoder
                .make_dev_cache(&enc_out, max_target, self.device, self.dtype)?;
        let mut state = WhisperDecodeState::new(self.device, self.dtype, self.config.vocab_size)?;

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
    fn decode_segment_ts(&self, enc_out: &Tensor, lang_token: u32, task: Task) -> Result<Vec<u32>> {
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
        Ok(self.transcribe_lang(audio, lang, task)?.0)
    }

    /// Как [`transcribe`], плюс код языка (заданный или определённый по первому сегменту).
    pub fn transcribe_lang(&self, audio: &[f32], lang: Option<&str>, task: Task) -> Result<(String, String)> {
        let mut text = String::new();
        let mut code = lang.map(str::to_string);
        let n_segments = audio.len().div_ceil(N_SAMPLES).max(1);
        for s in 0..n_segments {
            let start = s * N_SAMPLES;
            let end = (start + N_SAMPLES).min(audio.len());
            let mut seg = audio[start..end].to_vec();
            seg.resize(N_SAMPLES, 0.0);

            let mel = self.mel_tensor(&seg)?;
            let enc = self.encode(&mel)?;
            let lang_token = match lang {
                Some(l) => self
                    .language_token(l)
                    .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {l}")))?,
                None => {
                    let id = self.detect_language(&enc)?;
                    if code.is_none() {
                        code = Some(self.lang_code(id)?);
                    }
                    id
                }
            };
            let ids = self.decode_segment(&enc, lang_token, task)?;
            let chunk = self
                .tokenizer
                .decode(&ids, true)
                .map_err(|e| WhisperError::Tokenizer(e.to_string()))?;
            if !text.is_empty() && !chunk.is_empty() {
                text.push(' ');
            }
            text.push_str(chunk.trim());
        }
        Ok((text, code.unwrap_or_default()))
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
        self.decode_segment_ts(&self.encode(&mel)?, lang_token, task)
    }

    /// Транскрибация с временными метками: список сегментов (start,end,text).
    pub fn transcribe_timestamped(
        &self,
        audio: &[f32],
        lang: Option<&str>,
        task: Task,
    ) -> Result<Vec<TsSegment>> {
        Ok(self.transcribe_timestamped_lang(audio, lang, task)?.0)
    }

    pub fn transcribe_timestamped_lang(
        &self,
        audio: &[f32],
        lang: Option<&str>,
        task: Task,
    ) -> Result<(Vec<TsSegment>, String)> {
        let mut segments = Vec::new();
        let mut code = lang.map(str::to_string);
        let n_segments = audio.len().div_ceil(N_SAMPLES).max(1);
        for s in 0..n_segments {
            let start = s * N_SAMPLES;
            let end = (start + N_SAMPLES).min(audio.len());
            let mut seg = audio[start..end].to_vec();
            seg.resize(N_SAMPLES, 0.0);
            let offset = (s * 30) as f32;

            let mel = self.mel_tensor(&seg)?;
            let enc = self.encode(&mel)?;
            let lang_token = match lang {
                Some(l) => self
                    .language_token(l)
                    .ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {l}")))?,
                None => {
                    let id = self.detect_language(&enc)?;
                    if code.is_none() {
                        code = Some(self.lang_code(id)?);
                    }
                    id
                }
            };
            let seg_ids = self.decode_segment_ts(&enc, lang_token, task)?;
            segments.extend(self.parse_timestamps(&seg_ids, offset)?);
        }
        Ok((segments, code.unwrap_or_default()))
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

/// logsumexp по `v[..split]` и по всему `v` за один максимум.
fn lse_split(v: &[f32], split: usize) -> (f32, f32) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let a: f32 = v[..split].iter().map(|&x| (x - max).exp()).sum();
    let b: f32 = v[split..].iter().map(|&x| (x - max).exp()).sum();
    (max + a.ln(), max + (a + b).ln())
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

/// Подсказка с частыми заполнителями: без неё Whisper (turbo) выбрасывает
/// «um/uh» почти всегда; известная фраза openai («Umm, let me think like,
/// hmm…») на turbo не помогла, плотная подсказка вернула 14 из 17.
pub const VERBATIM_PROMPT: &str = "Um, uh, so, uh, I was, um, like, uh, thinking, um, that, uh, we could, um, go.";

/// Опции распознавания со словами.
#[derive(Debug, Clone)]
pub struct WordOptions {
    /// Подсказка [`VERBATIM_PROMPT`] через `<|startofprev|>`: Whisper реже
    /// выбрасывает заполнители пауз.
    pub verbatim: bool,
    /// Своя подсказка (перекрывает `verbatim`).
    pub initial_prompt: Option<String>,
    /// Подтягивать границы слов к паузам по энергии сигнала (по умолчанию да):
    /// DTW по вниманию размазывает слова на паузы.
    pub snap_to_pauses: bool,
}

impl Default for WordOptions {
    fn default() -> Self {
        Self { verbatim: false, initial_prompt: None, snap_to_pauses: true }
    }
}

/// Итог распознавания со словами.
#[derive(Debug, Clone, Default)]
pub struct WordsResult {
    pub text: String,
    pub language: String,
    /// Сегменты по окнам (от первого до последнего слова окна).
    pub segments: Vec<TsSegment>,
    pub words: Vec<WordTiming>,
}

/// Порог «тишины» как в openai: no_speech > 0.6 и средний logprob < -1.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;
/// Поправка таймкодов DTW: границы по вниманию в среднем на ~50 мс раньше
/// акустических (сверка с MFA на LibriSpeech и KSC).
const TIME_SHIFT: f32 = 0.04;
/// Минимальная пауза для подтяжки границ и допуск подтяжки (с).
const MIN_PAUSE: f32 = 0.15;
const PAUSE_SNAP: f32 = 0.3;
/// Сэмплов на кадр энкодера (20 мс).
const SAMPLES_PER_FRAME: usize = (SR as usize) / 50;

/// Выравнивание одного окна.
struct WindowAlign {
    words: Vec<RawWord>,
    no_speech: f32,
    avg_logprob: f32,
}

impl WhisperPipeline {
    fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(&format!(" {}", text.trim()), false)
            .map_err(|e| WhisperError::Tokenizer(e.to_string()))?;
        let mut ids: Vec<u32> = enc.ids.iter().copied().filter(|&t| t < self.special.eot).collect();
        // openai: не более max_target/2 - 1 токенов подсказки (хвост).
        let max = self.config.max_target_positions / 2 - 1;
        if ids.len() > max {
            ids.drain(..ids.len() - max);
        }
        Ok(ids)
    }

    fn decode_text(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Teacher-forced прогон по выбранным токенам: cross-attn голов выравнивания
    /// → DTW → слова (время от начала окна), вероятности токенов, no_speech.
    fn align_window(&self, enc: &Tensor, prefix: &[u32], text: &[u32], frames: usize, lang: &str) -> Result<WindowAlign> {
        let eot = self.special.eot;
        let mut tokens = prefix.to_vec();
        tokens.extend_from_slice(text);
        tokens.push(eot);
        let (logits, qk) = self.model.decoder.forward_prefix_attn(&tokens, enc, &self.alignment_heads)?;

        // Строки от sot: sot язык задача notimestamps текст… eot.
        let sot_pos = prefix.len() - 4;
        let rows = tokens.len() - sot_pos;
        let vocab = logits.dims()[2];
        let lg = logits.narrow(1, sot_pos, rows)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;

        let no_speech = match self.special.no_speech {
            Some(ns) => {
                let row = &lg[..vocab];
                (row[ns as usize] - logsumexp(row)).exp()
            }
            None => 0.0,
        };
        // Вероятности текстовых токенов: softmax по [..eot] (как openai) + logprob
        // по всему словарю для среднего (с eot).
        let rows_tf: Vec<u32> = text.iter().copied().chain(std::iter::once(eot)).collect();
        let lp: Vec<(f32, f32)> = std::thread::scope(|s| {
            let chunk = rows_tf.len().div_ceil(8).max(1);
            let hs: Vec<_> = rows_tf
                .chunks(chunk)
                .enumerate()
                .map(|(ci, ts)| {
                    let lg = &lg;
                    s.spawn(move || {
                        ts.iter()
                            .enumerate()
                            .map(|(k, &t)| {
                                let j = ci * chunk + k;
                                let row = &lg[(3 + j) * vocab..(4 + j) * vocab];
                                let (lse_text, lse_all) = lse_split(row, eot as usize);
                                (row[t as usize] - lse_text, row[t as usize] - lse_all)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().expect("logprobs")).collect()
        });
        let probs: Vec<f32> = lp[..text.len()].iter().map(|p| p.0.exp()).collect();
        let sum_lp: f32 = lp.iter().map(|p| p.1).sum();
        let avg_logprob = sum_lp / (text.len() + 1) as f32;

        if text.is_empty() {
            return Ok(WindowAlign { words: Vec::new(), no_speech, avg_logprob });
        }
        let t_enc = enc.dims()[1];
        let s = tokens.len();
        let qk_rows: Vec<Vec<f32>> = qk.iter().map(|h| h[sot_pos * t_enc..s * t_enc].to_vec()).collect();
        let full = alignment_matrix(&qk_rows, rows, t_enc, frames);
        let frames = frames.clamp(1, t_enc);
        // Строки notimestamps..последний текстовый (n_text + 1), стоимость = −матрица.
        let n = text.len() + 1;
        let cost: Vec<f32> = full[3 * frames..(3 + n) * frames].iter().map(|v| -v).collect();
        let (ti, fi) = dtw(&cost, n, frames);
        let mut jump_times = Vec::with_capacity(n);
        for k in 0..ti.len() {
            if k == 0 || ti[k] != ti[k - 1] {
                jump_times.push(fi[k] as f32 / TOKENS_PER_SECOND);
            }
        }

        let mut all = text.to_vec();
        all.push(eot);
        let dec = |ids: &[u32]| self.decode_text(ids);
        let (ws, groups) = split_to_words(&all, eot, lang, &dec);
        let mut words = Vec::with_capacity(ws.len());
        let mut b = 0usize;
        for (w, g) in ws.into_iter().zip(groups) {
            if g[0] >= eot {
                break;
            }
            let e = b + g.len();
            words.push(RawWord {
                word: w,
                start: jump_times[b.min(jump_times.len() - 1)],
                end: jump_times[e.min(jump_times.len() - 1)],
                probs: probs[b..e].to_vec(),
                tokens: g,
            });
            b = e;
        }
        Ok(WindowAlign { words, no_speech, avg_logprob })
    }

    /// Распознавание со словами: таймкоды слов (DTW по cross-attn), уверенность
    /// слова по вероятностям его токенов, сегменты, текст и язык. Аудио — 16 кГц
    /// моно любой длины: окна по 30 с, следующее окно начинается с конца
    /// последнего целого слова (недосказанное на границе окна слово переносится).
    pub fn transcribe_words(&self, audio: &[f32], lang: Option<&str>, opts: &WordOptions) -> Result<WordsResult> {
        let prompt = match (&opts.initial_prompt, opts.verbatim) {
            (Some(p), _) if !p.trim().is_empty() => self.encode_prompt(p)?,
            (_, true) => self.encode_prompt(VERBATIM_PROMPT)?,
            _ => Vec::new(),
        };
        let timing = std::env::var_os("SYNAPTIX_TIMING").is_some();
        let mut res = WordsResult::default();
        let mut lang_tok: Option<u32> = match lang {
            Some(l) => Some(self.language_token(l).ok_or_else(|| WhisperError::Tokenizer(format!("unknown language {l}")))?),
            None => None,
        };
        res.language = lang.map(str::to_string).unwrap_or_default();
        let min_window = SR as usize / 10;
        let total = audio.len() as f32 / SR as f32;
        let mut last_speech = 0.0f32;
        let mut seek = 0usize;
        while seek + min_window <= audio.len() || (seek == 0 && !audio.is_empty()) {
            let end = (seek + N_SAMPLES).min(audio.len());
            let mut seg = audio[seek..end].to_vec();
            let win_len = seg.len();
            seg.resize(N_SAMPLES, 0.0);
            let offset = seek as f32 / SR as f32;
            let last_window = end >= audio.len();

            let t0 = std::time::Instant::now();
            let mel = self.mel_tensor(&seg)?;
            let enc = self.encode(&mel)?;
            let lt = match lang_tok {
                Some(t) => t,
                None => {
                    let id = self.detect_language(&enc)?;
                    res.language = self.lang_code(id)?;
                    lang_tok = Some(id);
                    id
                }
            };
            let prefix = self.sot_prefix(lt, Task::Transcribe, &prompt);
            let t1 = std::time::Instant::now();
            let text = self.decode_prefixed(&enc, &prefix)?;
            let t2 = std::time::Instant::now();
            let frames = win_len / SAMPLES_PER_FRAME;
            let al = self.align_window(&enc, &prefix, &text, frames, &res.language)?;
            if timing {
                eprintln!(
                    "whisper words: окно {:.1} с: энкодер {:.0} мс, декод {:.0} мс ({} ток.), выравнивание {:.0} мс",
                    offset,
                    (t1 - t0).as_secs_f64() * 1e3,
                    (t2 - t1).as_secs_f64() * 1e3,
                    text.len(),
                    t2.elapsed().as_secs_f64() * 1e3
                );
            }
            if al.no_speech > NO_SPEECH_THRESHOLD && al.avg_logprob < LOGPROB_THRESHOLD {
                seek = end;
                continue;
            }
            let mut words = al.words;
            let (med, max_dur) = duration_limits(&words);
            truncate_at_sentence_ends(&mut words, max_dur);
            merge_punctuations(&mut words);
            words.retain(|w| !w.word.is_empty());

            // Не последнее окно: слово у правого края могло оборваться — переносим
            // его в следующее окно, которое начнётся с конца предыдущего слова.
            let mut next_seek = end;
            if !last_window && words.len() >= 2 {
                words.pop();
                let cut = words.last().unwrap().end;
                let s = seek + (cut * SR as f32) as usize;
                if s > seek + SR as usize {
                    next_seek = s;
                }
            }

            let mut out: Vec<WordTiming> = words
                .iter()
                .map(|w| {
                    let n = w.probs.len().max(1) as f32;
                    WordTiming {
                        text: w.word.trim().to_string(),
                        start: (offset + w.start + TIME_SHIFT).min(total),
                        end: (offset + w.end + TIME_SHIFT).min(total),
                        prob: w.probs.iter().sum::<f32>() / n,
                        prob_min: w.probs.iter().copied().fold(1.0, f32::min),
                    }
                })
                .collect();
            truncate_after_pause(&mut out, last_speech, med, max_dur);
            if opts.snap_to_pauses {
                let mut sil = silences(&audio[seek..end], SR, MIN_PAUSE);
                sil.iter_mut().for_each(|p| {
                    p.0 += offset;
                    p.1 += offset;
                });
                snap_to_silences(&mut out, &sil, PAUSE_SNAP);
            }
            if let (Some(f), Some(l)) = (out.first(), out.last()) {
                let mut seg_text = String::new();
                for w in &words {
                    seg_text.push_str(&w.word);
                }
                let seg_text = seg_text.trim().to_string();
                res.segments.push(TsSegment { start: f.start, end: l.end, text: seg_text.clone() });
                if !res.text.is_empty() {
                    res.text.push(' ');
                }
                res.text.push_str(&seg_text);
                last_speech = l.end;
            }
            res.words.extend(out);
            seek = next_seek;
            if last_window {
                break;
            }
        }
        Ok(res)
    }
}
