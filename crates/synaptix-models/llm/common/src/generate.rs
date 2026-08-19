use std::collections::HashSet;
use std::time::Instant;

use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::tensor::Tensor;
use synaptix_infer::sampling::{
    GreedySampler, LogitPipeline, MinPProcessor, MultinomialSampler, PresenceFrequencyProcessor,
    ProcessorContext, RepetitionPenaltyProcessor, Sampler, TemperatureProcessor, TopKProcessor,
    TopPProcessor,
};
use synaptix_ops::rng::Philox4x32;

use crate::model::{DecoderModel, KvCache, ModelError};

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// Окно штрафов (`repetition_penalty`, presence/frequency) по хвосту
    /// контекста в токенах. `0` — весь контекст.
    pub repeat_last_n: usize,
    /// Аддитивный штраф за сам факт появления токена в окне (OpenAI-семантика).
    pub presence_penalty: f32,
    /// Аддитивный штраф, пропорциональный числу повторов токена в окне.
    pub frequency_penalty: f32,
    pub seed: u64,
    pub eos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub max_seq: Option<usize>,
    pub prefill_batch: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            eos_token_id: None,
            eos_token_ids: Vec::new(),
            max_seq: None,
            prefill_batch: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_ms: u128,
    pub decode_ms: u128,
}

pub trait StreamSink {
    fn on_token(&mut self, token_id: u32) -> bool;
}

impl<F: FnMut(u32) -> bool> StreamSink for F {
    fn on_token(&mut self, token_id: u32) -> bool {
        self(token_id)
    }
}

struct NoopSink;

impl StreamSink for NoopSink {
    fn on_token(&mut self, _token_id: u32) -> bool {
        true
    }
}

pub struct TokenSampler {
    pipeline: LogitPipeline,
    sampler: Box<dyn Sampler>,
    rng: Philox4x32,
    context: Vec<u32>,
    step: usize,
    /// Сколько токенов хвоста контекста реально нужно процессорам штрафов:
    /// `None` — штрафы выключены, `Some(0)` — весь контекст, `Some(n)` — окно.
    ///
    /// Нужно не для семантики, а чтобы не копировать весь контекст в
    /// `ProcessorContext` на каждом шаге декода: на промпте агента в 30k
    /// токенов это 120 КБ memcpy и 30k итераций штрафа на каждый токен.
    penalty_window: Option<usize>,
}

impl TokenSampler {
    pub fn new(cfg: &GenerationConfig, prompt_ids: &[u32]) -> Self {
        let penalties_on = cfg.repetition_penalty != 1.0
            || cfg.presence_penalty != 0.0
            || cfg.frequency_penalty != 0.0;
        Self {
            pipeline: build_logit_pipeline(cfg),
            sampler: build_sampler(cfg),
            rng: Philox4x32::new(cfg.seed),
            context: prompt_ids.to_vec(),
            step: 0,
            penalty_window: penalties_on.then_some(cfg.repeat_last_n),
        }
    }

    /// Хвост контекста для процессоров штрафов (см. `penalty_window`).
    fn penalty_ids(&self) -> Vec<u32> {
        match self.penalty_window {
            None => Vec::new(),
            Some(0) => self.context.clone(),
            Some(n) => self.context[self.context.len().saturating_sub(n)..].to_vec(),
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32, ModelError> {
        let mut v = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let ctx = ProcessorContext {
            input_ids: self.penalty_ids(),
            step: self.step,
            batch_idx: 0,
        };
        self.pipeline
            .process(&mut v, &ctx)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let tok = self
            .sampler
            .sample(&v, &mut self.rng)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        self.context.push(tok);
        self.step += 1;
        Ok(tok)
    }

    pub fn probs(&mut self, logits: &Tensor) -> Result<Vec<f32>, ModelError> {
        let mut v = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let ctx = ProcessorContext {
            input_ids: self.penalty_ids(),
            step: self.step,
            batch_idx: 0,
        };
        self.pipeline
            .process(&mut v, &ctx)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            *x = if x.is_finite() { (*x - max).exp() } else { 0.0 };
            sum += *x;
        }
        if sum > 0.0 {
            for x in v.iter_mut() {
                *x /= sum;
            }
        }
        Ok(v)
    }

    pub fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let u = self.rng.next_f32_uniform();
        let mut acc = 0.0f32;
        for (i, p) in probs.iter().enumerate() {
            acc += *p;
            if u <= acc {
                return i as u32;
            }
        }
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    pub fn uniform(&mut self) -> f32 {
        self.rng.next_f32_uniform()
    }

    pub fn commit(&mut self, tok: u32) {
        self.context.push(tok);
        self.step += 1;
    }
}

pub fn eos_set(cfg: &GenerationConfig) -> HashSet<u32> {
    let mut eos: HashSet<u32> = cfg.eos_token_ids.iter().copied().collect();
    if let Some(e) = cfg.eos_token_id {
        eos.insert(e);
    }
    eos
}

pub fn generate(
    model: &DecoderModel,
    prompt_ids: &[u32],
    cfg: &GenerationConfig,
) -> Result<(Vec<u32>, GenerationStats), ModelError> {
    let mut sink = NoopSink;
    generate_streaming(model, prompt_ids, cfg, &mut sink)
}

pub fn generate_streaming(
    model: &DecoderModel,
    prompt_ids: &[u32],
    cfg: &GenerationConfig,
    sink: &mut dyn StreamSink,
) -> Result<(Vec<u32>, GenerationStats), ModelError> {
    if prompt_ids.is_empty() {
        return Err(ModelError::Forward("empty prompt".into()));
    }
    let kv_max = cfg.max_seq.unwrap_or(prompt_ids.len() + cfg.max_new_tokens);
    let mut kv = model.make_kv_cache(1, kv_max)?;
    generate_streaming_resume(model, &mut kv, prompt_ids, cfg, sink)
}

/// Как [`generate_streaming`], но prefill начинается с `kv.seq_len` (prefix-KV-кэш):
/// первые `kv.seq_len` токенов промпта считаются уже материализованными в `kv`
/// (caller гарантирует совпадение). Прифилливается только хвост `prompt_ids[seq..]`
/// — для него обязан остаться ≥1 токен. Decode дописывает в тот же `kv`.
pub fn generate_streaming_resume(
    model: &DecoderModel,
    kv: &mut KvCache,
    prompt_ids: &[u32],
    cfg: &GenerationConfig,
    sink: &mut dyn StreamSink,
) -> Result<(Vec<u32>, GenerationStats), ModelError> {
    if prompt_ids.is_empty() {
        return Err(ModelError::Forward("empty prompt".into()));
    }
    let device = model.device;
    let prompt_len = prompt_ids.len();
    let prefix = kv.seq_len.min(prompt_len.saturating_sub(1));
    kv.seq_len = prefix;

    let eos = eos_set(cfg);
    let mut sampler = TokenSampler::new(cfg, prompt_ids);

    let chunk = if cfg.prefill_batch == 0 {
        prompt_len
    } else {
        cfg.prefill_batch.max(1)
    };
    let t0 = Instant::now();
    let mut last_logits: Option<Tensor> = None;
    let mut off = prefix;
    while off < prompt_len {
        let end = (off + chunk).min(prompt_len);
        let slice = &prompt_ids[off..end];
        let t = Tensor::from_vec(slice.to_vec(), vec![1usize, slice.len()], device)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let logits = no_grad(|| model.forward(&t, &mut *kv))?;
        last_logits = Some(logits);
        off = end;
    }
    let prefill_ms = t0.elapsed().as_millis();
    let logits = last_logits.expect("prompt non-empty checked above");

    let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
    let first = sampler.sample(&logits)?;
    out.push(first);
    let mut cancelled = !sink.on_token(first);

    let dec_t0 = Instant::now();
    while !cancelled && out.len() < cfg.max_new_tokens {
        let last = *out.last().unwrap();
        if eos.contains(&last) {
            break;
        }
        let t = Tensor::from_vec(vec![last], vec![1usize, 1], device)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let logits = no_grad(|| model.forward(&t, &mut *kv))?;
        let tok = sampler.sample(&logits)?;
        out.push(tok);
        cancelled = !sink.on_token(tok);
    }
    let decode_ms = dec_t0.elapsed().as_millis();

    let stats = GenerationStats {
        prompt_tokens: prompt_len,
        new_tokens: out.len(),
        prefill_ms,
        decode_ms,
    };
    Ok((out, stats))
}

fn build_logit_pipeline(cfg: &GenerationConfig) -> LogitPipeline {
    let mut p = LogitPipeline::new();
    if cfg.repetition_penalty != 1.0 {
        p = p.add(RepetitionPenaltyProcessor {
            penalty: cfg.repetition_penalty,
            last_n: cfg.repeat_last_n,
        });
    }
    if cfg.presence_penalty != 0.0 || cfg.frequency_penalty != 0.0 {
        p = p.add(PresenceFrequencyProcessor {
            presence: cfg.presence_penalty,
            frequency: cfg.frequency_penalty,
            last_n: cfg.repeat_last_n,
        });
    }
    if cfg.temperature > 0.0 && cfg.temperature != 1.0 {
        p = p.add(TemperatureProcessor { temperature: cfg.temperature });
    }
    if cfg.top_k > 0 {
        p = p.add(TopKProcessor { k: cfg.top_k });
    }
    if cfg.top_p < 1.0 {
        p = p.add(TopPProcessor { p: cfg.top_p });
    }
    if cfg.min_p > 0.0 {
        p = p.add(MinPProcessor { min_p: cfg.min_p });
    }
    p
}

fn build_sampler(cfg: &GenerationConfig) -> Box<dyn Sampler> {
    if cfg.temperature <= 0.0 {
        Box::new(GreedySampler)
    } else {
        Box::new(MultinomialSampler)
    }
}
