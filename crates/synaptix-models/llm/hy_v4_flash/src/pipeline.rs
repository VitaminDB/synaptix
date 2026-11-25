//! Генерация Hy4-flash: токенизатор, сессия с префиксным кэшем, префилл чанками.
//!
//! KV индексированных слоёв адресуется позицией и назад откатывается простым
//! укорочением; кольца SWA (окно + чанк) — только пока после точки отката
//! записано меньше чанка, иначе ход считается с нуля.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::expert_mirror as em;
use synaptix_llm_common::generate::{eos_set, GenerationConfig, GenerationStats, StreamSink, TokenSampler, INTERRUPTED};
use synaptix_llm_common::moe::{ExpertCache, ExpertSource};
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Hy4Config;
use crate::loader::{Hy4Weights, LoadError};
use crate::model::{Hy4Model, ModelCache};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("model: {0}")]
    Model(#[from] ModelError),
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// Чанк префилла: `SYN_HY4_CHUNK` (2048). Кольца SWA держат окно + чанк.
pub fn prefill_chunk() -> usize {
    env_usize("SYN_HY4_CHUNK").unwrap_or(2048).max(16)
}

pub struct Hy4Pipeline {
    /// Закреплённое зеркало стопок экспертов в RAM (сколько влезло). Первое
    /// поле: поля роняются по порядку, и копировщики зеркала должны встать
    /// раньше, чем модель отпустит mmap бандла, из которого они читают.
    _mirror: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    pub model: Hy4Model,
    pub tokenizer: HfTokenizer,
    pub chat_template: Option<String>,
    pub config: Hy4Config,
    pub path: PathBuf,
    max_seq: usize,
}

pub struct Hy4Session {
    pub cache: ModelCache,
    resume_at: Option<usize>,
    parked: bool,
    device: Device,
}

impl Hy4Session {
    pub fn ctx_tokens(&self) -> usize {
        self.cache.max_seq
    }
    pub fn cached_tokens(&self) -> usize {
        self.cache.seq_len()
    }
    pub fn set_resume_at(&mut self, at: Option<usize>) {
        self.resume_at = at;
    }
    pub fn resume_at(&self) -> Option<usize> {
        self.resume_at
    }
    pub fn invalidate(&mut self) {
        self.cache.truncate(0);
    }
    pub fn device_bytes(&self) -> usize {
        if self.parked { 0 } else { self.cache.device_bytes() }
    }
    /// Увести KV в RAM (освободить карту под вложенную генерацию).
    pub fn park_to_host(&mut self) -> Result<usize, PipelineError> {
        if self.parked {
            return Ok(0);
        }
        let n = self.cache.device_bytes();
        self.device = self.cache.move_to(Device::Cpu)?;
        self.parked = true;
        Ok(n)
    }
    /// Вернуть KV на карту.
    pub fn unpark_to(&mut self, device: Device) -> Result<usize, PipelineError> {
        if !self.parked {
            return Ok(0);
        }
        self.cache.move_to(device)?;
        self.parked = false;
        Ok(self.cache.device_bytes())
    }
    pub fn is_parked(&self) -> bool {
        self.parked
    }
    /// Сколько токенов `prompt_ids` можно взять из кэша.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        let ids = &self.cache.ids;
        let n = ids.iter().zip(prompt_ids).take_while(|(a, b)| a == b).count();
        // Последний токен промпта всегда считаем заново (нужны его логиты);
        // кольцо SWA не должно быть перезаписано после точки отката.
        let n = n.min(prompt_ids.len().saturating_sub(1));
        if ids.len() - n > prefill_chunk() { 0 } else { n }
    }
}

/// Кэш экспертов на карте: `SYN_HY4_EXPERT_CACHE_GB` (0 — все на карту).
/// Без переменной — вся свободная VRAM за вычетом запаса [`vram_reserve`]
/// (ёмкость подгоняется после загрузки и перед каждым ходом).
fn expert_cache(device: Device) -> Option<Arc<ExpertCache>> {
    let Device::Cuda(ord) = device else { return None };
    let bytes = match std::env::var("SYN_HY4_EXPERT_CACHE_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        Some(gb) if gb <= 0.0 => return None,
        Some(gb) => (gb * (1u64 << 30) as f64) as usize,
        None => synaptix_core::device::cuda::mem_info(ord).map(|(_, total)| total).unwrap_or(8 << 30),
    };
    Some(ExpertCache::new(device, bytes))
}

/// Запас VRAM под KV и активации хода: `SYN_HY4_VRAM_RESERVE_GB` (3).
fn vram_reserve() -> usize {
    let gb = std::env::var("SYN_HY4_VRAM_RESERVE_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(3.0);
    (gb * (1u64 << 30) as f64) as usize
}

/// Ёмкость кэша экспертов под свободную VRAM (только в авто-режиме).
fn fit_cache(model: &Hy4Model) {
    if std::env::var_os("SYN_HY4_EXPERT_CACHE_GB").is_some() {
        return;
    }
    if let Some(c) = model.expert_cache() {
        c.fit_to_vram(vram_reserve());
    }
}

impl Hy4Pipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, max_seq: Option<usize>) -> Result<Self, PipelineError> {
        let path = path.as_ref().to_path_buf();
        let le = |e: LoadError| PipelineError::Load(e.to_string());
        let w = Hy4Weights::open(&path).map_err(le)?;
        let tokenizer = HfTokenizer::from_bytes(&w.tokenizer_json).map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let cache = expert_cache(device);
        let stacks = w.expert_stacks().map(Arc::new);
        let first_sparse = w.config.dense.iter().position(|d| !d).unwrap_or(0);
        let lazy = cache.is_some() && stacks.as_deref().is_some_and(|s| em::has_lazy_stacks(s, first_sparse));
        let source: Option<Arc<dyn ExpertSource>> = lazy.then(|| Arc::new(em::BundleExperts::new(stacks.clone().expect("lazy"))) as Arc<dyn ExpertSource>);
        let mirror = if lazy && std::env::var("SYN_HY4_MIRROR").as_deref() != Ok("0") {
            let s = &**stacks.as_ref().expect("lazy");
            em::host_mirror(&em::mirror_ranges(s, device), "hy4").map(|(g, _)| g)
        } else {
            None
        };
        let t0 = Instant::now();
        let guard = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let model = no_grad(|| Hy4Model::build(&w, device, cache, source, prefill_chunk()))?;
        drop(guard);
        if let Some(m) = &mirror {
            m.resume();
        }
        fit_cache(&model);
        eprintln!(
            "[hy4] модель поднята за {:.1} с; эксперты {}",
            t0.elapsed().as_secs_f64(),
            if lazy { "читаются из бандла (кэш на карте)" } else { "на карте" }
        );
        if let Some(c) = model.expert_cache() {
            eprintln!("[hy4] кэш экспертов на карте: {:.1} ГБ", c.capacity_bytes() as f64 / (1u64 << 30) as f64);
        }
        let max_seq = max_seq.unwrap_or(32768).min(w.config.max_position_embeddings);
        Ok(Self { _mirror: mirror, model, tokenizer, chat_template: w.chat_template.clone(), config: w.config.clone(), path, max_seq })
    }

    /// Готовность зеркала экспертов в RAM: (готово байт, всего), `None` — зеркала нет.
    pub fn mirror_progress(&self) -> Option<(usize, usize)> {
        self._mirror.as_ref().map(|m| m.progress())
    }

    /// Копировщики зеркала закончили (или зеркала нет).
    pub fn mirror_finished(&self) -> bool {
        self._mirror.as_ref().is_none_or(|m| m.finished())
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.config.eos.clone()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, PipelineError> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| PipelineError::Tokenize(e.to_string()))?.ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer.decode(ids, false).map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        self.model.kv_bytes_per_token()
    }

    pub fn new_session(&self, ctx: usize) -> Result<Hy4Session, PipelineError> {
        Ok(Hy4Session { cache: self.model.make_cache(ctx.max(2))?, resume_at: None, parked: false, device: self.model.device })
    }

    /// Дорастить сессию: кэш пересоздаётся, посчитанное пересчитается в следующем ходе.
    pub fn grow_session(&self, s: &mut Hy4Session, ctx: usize) -> Result<(), PipelineError> {
        if ctx > s.cache.max_seq {
            s.cache = self.model.make_cache(ctx)?;
        }
        Ok(())
    }

    pub fn generate_streaming(&self, prompt_ids: &[u32], cfg: &GenerationConfig, sink: &mut dyn StreamSink) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let ctx = cfg.max_seq.unwrap_or(prompt_ids.len() + cfg.max_new_tokens + 1).min(prompt_ids.len() + cfg.max_new_tokens + 1);
        let mut s = self.new_session(ctx)?;
        let (out, st, _) = self.generate_cached_streaming(&mut s, prompt_ids, cfg, sink)?;
        Ok((out, st))
    }

    /// Префилл промпта в сессии (с переиспользованием префикса): логиты
    /// последней позиции `[1, V]` F32 и сколько токенов взято из кэша.
    pub fn prefill(&self, s: &mut Hy4Session, prompt_ids: &[u32], sink: &mut dyn StreamSink) -> Result<(Tensor, usize), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let reuse = s.reusable(prompt_ids);
        s.cache.truncate(reuse);
        let chunk = prefill_chunk();
        let mut hidden = None;
        let mut at = reuse;
        while at < prompt_ids.len() {
            if sink.interrupted() {
                s.cache.truncate(at);
                return Err(PipelineError::Model(ModelError::Forward(INTERRUPTED.into())));
            }
            let end = (at + chunk).min(prompt_ids.len());
            let last = end == prompt_ids.len();
            let h = no_grad(|| self.model.forward(&prompt_ids[at..end], &mut s.cache, Some(1)))?;
            if last {
                hidden = Some(h);
            }
            at = end;
        }
        let logits = no_grad(|| self.model.logits(hidden.as_ref().expect("последний чанк")))?;
        Ok((logits, reuse))
    }

    pub fn generate_cached_streaming(
        &self,
        s: &mut Hy4Session,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        if s.parked {
            s.unpark_to(self.model.device)?;
        }
        let need = prompt_ids.len() + cfg.max_new_tokens + 1;
        if need > s.cache.max_seq {
            self.grow_session(s, need.min(self.max_seq.max(need)))?;
        }
        fit_cache(&self.model);
        let eos = eos_set(cfg);
        let mut sampler = TokenSampler::new(cfg, prompt_ids);
        let t0 = Instant::now();
        let (logits, reuse) = self.prefill(s, prompt_ids, sink)?;
        let prefill_ms = t0.elapsed().as_millis();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        if cfg.max_new_tokens == 0 {
            return Ok((out, GenerationStats { prompt_tokens: prompt_ids.len(), new_tokens: 0, prefill_ms, decode_ms: 0 }, reuse));
        }
        let t1 = Instant::now();
        let mut tok = sampler.sample(&logits)?;
        loop {
            out.push(tok);
            if !sink.on_token(tok) || eos.contains(&tok) || out.len() >= cfg.max_new_tokens || s.cache.seq_len() + 1 >= s.cache.max_seq {
                break;
            }
            let h = no_grad(|| self.model.forward(&[tok], &mut s.cache, Some(1)))?;
            let l = no_grad(|| self.model.logits(&h))?;
            tok = sampler.sample(&l)?;
        }
        let st = GenerationStats { prompt_tokens: prompt_ids.len(), new_tokens: out.len(), prefill_ms, decode_ms: t1.elapsed().as_millis() };
        Ok((out, st, reuse))
    }
}
