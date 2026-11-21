//! Генерация Inkling-2: токенизатор, шаблон чата, сессия с префиксным кэшем.
//!
//! Состояние коротких свёрток назад не откатывается, поэтому кэш сессии
//! годен, только если новый промпт целиком продолжает уже посчитанные токены
//! (как у LFM3); иначе ход считается с нуля.

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_llm_common::generate::{eos_set, GenerationConfig, GenerationStats, StreamSink, TokenSampler, INTERRUPTED};
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Inkling2Config;
use crate::loader::Inkling2Weights;
use crate::model::{Inkling2Model, ModelCache};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("model: {0}")]
    Model(#[from] ModelError),
}

pub struct Inkling2Pipeline {
    pub model: Inkling2Model,
    pub tokenizer: HfTokenizer,
    pub chat_template: Option<String>,
    pub config: Inkling2Config,
    pub path: PathBuf,
}

/// Кэш хода между репликами чата.
pub struct Inkling2Session {
    pub cache: ModelCache,
    /// Токены, чьё состояние лежит в `cache` (промпт и ответ прошлого хода).
    pub ids: Vec<u32>,
    /// Байт KV на позицию контекста (для учёта VRAM фасадом).
    kv_bytes_per_token: usize,
}

impl Inkling2Session {
    pub fn ctx_tokens(&self) -> usize {
        self.cache.capacity
    }

    pub fn cached_tokens(&self) -> usize {
        self.ids.len()
    }

    /// Сколько токенов `prompt_ids` возьмётся из кэша: всё посчитанное, если
    /// промпт его целиком продолжает (и хоть один токен остаётся на
    /// префилл), иначе 0 — состояние свёрток назад не откатить.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        let n = self.ids.len();
        if n > 0 && n < prompt_ids.len() && prompt_ids[..n] == self.ids[..] && self.cache.seq_len == n {
            n
        } else {
            0
        }
    }

    /// Забыть посчитанное: следующий ход считается с нуля (обнуление
    /// состояний — в начале хода).
    pub fn invalidate(&mut self) {
        self.ids.clear();
        self.cache.seq_len = 0;
    }

    /// VRAM под KV сессии (состояния свёрток — копейки, не в счёт).
    pub fn device_bytes(&self) -> usize {
        self.cache.capacity * self.kv_bytes_per_token
    }
}

/// Длина чанка префилла: внимание считается по всему контексту чанка.
fn prefill_chunk() -> usize {
    std::env::var("SYN_INKLING2_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512)
}

impl Inkling2Pipeline {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self, PipelineError> {
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let path = path.as_ref().to_path_buf();
        let w = Inkling2Weights::open(&path).map_err(|e| PipelineError::Load(e.to_string()))?;
        let model = Inkling2Model::load(&w, device).map_err(|e| PipelineError::Load(e.to_string()))?;
        let tokenizer = HfTokenizer::from_bytes(&w.tokenizer_json).map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        Ok(Self { model, tokenizer, chat_template: w.chat_template.clone(), config: w.config.clone(), path })
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.config.eos_ids.clone()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, PipelineError> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| PipelineError::Tokenize(e.to_string()))?.ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer.decode(ids, false).map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    /// Байт KV и состояний на токен контекста (все слоты цикла).
    pub fn kv_bytes_per_token(&self) -> usize {
        let t = &self.config.text;
        self.config.num_slots() * 2 * t.num_key_value_heads * t.head_dim * 2
    }

    pub fn new_session(&self, ctx: usize) -> Result<Inkling2Session, PipelineError> {
        Ok(Inkling2Session {
            cache: self.model.make_cache(ctx.max(2))?,
            ids: Vec::new(),
            kv_bytes_per_token: self.kv_bytes_per_token(),
        })
    }

    /// Дорастить сессию: KV копируется в больший кэш, позиции сохраняются.
    pub fn grow_session(&self, s: &mut Inkling2Session, ctx: usize) -> Result<(), PipelineError> {
        if ctx <= s.cache.capacity {
            return Ok(());
        }
        self.model.grow_cache(&mut s.cache, ctx)?;
        Ok(())
    }

    /// Разовая генерация с нуля.
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let ctx = cfg.max_seq.unwrap_or(prompt_ids.len() + cfg.max_new_tokens + 1);
        let mut s = self.new_session(ctx)?;
        let (out, st, _) = self.generate_cached_streaming(&mut s, prompt_ids, cfg, sink)?;
        Ok((out, st))
    }

    /// Ход в сессии: переиспользует кэш, если промпт продолжает посчитанные
    /// токены. Возвращает (ответ, статистика, сколько токенов взято из кэша).
    pub fn generate_cached_streaming(
        &self,
        s: &mut Inkling2Session,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, usize), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Model(ModelError::Forward("пустой промпт".into())));
        }
        let need = prompt_ids.len() + cfg.max_new_tokens + 1;
        if need > s.cache.capacity {
            self.grow_session(s, need)?;
        }
        // Кэш годен, только если промпт продолжает его целиком и хоть один токен остаётся на префилл.
        let reuse = s.reusable(prompt_ids);
        if reuse == 0 {
            s.cache.reset()?;
            s.ids.clear();
        }
        let eos = eos_set(cfg);
        let mut sampler = TokenSampler::new(cfg, prompt_ids);
        let t0 = Instant::now();
        let chunk = prefill_chunk().max(1);
        let mut off = reuse;
        let mut logits = None;
        s.ids.truncate(reuse);
        while off < prompt_ids.len() {
            if sink.interrupted() {
                s.ids.clear();
                s.cache.seq_len = 0;
                return Err(PipelineError::Model(ModelError::Forward(INTERRUPTED.into())));
            }
            let end = (off + chunk).min(prompt_ids.len());
            let res = no_grad(|| self.model.forward(&prompt_ids[off..end], &mut s.cache, false));
            match res {
                Ok(l) => logits = Some(l),
                Err(e) => {
                    s.ids.clear();
                    s.cache.seq_len = 0;
                    return Err(e.into());
                }
            }
            s.ids.extend_from_slice(&prompt_ids[off..end]);
            synaptix_llm_common::prefill_progress::add(end - off);
            off = end;
        }
        let prefill_ms = t0.elapsed().as_millis();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        let stats = |out: &Vec<u32>, decode_ms| GenerationStats { prompt_tokens: prompt_ids.len(), new_tokens: out.len(), prefill_ms, decode_ms };
        if cfg.max_new_tokens == 0 {
            let st = stats(&out, 0);
            return Ok((out, st, reuse));
        }
        let logits = logits.expect("префилл непустой");
        let t1 = Instant::now();
        let mut tok = sampler.sample(&logits)?;
        loop {
            out.push(tok);
            if !sink.on_token(tok) || eos.contains(&tok) || out.len() >= cfg.max_new_tokens {
                break;
            }
            let l = no_grad(|| self.model.forward(&[tok], &mut s.cache, false))?;
            s.ids.push(tok);
            tok = sampler.sample(&l)?;
        }
        Ok((out.clone(), stats(&out, t1.elapsed().as_millis()), reuse))
    }
}
