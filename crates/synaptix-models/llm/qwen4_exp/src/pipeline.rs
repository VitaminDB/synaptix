use std::path::Path;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use std::sync::Arc;

use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink, TokenSampler};
use synaptix_llm_common::moe::{ExpertCache, ExpertCacheStats, ExpertSource};
use synaptix_llm_common::ModelError;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::config::Qwen4ExpConfig;
use crate::loader::{BundleExperts, Qwen4ExpWeights};
use crate::model::{ModelCache, Qwen4ExpModel};

/// Чанк префилла. Чем он крупнее, тем меньше проходов по всей стопке
/// экспертов: любой чанк длиннее сотни токенов задевает почти все 512
/// экспертов слоя, так что стоимость префилла — это число чанков, умноженное
/// на объём экспертов. Меняется `SYN_QWEN4EXP_PREFILL_CHUNK`.
pub const DEFAULT_PREFILL_CHUNK: usize = 4096;

/// Ёмкость кэша экспертов на время префилла. Чанк задевает почти всех
/// экспертов слоя — кэш всё равно вытеснится целиком, а память нужна
/// активациям; на декоде ёмкость возвращается.
const PREFILL_CACHE_BYTES: usize = 3 << 30;

pub struct Qwen4ExpPipeline {
    pub model: Qwen4ExpModel,
    pub config: Qwen4ExpConfig,
    pub chat_template: Option<String>,
    tokenizer: Option<HfTokenizer>,
    max_seq: usize,
}

impl Qwen4ExpPipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        Self::load_with_precision(path, device, PrecisionConfig::dense(dtype), None)
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let weights = Arc::new(
            Qwen4ExpWeights::open(path, device, precision.compute)
                .map_err(|e| PipelineError::Load(e.to_string()))?,
        );
        let mut config = weights.config.clone();
        let tokenizer = if weights.tokenizer_json.is_empty() {
            None
        } else {
            Some(
                HfTokenizer::from_bytes(&weights.tokenizer_json)
                    .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?,
            )
        };
        let cap = max_seq.unwrap_or_else(|| config.max_position_embeddings.min(4096));
        // MoE считает вход своими под-чанками; если они мельче чанка префилла,
        // каждый под-чанк заново поднимает почти всех экспертов слоя — на
        // длинном промпте это кратный перечит всей стопки.
        config.moe.chunk = config.moe.chunk.max(prefill_chunk());
        let expert_cache = expert_cache_for(&config, device);
        let lazy = expert_cache.is_some() && weights.has_lazy_experts(0);
        let expert_source: Option<Arc<dyn ExpertSource>> = lazy
            .then(|| Arc::new(BundleExperts::new(weights.clone())) as Arc<dyn ExpertSource>);
        if let Some(cache) = &expert_cache {
            eprintln!(
                "[qwen4_exp] эксперты {}, на карте кэш {:.1} ГБ",
                if lazy { "читаются из бандла по одному" } else { "в системной памяти" },
                cache.capacity_bytes() as f64 / (1 << 30) as f64
            );
        }
        let kv_reserve = model_kv_reserve(&config, cap);
        let model = Qwen4ExpModel::build_with_cache(
            &config,
            &*weights,
            device,
            precision.compute,
            precision.mlp_w,
            cap,
            &|layer| weights.ngram_rows(layer),
            expert_cache,
            expert_source,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?;
        if let Some(cache) = model.expert_cache() {
            warn_if_cache_too_big(cache, device, kv_reserve);
        }
        let chat_template = weights.chat_template.clone();
        Ok(Self { model, config, chat_template, tokenizer, max_seq: cap })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PipelineError::Tokenize("бандл без tokenizer.json".into()))?;
        let enc = tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PipelineError::Tokenize("бандл без tokenizer.json".into()))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    /// Отдать память карты, которую держат резидентные эксперты. Нужно при
    /// выгрузке модели: кэш живёт в самой модели, но её `Drop` может
    /// задержаться, пока кто-то держит ссылку, а память освободить надо сразу.
    pub fn release_device_caches(&self) {
        if let Some(cache) = self.model.expert_cache() {
            cache.clear();
        }
    }

    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.model.expert_cache_stats()
    }

    pub fn rope_capacity(&self) -> usize {
        self.max_seq
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        let cfg = &self.config;
        let elem = (self.model.compute.size_in_bits() / 8).max(1);
        let qsa_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, crate::config::LayerType::Qsa))
            .count();
        let kv = 2 * cfg.num_key_value_heads * cfg.head_dim * elem;
        let indexer = if cfg.indexer.compress_ratio > 0 {
            cfg.indexer.head_dim * elem / cfg.indexer.compress_ratio
        } else {
            0
        };
        qsa_layers * (kv + indexer)
    }

    pub fn kv_fixed_bytes(&self, batch: usize, max_seq: usize) -> usize {
        let cfg = &self.config;
        let elem = 4;
        let linear_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, crate::config::LayerType::Linear))
            .count();
        let state = cfg.linear.num_value_heads * cfg.linear.key_head_dim * cfg.linear.value_head_dim * elem;
        let conv = (cfg.linear.conv_kernel - 1) * cfg.linear.conv_dim() * elem;
        batch * (linear_layers * (state + conv) + max_seq * self.kv_bytes_per_token())
    }

    pub fn make_cache(&self, tokens: usize) -> Result<ModelCache, PipelineError> {
        self.model
            .make_cache(tokens.max(1).min(self.max_seq))
            .map_err(PipelineError::from)
    }

    fn prepare(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_token_ids.clone();
        }
        // Цена префилла здесь — число чанков, умноженное на объём экспертов:
        // любой чанк длиннее сотни токенов задевает почти все 512 экспертов
        // слоя, поэтому мелкий чанк означает лишний полный прогон весов через
        // шину. Внешнюю настройку (её ставят ради пика VRAM у плотных
        // моделей) поднимаем до своего минимума, но не опускаем ниже неё.
        let want = prefill_chunk();
        if cfg.prefill_batch < want {
            if cfg.prefill_batch > 0 {
                eprintln!(
                    "[qwen4_exp] чанк префилла поднят с {} до {want}: на мелких чанках \
                     эксперты перечитываются целиком на каждый",
                    cfg.prefill_batch
                );
            }
            cfg.prefill_batch = want;
        }
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        self.generate_streaming(prompt_ids, cfg, &mut |_: u32| true)
    }

    pub fn generate_text(&self, prompt: &str, cfg: GenerationConfig) -> Result<String, PipelineError> {
        let ids = self.encode(prompt)?;
        let (out, _) = self.generate(&ids, cfg)?;
        self.decode(&out)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.prepare(cfg);
        let budget = prompt_ids.len() + cfg.max_new_tokens;
        if budget > self.max_seq {
            return Err(PipelineError::Model(format!(
                "промпт {} + {} новых токенов больше окна {}",
                prompt_ids.len(),
                cfg.max_new_tokens,
                self.max_seq
            )));
        }
        let mut cache = self.make_cache(budget)?;
        let mut sampler = TokenSampler::new(&cfg, prompt_ids);
        let eos: Vec<u32> = if cfg.eos_token_ids.is_empty() {
            cfg.eos_token_id.into_iter().collect()
        } else {
            cfg.eos_token_ids.clone()
        };

        let cache_capacity = self.model.expert_cache().map(|c| c.capacity_bytes());
        if let (Some(cache), Some(full)) = (self.model.expert_cache(), cache_capacity) {
            cache.set_capacity(PREFILL_CACHE_BYTES.min(full));
        }
        let prefill_start = Instant::now();
        let mut logits = no_grad(|| -> Result<_, ModelError> {
            let mut last = None;
            let mut offset = 0usize;
            while offset < prompt_ids.len() {
                let take = cfg.prefill_batch.min(prompt_ids.len() - offset);
                let chunk = &prompt_ids[offset..offset + take];
                last = Some(self.model.forward_last(chunk, &mut cache)?);
                offset += take;
            }
            last.ok_or_else(|| ModelError::Forward("пустой префилл".into()))
        })?;
        let prefill_ms = prefill_start.elapsed().as_millis();
        if let (Some(cache), Some(full)) = (self.model.expert_cache(), cache_capacity) {
            cache.set_capacity(full);
        }

        let decode_start = Instant::now();
        let mut out = Vec::with_capacity(cfg.max_new_tokens);
        for _ in 0..cfg.max_new_tokens {
            let token = sampler.sample(&logits)?;
            out.push(token);
            if !sink.on_token(token) {
                break;
            }
            if eos.contains(&token) {
                break;
            }
            if cache.seq_len >= self.max_seq {
                break;
            }
            logits = no_grad(|| self.model.forward_last(&[token], &mut cache))?;
        }
        let decode_ms = decode_start.elapsed().as_millis();

        Ok((
            out.clone(),
            GenerationStats {
                prompt_tokens: prompt_ids.len(),
                new_tokens: out.len(),
                prefill_ms,
                decode_ms,
            },
        ))
    }
}

fn prefill_chunk() -> usize {
    std::env::var("SYN_QWEN4EXP_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_PREFILL_CHUNK)
}

/// Сколько VRAM уйдёт под KV и ключи индексатора при полном окне.
fn model_kv_reserve(cfg: &Qwen4ExpConfig, max_seq: usize) -> usize {
    let qsa = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, crate::config::LayerType::Qsa))
        .count();
    let per_token = 2 * cfg.num_key_value_heads * cfg.head_dim * 2
        + if cfg.indexer.compress_ratio > 0 {
            cfg.indexer.head_dim * 2 / cfg.indexer.compress_ratio
        } else {
            0
        };
    qsa * max_seq * per_token
}

/// Предупредить, если под кэш просят больше, чем видно свободного. Сам размер
/// не трогаем: `cuMemGetInfo` не знает про уже зарезервированное пулами, и
/// автоподгонка по нему обнуляла кэш там, где он прекрасно помещался.
fn warn_if_cache_too_big(cache: &Arc<ExpertCache>, device: Device, kv_reserve: usize) {
    let Device::Cuda(ordinal) = device else { return };
    let Ok((free, _total)) = synaptix_core::device::cuda::mem_info(ordinal) else {
        return;
    };
    let need = cache.capacity_bytes() + kv_reserve;
    if need <= free {
        return;
    }
    // Свободного меньше, чем просят: карту делят с другими моделями. Ужимаем,
    // но не в ноль — без кэша каждый токен перечитывает всех своих экспертов.
    // `cuMemGetInfo` не знает про уже зарезервированное пулами, поэтому
    // нижнюю границу держим щедрой.
    let room = free.saturating_sub(kv_reserve + (1 << 30));
    let capacity = cache.capacity_bytes().min(room).max(2 << 30);
    eprintln!(
        "[qwen4_exp] кэш экспертов ужат до {:.1} ГБ: свободно {:.1} ГБ, под KV нужно {:.1} ГБ",
        capacity as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        kv_reserve as f64 / (1u64 << 30) as f64,
    );
    cache.set_capacity(capacity);
}

/// Кэш резидентных экспертов: на CUDA держим часть экспертов на карте, всё
/// остальное — в системной памяти. Размер задаётся
/// `SYN_QWEN4EXP_EXPERT_CACHE_GB` (0 — грузить эксперты целиком на
/// устройство, как раньше); по умолчанию 12 ГБ, и только для моделей, чьи
/// эксперты заведомо не влезают в память карты.
fn expert_cache_for(cfg: &Qwen4ExpConfig, device: Device) -> Option<Arc<ExpertCache>> {
    if !matches!(device, Device::Cuda(_)) {
        return None;
    }
    let requested = std::env::var("SYN_QWEN4EXP_EXPERT_CACHE_GB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok());
    let big = cfg.moe.num_experts * cfg.num_hidden_layers >= 1024;
    let gb = match requested {
        Some(v) if v <= 0.0 => return None,
        Some(v) => v,
        None if big => 12.0,
        None => return None,
    };
    Some(ExpertCache::new(device, (gb * (1u64 << 30) as f64) as usize))
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        PipelineError::Model(e.to_string())
    }
}
