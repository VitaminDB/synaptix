use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Qwen3Config;
use crate::loader::Qwen3Weights;
use crate::model::{DecoderModel, ModelError};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

pub struct Qwen3Pipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Qwen3Config,
}

impl Qwen3Pipeline {
    pub fn load(model_dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        Self::load_with_max_seq(model_dir, device, dtype, None)
    }

    /// Как `load`, но RoPE-кэш строится на `max_seq` позиций (`--max-seq` для
    /// long-context). `None` → `config.max_position_embeddings`. KV-кеш BF16.
    pub fn load_with_max_seq(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        Self::load_with_opts(model_dir, device, dtype, max_seq, dtype)
    }

    /// Полный вариант: `kv_dtype` (`MXFP8` block-scale для 256K-контекста в 24GB) отдельно
    /// от compute `dtype`. `kv_dtype == dtype` → обычный BF16 KV-кеш.
    pub fn load_with_opts(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
        kv_dtype: DType,
    ) -> Result<Self, PipelineError> {
        let mut precision = PrecisionConfig::dense(dtype);
        precision.kv = kv_dtype;
        Self::load_with_precision(model_dir, device, precision, max_seq)
    }

    /// Per-component precision (квант весов attn/mlp в NVFP4, compute=F16 и т.д.).
    /// Веса грузятся в `precision.compute`; квант-группы квантуются при загрузке.
    pub fn load_with_precision(
        model_dir: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        precision.validate().map_err(PipelineError::Load)?;
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let weights = Qwen3Weights::load(&dir, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let rope_capacity = max_seq.unwrap_or(config.max_position_embeddings);
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(
            &dcfg, &weights, device, precision.compute, precision.attn_w, precision.mlp_w, precision.lm_head, precision.embed, rope_capacity,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        .with_kv_cache_dtype(precision.kv);
        Ok(Self { model, tokenizer, config })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    fn cfg_with_eos(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() {
            cfg.eos_token_id = self.config.eos_token_id;
        }
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate(&self.model, prompt_ids, &cfg)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        let text = self.decode(&new_ids)?;
        Ok((text, stats))
    }

    /// CUDA-graph decode (P6.3): тело single-token forward'а захватывается в один
    /// `CudaGraph` и реплеится на каждом шаге — устраняет launch-overhead десятков
    /// мелких ядер/токен (главный лимит decode, см. session 12 Phase 2). Prefill —
    /// обычный батч-forward; затем warmup-шаги (праймят pool/кеши), capture одного
    /// `forward_decode_dev`, и replay-loop (обновить device-буферы → launch → dtoh
    /// logits → host-sample). Greedy совпадает с [`Self::generate`] (с точностью до
    /// F16-rope-таблиц). Требует CUDA-устройство и не-FP8 KV.
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let mut noop = |_: u32| true;
        self.generate_with_graph_streaming(prompt_ids, gen_cfg, &mut noop)
    }

    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, prompt_ids, gen_cfg, sink)
    }

    /// Как [`Self::generate_with_graph_streaming`], но prefill стартует с `kv.seq_len`
    /// (prefix-KV-кэш) — `kv` переиспользуется между ходами чата.
    pub fn generate_with_graph_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let gen_cfg = self.cfg_with_eos(gen_cfg);
        let eos = synaptix_llm_common::generate::eos_set(&gen_cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&gen_cfg, prompt_ids);
        let device = self.model.device;
        let ord = match device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("generate_with_graph requires CUDA device".into())),
        };
        let l = prompt_ids.len();
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;

        // Prefill хвоста prompt_ids[prefix..] чанками — ограничивает пик памяти
        // активаций/attn на длинном промпте (KV переносится между чанками в `kv`).
        // Каждый chunk идёт через `model.forward` (FA-prefill: FA-4 на sm_120,
        // Q-тайлы 16-64 ток × тензор-коры). Per-chunk host-loop одноразовый и не
        // оправдывает CUDA-graph capture (device-резидентный replay на chunk=256
        // оказался ~4× медленнее baseline forward + ~400 мс capture-overhead).
        let suffix = &prompt_ids[prefix..];
        let chunk = if gen_cfg.prefill_batch > 0 { gen_cfg.prefill_batch } else { 256 };
        let t0 = std::time::Instant::now();
        let mut logits_opt: Option<Tensor> = None;
        let mut off = 0usize;
        while off < suffix.len() {
            let end = (off + chunk).min(suffix.len());
            let part = Tensor::from_vec(suffix[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = no_grad(|| self.model.forward(&part, kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            logits_opt = Some(lg);
            off = end;
        }
        let logits = logits_opt.ok_or_else(|| PipelineError::Forward("empty prefill suffix".into()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let tok0 = sampler.sample(&logits).map_err(PipelineError::from)?;
        out.push(tok0);
        let mut cancelled = !sink.on_token(tok0);

        // DecodeState: вход = tok0, позиция = L (warmup/capture идемпотентны на slot L).
        let mut state = self
            .model
            .make_decode_state()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        state.update(tok0, l as u32).map_err(|e| PipelineError::Forward(e.to_string()))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Forward(format!("stream: {e}")))?;
        let mut capturer = GraphCapturer::new(3);

        let dec_t0 = std::time::Instant::now();
        let graph = {
            let model = &self.model;
            let state_ref = &mut state;
            let kv_ref = &mut *kv;
            no_grad(|| {
                capturer.capture_with(&stream, |_s| {
                    model
                        .forward_decode_dev(state_ref, kv_ref)
                        .map_err(|e| InferError::Other(e.to_string()))
                })
            })
        }
        .map_err(|e| PipelineError::Forward(format!("graph capture: {e}")))?;
        let _ = graph.upload();

        // Capture-шаг уже посчитал logits для предсказания out[1] (вход tok0 @ pos L).
        if !cancelled && out.len() < gen_cfg.max_new_tokens && !eos.contains(&tok0) {
            let tok1 = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok1);
            cancelled = !sink.on_token(tok1);
        }
        // Replay loop: обрабатываем out[len-1] на позиции L+len-1, получаем следующий.
        while !cancelled && out.len() < gen_cfg.max_new_tokens {
            let last = *out.last().unwrap();
            if eos.contains(&last) {
                break;
            }
            let pos = (l + out.len() - 1) as u32;
            if (pos as usize) >= kv.max_seq {
                break;
            }
            // update (htod) и launch — на одном stream'е (default_stream), порядок
            // гарантирован → pre-launch sync не нужен. post-launch sync обязателен:
            // запись logits графом НЕ event-tracked (capture с выкл. tracking), поэтому
            // host-dtoh в sample обогнал бы граф без явного барьера.
            state.update(last, pos).map_err(|e| PipelineError::Forward(e.to_string()))?;
            graph
                .launch()
                .map_err(|e| PipelineError::Forward(format!("graph launch: {e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| PipelineError::Forward(format!("sync post-launch: {e:?}")))?;
            let tok = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok);
            cancelled = !sink.on_token(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();

        // Host-реконсиляция длины KV: записаны slots [0, L + out.len() - 1)
        // (prompt + все токены кроме последнего, ещё не обработанного).
        kv.seq_len = (l + out.len() - 1).min(kv.max_seq);

        let stats = GenerationStats {
            prompt_tokens: l,
            new_tokens: out.len(),
            prefill_ms,
            decode_ms,
        };
        Ok((out, stats))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("forward: {0}")]
    Forward(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self { Self::Model(e.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn qwen3_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/Qwen/Qwen3-1.7B");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    #[test]
    fn pipeline_loads_and_encodes() {
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load pipeline");
        let ids = p.encode("Hello").unwrap();
        assert!(!ids.is_empty());
        let back = p.decode(&ids).unwrap();
        assert!(back.contains("Hello"), "round-trip lost text: '{back}'");
    }

    #[test]
    fn pipeline_generates_one_token_greedy() {
        // Долго (~40s на prefill), пропускается если SYN_QWEN3_GENERATE не set.
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("The capital of France is").unwrap();
        let (new_ids, stats) = p.generate(
            &ids,
            GenerationConfig { max_new_tokens: 1, temperature: 0.0, ..Default::default() },
        ).unwrap();
        assert_eq!(new_ids.len(), 1);
        let txt = p.decode(&new_ids).unwrap();
        eprintln!("[qwen3 gen] new='{txt}' prefill_ms={} decode_ms={}", stats.prefill_ms, stats.decode_ms);
    }

    #[test]
    fn chunked_prefill_matches_single_shot() {
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("The capital of France is the city of").unwrap();
        let base = GenerationConfig { max_new_tokens: 8, temperature: 0.0, ..Default::default() };

        let (single, _) = p
            .generate(&ids, GenerationConfig { prefill_batch: 0, ..base.clone() })
            .unwrap();
        let (chunked, _) = p
            .generate(&ids, GenerationConfig { prefill_batch: 4, ..base })
            .unwrap();
        assert_eq!(single, chunked, "chunked prefill (batch=4) разошёлся с single-shot greedy");
    }

    #[test]
    fn prefix_cache_resume_matches_fresh() {
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let full = p.encode("The capital of France is the city of").unwrap();
        let cap = full.len() + 16;
        let cfg = GenerationConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            max_seq: Some(cap),
            ..Default::default()
        };

        let (fresh, _) = p.generate(&full, cfg.clone()).unwrap();

        let half = full.len() / 2;
        let mut kv = p.model.make_kv_cache(1, cap).unwrap();
        let t = synaptix_core::tensor::Tensor::from_vec(full[..half].to_vec(), vec![1, half], p.model.device)
            .unwrap();
        synaptix_core::grad::no_grad(|| p.model.forward(&t, &mut kv)).unwrap();
        assert_eq!(kv.seq_len, half);
        let mut noop = |_: u32| true;
        let (cached, _) = p.generate_streaming_resume(&mut kv, &full, cfg, &mut noop).unwrap();

        assert_eq!(fresh, cached, "prefix-cache resume (prefix=half) разошёлся с fresh greedy");
    }
}
