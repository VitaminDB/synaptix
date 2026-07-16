use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
#[cfg(feature = "cuda")]
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::HybridConfig;
use crate::loader::HybridWeights;
use crate::model::{DecoderModel, ModelError};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

pub struct HybridPipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: HybridConfig,
    pub add_bos: bool,
}

impl HybridPipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        let weights =
            HybridWeights::load(path, device, dtype).map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_bytes(&weights.tokenizer_json)
            .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?;
        let cap = config.max_position_embeddings.min(4096);
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(&dcfg, &weights, device, dtype, dtype, dtype, dtype, cap)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        Ok(Self { model, tokenizer, config, add_bos: false })
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        let weights = HybridWeights::load(path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_bytes(&weights.tokenizer_json)
            .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?;
        let cap = max_seq.unwrap_or_else(|| config.max_position_embeddings.min(4096));
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(
            &dcfg, &weights, device, precision.compute, precision.attn_w, precision.mlp_w, precision.lm_head, cap,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        // БАГ-ФИКС: пробросить kv-dtype (--kv-dtype mxfp8) в модель. Без этого
        // model.kv_dtype оставался compute(F16) → MXFP8-KV игнорировался, KV
        // аллоцировался F16 и decode шёл по f16-flash (qwen3-pipeline это делал).
        .with_kv_cache_dtype(precision.kv);
        Ok(Self { model, tokenizer, config, add_bos: false })
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

    fn maybe_prepend_bos(&self, prompt_ids: &[u32]) -> Vec<u32> {
        match self.config.bos_token_id {
            Some(bos) if self.add_bos && prompt_ids.first() != Some(&bos) => {
                let mut v = Vec::with_capacity(prompt_ids.len() + 1);
                v.push(bos);
                v.extend_from_slice(prompt_ids);
                v
            }
            _ => prompt_ids.to_vec(),
        }
    }

    fn prepare_cfg(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_ids();
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
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate(&self.model, &prompt, &cfg)
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
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, &prompt, &cfg, sink)
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
        let cfg = self.prepare_cfg(gen_cfg);
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

    /// CUDA-graph decode для гибрида (linear + full слои). Prefill — обычный
    /// батч-forward (host-scan для linear строит host-состояние + device KV).
    /// Затем device-зеркала linear-state сеются из host (S0); warmup+capture
    /// одного `forward_decode_dev` (продвигает dev-state, т.к. рекуррентность НЕ
    /// идемпотентна — но host-векторы не тронуты); dev-state восстанавливается в
    /// S0 пере-сеянием; replay-loop обрабатывает токены начиная с tok0@L (по
    /// одному advance state за launch). Greedy совпадает с [`Self::generate`] с
    /// точностью до F16-compute. Требует CUDA-устройство, **compute=F16** (ядра
    /// linear-decode F16-нативные) и не-FP8 KV.
    #[cfg(feature = "cuda")]
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let mut noop = |_: u32| true;
        self.generate_with_graph_streaming(prompt_ids, gen_cfg, &mut noop)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, &prompt, gen_cfg, sink)
    }

    /// Как [`Self::generate_with_graph_streaming`], но prefill стартует с `kv.seq_len`
    /// (prefix-KV-кэш). После decode синкает device→host linear-состояние, чтобы
    /// следующий ход продолжил host-scan корректно. `prompt_ids` — уже с BOS (если
    /// нужен): caller отвечает за совпадение с кэшированным префиксом.
    #[cfg(feature = "cuda")]
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
        if self.model.dtype != DType::F16 {
            return Err(PipelineError::Forward(format!(
                "generate_with_graph: linear-decode ядра требуют compute=F16, получено {:?}",
                self.model.dtype
            )));
        }
        let device = self.model.device;
        let ord = match device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("generate_with_graph requires CUDA device".into())),
        };
        let l = prompt_ids.len();
        let gen_cfg = self.prepare_cfg(gen_cfg);
        let eos = synaptix_llm_common::generate::eos_set(&gen_cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&gen_cfg, prompt_ids);
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;

        // Prefill хвоста prompt_ids[prefix..] чанками: device chunked-scan для linear
        // + device KV для full. Чанкуем, чтобы ограничить пик памяти (буферы scan-
        // оркестратора и активации растут с длиной чанка, а не всего промпта) —
        // иначе длинный первый prefill упирается в VRAM. Стейт linear/conv и KV
        // переносятся между чанками внутри одного `kv`. Берём логиты последнего чанка.
        let suffix = &prompt_ids[prefix..];
        // КОРРЕКТНОСТЬ: prefill_batch=0 → single-shot (весь промпт за один forward),
        // как документировано. Чанкование (>1 чанк) сейчас НЕ bit-exact к single
        // (теряет ранний контекст — known chunked-prefill баг), поэтому дефолт = весь
        // промпт. --prefill-batch (gen_cfg.prefill_batch) — для перф-тестов large-M.
        let chunk = if gen_cfg.prefill_batch > 0 {
            gen_cfg.prefill_batch
        } else {
            suffix.len().max(1)
        };
        let t0 = std::time::Instant::now();
        let mut logits_opt = None;
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

        // Prefill держит linear-state device-резидентным (без per-chunk host
        // round-trip). Обновляем host из dev ОДИН раз — для decode-handoff и
        // continuation. No-op, если prefill не шёл (dev=None) → dev засеется
        // из host ниже.
        self.model
            .sync_decode_host_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        // Засеять device-зеркала linear-state из host (post-prefill S0).
        self.model
            .sync_decode_dev_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        let mut state = self
            .model
            .make_decode_state()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        state.update(tok0, l as u32).map_err(|e| PipelineError::Forward(e.to_string()))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Forward(format!("stream: {e}")))?;

        let mut capturer = GraphCapturer::new(3);

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

        // Восстановить S0: warmup+capture продвинули dev linear-state (не
        // идемпотентно), но host-векторы нетронуты → пере-сеять. KV full-слоёв
        // идемпотентен (slot L переписан tok0), восстановления не требует.
        self.model
            .sync_decode_dev_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        // decode_ms измеряет ЧИСТЫЙ replay (steady-state throughput); capture —
        // одноразовая стоимость (warmup×3 + capture + instantiate), не в метрике.
        let dec_t0 = std::time::Instant::now();
        // Replay-loop: обрабатываем out[len-1] на позиции L+len-1 (старт tok0@L),
        // каждый launch продвигает linear-state на один шаг.
        while !cancelled && out.len() < gen_cfg.max_new_tokens {
            let last = *out.last().unwrap();
            if eos.contains(&last) {
                break;
            }
            let pos = (l + out.len() - 1) as u32;
            if (pos as usize) >= kv.max_seq {
                break;
            }
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
        kv.seq_len = (l + out.len() - 1).min(kv.max_seq);
        // graph-decode продвинул только device linear-state → вернуть в host, чтобы
        // следующий ход (prefix-KV-кэш) продолжил host-scan с верного состояния.
        self.model
            .sync_decode_host_state(kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

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
    fn from(e: ModelError) -> Self {
        Self::Model(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn bundle_path() -> Option<PathBuf> {
        let p = PathBuf::from("models/qwen3.6 27B.syn");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn nvfp4_precision() -> PrecisionConfig {
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        }
    }

    #[test]
    fn cuda_nvfp4_generates() {
        if std::env::var("SYN_QWEN_NEXT_CUDA").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let p = HybridPipeline::load_with_precision(&path, Device::Cuda(0), nvfp4_precision(), Some(2048))
            .expect("load nvfp4 cuda");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig {
                    max_new_tokens: 32,
                    temperature: 0.0,
                    max_seq: Some(2048),
                    ..Default::default()
                },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!(
            "[qwen3-next cuda nvfp4] ids={new_ids:?}\n  '{txt}'\n  prefill_ms={} decode_ms={}",
            stats.prefill_ms, stats.decode_ms
        );
        assert!(!new_ids.is_empty());
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_f16_precision() -> PrecisionConfig {
        PrecisionConfig {
            compute: DType::F16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::F16,
            embed: DType::F16,
            kv: DType::F16,
        }
    }

    /// Сверка CUDA-graph decode против host-reference (та же модель, greedy):
    /// токены должны совпасть (с точностью до F16-compute). Также печатает
    /// tok/s обоих путей. Gated на `SYN_QWEN_NEXT_GRAPH` + наличие бандла.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_graph_matches_host() {
        if std::env::var("SYN_QWEN_NEXT_GRAPH").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let p = HybridPipeline::load_with_precision(&path, Device::Cuda(0), nvfp4_f16_precision(), Some(2048))
            .expect("load nvfp4 f16 cuda");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let cfg = GenerationConfig {
            max_new_tokens: 96,
            temperature: 0.0,
            max_seq: Some(2048),
            ..Default::default()
        };

        let (host_ids, hstats) = p.generate(&ids, cfg.clone()).expect("host generate");
        let host_tps = host_ids.len() as f64 / (hstats.decode_ms.max(1) as f64 / 1000.0);
        let (graph_ids, gstats) = p.generate_with_graph(&ids, cfg).expect("graph generate");
        // decode_ms графа — чистый replay (capture не входит).
        let graph_tps = graph_ids.len() as f64 / (gstats.decode_ms.max(1) as f64 / 1000.0);

        eprintln!("[host ] decode_ms={} ({host_tps:.1} tok/s)\n  '{}'",
            hstats.decode_ms, p.decode(&host_ids).unwrap());
        eprintln!("[graph] decode_ms={} ({graph_tps:.1} tok/s replay)\n  '{}'",
            gstats.decode_ms, p.decode(&graph_ids).unwrap());

        // F16-graph vs F32-host greedy: префикс совпадает, дальше дрейфует
        // (как в qwen3-графе). Корректность = разумный префикс + связный вывод.
        let common = host_ids.iter().zip(&graph_ids).take_while(|(a, b)| a == b).count();
        eprintln!("[match] {common}/{} токенов префикса совпали; graph speedup ×{:.2}",
            host_ids.len(), graph_tps / host_tps.max(1e-6));
        assert!(!graph_ids.is_empty());
        assert!(common >= 8, "graph разошёлся слишком рано (токен {common}) — вероятен баг, не дрейф");
    }

    #[test]
    fn pipeline_generates_greedy() {
        if std::env::var("SYN_QWEN_NEXT_GENERATE").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = HybridPipeline::load(&path, Device::Cpu, DType::BF16).expect("load");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig {
                    max_new_tokens: 16,
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!(
            "[qwen3-next cpu bf16] ids={new_ids:?}\n  '{txt}'\n  prefill_ms={} decode_ms={}",
            stats.prefill_ms, stats.decode_ms
        );
        assert!(!new_ids.is_empty());
    }
}
