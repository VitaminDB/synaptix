use std::collections::HashSet;
use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::LlamaConfig;
use crate::loader::LlamaWeights;
use crate::model::{DecoderModel, ModelError};

pub struct LlamaPipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: LlamaConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    /// Override EOS. `None` → берётся `config.eos_ids()` (Llama-3 имеет несколько).
    pub eos_token_id: Option<u32>,
    pub max_seq: Option<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 64, temperature: 0.0, seed: 0, eos_token_id: None, max_seq: None }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_ms: u128,
    pub decode_ms: u128,
}

impl LlamaPipeline {
    pub fn load(model_dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        Self::load_with_max_seq(model_dir, device, dtype, None)
    }

    pub fn load_with_max_seq(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        Self::load_with_opts(model_dir, device, dtype, max_seq, dtype)
    }

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

    /// Per-component precision (NVFP4-квант весов attn/mlp, compute=F16 и т.д.).
    /// MLX int4-веса дектвантятся в `precision.compute`; при NVFP4 затем
    /// реквантуются (см. [`crate::model::QLinear`]).
    pub fn load_with_precision(
        model_dir: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        precision.validate().map_err(PipelineError::Load)?;
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let weights = LlamaWeights::load(&dir, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let rope_capacity = max_seq.unwrap_or(config.max_position_embeddings);
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build(
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

    fn eos_set(&self, override_eos: Option<u32>) -> HashSet<u32> {
        match override_eos {
            Some(e) => HashSet::from([e]),
            None => self.config.eos_ids().into_iter().collect(),
        }
    }

    /// Greedy/temperature generate. Возвращает только новые токены (без prompt).
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }

        let eos = self.eos_set(gen_cfg.eos_token_id);
        let device = self.model.device;
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mut rng_state = gen_cfg.seed;

        let prompt_tensor =
            Tensor::from_vec(prompt_ids.to_vec(), vec![1usize, prompt_ids.len()], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let t0 = std::time::Instant::now();
        let logits = synaptix_core::grad::no_grad(|| self.model.forward(&prompt_tensor, &mut kv))
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut new_tokens = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut tok = sample_logits(&logits, gen_cfg.temperature, &mut rng_state)?;
        new_tokens.push(tok);

        let dec_t0 = std::time::Instant::now();
        for _ in 1..gen_cfg.max_new_tokens {
            if eos.contains(&tok) {
                break;
            }
            let next = Tensor::from_vec(vec![tok], vec![1usize, 1], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let logits = synaptix_core::grad::no_grad(|| self.model.forward(&next, &mut kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            tok = sample_logits(&logits, gen_cfg.temperature, &mut rng_state)?;
            new_tokens.push(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();

        let stats = GenerationStats {
            prompt_tokens: prompt_ids.len(),
            new_tokens: new_tokens.len(),
            prefill_ms,
            decode_ms,
        };
        Ok((new_tokens, stats))
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

    /// CUDA-graph decode (зеркально qwen3): тело single-token forward'а
    /// захватывается в `CudaGraph` и реплеится — устраняет launch-overhead. Prefill —
    /// обычный батч-forward. Требует CUDA-устройство и не-FP8 KV.
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let eos = self.eos_set(gen_cfg.eos_token_id);
        let device = self.model.device;
        let ord = match device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("generate_with_graph requires CUDA device".into())),
        };
        let l = prompt_ids.len();
        let kv_max = gen_cfg.max_seq.unwrap_or(l + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mut rng_state = gen_cfg.seed;

        let prompt_tensor = Tensor::from_vec(prompt_ids.to_vec(), vec![1usize, l], device)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let t0 = std::time::Instant::now();
        let logits = no_grad(|| self.model.forward(&prompt_tensor, &mut kv))
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let tok0 = sample_logits(&logits, gen_cfg.temperature, &mut rng_state)?;
        out.push(tok0);

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
            let kv_ref = &mut kv;
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

        if out.len() < gen_cfg.max_new_tokens && !eos.contains(&tok0) {
            let tok1 = sample_logits(&state.logits, gen_cfg.temperature, &mut rng_state)?;
            out.push(tok1);
        }
        while out.len() < gen_cfg.max_new_tokens {
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
            let tok = sample_logits(&state.logits, gen_cfg.temperature, &mut rng_state)?;
            out.push(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();

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

fn sample_logits(logits: &Tensor, temperature: f32, rng_state: &mut u64) -> Result<u32, PipelineError> {
    let l = logits
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| PipelineError::Forward(e.to_string()))?;
    if temperature <= 0.0 {
        let (am, _) = l
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(ai, av), (i, &v)| if v > av { (i, v) } else { (ai, av) });
        return Ok(am as u32);
    }
    let inv_t = 1.0 / temperature;
    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = l.iter().map(|&v| ((v - max) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum.max(1.0e-30);
    for p in &mut probs {
        *p *= inv_sum;
    }
    let u = next_uniform(rng_state);
    let mut cum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

fn next_uniform(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (*state >> 11) as u64;
    (r as f32) / (1u64 << 53) as f32
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

    fn llama_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/mlx-community/Llama-3.2-1B-Instruct-4bit");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    #[test]
    fn pipeline_loads_and_encodes() {
        let Some(dir) = llama_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = LlamaPipeline::load(&dir, Device::Cpu, DType::F32).expect("load pipeline");
        let ids = p.encode("Hello").unwrap();
        assert!(!ids.is_empty());
        let back = p.decode(&ids).unwrap();
        assert!(back.contains("Hello"), "round-trip lost text: '{back}'");
    }

    #[test]
    fn pipeline_generates_greedy() {
        // Долго на CPU — пропускается без SYN_LLAMA_GENERATE.
        if std::env::var("SYN_LLAMA_GENERATE").is_err() {
            return;
        }
        let Some(dir) = llama_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let dtype = match std::env::var("SYN_LLAMA_DTYPE").ok().as_deref() {
            Some("bf16") => DType::BF16,
            Some("f16") => DType::F16,
            _ => DType::F32,
        };
        let p = LlamaPipeline::load(&dir, Device::Cpu, dtype).expect("load");
        let ids = p.encode("The capital of France is").unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig { max_new_tokens: 8, temperature: 0.0, seed: 0, eos_token_id: None, max_seq: None },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!("[llama gen] new='{txt}' prefill_ms={} decode_ms={}", stats.prefill_ms, stats.decode_ms);
        assert!(!new_ids.is_empty());
    }
}
