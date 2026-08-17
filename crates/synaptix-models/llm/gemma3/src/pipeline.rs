use std::collections::HashSet;
use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Gemma3Config;
use crate::loader::GemmaWeights;
use crate::model::{DecoderModel, ModelError};

pub struct GemmaPipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Gemma3Config,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
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

impl GemmaPipeline {
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
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let weights = GemmaWeights::load(&dir, device, dtype)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let rope_capacity = max_seq.unwrap_or(config.max_position_embeddings);
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build(&dcfg, &weights, device, dtype, dtype, dtype, dtype, dtype, rope_capacity)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        Ok(Self { model, tokenizer, config })
    }

    /// Per-component precision (`--quant nvfp4` → compute=F16, attn/mlp веса NVFP4).
    /// Веса читаются лениво и квантуются на лету сразу на `device` (см.
    /// [`GemmaModel::from_weights_with_precision`]).
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
        // НЕ validate(): Gemma сознательно использует смешанную точность —
        // residual BF16 (massive activations >65504 переполняют F16) + NVFP4-веса,
        // что `validate` (требует compute==F16 при кванте) отверг бы. Каст
        // bf16↔f16 вокруг NVFP4-ядра делает QLinear::Quant.
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        // tokenizer.json (33MB, 262k-вокаб) десериализуется tokenizers-crate'ом
        // ~13s — дольше, чем веса+build (5.8s); ни от чего не зависит → фоновый
        // поток, join после сборки модели.
        let tok_dir = dir.clone();
        let tok_handle = std::thread::spawn(move || {
            HfTokenizer::from_file(tok_dir.join("tokenizer.json"))
        });
        let weights = GemmaWeights::load(&dir, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let rope_capacity = max_seq.unwrap_or(config.max_position_embeddings);
        let dcfg = config.to_decoder_config();
        // Dense-веса (bf16/f16, без кванта), не влезающие в свободную VRAM с
        // запасом → host-stream блоков (CPU-резидент + per-block pinned-H2D в
        // forward): bf16-Gemma 24GB работает text-энкодером на 24GB-карте.
        // Квант-пути (mxfp8/nvfp4 ~6-12GB) остаются резидентными как раньше.
        let block_device = if !precision.attn_w.is_quantized() && device.is_cuda() {
            let h = dcfg.hidden_size;
            let heads_dim = dcfg.num_attention_heads * dcfg.head_dim;
            let kv_dim = dcfg.num_key_value_heads * dcfg.head_dim;
            let per_layer = 2 * h * heads_dim + 2 * h * kv_dim + 3 * h * dcfg.intermediate_size;
            let esz = (precision.compute.size_in_bits() / 8).max(1) as usize;
            let blocks_bytes = dcfg.num_hidden_layers * per_layer * esz;
            let free = synaptix_core::device::cuda::mem_info(device.ordinal())
                .map(|(f, _)| f)
                .unwrap_or(usize::MAX);
            if blocks_bytes + (6usize << 30) > free {
                eprintln!(
                    "  Gemma dense {:.1}GB + запас > свободно {:.1}GB → host-stream блоков",
                    blocks_bytes as f64 / 1e9, free as f64 / 1e9
                );
                Some(Device::Cpu)
            } else {
                None
            }
        } else {
            None
        };
        let model = DecoderModel::build_ext(
            &dcfg, &weights, device, block_device, precision.compute, precision.attn_w, precision.mlp_w, precision.lm_head, precision.embed, rope_capacity,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        // Как в qwen3/llama/hybrid: пробросить --kv-dtype в модель (иначе KV молча
        // F16). NB: gemma3 — sliding-window; mxfp8-KV в sliding-слоях может быть не
        // полностью разведён (отдельный путь), но флаг не должен теряться тихо.
        .with_kv_cache_dtype(precision.kv);
        let tokenizer = tok_handle
            .join()
            .map_err(|_| PipelineError::Load("tokenizer thread panicked".into()))?
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
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

    /// Кодирование промпта в 49 hidden states Gemma для LTX-2.3 text-conditioning.
    /// Токенизация → BOS → LEFT-pad до `max_seq` (конвенция LTX: pad слева,
    /// mask 1=valid/0=pad) → `forward_hidden_states`. Возвращает `(states×49,
    /// mask[max_seq] u32)` для подачи в Video/AudioTextConditioner.
    pub fn encode_for_ltx(
        &self,
        prompt: &str,
        max_seq: usize,
        device: synaptix_core::device::Device,
    ) -> Result<(Vec<synaptix_core::tensor::Tensor>, Vec<u32>), PipelineError> {
        use synaptix_core::tensor::Tensor;
        let mut ids = self.encode(prompt)?;
        if let Some(bos) = self.config.bos_token_id {
            ids.insert(0, bos);
        }
        if ids.len() > max_seq {
            ids.truncate(max_seq); // оставляем BOS + начало промпта
        }
        let real = ids.len();
        let pad = max_seq - real;
        // LEFT-pad: [pad×pad_id, ids...]; mask: [pad×0, real×1]
        let mut padded = vec![0u32; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0u32; pad];
        mask.extend(std::iter::repeat(1u32).take(real));
        let ids_t = Tensor::from_vec(padded, vec![1, max_seq], device)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let mask_f32: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
        let mask_t = Tensor::from_vec(mask_f32, vec![1, max_seq], device)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        let states = synaptix_core::grad::no_grad(|| {
            self.model.forward_hidden_states(&ids_t, Some(&mask_t))
        })
        .map_err(|e| PipelineError::Model(e.to_string()))?;
        Ok((states, mask))
    }

    fn eos_set(&self, override_eos: Option<u32>) -> HashSet<u32> {
        match override_eos {
            Some(e) => HashSet::from([e]),
            None => self.config.eos_ids().into_iter().collect(),
        }
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        // Gemma требует BOS в начале — без него выход вырождается. Добавляем, если
        // его ещё нет (caller мог подать токены с уже добавленным спец-токеном).
        let prompt: Vec<u32> = match self.config.bos_token_id {
            Some(bos) if prompt_ids.first() != Some(&bos) => {
                let mut v = Vec::with_capacity(prompt_ids.len() + 1);
                v.push(bos);
                v.extend_from_slice(prompt_ids);
                v
            }
            _ => prompt_ids.to_vec(),
        };
        let prompt_ids = &prompt[..];
        let eos = self.eos_set(gen_cfg.eos_token_id);
        let device = self.model.device;
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mut rng_state = gen_cfg.seed;

        let prompt_tensor = Tensor::from_vec(prompt_ids.to_vec(), vec![1usize, prompt_ids.len()], device)
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
    fn from(e: ModelError) -> Self {
        Self::Model(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/gemma-3-12b-qat");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    /// LTX-2.3 Фаза 2: `DecoderModel::forward_hidden_states` (все 49 hidden states
    /// Gemma) против эталона из официального LTX-кода. Веса MXFP8 (12B→~12GB,
    /// влезает в 24GB GPU; cos≈1.0 к bf16), bf16-compute. Сравнение per-row cos +
    /// rel-L2 на ВАЛИДНЫХ (не-pad) позициях. Гейт: SYN_LTX_GEMMA.
    #[test]
    fn ltx_gemma_hidden_states_match() {
        use synaptix_io::weights::safetensors::SafetensorsLoader;
        use synaptix_io::weights::WeightLoader;
        if std::env::var("SYN_LTX_GEMMA").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        let ref_path = PathBuf::from(
            "tests/reference_data/ltx_gemma/gemma_ref_s128_bf16.safetensors",
        );
        if !ref_path.exists() {
            eprintln!("skip ltx_gemma_hidden_states_match: ref absent at {}", ref_path.display());
            return;
        }
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let dev = Device::Cuda(0);

        let rl = SafetensorsLoader::open(&ref_path).expect("open ref");
        let ids_i64: Vec<i64> = rl.load("input_ids").unwrap().to_vec1::<i64>().unwrap();
        let mask_i64: Vec<i64> = rl.load("attention_mask").unwrap().to_vec1::<i64>().unwrap();
        let hs_ref = rl.load("hidden_states").unwrap(); // [49, S, H] f32 (cpu)
        let (n_states, s, h) = (hs_ref.dims()[0], hs_ref.dims()[1], hs_ref.dims()[2]);
        let seq = ids_i64.len();
        assert_eq!(s, seq, "ref seq mismatch");
        let hs_ref_v: Vec<f32> =
            hs_ref.reshape(vec![n_states * s * h]).unwrap().to_vec1::<f32>().unwrap();

        let ids_u32: Vec<u32> = ids_i64.iter().map(|&x| x as u32).collect();
        let mask_f32: Vec<f32> = mask_i64.iter().map(|&x| x as f32).collect();
        let ids_t = Tensor::from_vec(ids_u32, vec![1, seq], dev).unwrap();
        let mask_t = Tensor::from_vec(mask_f32.clone(), vec![1, seq], dev).unwrap();

        let prec = PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::MXFP8,
            mlp_w: DType::MXFP8,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        };
        let p = GemmaPipeline::load_with_precision(&dir, dev, prec, Some(seq)).expect("load mxfp8");
        let states = synaptix_core::grad::no_grad(|| {
            p.model.forward_hidden_states(&ids_t, Some(&mask_t))
        })
        .expect("forward_hidden_states");
        assert_eq!(states.len(), n_states, "state count (ожидалось {n_states})");

        let valid: Vec<usize> = (0..seq).filter(|&t| mask_f32[t] > 0.5).collect();
        assert!(!valid.is_empty(), "no valid tokens");
        let mut global_min_cos = f32::INFINITY;
        let mut worst = (0usize, 0usize, 1.0f32);
        for (i, st) in states.iter().enumerate() {
            let ours = st
                .reshape(vec![s * h]).unwrap()
                .to_dtype(DType::F32).unwrap()
                .to_vec1::<f32>().unwrap();
            let mut min_cos = f32::INFINITY;
            let mut max_rel = 0.0f32;
            for &t in &valid {
                let off_r = i * s * h + t * h;
                let off_o = t * h;
                let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for k in 0..h {
                    let r = hs_ref_v[off_r + k] as f64;
                    let o = ours[off_o + k] as f64;
                    dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
                }
                let cos = (dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32;
                let rel = (dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32;
                if cos < min_cos { min_cos = cos; }
                if rel > max_rel { max_rel = rel; }
                if cos < worst.2 { worst = (i, t, cos); }
            }
            if min_cos < global_min_cos { global_min_cos = min_cos; }
            eprintln!("state[{i:2}] valid-row min_cos={min_cos:.5} max_rel={max_rel:.4}");
        }
        eprintln!(
            "GLOBAL min_cos={global_min_cos:.5} (worst state {} tok {} cos {:.5})",
            worst.0, worst.1, worst.2
        );
        // структурная проверка раскладки 49 состояний.
        assert_eq!(n_states, 49);
        // Гейт логики: per-row cos на валидных позициях (MXFP8-веса → допуск).
        assert!(global_min_cos > 0.99, "per-row cos слишком низкий: {global_min_cos}");
    }

    #[test]
    fn pipeline_loads_and_encodes() {
        // Загрузка 24GB bf16 — только при SYN_GEMMA_LOAD.
        if std::env::var("SYN_GEMMA_LOAD").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = GemmaPipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("Hello").unwrap();
        assert!(!ids.is_empty());
    }

    fn vram_used_mib() -> i64 {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .expect("nvidia-smi");
        String::from_utf8_lossy(&out.stdout).lines().next().unwrap().trim().parse().unwrap()
    }

    #[test]
    fn cuda_trim_demo() {
        if std::env::var("SYN_GEMMA_TRIMDEMO").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let base = vram_used_mib();
        eprintln!("[trim demo] VRAM до загрузки: {base} MiB");
        let prec = PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        };
        let _p = GemmaPipeline::load_with_precision(&dir, Device::Cuda(0), prec, Some(4096))
            .expect("load nvfp4 cuda");
        let after_load = vram_used_mib();
        eprintln!("[trim demo] VRAM после загрузки (до trim): {after_load} MiB (модель {} MiB)", after_load - base);
        synaptix_core::memory::cuda_pool::trim_cuda_mempool();
        let after_trim = vram_used_mib();
        eprintln!(
            "[trim demo] VRAM после trim: {after_trim} MiB | освобождено trim'ом: {} MiB",
            after_load - after_trim
        );
    }

    #[test]
    fn cuda_nvfp4_generates() {
        if std::env::var("SYN_GEMMA_CUDA").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let dev = Device::Cuda(0);
        // Gemma смешанная точность: residual/нормы/attention BF16 (massive activations),
        // attn/mlp веса NVFP4 (каст bf16↔f16 вокруг ядра внутри QLinear::Quant).
        let prec = PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        };
        let p = GemmaPipeline::load_with_precision(&dir, dev, prec, Some(4096)).expect("load nvfp4 cuda");
        let ids = p.encode("The capital of France is").unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig { max_new_tokens: 16, temperature: 0.0, seed: 0, eos_token_id: None, max_seq: Some(4096) },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!("[gemma cuda nvfp4] ids={new_ids:?} '{txt}' prefill_ms={} decode_ms={}", stats.prefill_ms, stats.decode_ms);
        assert!(!new_ids.is_empty());
    }

    #[test]
    fn pipeline_generates_greedy() {
        if std::env::var("SYN_GEMMA_GENERATE").is_err() {
            return;
        }
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = GemmaPipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("The capital of France is").unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig { max_new_tokens: 8, temperature: 0.0, seed: 0, eos_token_id: None, max_seq: None },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!("[gemma gen] new='{txt}' prefill_ms={} decode_ms={}", stats.prefill_ms, stats.decode_ms);
        assert!(!new_ids.is_empty());
    }
}
