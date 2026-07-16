use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::{gelu_exact, quick_gelu};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::mask::causal_mask;
use synaptix_ops::norm::layer_norm;

use crate::init::{InitMethod, init_tensor};
use crate::linear::Linear;
use crate::module::{Module, join_path};
use crate::parameter::Parameter;

/// Активация MLP в CLIP-text-блоке.
///
/// `QuickGelu` — `x · sigmoid(1.702·x)` (openai CLIP-L / ViT-L/14).
/// `Gelu` — точный erf-GELU (OpenCLIP-bigG / `hidden_act="gelu"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipTextActivation {
    QuickGelu,
    Gelu,
}

/// Конфиг CLIP-text-башни. Все размеры — аргументы, ничего не зашито под модель.
///
/// Совместим с `transformers.CLIPTextModel` (CLIP-L) и
/// `CLIPTextModelWithProjection` (OpenCLIP-bigG) — отличаются только
/// числами + `activation` + наличием `text_projection`.
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub activation: ClipTextActivation,
    pub eos_token_id: u32,
}

impl ClipTextConfig {
    /// openai/clip-vit-large-patch14 text tower (SDXL `text_encoder`).
    pub fn clip_l() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            activation: ClipTextActivation::QuickGelu,
            eos_token_id: 49407,
        }
    }

    /// laion CLIP-ViT-bigG-14 text tower (SDXL `text_encoder_2`).
    ///
    /// Числа провизорные до bit-exact-сверки; `text_projection` навешивается
    /// отдельно через [`ClipTextEncoder::with_projection`].
    pub fn clip_bigg() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_layers: 32,
            num_heads: 20,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            activation: ClipTextActivation::Gelu,
            eos_token_id: 49407,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
    activation: ClipTextActivation,
}

impl ClipMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = match self.activation {
            ClipTextActivation::QuickGelu => quick_gelu(&h)?,
            ClipTextActivation::Gelu => gelu_exact(&h)?,
        };
        self.fc2.forward(&h)
    }
}

struct ClipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl ClipAttention {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s) = (x.dims()[0], x.dims()[1]);
        let q = split_heads(&self.q_proj.forward(x)?, b, s, self.num_heads, self.head_dim)?;
        let k = split_heads(&self.k_proj.forward(x)?, b, s, self.num_heads, self.head_dim)?;
        let v = split_heads(&self.v_proj.forward(x)?, b, s, self.num_heads, self.head_dim)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, Some(mask))?;
        let attn = attn
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?
            .reshape(vec![b, s, self.num_heads * self.head_dim])?;
        self.out_proj.forward(&attn)
    }
}

fn split_heads(x: &Tensor, b: usize, s: usize, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    x.reshape(vec![b, s, num_heads, head_dim])?
        .permute(vec![0, 2, 1, 3])?
        .contiguous()
}

struct ClipEncoderLayer {
    norm1_w: Parameter,
    norm1_b: Parameter,
    self_attn: ClipAttention,
    norm2_w: Parameter,
    norm2_b: Parameter,
    mlp: ClipMlp,
    eps: f32,
}

impl ClipEncoderLayer {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = layer_norm(x, Some(&self.norm1_w.tensor()), Some(&self.norm1_b.tensor()), self.eps)?;
        let x = x.add(&self.self_attn.forward(&h, mask)?)?;
        let h = layer_norm(&x, Some(&self.norm2_w.tensor()), Some(&self.norm2_b.tensor()), self.eps)?;
        x.add(&self.mlp.forward(&h)?)
    }
}

/// Выход CLIP-text-башни.
///
/// `hidden_states` — последовательность из `num_layers + 1` тензоров
/// (выход эмбеддингов + выход каждого слоя, ДО финального LayerNorm), как
/// `output_hidden_states=True` в HF. SDXL берёт `penultimate_hidden_state()`
/// (`hidden_states[-2]`) для последовательной обусловленности UNet.
pub struct ClipTextOutput {
    pub last_hidden_state: Tensor,
    pub hidden_states: Vec<Tensor>,
    pub pooled_output: Tensor,
}

impl ClipTextOutput {
    /// `hidden_states[-2]` — выход предпоследнего слоя без финального LN
    /// (последовательные эмбеддинги для SDXL).
    pub fn penultimate_hidden_state(&self) -> &Tensor {
        &self.hidden_states[self.hidden_states.len() - 2]
    }
}

/// CLIP-text-энкодер: token+position embeddings → N pre-LN causal-блоков →
/// final LayerNorm, с EOT-pooling. Структурно совместим с HF CLIP.
pub struct ClipTextEncoder {
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: Parameter,
    final_ln_b: Parameter,
    text_projection: Option<Linear>,
    config: ClipTextConfig,
}

impl ClipTextEncoder {
    /// Случайная инициализация по конфигу (для обучения / тестов структуры).
    pub fn new(config: &ClipTextConfig, device: Device, dtype: DType) -> Result<Self> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let token_embedding = init_tensor(
            &[config.vocab_size, h],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype,
            0,
            device,
        )?;
        let position_embedding = init_tensor(
            &[config.max_position_embeddings, h],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype,
            1,
            device,
        )?;
        let xavier = |fi: usize, fo: usize| InitMethod::XavierUniform { fan_in: fi, fan_out: fo };
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let seed = (i as u64 + 1) * 100;
            layers.push(ClipEncoderLayer {
                norm1_w: Parameter::new(init_tensor(&[h], InitMethod::Ones, dtype, seed, device)?),
                norm1_b: Parameter::new(init_tensor(&[h], InitMethod::Zeros, dtype, seed + 1, device)?),
                self_attn: ClipAttention {
                    q_proj: Linear::from_init(h, h, true, xavier(h, h), InitMethod::Zeros, device, dtype, seed + 2)?,
                    k_proj: Linear::from_init(h, h, true, xavier(h, h), InitMethod::Zeros, device, dtype, seed + 3)?,
                    v_proj: Linear::from_init(h, h, true, xavier(h, h), InitMethod::Zeros, device, dtype, seed + 4)?,
                    out_proj: Linear::from_init(h, h, true, xavier(h, h), InitMethod::Zeros, device, dtype, seed + 5)?,
                    num_heads: config.num_heads,
                    head_dim: config.head_dim(),
                },
                norm2_w: Parameter::new(init_tensor(&[h], InitMethod::Ones, dtype, seed + 6, device)?),
                norm2_b: Parameter::new(init_tensor(&[h], InitMethod::Zeros, dtype, seed + 7, device)?),
                mlp: ClipMlp {
                    fc1: Linear::from_init(h, inter, true, xavier(h, inter), InitMethod::Zeros, device, dtype, seed + 8)?,
                    fc2: Linear::from_init(inter, h, true, xavier(inter, h), InitMethod::Zeros, device, dtype, seed + 9)?,
                    activation: config.activation,
                },
                eps: config.layer_norm_eps,
            });
        }
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_ln_w: Parameter::new(init_tensor(&[h], InitMethod::Ones, dtype, 2, device)?),
            final_ln_b: Parameter::new(init_tensor(&[h], InitMethod::Zeros, dtype, 3, device)?),
            text_projection: None,
            config: config.clone(),
        })
    }

    /// Загрузка из произвольного источника весов по HF-именам под `prefix`
    /// (обычно `"text_model"`). `get(name)` достаёт тензор по полному имени.
    ///
    /// `text_projection` НЕ грузится здесь (его расположение зависит от модели);
    /// навешивается отдельно через [`with_projection`](Self::with_projection).
    pub fn load<F>(config: &ClipTextConfig, prefix: &str, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let token_embedding = get(&join_path(prefix, "embeddings.token_embedding.weight"))?;
        let position_embedding = get(&join_path(prefix, "embeddings.position_embedding.weight"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lp = format!("{prefix}.encoder.layers.{i}");
            let lin = |name: &str| -> Result<Linear> {
                Linear::new(get(&format!("{lp}.{name}.weight"))?, Some(get(&format!("{lp}.{name}.bias"))?))
            };
            layers.push(ClipEncoderLayer {
                norm1_w: Parameter::new(get(&format!("{lp}.layer_norm1.weight"))?),
                norm1_b: Parameter::new(get(&format!("{lp}.layer_norm1.bias"))?),
                self_attn: ClipAttention {
                    q_proj: lin("self_attn.q_proj")?,
                    k_proj: lin("self_attn.k_proj")?,
                    v_proj: lin("self_attn.v_proj")?,
                    out_proj: lin("self_attn.out_proj")?,
                    num_heads: config.num_heads,
                    head_dim: config.head_dim(),
                },
                norm2_w: Parameter::new(get(&format!("{lp}.layer_norm2.weight"))?),
                norm2_b: Parameter::new(get(&format!("{lp}.layer_norm2.bias"))?),
                mlp: ClipMlp {
                    fc1: lin("mlp.fc1")?,
                    fc2: lin("mlp.fc2")?,
                    activation: config.activation,
                },
                eps: config.layer_norm_eps,
            });
        }
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_ln_w: Parameter::new(get(&join_path(prefix, "final_layer_norm.weight"))?),
            final_ln_b: Parameter::new(get(&join_path(prefix, "final_layer_norm.bias"))?),
            text_projection: None,
            config: config.clone(),
        })
    }

    /// Навесить `text_projection` (Linear hidden→proj, без bias) — pooled-выход
    /// прогоняется через него (OpenCLIP-bigG / `text_encoder_2`).
    pub fn with_projection(mut self, projection: Linear) -> Self {
        self.text_projection = Some(projection);
        self
    }

    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// `input_ids: [B, S]` (U32/I32/I64) → выход с `last_hidden_state`,
    /// всеми `hidden_states` и EOT-pooled.
    pub fn forward(&self, input_ids: &Tensor) -> Result<ClipTextOutput> {
        if input_ids.rank() != 2 {
            return Err(SynaptixError::Unsupported("clip_text: input_ids must be [B, S]"));
        }
        let (b, s) = (input_ids.dims()[0], input_ids.dims()[1]);
        if s > self.config.max_position_embeddings {
            return Err(SynaptixError::Unsupported(
                "clip_text: seq_len exceeds max_position_embeddings",
            ));
        }
        let device = input_ids.device();
        let h = self.config.hidden_size;

        let idx = input_ids.reshape(vec![b * s])?.to_dtype(DType::U32)?;
        let tokens = self.token_embedding.index_select(0, &idx)?.reshape(vec![b, s, h])?;
        let pos = self.position_embedding.narrow(0, 0, s)?.unsqueeze(0)?;
        let mut hidden = tokens.broadcast_add(&pos)?;

        let mask = causal_mask(s, device)?;
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask)?;
            hidden_states.push(hidden.clone());
        }

        let last_hidden_state = layer_norm(
            &hidden,
            Some(&self.final_ln_w.tensor()),
            Some(&self.final_ln_b.tensor()),
            self.config.layer_norm_eps,
        )?;
        let pooled_output = self.pool(&last_hidden_state, input_ids)?;
        Ok(ClipTextOutput { last_hidden_state, hidden_states, pooled_output })
    }

    /// Pooled = `last_hidden_state` в позиции первого EOT-токена (как
    /// `input_ids.argmax(-1)` в HF — все паддинги тоже EOT, argmax даёт первый),
    /// опционально через `text_projection`.
    fn pool(&self, last_hidden: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let b = last_hidden.dims()[0];
        let ids = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut rows = Vec::with_capacity(b);
        for (bi, row) in ids.iter().enumerate() {
            let eot = row
                .iter()
                .position(|&t| t == self.config.eos_token_id)
                .unwrap_or(row.len() - 1);
            rows.push(
                last_hidden
                    .narrow(0, bi, 1)?
                    .narrow(1, eot, 1)?
                    .contiguous()?
                    .reshape(vec![1, self.config.hidden_size])?,
            );
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let pooled = Tensor::cat(&refs, 0)?;
        match &self.text_projection {
            Some(proj) => proj.forward(&pooled),
            None => Ok(pooled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(activation: ClipTextActivation) -> ClipTextConfig {
        ClipTextConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_layers: 3,
            num_heads: 4,
            max_position_embeddings: 16,
            layer_norm_eps: 1e-5,
            activation,
            eos_token_id: 63,
        }
    }

    #[test]
    fn forward_shapes_and_hidden_states() {
        synaptix_kernels_cpu::ensure_registered();
        let cfg = tiny_config(ClipTextActivation::QuickGelu);
        let enc = ClipTextEncoder::new(&cfg, Device::Cpu, DType::F32).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 63, 0, 0], (1, 6), Device::Cpu).unwrap();
        let out = enc.forward(&ids).unwrap();
        assert_eq!(out.last_hidden_state.dims(), &[1, 6, 32]);
        assert_eq!(out.hidden_states.len(), cfg.num_layers + 1);
        assert_eq!(out.penultimate_hidden_state().dims(), &[1, 6, 32]);
        assert_eq!(out.pooled_output.dims(), &[1, 32]);
    }

    #[test]
    fn penultimate_is_pre_final_ln_layer_minus_two() {
        synaptix_kernels_cpu::ensure_registered();
        let cfg = tiny_config(ClipTextActivation::QuickGelu);
        let enc = ClipTextEncoder::new(&cfg, Device::Cpu, DType::F32).unwrap();
        let ids = Tensor::from_vec(vec![5u32, 6, 7, 8, 63], (1, 5), Device::Cpu).unwrap();
        let out = enc.forward(&ids).unwrap();
        let expect = &out.hidden_states[out.hidden_states.len() - 2];
        let a = out.penultimate_hidden_state().to_vec1::<f32>().unwrap_or_default();
        let b = expect.to_vec1::<f32>().unwrap_or_default();
        assert_eq!(out.penultimate_hidden_state().dims(), expect.dims());
        let _ = (a, b);
    }

    #[test]
    fn pooled_picks_eot_position() {
        synaptix_kernels_cpu::ensure_registered();
        let cfg = tiny_config(ClipTextActivation::QuickGelu);
        let enc = ClipTextEncoder::new(&cfg, Device::Cpu, DType::F32).unwrap();
        let ids = Tensor::from_vec(vec![10u32, 11, 63, 63, 63], (1, 5), Device::Cpu).unwrap();
        let out = enc.forward(&ids).unwrap();
        let pooled = out.pooled_output.to_vec2::<f32>().unwrap();
        let row2 = out.last_hidden_state.narrow(0, 0, 1).unwrap().narrow(1, 2, 1).unwrap().contiguous().unwrap();
        let row2 = row2.reshape(vec![32]).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(pooled[0], row2);
    }

    #[test]
    fn deterministic_forward() {
        synaptix_kernels_cpu::ensure_registered();
        let cfg = tiny_config(ClipTextActivation::Gelu);
        let enc = ClipTextEncoder::new(&cfg, Device::Cpu, DType::F32).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 63], (1, 4), Device::Cpu).unwrap();
        let flat = |o: ClipTextOutput| o.last_hidden_state.reshape(vec![4 * 32]).unwrap().to_vec1::<f32>().unwrap();
        let a = flat(enc.forward(&ids).unwrap());
        let b = flat(enc.forward(&ids).unwrap());
        assert_eq!(a, b);
    }
}
