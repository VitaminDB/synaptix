//! `config.json` Inkling-2 (`model_type` `inkling2_loop`, текст — `text_config`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub swa_num_attention_heads: Option<usize>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub swa_head_dim: Option<usize>,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f32,
    pub log_scaling_alpha: f32,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub final_logit_softcapping: f32,
    pub sliding_window_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    #[serde(default)]
    pub model_max_length: Option<usize>,
    pub loop_config: Option<LoopConfig>,
}

/// `vision_config`: ViT с осевым 2D RoPE, слиянием 2×2 патчей и MLP-проекцией в текст.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "vision_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(alias = "patch_size")]
    pub native_patch_size: usize,
    #[serde(alias = "spatial_merge_size")]
    pub adapter_merge_size: usize,
    pub posemb_grid_size: usize,
    pub projection_hidden_dim: usize,
    #[serde(default)]
    pub projector_bias: bool,
    pub text_hidden_size: usize,
    #[serde(default = "vision_ln_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "vision_rope_theta")]
    pub rope_theta: f32,
}

fn vision_mlp_ratio() -> f32 {
    4.0
}
fn vision_ln_eps() -> f32 {
    1e-5
}
fn vision_rope_theta() -> f32 {
    10000.0
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }
    /// Сторона клетки картинки на один токен текста (40 px).
    pub fn cell(&self) -> usize {
        self.native_patch_size * self.adapter_merge_size
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Root {
    text_config: TextConfig,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    model_type: String,
}

#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub text: TextConfig,
    pub vision: Option<VisionConfig>,
    pub eos_ids: Vec<u32>,
    pub model_type: String,
}

/// Место применения слоя в расписании прохода: физический слой и слот кэша
/// (у каждого прохода цикла свой KV и свои свёртки).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    pub phys: usize,
    pub slot: usize,
    /// Номер прохода цикла: 0 — пролог/эпилог и первый проход ядра.
    pub pass: usize,
}

impl Inkling2Config {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let root: Root = serde_json::from_slice(bytes).map_err(|e| format!("config.json: {e}"))?;
        let eos_ids = match root.eos_token_id {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().unwrap_or(0) as u32],
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
            _ => Vec::new(),
        };
        let cfg = Self { text: root.text_config, vision: root.vision_config, eos_ids, model_type: root.model_type };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        let t = &self.text;
        if t.n_shared_experts != 0 {
            return Err("Inkling2: shared-эксперты не поддержаны".into());
        }
        if t.swa_head_dim.unwrap_or(t.head_dim) != t.head_dim
            || t.swa_num_attention_heads.unwrap_or(t.num_attention_heads) != t.num_attention_heads
            || t.swa_num_key_value_heads.unwrap_or(t.num_key_value_heads) != t.num_key_value_heads
        {
            return Err("Inkling2: разные формы голов у SWA и полных слоёв не поддержаны".into());
        }
        if t.num_attention_heads % t.num_key_value_heads != 0 {
            return Err("Inkling2: heads % kv_heads != 0".into());
        }
        if let Some(l) = &t.loop_config {
            let b = t.n_layers_per_attn_res_block;
            if l.is_varloop || l.num_passes < 2 || l.core_start == 0 || l.core_start >= l.core_end
                || l.core_end > t.num_hidden_layers || l.core_start % b != 0 || l.core_end % b != 0
            {
                return Err(format!("Inkling2: неподдержанный loop_config {l:?}"));
            }
        }
        Ok(())
    }

    pub fn is_swa(&self, phys: usize) -> bool {
        self.text.local_layer_ids.contains(&phys)
    }

    pub fn rel_extent(&self, phys: usize) -> usize {
        if self.is_swa(phys) { self.text.sliding_window_size } else { self.text.rel_extent }
    }

    pub fn num_passes(&self) -> usize {
        self.text.loop_config.as_ref().map_or(1, |l| l.num_passes)
    }

    pub fn core(&self) -> (usize, usize) {
        self.text.loop_config.as_ref().map_or((0, 0), |l| (l.core_start, l.core_end))
    }

    /// Число слотов кэша: физические слои + повторные проходы ядра.
    pub fn num_slots(&self) -> usize {
        let (s, e) = self.core();
        self.text.num_hidden_layers + (self.num_passes().saturating_sub(1)) * (e - s)
    }

    /// Слот кэша для `phys` на проходе `pass` (как `il` в llama.cpp).
    pub fn slot(&self, phys: usize, pass: usize) -> usize {
        let (s, e) = self.core();
        if pass == 0 { phys } else { self.text.num_hidden_layers + (pass - 1) * (e - s) + (phys - s) }
    }

    pub fn loop_layer(&self, phys: usize) -> bool {
        let (s, e) = self.core();
        phys >= s && phys < e
    }
}
