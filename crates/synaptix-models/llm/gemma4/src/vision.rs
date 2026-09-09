//! Башня зрения Gemma-4 и препроцессинг картинки.
//!
//! Отличия от привычных ViT:
//!
//! * **произвольное соотношение сторон.** Картинка не режется на тайлы и не
//!   доводится до квадрата: её масштабируют так, чтобы патчей было не больше
//!   бюджета, а стороны делились на `pooling_kernel_size · patch_size`;
//! * **позиция патча — пара (x, y)** и складывается из ДВУХ таблиц
//!   эмбеддингов, а не берётся из одной по индексу в последовательности;
//! * **двумерный RoPE**: половина головы вращается позицией по x, половина —
//!   по y, у каждой половины свой набор частот от `head_dim / 2`;
//! * **RMS-норма без веса поверх V** и масштаб внимания 1.0 — как в текстовой
//!   башне;
//! * на выходе патчи усредняются блоками `k × k` (k = `pooling_kernel_size`),
//!   домножаются на `sqrt(hidden)`, стандартизуются и проецируются в
//!   пространство языковой модели.
//!
//! Итог: `encode_image` отдаёт `[soft_tokens, text_hidden]` — ровно то, что
//! подставляется вместо прогона `<|image|>` в промпте.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::attention::softmax::scaled_dot::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::loader::Gemma4Weights;

/// Разрешённые бюджеты мягких токенов на картинку (`_SUPPORTED_SOFT_TOKENS`).
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub standardize: bool,
    /// Потолок мягких токенов на картинку (`vision_soft_tokens_per_image`).
    pub max_soft_tokens: usize,
}

impl VisionConfig {
    /// Разбирает `vision_config` из `config.json`. `None` — башни у модели нет.
    pub fn from_hf_bytes(bytes: &[u8]) -> Option<Self> {
        let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        let v = root.get("vision_config")?;
        if v.is_null() {
            return None;
        }
        let u = |k: &str, d: usize| {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d)
        };
        let hidden_size = u("hidden_size", 1152);
        let heads = u("num_attention_heads", 16);
        Some(Self {
            hidden_size,
            intermediate_size: u("intermediate_size", 4304),
            num_hidden_layers: u("num_hidden_layers", 27),
            num_attention_heads: heads,
            num_key_value_heads: u("num_key_value_heads", heads),
            head_dim: u("head_dim", hidden_size / heads.max(1)),
            patch_size: u("patch_size", 16),
            pooling_kernel_size: u("pooling_kernel_size", 3),
            position_embedding_size: u("position_embedding_size", 10240),
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-6),
            rope_theta: v
                .get("rope_parameters")
                .and_then(|p| p.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(100.0),
            standardize: v.get("standardize").and_then(|x| x.as_bool()).unwrap_or(true),
            max_soft_tokens: root
                .get("vision_soft_tokens_per_image")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(280),
        })
    }

    /// Сторона, на которую обязаны делиться размеры после ресайза.
    pub fn side_multiple(&self) -> usize {
        self.pooling_kernel_size * self.patch_size
    }
}

/// Размер после ресайза с сохранением пропорций: самый крупный, который даёт
/// не больше `max_patches` патчей и делится на `pooling_kernel_size ·
/// patch_size` по обеим сторонам.
pub fn aspect_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), ModelError> {
    if height == 0 || width == 0 {
        return Err(ModelError::Forward("картинка нулевого размера".into()));
    }
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side = pooling_kernel_size * patch_size;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;

    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side;
    if th == 0 && tw == 0 {
        return Err(ModelError::Forward(format!(
            "ресайз {height}×{width} даёт 0×0 при кратности {side}"
        )));
    }
    if th == 0 {
        th = side;
        tw = ((width / height) * side).min(max_side).max(side);
    } else if tw == 0 {
        tw = side;
        th = ((height / width) * side).min(max_side).max(side);
    }
    Ok((th, tw))
}

/// Патчи `[N, patch²·3]` и их позиции `(x, y)` из CHW-картинки в `[0, 1]`.
///
/// Раскладка канала внутри патча — как в HF (`permute(1, 3, 2, 4, 0)`):
/// строка патча, столбец патча, а внутри — строка, столбец, канал.
fn patchify(chw: &[f32], h: usize, w: usize, patch: usize) -> (Vec<f32>, Vec<u32>) {
    let ph = h / patch;
    let pw = w / patch;
    let feat = patch * patch * 3;
    let mut out = vec![0f32; ph * pw * feat];
    let plane = h * w;
    for py in 0..ph {
        for px in 0..pw {
            let base = (py * pw + px) * feat;
            let mut k = 0usize;
            for iy in 0..patch {
                for ix in 0..patch {
                    let pix = (py * patch + iy) * w + px * patch + ix;
                    for c in 0..3 {
                        out[base + k] = chw[c * plane + pix];
                        k += 1;
                    }
                }
            }
        }
    }
    let mut pos = vec![0u32; ph * pw * 2];
    for py in 0..ph {
        for px in 0..pw {
            let i = (py * pw + px) * 2;
            pos[i] = px as u32;
            pos[i + 1] = py as u32;
        }
    }
    (out, pos)
}

struct VisionLayer {
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ffn_norm: Tensor,
    post_ffn_norm: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

pub struct VisionTower {
    pub cfg: VisionConfig,
    input_proj: Linear,
    /// `[2, position_embedding_size, hidden]`: отдельная таблица на x и на y.
    pos_table: Tensor,
    layers: Vec<VisionLayer>,
    std_bias: Option<Tensor>,
    std_scale: Option<Tensor>,
    /// Проекция мягких токенов в пространство языковой модели.
    embed_proj: Linear,
    /// Вектор единиц длиной `head_dim` — RMS-норма V идёт без обучаемого веса.
    v_norm_ones: Tensor,
    device: Device,
    dtype: DType,
}

impl VisionTower {
    /// Веса башни лежат в чекпойнте под `model.vision_tower.*`, проекция — под
    /// `model.embed_vision.*`; ни то, ни другое не переименовывается.
    pub fn load(
        weights: &Gemma4Weights,
        cfg: VisionConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, ModelError> {
        let t = |key: &str| -> Result<Tensor, ModelError> { weights.raw_tensor(key, device, dtype) };
        let lin = |key: &str| -> Result<Linear, ModelError> {
            Linear::new(t(key)?, None).map_err(|e| ModelError::Load(e.to_string()))
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.vision_tower.encoder.layers.{l}");
            layers.push(VisionLayer {
                input_norm: t(&format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: t(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_ffn_norm: t(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ffn_norm: t(&format!("{p}.post_feedforward_layernorm.weight"))?,
                q_proj: lin(&format!("{p}.self_attn.q_proj.linear.weight"))?,
                k_proj: lin(&format!("{p}.self_attn.k_proj.linear.weight"))?,
                v_proj: lin(&format!("{p}.self_attn.v_proj.linear.weight"))?,
                o_proj: lin(&format!("{p}.self_attn.o_proj.linear.weight"))?,
                q_norm: t(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: t(&format!("{p}.self_attn.k_norm.weight"))?,
                gate_proj: lin(&format!("{p}.mlp.gate_proj.linear.weight"))?,
                up_proj: lin(&format!("{p}.mlp.up_proj.linear.weight"))?,
                down_proj: lin(&format!("{p}.mlp.down_proj.linear.weight"))?,
            });
        }
        let (std_bias, std_scale) = if cfg.standardize {
            (
                Some(t("model.vision_tower.std_bias")?.to_dtype(DType::F32).coerr()?),
                Some(t("model.vision_tower.std_scale")?.to_dtype(DType::F32).coerr()?),
            )
        } else {
            (None, None)
        };
        let v_norm_ones = Tensor::from_vec(vec![1.0_f32; cfg.head_dim], vec![cfg.head_dim], device)
            .and_then(|x| x.to_dtype(dtype))
            .map_err(|e| ModelError::Load(e.to_string()))?;
        Ok(Self {
            input_proj: lin("model.vision_tower.patch_embedder.input_proj.weight")?,
            pos_table: t("model.vision_tower.patch_embedder.position_embedding_table")?,
            layers,
            std_bias,
            std_scale,
            embed_proj: lin("model.embed_vision.embedding_projection.weight")?,
            v_norm_ones,
            cfg,
            device,
            dtype,
        })
    }

    /// Сколько мягких токенов даст картинка такого размера.
    pub fn soft_tokens_for(&self, height: usize, width: usize, max_soft: usize) -> usize {
        let pool = self.cfg.pooling_kernel_size;
        let max_patches = max_soft * pool * pool;
        match aspect_preserving_size(height, width, self.cfg.patch_size, max_patches, pool) {
            Ok((th, tw)) => (th / self.cfg.patch_size) * (tw / self.cfg.patch_size) / (pool * pool),
            Err(_) => 0,
        }
    }

    /// Картинка (CHW в `[0, 1]`, RGB) → мягкие токены в пространстве языковой
    /// модели: `[soft_tokens, text_hidden]`.
    pub fn encode_chw(
        &self,
        chw: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Tensor, ModelError> {
        let patch = self.cfg.patch_size;
        if height % patch != 0 || width % patch != 0 {
            return Err(ModelError::Forward(format!(
                "картинка {height}×{width} не кратна патчу {patch}"
            )));
        }
        let (pixels, positions) = patchify(chw, height, width, patch);
        let n = positions.len() / 2;
        let feat = patch * patch * 3;
        let pixel = Tensor::from_vec(pixels, vec![n, feat], self.device)
            .and_then(|x| x.to_dtype(self.dtype))
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        self.forward_patches(&pixel, &positions, width / patch)
    }

    /// Голый проход по уже нарезанным патчам — точка входа для сверки с
    /// эталоном (`pixel_values` из HF-препроцессора подаются как есть).
    pub fn forward_patches(
        &self,
        pixel: &Tensor,
        positions: &[u32],
        patch_cols: usize,
    ) -> Result<Tensor, ModelError> {
        let n = pixel.dims()[0];
        if positions.len() != 2 * n {
            return Err(ModelError::Forward(format!(
                "позиций {} на {n} патчей",
                positions.len()
            )));
        }
        // HF не нормирует пиксели, а растягивает их в [-1, 1] уже в модели.
        let x = pixel
            .contiguous()
            .and_then(|t| t.mul_scalar(2.0))
            .and_then(|t| t.add_scalar(-1.0))
            .coerr()?;
        let mut h = self.input_proj.forward(&x).coerr()?;
        h = h.add(&self.position_embeddings(positions, n)?).coerr()?;

        let (cos_x, sin_x, cos_y, sin_y) = self.rope_tables(positions, n)?;
        for layer in &self.layers {
            h = self.layer_forward(layer, &h, n, &cos_x, &sin_x, &cos_y, &sin_y)?;
        }
        let pooled = self.pool(&h, positions, n, patch_cols)?;
        self.project(&pooled)
    }

    fn position_embeddings(&self, positions: &[u32], n: usize) -> Result<Tensor, ModelError> {
        let hidden = self.cfg.hidden_size;
        let p = self.cfg.position_embedding_size;
        let xs: Vec<u32> = (0..n).map(|i| positions[2 * i]).collect();
        let ys: Vec<u32> = (0..n).map(|i| positions[2 * i + 1]).collect();
        let table = |axis: usize, idx: Vec<u32>| -> Result<Tensor, ModelError> {
            let t = self
                .pos_table
                .narrow(0, axis, 1)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape(vec![p, hidden]))
                .coerr()?;
            let ids = Tensor::from_vec(idx, vec![n], self.device).coerr()?;
            t.index_select(0, &ids).coerr()
        };
        let ex = table(0, xs)?;
        let ey = table(1, ys)?;
        ex.add(&ey).coerr()
    }

    /// Таблицы cos/sin двумерного RoPE: своя пара на каждую ось, частоты
    /// считаются от ПОЛОВИНЫ головы (`head_dim / 2`), а не от всей.
    fn rope_tables(
        &self,
        positions: &[u32],
        n: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), ModelError> {
        let spatial = self.cfg.head_dim / 2;
        let half = spatial / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|j| self.cfg.rope_theta.powf(-(2.0 * j as f32) / spatial as f32))
            .collect();
        let build = |axis: usize| -> Result<(Tensor, Tensor), ModelError> {
            let mut cos = vec![0f32; n * half];
            let mut sin = vec![0f32; n * half];
            for i in 0..n {
                let pos = positions[2 * i + axis] as f32;
                for (j, f) in freqs.iter().enumerate() {
                    let a = pos * f;
                    cos[i * half + j] = a.cos();
                    sin[i * half + j] = a.sin();
                }
            }
            Ok((
                Tensor::from_vec(cos, vec![n, half], self.device).coerr()?,
                Tensor::from_vec(sin, vec![n, half], self.device).coerr()?,
            ))
        };
        let (cx, sx) = build(0)?;
        let (cy, sy) = build(1)?;
        Ok((cx, sx, cy, sy))
    }

    /// `[1, heads, N, head_dim]` → RoPE по x на первой половине головы и по y
    /// на второй (HF `apply_multidimensional_rope`).
    fn rope2d(
        &self,
        x: &Tensor,
        cos_x: &Tensor,
        sin_x: &Tensor,
        cos_y: &Tensor,
        sin_y: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let spatial = self.cfg.head_dim / 2;
        let part = |off: usize, cos: &Tensor, sin: &Tensor| -> Result<Tensor, ModelError> {
            let slice = x.narrow(3, off, spatial).and_then(|t| t.contiguous()).coerr()?;
            apply_rope_with_cossin(&slice, cos, sin, RopeLayout::Split).coerr()
        };
        let a = part(0, cos_x, sin_x)?;
        let b = part(spatial, cos_y, sin_y)?;
        Tensor::cat(&[&a, &b], 3).coerr()
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &VisionLayer,
        h: &Tensor,
        n: usize,
        cos_x: &Tensor,
        sin_x: &Tensor,
        cos_y: &Tensor,
        sin_y: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let eps = self.cfg.rms_norm_eps;
        let heads = self.cfg.num_attention_heads;
        let hd = self.cfg.head_dim;
        let hidden = self.cfg.hidden_size;

        let residual = h.clone();
        let x = rms_norm(h, &layer.input_norm, eps).coerr()?;

        let heads_view = |t: Tensor| -> Result<Tensor, ModelError> {
            t.reshape(vec![1usize, n, heads, hd])
                .and_then(|t| t.permute(vec![0, 2, 1, 3]))
                .and_then(|t| t.contiguous())
                .coerr()
        };
        let q = heads_view(layer.q_proj.forward(&x).coerr()?)?;
        let k = heads_view(layer.k_proj.forward(&x).coerr()?)?;
        let v = heads_view(layer.v_proj.forward(&x).coerr()?)?;

        let q = rms_norm(&q, &layer.q_norm, eps).coerr()?;
        let k = rms_norm(&k, &layer.k_norm, eps).coerr()?;
        let v = rms_norm(&v, &self.v_norm_ones, eps).coerr()?;

        let q = self.rope2d(&q, cos_x, sin_x, cos_y, sin_y)?;
        let k = self.rope2d(&k, cos_x, sin_x, cos_y, sin_y)?;

        // Внимание двустороннее (маски нет: паддинга у нас тоже нет) и без
        // масштаба 1/sqrt(d) — его роль играет обучаемая Q-норма.
        let attn = scaled_dot_attention(&q, &k, &v, 1.0, None).coerr()?;
        let attn = attn
            .permute(vec![0, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![n, hidden]))
            .coerr()?;
        let attn = layer.o_proj.forward(&attn).coerr()?;
        let attn = rms_norm(&attn, &layer.post_attn_norm, eps).coerr()?;
        let h = residual.add(&attn).coerr()?;

        let residual = h.clone();
        let x = rms_norm(&h, &layer.pre_ffn_norm, eps).coerr()?;
        let gate = layer.gate_proj.forward(&x).coerr()?;
        let up = layer.up_proj.forward(&x).coerr()?;
        let act = gate.gelu_tanh().and_then(|g| g.mul(&up)).coerr()?;
        let ffn = layer.down_proj.forward(&act).coerr()?;
        let ffn = rms_norm(&ffn, &layer.post_ffn_norm, eps).coerr()?;
        residual.add(&ffn).coerr()
    }

    /// Усреднение по блокам `k × k` в координатах патчей плюс масштаб
    /// `sqrt(hidden)`. Считается в F32: масштаб выносит активации за предел F16.
    fn pool(
        &self,
        h: &Tensor,
        positions: &[u32],
        n: usize,
        patch_cols: usize,
    ) -> Result<Tensor, ModelError> {
        let k = self.cfg.pooling_kernel_size;
        let k2 = k * k;
        if n % k2 != 0 {
            return Err(ModelError::Forward(format!(
                "патчей {n} не кратно {k2} — усреднить блоками нечем"
            )));
        }
        let groups = n / k2;
        let cols = patch_cols / k;
        // Веса блока: строка «патч → группа», делённая на k². Один matmul
        // вместо ручной развёртки индексов.
        let mut w = vec![0f32; groups * n];
        for i in 0..n {
            let (x, y) = (positions[2 * i] as usize, positions[2 * i + 1] as usize);
            let g = x / k + cols * (y / k);
            if g >= groups {
                return Err(ModelError::Forward(format!(
                    "патч ({x}, {y}) попал в группу {g} при {groups} группах"
                )));
            }
            w[g * n + i] = 1.0 / k2 as f32;
        }
        let weights = Tensor::from_vec(w, vec![groups, n], self.device).coerr()?;
        let hf = h.to_dtype(DType::F32).coerr()?;
        let pooled = weights.matmul(&hf).coerr()?;
        pooled.mul_scalar((self.cfg.hidden_size as f32).sqrt()).coerr()
    }

    /// Стандартизация и проекция в пространство языковой модели.
    fn project(&self, pooled: &Tensor) -> Result<Tensor, ModelError> {
        let mut x = pooled.clone();
        if let (Some(bias), Some(scale)) = (&self.std_bias, &self.std_scale) {
            x = x
                .broadcast_sub(bias)
                .and_then(|t| t.broadcast_mul(scale))
                .coerr()?;
        }
        let x = x.to_dtype(self.dtype).coerr()?;
        // Норма перед проекцией — без обучаемого веса.
        let ones = Tensor::from_vec(
            vec![1.0_f32; self.cfg.hidden_size],
            vec![self.cfg.hidden_size],
            self.device,
        )
        .and_then(|t| t.to_dtype(self.dtype))
        .coerr()?;
        let normed = rms_norm(&x, &ones, self.cfg.rms_norm_eps).coerr()?;
        self.embed_proj.forward(&normed).coerr()
    }
}

trait CoreErr<T> {
    fn coerr(self) -> Result<T, ModelError>;
}
impl<T> CoreErr<T> for synaptix_core::error::Result<T> {
    fn coerr(self) -> Result<T, ModelError> {
        self.map_err(|e| ModelError::Forward(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_size_keeps_ratio_and_budget() {
        // 280 мягких токенов × 9 = 2520 патчей по 16 пикселей.
        let (h, w) = aspect_preserving_size(1000, 2000, 16, 2520, 3).unwrap();
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        assert!((h / 16) * (w / 16) <= 2520, "{h}×{w}");
        // Пропорция 1:2 сохранена с точностью до кратности 48.
        assert!((w as f32 / h as f32 - 2.0).abs() < 0.2, "{h}×{w}");
    }

    #[test]
    fn small_image_is_not_upscaled_past_the_budget() {
        let (h, w) = aspect_preserving_size(96, 144, 16, 2520, 3).unwrap();
        assert!((h / 16) * (w / 16) <= 2520);
        assert_eq!((h % 48, w % 48), (0, 0));
    }

    #[test]
    fn patchify_matches_hf_channel_order() {
        // 1×(2 патча по горизонтали) при patch=2: 3 канала, 2×4 пикселя.
        let (h, w, p) = (2usize, 4usize, 2usize);
        let plane = h * w;
        let chw: Vec<f32> = (0..3 * plane).map(|i| i as f32).collect();
        let (patches, pos) = patchify(&chw, h, w, p);
        assert_eq!(patches.len(), 2 * p * p * 3);
        assert_eq!(pos, vec![0, 0, 1, 0]);
        // Первый патч, первый пиксель (0,0): каналы 0, plane, 2·plane.
        assert_eq!(&patches[0..3], &[0.0, plane as f32, 2.0 * plane as f32]);
        // Второй пиксель патча — (0,1).
        assert_eq!(&patches[3..6], &[1.0, plane as f32 + 1.0, 2.0 * plane as f32 + 1.0]);
    }
}
