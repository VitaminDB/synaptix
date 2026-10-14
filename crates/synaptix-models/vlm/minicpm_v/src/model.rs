//! Башня зрения MiniCPM-V 4.6/4.7 — порт `MiniCPMV4_6VisionModel` +
//! `MiniCPMV4_6Merger`.
//!
//! Кроп `h×w` патчей → патч-эмбеддинг + позиционный эмбеддинг по ближайшему
//! узлу сетки 70×70 → 27 слоёв SigLIP (pre-LN, полное двунаправленное
//! внимание, MLP с gelu_tanh). После слоя `insert_layer_id` (режим `16x`)
//! стоит оконный объединитель `vit_merger`: внимание внутри окон 2×2 и MLP,
//! склеивающий 4 токена в один (+ среднее четырёх как остаток) — сетка
//! становится `h/2 × w/2`, остальные слои идут по ней. Затем
//! `post_layernorm` и объединитель 2×2 в пространство LLM (LayerNorm →
//! Linear → точный GELU → Linear). Итог — `h·w/16` строк на кроп.
//!
//! Кропы считаются поштучно: в HF они упакованы NaViT-ом с `cu_seqlens`, и
//! внимание не пересекает границы кропов — результат тот же.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::config::MiniCpmConfig;
use crate::preprocess::{CropGrid, PreparedImage};

/// Префикс тензоров башни в HF-чекпоинте.
pub const TOWER: &str = "model.vision_tower";
/// Префикс объединителя в пространство LLM.
pub const MERGER: &str = "model.merger.mlp.0";

pub trait VisionWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError>;
}

type CoreResult<T> = Result<T, synaptix_core::error::SynaptixError>;

fn fwd<T>(r: CoreResult<T>) -> Result<T, VisionError> {
    r.map_err(|e| VisionError::Forward(e.to_string()))
}

fn ld<T>(r: CoreResult<T>) -> Result<T, VisionError> {
    r.map_err(|e| VisionError::Load(e.to_string()))
}

/// Линейный слой: вес хранится транспонированным (`[in, out]`), bias — всегда.
struct Lin {
    wt: Tensor,
    bias: Tensor,
}

impl Lin {
    fn load(w: &dyn VisionWeights, key: &str, device: Device, dtype: DType) -> Result<Self, VisionError> {
        let raw = w.tensor(&format!("{key}.weight"), device, dtype)?;
        Self::from_raw(raw, w.tensor(&format!("{key}.bias"), device, dtype)?)
    }

    /// Из веса `[out, in]` и bias `[out]`.
    fn from_raw(weight: Tensor, bias: Tensor) -> Result<Self, VisionError> {
        let wt = ld(weight.transpose(0, 1).and_then(|t| t.contiguous()))?;
        Ok(Self { wt, bias })
    }

    /// Склейка q/k/v в один слой `[in, 3·out]` — один GEMM вместо трёх.
    fn load_qkv(w: &dyn VisionWeights, prefix: &str, device: Device, dtype: DType) -> Result<Self, VisionError> {
        let mut ws = Vec::with_capacity(3);
        let mut bs = Vec::with_capacity(3);
        for n in ["q_proj", "k_proj", "v_proj"] {
            ws.push(w.tensor(&format!("{prefix}.{n}.weight"), device, dtype)?);
            bs.push(w.tensor(&format!("{prefix}.{n}.bias"), device, dtype)?);
        }
        let wr: Vec<&Tensor> = ws.iter().collect();
        let br: Vec<&Tensor> = bs.iter().collect();
        let weight = ld(Tensor::cat(&wr, 0))?;
        let bias = ld(Tensor::cat(&br, 0))?;
        Self::from_raw(weight, bias)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, VisionError> {
        fwd(x.matmul(&self.wt).and_then(|y| y.broadcast_add(&self.bias)))
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn load(w: &dyn VisionWeights, key: &str, device: Device, dtype: DType) -> Result<Self, VisionError> {
        Ok(Self {
            weight: w.tensor(&format!("{key}.weight"), device, dtype)?,
            bias: w.tensor(&format!("{key}.bias"), device, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor, eps: f32) -> Result<Tensor, VisionError> {
        fwd(layer_norm(x, Some(&self.weight), Some(&self.bias), eps))
    }
}

struct Attention {
    qkv: Lin,
    out: Lin,
}

impl Attention {
    fn load(w: &dyn VisionWeights, prefix: &str, device: Device, dtype: DType) -> Result<Self, VisionError> {
        Ok(Self {
            qkv: Lin::load_qkv(w, prefix, device, dtype)?,
            out: Lin::load(w, &format!("{prefix}.out_proj"), device, dtype)?,
        })
    }

    /// Полное внимание по `[n, D]`.
    fn forward_full(&self, h: &Tensor, nh: usize, hd: usize) -> Result<Tensor, VisionError> {
        let n = h.dims()[0];
        let qkv = self.qkv.forward(h)?;
        let qkv = fwd(qkv.reshape(vec![n, 3, nh, hd]))?;
        let take = |i: usize| -> Result<Tensor, VisionError> {
            let t = fwd(qkv.narrow(1, i, 1).and_then(|t| t.contiguous()))?;
            let t = fwd(t.reshape(vec![n, nh, hd]))?;
            fwd(t.permute(vec![1, 0, 2]).and_then(|t| t.contiguous()))
        };
        let (q, k, v) = (take(0)?, take(1)?, take(2)?);
        let scale = 1.0 / (hd as f32).sqrt();
        let a = attention_chunked(&q, &k, &v, scale, ATTN_Q_CHUNK)?;
        let a = fwd(a.permute(vec![1, 0, 2]).and_then(|t| t.contiguous()))?;
        let a = fwd(a.reshape(vec![n, nh * hd]))?;
        self.out.forward(&a)
    }

    /// Внимание внутри окон: `[n_win·ws, D]` → батч окон `[n_win, nh, ws, hd]`.
    fn forward_windows(&self, h: &Tensor, n_win: usize, ws: usize, nh: usize, hd: usize) -> Result<Tensor, VisionError> {
        let qkv = self.qkv.forward(h)?;
        let qkv = fwd(qkv.reshape(vec![n_win, ws, 3, nh, hd]))?;
        let take = |i: usize| -> Result<Tensor, VisionError> {
            let t = fwd(qkv.narrow(2, i, 1).and_then(|t| t.contiguous()))?;
            let t = fwd(t.reshape(vec![n_win, ws, nh, hd]))?;
            fwd(t.permute(vec![0, 2, 1, 3]).and_then(|t| t.contiguous()))
        };
        let (q, k, v) = (take(0)?, take(1)?, take(2)?);
        let scale = 1.0 / (hd as f32).sqrt();
        let a = fwd(scaled_dot_attention(&q, &k, &v, scale, None))?;
        let a = fwd(a.permute(vec![0, 2, 1, 3]).and_then(|t| t.contiguous()))?;
        let a = fwd(a.reshape(vec![n_win * ws, nh * hd]))?;
        self.out.forward(&a)
    }
}

struct Block {
    norm1: Norm,
    attn: Attention,
    norm2: Norm,
    fc1: Lin,
    fc2: Lin,
}

impl Block {
    fn forward(&self, x: &Tensor, nh: usize, hd: usize, eps: f32) -> Result<Tensor, VisionError> {
        let h = self.norm1.forward(x, eps)?;
        let a = self.attn.forward_full(&h, nh, hd)?;
        let x = fwd(x.add(&a))?;
        let h = self.norm2.forward(&x, eps)?;
        let h = self.fc1.forward(&h)?;
        let h = fwd(h.gelu_tanh())?;
        let h = self.fc2.forward(&h)?;
        fwd(x.add(&h))
    }
}

/// Оконный объединитель `vit_merger`.
struct WindowMerger {
    norm1: Norm,
    attn: Attention,
    pre_norm: Norm,
    linear_1: Lin,
    linear_2: Lin,
}

/// Объединитель 2×2 в пространство LLM (`DownsampleMLP`).
struct Merger {
    pre_norm: Norm,
    linear_1: Lin,
    linear_2: Lin,
}

pub struct MiniCpmVision {
    pub config: MiniCpmConfig,
    pub device: Device,
    pub dtype: DType,
    patch_embed: Lin,
    /// `[side², D]` на устройстве — строки выбираются по id позиции.
    pos_embed: Tensor,
    blocks: Vec<Block>,
    vit_merger: Option<WindowMerger>,
    post_norm: Norm,
    merger: Merger,
}

impl MiniCpmVision {
    pub fn build(
        config: MiniCpmConfig,
        weights: &dyn VisionWeights,
        device: Device,
        dtype: DType,
    ) -> Result<Self, VisionError> {
        let feats = config.patch_features();
        let patch_raw = weights.tensor(&format!("{TOWER}.embeddings.patch_embedding.weight"), device, dtype)?;
        let flat = ld(patch_raw.reshape(vec![config.hidden_size, feats]))?;
        let patch_embed = Lin::from_raw(
            flat,
            weights.tensor(&format!("{TOWER}.embeddings.patch_embedding.bias"), device, dtype)?,
        )?;
        let pos_embed = weights.tensor(&format!("{TOWER}.embeddings.position_embedding.weight"), device, dtype)?;
        let side = config.num_patches_per_side();
        if pos_embed.dims() != [side * side, config.hidden_size] {
            return Err(VisionError::Load(format!(
                "position_embedding {:?}, ожидалось [{}, {}]",
                pos_embed.dims(),
                side * side,
                config.hidden_size
            )));
        }
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("{TOWER}.encoder.layers.{i}");
            blocks.push(Block {
                norm1: Norm::load(weights, &format!("{p}.layer_norm1"), device, dtype)?,
                attn: Attention::load(weights, &format!("{p}.self_attn"), device, dtype)?,
                norm2: Norm::load(weights, &format!("{p}.layer_norm2"), device, dtype)?,
                fc1: Lin::load(weights, &format!("{p}.mlp.fc1"), device, dtype)?,
                fc2: Lin::load(weights, &format!("{p}.mlp.fc2"), device, dtype)?,
            });
        }
        let vit_merger = if config.use_vit_merger() {
            let p = format!("{TOWER}.vit_merger");
            Some(WindowMerger {
                norm1: Norm::load(weights, &format!("{p}.layer_norm1"), device, dtype)?,
                attn: Attention::load(weights, &format!("{p}.self_attn"), device, dtype)?,
                pre_norm: Norm::load(weights, &format!("{p}.pre_norm"), device, dtype)?,
                linear_1: Lin::load(weights, &format!("{p}.linear_1"), device, dtype)?,
                linear_2: Lin::load(weights, &format!("{p}.linear_2"), device, dtype)?,
            })
        } else {
            None
        };
        let post_norm = Norm::load(weights, &format!("{TOWER}.post_layernorm"), device, dtype)?;
        let merger = Merger {
            pre_norm: Norm::load(weights, &format!("{MERGER}.pre_norm"), device, dtype)?,
            linear_1: Lin::load(weights, &format!("{MERGER}.linear_1"), device, dtype)?,
            linear_2: Lin::load(weights, &format!("{MERGER}.linear_2"), device, dtype)?,
        };
        Ok(Self { config, device, dtype, patch_embed, pos_embed, blocks, vit_merger, post_norm, merger })
    }

    /// Эмбеддинги каждого кропа `[tokens, llm_hidden]` в `dtype` башни.
    pub fn forward(&self, crops: &[(Tensor, CropGrid)]) -> Result<Vec<Tensor>, VisionError> {
        crops.iter().map(|(t, g)| self.forward_crop(t, *g)).collect()
    }

    /// Все кропы картинки одним тензором `[Σtokens, llm_hidden]` — в порядке
    /// заполнителя (миниатюра, затем срезы построчно).
    pub fn forward_image(&self, prepared: &PreparedImage) -> Result<Tensor, VisionError> {
        let grids = prepared.layout.crops();
        if grids.len() != prepared.crops.len() {
            return Err(VisionError::Forward(format!(
                "кропов {} при раскладке на {}",
                prepared.crops.len(),
                grids.len()
            )));
        }
        let mut outs = Vec::with_capacity(grids.len());
        for (t, g) in prepared.crops.iter().zip(grids) {
            outs.push(self.forward_crop(t, g)?);
        }
        if outs.len() == 1 {
            return Ok(outs.pop().expect("один кроп"));
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        fwd(Tensor::cat(&refs, 0))
    }

    fn forward_crop(&self, patches: &Tensor, grid: CropGrid) -> Result<Tensor, VisionError> {
        let cfg = &self.config;
        let f = cfg.merge_factor();
        if grid.h == 0 || grid.w == 0 || grid.h % f != 0 || grid.w % f != 0 {
            return Err(VisionError::Forward(format!(
                "сетка {}×{} патчей не кратна {f}",
                grid.h, grid.w
            )));
        }
        let n = grid.patches();
        if patches.dims() != [n, cfg.patch_features()] {
            return Err(VisionError::Forward(format!(
                "патчи {:?}, ожидалось [{n}, {}]",
                patches.dims(),
                cfg.patch_features()
            )));
        }
        let (nh, hd, eps) = (cfg.num_attention_heads, cfg.head_dim(), cfg.layer_norm_eps);

        let x = fwd(patches.to_device(self.device).and_then(|t| t.to_dtype(self.dtype)))?;
        let mut x = self.patch_embed.forward(&x)?;
        let ids = nearest_position_ids(grid, cfg.num_patches_per_side());
        let ids = fwd(Tensor::from_vec(ids, vec![n], self.device))?;
        let pos = fwd(self.pos_embed.index_select(0, &ids))?;
        x = fwd(x.add(&pos))?;

        let mut cur = grid;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, nh, hd, eps)?;
            if i == cfg.insert_layer_id {
                if let Some(m) = &self.vit_merger {
                    x = self.window_merge(m, &x, cur, nh, hd, eps)?;
                    cur = CropGrid { h: cur.h / cfg.window_kernel_size.0, w: cur.w / cfg.window_kernel_size.1 };
                }
            }
        }
        let x = self.post_norm.forward(&x, eps)?;
        self.merge(&x, cur)
    }

    /// Оконный объединитель: внимание в окнах 2×2 и MLP 4D → D.
    fn window_merge(
        &self,
        m: &WindowMerger,
        x: &Tensor,
        grid: CropGrid,
        nh: usize,
        hd: usize,
        eps: f32,
    ) -> Result<Tensor, VisionError> {
        let (wh, ww) = self.config.window_kernel_size;
        let ws = wh * ww;
        let n_win = (grid.h / wh) * (grid.w / ww);
        let d = self.config.hidden_size;
        let idx = block_index(grid, (wh, ww));
        let idx = fwd(Tensor::from_vec(idx, vec![n_win * ws], self.device))?;
        let xg = fwd(x.index_select(0, &idx))?;
        let h = m.norm1.forward(&xg, eps)?;
        let a = m.attn.forward_windows(&h, n_win, ws, nh, hd)?;
        let h = fwd(xg.add(&a))?;
        let patch = fwd(h.reshape(vec![n_win, ws, d]))?;
        let residual = fwd(patch.sum([1usize]).and_then(|t| t.mul_scalar(1.0 / ws as f32)))?;
        let residual = fwd(residual.reshape(vec![n_win, d]))?;
        let flat = fwd(h.reshape(vec![n_win, ws * d]))?;
        let y = m.pre_norm.forward(&flat, eps)?;
        let y = m.linear_1.forward(&y)?;
        let y = fwd(y.gelu_tanh())?;
        let y = m.linear_2.forward(&y)?;
        fwd(y.add(&residual))
    }

    /// Объединитель 2×2 → `[n/4, llm_hidden]` (LayerNorm с eps 1e-6, как `DownsampleMLP`).
    fn merge(&self, x: &Tensor, grid: CropGrid) -> Result<Tensor, VisionError> {
        let (mh, mw) = self.config.merge_kernel_size;
        let unit = mh * mw;
        let n_out = (grid.h / mh) * (grid.w / mw);
        let d = self.config.hidden_size;
        let idx = block_index(grid, (mh, mw));
        let idx = fwd(Tensor::from_vec(idx, vec![n_out * unit], self.device))?;
        let xg = fwd(x.index_select(0, &idx))?;
        let flat = fwd(xg.reshape(vec![n_out, unit * d]))?;
        let y = self.merger.pre_norm.forward(&flat, 1e-6)?;
        let y = self.merger.linear_1.forward(&y)?;
        let y = fwd(y.gelu_exact())?;
        self.merger.linear_2.forward(&y)
    }
}

/// Индексы токенов растровой сетки `grid`, собранные блоками `bh×bw`: блоки
/// в растровом порядке по сетке блоков, внутри блока — строка за строкой
/// (`get_vision_window_index` при merge 1 / окно 2 и `view(…).permute(0,2,1,3,4)`
/// объединителя дают одну и ту же перестановку).
pub fn block_index(grid: CropGrid, (bh, bw): (usize, usize)) -> Vec<u32> {
    let mut out = Vec::with_capacity(grid.patches());
    for by in 0..grid.h / bh {
        for bx in 0..grid.w / bw {
            for dy in 0..bh {
                for dx in 0..bw {
                    out.push(((by * bh + dy) * grid.w + bx * bw + dx) as u32);
                }
            }
        }
    }
    out
}

/// `get_vision_nearest_position_ids`: для строки `i` из `h` — число границ
/// `k/side` (k = 1..side−1), не превосходящих `i/h` (`bucketize(right=True)`),
/// та же f32-арифметика, что у torch: границы — `start + k·step` в f64,
/// приведённые к f32, координаты — `i as f32 / h as f32`.
pub fn nearest_position_ids(grid: CropGrid, side: usize) -> Vec<u32> {
    let step = 1.0f64 / side as f64;
    let boundaries: Vec<f32> = (0..side - 1).map(|k| (step + k as f64 * step) as f32).collect();
    let bucket = |i: usize, len: usize| -> u32 {
        let v = i as f32 / len as f32;
        boundaries.iter().filter(|b| **b <= v).count() as u32
    };
    let bh: Vec<u32> = (0..grid.h).map(|i| bucket(i, grid.h)).collect();
    let bw: Vec<u32> = (0..grid.w).map(|j| bucket(j, grid.w)).collect();
    let mut out = Vec::with_capacity(grid.patches());
    for y in &bh {
        for x in &bw {
            out.push(y * side as u32 + x);
        }
    }
    out
}

/// Сколько query-строк считать за один вызов `scaled_dot_attention` (score
/// `[heads, Nq, Nk]` в F32 целиком — см. qwen3_vl).
pub const ATTN_Q_CHUNK: usize = 1024;

/// Внимание `[heads, N, hd]` с разрезом по строкам запросов.
pub fn attention_chunked(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, chunk: usize) -> Result<Tensor, VisionError> {
    let n = q.dims()[1];
    let chunk = chunk.max(1);
    if n <= chunk {
        return fwd(scaled_dot_attention(q, k, v, scale, None));
    }
    let mut parts: Vec<Tensor> = Vec::with_capacity(n.div_ceil(chunk));
    let mut off = 0usize;
    while off < n {
        let len = chunk.min(n - off);
        let qc = fwd(q.narrow(1, off, len).and_then(|t| t.contiguous()))?;
        parts.push(fwd(scaled_dot_attention(&qc, k, v, scale, None))?);
        off += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    fwd(Tensor::cat(&refs, 1))
}

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("vision load: {0}")]
    Load(String),
    #[error("vision forward: {0}")]
    Forward(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn block_index_windows_raster() {
        // Сетка 4×6, окна 2×2: окно (0,1) — токены 2,3,8,9.
        let idx = block_index(CropGrid { h: 4, w: 6 }, (2, 2));
        assert_eq!(idx.len(), 24);
        assert_eq!(&idx[0..4], &[0, 1, 6, 7]);
        assert_eq!(&idx[4..8], &[2, 3, 8, 9]);
        assert_eq!(&idx[12..16], &[12, 13, 18, 19]);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..24).collect::<Vec<u32>>());
    }

    #[test]
    fn position_ids_match_bucketize_definition() {
        // Прямое определение: bucketize(right=True) = индекс первой границы > v.
        let side = 70usize;
        let step = 1.0f64 / side as f64;
        let b: Vec<f32> = (0..side - 1).map(|k| (step + k as f64 * step) as f32).collect();
        for h in [4usize, 8, 16, 24, 32, 44, 70, 140] {
            let ids = nearest_position_ids(CropGrid { h, w: 1 }, side);
            for (i, id) in ids.iter().enumerate() {
                let v = i as f32 / h as f32;
                let expect = b.iter().position(|x| *x > v).unwrap_or(side - 1) as u32;
                assert_eq!(*id, expect * side as u32, "h={h} i={i}");
            }
        }
        // 70 строк — тождество; 140 — попарно.
        let ids = nearest_position_ids(CropGrid { h: 70, w: 70 }, 70);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[70 * 35 + 35], 35 * 70 + 35);
        assert_eq!(ids[70 * 69 + 69], 69 * 70 + 69);
        let ids = nearest_position_ids(CropGrid { h: 140, w: 2 }, 70);
        assert_eq!(ids[2 * 139], 69 * 70);
        assert_eq!(ids[2 * 139 + 1], 69 * 70 + 35);
    }

    /// Случайные веса малой башни: прогон на CPU даёт нужные формы, а
    /// оконное внимание батчем совпадает с внимательным счётом по одному окну.
    struct RandWeights {
        seed: std::cell::Cell<u32>,
        shapes: HashMap<String, Vec<usize>>,
    }

    impl RandWeights {
        fn rnd(&self) -> f32 {
            let mut s = self.seed.get();
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            self.seed.set(s);
            ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        }
    }

    impl VisionWeights for RandWeights {
        fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError> {
            let shape = self
                .shapes
                .get(key)
                .cloned()
                .ok_or_else(|| VisionError::Load(format!("нет формы для {key}")))?;
            let n: usize = shape.iter().product();
            let scale = if key.ends_with("norm.weight") || key.ends_with("layer_norm1.weight") || key.ends_with("layer_norm2.weight") || key.ends_with("post_layernorm.weight") { 0.0 } else { 0.2 };
            let data: Vec<f32> = (0..n).map(|_| self.rnd() * scale + if scale == 0.0 { 1.0 } else { 0.0 }).collect();
            let t = Tensor::from_vec(data, shape, device).map_err(|e| VisionError::Load(e.to_string()))?;
            t.to_dtype(dtype).map_err(|e| VisionError::Load(e.to_string()))
        }
    }

    fn tiny_cfg() -> MiniCpmConfig {
        let s = crate::config::tests::sample()
            .replace("\"hidden_size\": 1152", "\"hidden_size\": 16")
            .replace("\"intermediate_size\": 4304", "\"intermediate_size\": 24")
            .replace("\"num_hidden_layers\": 27", "\"num_hidden_layers\": 2")
            .replace("\"num_attention_heads\": 16", "\"num_attention_heads\": 2")
            .replace("\"insert_layer_id\": 6", "\"insert_layer_id\": 0")
            .replace("\"text_config\": {\"hidden_size\": 1024", "\"text_config\": {\"hidden_size\": 12");
        MiniCpmConfig::from_hf_bytes(s.as_bytes()).unwrap()
    }

    fn rand_weights(cfg: &MiniCpmConfig) -> RandWeights {
        let d = cfg.hidden_size;
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        fn lin(shapes: &mut HashMap<String, Vec<usize>>, key: &str, out: usize, inp: usize) {
            shapes.insert(format!("{key}.weight"), if inp == 0 { vec![out] } else { vec![out, inp] });
            shapes.insert(format!("{key}.bias"), vec![out]);
        }
        let s = &mut shapes;
        lin(s, &format!("{TOWER}.embeddings.patch_embedding"), d, 588);
        s.insert(format!("{TOWER}.embeddings.position_embedding.weight"), vec![4900, d]);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{TOWER}.encoder.layers.{i}");
            lin(s, &format!("{p}.layer_norm1"), d, 0);
            lin(s, &format!("{p}.layer_norm2"), d, 0);
            for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                lin(s, &format!("{p}.self_attn.{n}"), d, d);
            }
            lin(s, &format!("{p}.mlp.fc1"), cfg.intermediate_size, d);
            lin(s, &format!("{p}.mlp.fc2"), d, cfg.intermediate_size);
        }
        let p = format!("{TOWER}.vit_merger");
        lin(s, &format!("{p}.layer_norm1"), d, 0);
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(s, &format!("{p}.self_attn.{n}"), d, d);
        }
        lin(s, &format!("{p}.pre_norm"), 4 * d, 0);
        lin(s, &format!("{p}.linear_1"), 4 * cfg.intermediate_size, 4 * d);
        lin(s, &format!("{p}.linear_2"), d, 4 * cfg.intermediate_size);
        lin(s, &format!("{TOWER}.post_layernorm"), d, 0);
        lin(s, &format!("{MERGER}.pre_norm"), 4 * d, 0);
        lin(s, &format!("{MERGER}.linear_1"), 4 * d, 4 * d);
        lin(s, &format!("{MERGER}.linear_2"), cfg.llm_hidden_size, 4 * d);
        RandWeights { seed: std::cell::Cell::new(7), shapes }
    }

    #[test]
    fn tiny_tower_shapes_and_window_attention() {
        synaptix_kernels_cpu::ensure_registered();
        let cfg = tiny_cfg();
        let w = rand_weights(&cfg);
        let tower = MiniCpmVision::build(cfg.clone(), &w, Device::Cpu, DType::F32).unwrap();
        let grid = CropGrid { h: 8, w: 12 };
        let n = grid.patches();
        let data: Vec<f32> = (0..n * 588).map(|_| w.rnd()).collect();
        let patches = Tensor::from_vec(data, vec![n, 588], Device::Cpu).unwrap();
        let out = tower.forward(&[(patches.clone(), grid)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dims(), &[cfg.tokens_for(8, 12), cfg.llm_hidden_size]);
        assert_eq!(cfg.tokens_for(8, 12), 6);
        let v = out[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));

        // Оконное внимание батчем == поокно через полное внимание.
        let m = tower.vit_merger.as_ref().unwrap();
        let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim());
        let hdata: Vec<f32> = (0..n * cfg.hidden_size).map(|_| w.rnd()).collect();
        let h = Tensor::from_vec(hdata, vec![n, cfg.hidden_size], Device::Cpu).unwrap();
        let batched = m.attn.forward_windows(&h, n / 4, 4, nh, hd).unwrap();
        let bv = batched.to_vec2::<f32>().unwrap();
        for wi in 0..n / 4 {
            let win = h.narrow(0, wi * 4, 4).unwrap().contiguous().unwrap();
            let single = m.attn.forward_full(&win, nh, hd).unwrap().to_vec2::<f32>().unwrap();
            for t in 0..4 {
                for (a, b) in bv[wi * 4 + t].iter().zip(&single[t]) {
                    assert!((a - b).abs() < 1e-4, "{a} vs {b}");
                }
            }
        }

        // Раскладка картинки с миниатюрой и срезами склеивается в один тензор.
        let layout = crate::preprocess::ImageLayout {
            thumb: grid,
            slice: Some(CropGrid { h: 4, w: 4 }),
            rows: 1,
            cols: 2,
        };
        let crops = vec![
            patches,
            Tensor::from_vec((0..16 * 588).map(|_| w.rnd()).collect::<Vec<f32>>(), vec![16, 588], Device::Cpu).unwrap(),
            Tensor::from_vec((0..16 * 588).map(|_| w.rnd()).collect::<Vec<f32>>(), vec![16, 588], Device::Cpu).unwrap(),
        ];
        let prep = PreparedImage { crops, layout: layout.clone() };
        let all = tower.forward_image(&prep).unwrap();
        assert_eq!(all.dims(), &[layout.tokens(4), cfg.llm_hidden_size]);
    }
}
