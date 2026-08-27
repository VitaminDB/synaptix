use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::config::VisionConfig;
use crate::preprocess::ImageGrid;

pub const VIS: &str = "model.visual";
const ROPE_THETA: f32 = 10_000.0;

pub trait VisionWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError>;
}

struct Lin {
    wt: Tensor,
    bias: Option<Tensor>,
}

impl Lin {
    fn load(
        w: &dyn VisionWeights,
        key: &str,
        device: Device,
        dtype: DType,
        with_bias: bool,
    ) -> Result<Self, VisionError> {
        let raw = w.tensor(&format!("{key}.weight"), device, dtype)?;
        let wt = raw
            .transpose(0, 1)
            .and_then(|t| t.contiguous())
            .map_err(|e| VisionError::Load(e.to_string()))?;
        let bias = if with_bias {
            Some(w.tensor(&format!("{key}.bias"), device, dtype)?)
        } else {
            None
        };
        Ok(Self { wt, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, VisionError> {
        let y = x.matmul(&self.wt).map_err(|e| VisionError::Forward(e.to_string()))?;
        match &self.bias {
            Some(b) => y
                .broadcast_add(b)
                .map_err(|e| VisionError::Forward(e.to_string())),
            None => Ok(y),
        }
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn load(
        w: &dyn VisionWeights,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self, VisionError> {
        Ok(Self {
            weight: w.tensor(&format!("{key}.weight"), device, dtype)?,
            bias: w.tensor(&format!("{key}.bias"), device, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor, eps: f32) -> Result<Tensor, VisionError> {
        layer_norm(x, Some(&self.weight), Some(&self.bias), eps)
            .map_err(|e| VisionError::Forward(e.to_string()))
    }
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

pub struct VisionTower {
    pub config: VisionConfig,
    pub device: Device,
    pub dtype: DType,
    patch_embed: Lin,
    pos_embed: Vec<f32>,
    blocks: Vec<Block>,
    merger_norm: Norm,
    merger_fc1: Lin,
    merger_fc2: Lin,
    deepstack: Vec<Merger>,
    use_rope: bool,
}

pub struct Merger {
    norm: Norm,
    fc1: Lin,
    fc2: Lin,
}

impl Merger {
    fn load(
        weights: &dyn VisionWeights,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self, VisionError> {
        Ok(Self {
            norm: Norm::load(weights, &format!("{prefix}.norm"), device, dtype)?,
            fc1: Lin::load(weights, &format!("{prefix}.linear_fc1"), device, dtype, true)?,
            fc2: Lin::load(weights, &format!("{prefix}.linear_fc2"), device, dtype, true)?,
        })
    }

    fn forward(&self, x: &Tensor, eps: f32) -> Result<Tensor, VisionError> {
        let h = self.norm.forward(x, eps)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_tanh().map_err(|e| VisionError::Forward(e.to_string()))?;
        self.fc2.forward(&h)
    }
}

impl VisionTower {
    pub fn build(
        config: VisionConfig,
        weights: &dyn VisionWeights,
        device: Device,
        dtype: DType,
    ) -> Result<Self, VisionError> {
        let patch_raw = weights.tensor(
            &format!("{VIS}.patch_embed.proj.weight"),
            device,
            dtype,
        )?;
        let feats = config.patch_features();
        let flat = patch_raw
            .reshape(vec![config.hidden_size, feats])
            .map_err(|e| VisionError::Load(e.to_string()))?;
        let patch_embed = Lin {
            wt: flat
                .transpose(0, 1)
                .and_then(|t| t.contiguous())
                .map_err(|e| VisionError::Load(e.to_string()))?,
            bias: Some(weights.tensor(&format!("{VIS}.patch_embed.proj.bias"), device, dtype)?),
        };

        let pos = weights.tensor(&format!("{VIS}.pos_embed.weight"), Device::Cpu, DType::F32)?;
        let pos_embed = pos
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| VisionError::Load(e.to_string()))?;

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let p = format!("{VIS}.blocks.{i}");
            blocks.push(Block {
                norm1: Norm::load(weights, &format!("{p}.norm1"), device, dtype)?,
                norm2: Norm::load(weights, &format!("{p}.norm2"), device, dtype)?,
                qkv: Lin::load(weights, &format!("{p}.attn.qkv"), device, dtype, true)?,
                proj: Lin::load(weights, &format!("{p}.attn.proj"), device, dtype, true)?,
                fc1: Lin::load(weights, &format!("{p}.mlp.linear_fc1"), device, dtype, true)?,
                fc2: Lin::load(weights, &format!("{p}.mlp.linear_fc2"), device, dtype, true)?,
            });
        }

        let use_rope = std::env::var("SYN_QWEN3VL_ROPE")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);

        let mut deepstack = Vec::with_capacity(config.deepstack_visual_indexes.len());
        for i in 0..config.deepstack_visual_indexes.len() {
            deepstack.push(Merger::load(
                weights,
                &format!("{VIS}.deepstack_merger_list.{i}"),
                device,
                dtype,
            )?);
        }

        Ok(Self {
            merger_norm: Norm::load(weights, &format!("{VIS}.merger.norm"), device, dtype)?,
            merger_fc1: Lin::load(weights, &format!("{VIS}.merger.linear_fc1"), device, dtype, true)?,
            merger_fc2: Lin::load(weights, &format!("{VIS}.merger.linear_fc2"), device, dtype, true)?,
            config,
            device,
            dtype,
            patch_embed,
            pos_embed,
            blocks,
            deepstack,
            use_rope,
        })
    }

    pub fn token_coords(&self, grid: ImageGrid) -> Vec<(usize, usize)> {
        let m = self.config.spatial_merge_size;
        let mut out = Vec::with_capacity(grid.patches());
        for _t in 0..grid.t {
            for bh in 0..grid.h / m {
                for bw in 0..grid.w / m {
                    for mh in 0..m {
                        for mw in 0..m {
                            out.push((bh * m + mh, bw * m + mw));
                        }
                    }
                }
            }
        }
        out
    }

    fn interpolated_pos(&self, grid: ImageGrid) -> Vec<f32> {
        let side = self.config.pos_grid();
        let dim = self.config.hidden_size;
        let coords = self.token_coords(grid);
        let mut out = vec![0f32; coords.len() * dim];
        for (i, (ph, pw)) in coords.iter().enumerate() {
            let sy = ((*ph as f32 + 0.5) * side as f32 / grid.h as f32 - 0.5).max(0.0);
            let sx = ((*pw as f32 + 0.5) * side as f32 / grid.w as f32 - 0.5).max(0.0);
            let y0 = (sy.floor() as usize).min(side - 1);
            let x0 = (sx.floor() as usize).min(side - 1);
            let y1 = (y0 + 1).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            let dy = sy - y0 as f32;
            let dx = sx - x0 as f32;
            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;
            let base = i * dim;
            let r00 = (y0 * side + x0) * dim;
            let r01 = (y0 * side + x1) * dim;
            let r10 = (y1 * side + x0) * dim;
            let r11 = (y1 * side + x1) * dim;
            for d in 0..dim {
                out[base + d] = w00 * self.pos_embed[r00 + d]
                    + w01 * self.pos_embed[r01 + d]
                    + w10 * self.pos_embed[r10 + d]
                    + w11 * self.pos_embed[r11 + d];
            }
        }
        out
    }

    fn rope_tables(&self, grid: ImageGrid) -> (Vec<f32>, Vec<f32>) {
        let hd = self.config.head_dim();
        let half = hd / 2;
        let quarter = half / 2;
        let coords = self.token_coords(grid);
        let n = coords.len();
        let mut cos = vec![0f32; n * hd];
        let mut sin = vec![0f32; n * hd];
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / half as f32))
            .collect();
        for (i, (ph, pw)) in coords.iter().enumerate() {
            let base = i * hd;
            for j in 0..quarter {
                let a = *ph as f32 * inv[j];
                let b = *pw as f32 * inv[j];
                for (k, v) in [(j, a), (quarter + j, b)] {
                    cos[base + k] = v.cos();
                    sin[base + k] = v.sin();
                    cos[base + half + k] = v.cos();
                    sin[base + half + k] = v.sin();
                }
            }
        }
        (cos, sin)
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, VisionError> {
        let dims = x.dims().to_vec();
        let hd = dims[dims.len() - 1];
        let half = hd / 2;
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };
        let x1 = e(x.narrow(dims.len() - 1, 0, half).and_then(|t| t.contiguous()))?;
        let x2 = e(x.narrow(dims.len() - 1, half, half).and_then(|t| t.contiguous()))?;
        let neg = e(x2.mul_scalar(-1.0))?;
        let rot = e(Tensor::cat(&[&neg, &x1], dims.len() - 1))?;
        let a = e(x.broadcast_mul(cos))?;
        let b = e(rot.broadcast_mul(sin))?;
        e(a.add(&b))
    }

    pub fn forward(&self, patches: &Tensor, grid: ImageGrid) -> Result<Tensor, VisionError> {
        Ok(self.forward_deepstack(patches, grid)?.0)
    }

    pub fn deepstack_len(&self) -> usize {
        self.deepstack.len()
    }

    /// Прямой проход башни. У видео (`grid.t > 1`) внимание живёт внутри
    /// группы кадров — у HF это `cu_seqlens` по t, — поэтому группы
    /// кодируются независимо и склеиваются по порядку; так же склеиваются
    /// deepstack-фичи каждого слота.
    pub fn forward_deepstack(
        &self,
        patches: &Tensor,
        grid: ImageGrid,
    ) -> Result<(Tensor, Vec<Tensor>), VisionError> {
        if grid.t <= 1 {
            return self.forward_group(patches, grid);
        }
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|err| VisionError::Forward(err.to_string()))
        };
        let per = grid.h * grid.w;
        let single = ImageGrid { t: 1, h: grid.h, w: grid.w };
        let mut outs: Vec<Tensor> = Vec::with_capacity(grid.t);
        let mut taps: Vec<Vec<Tensor>> = Vec::new();
        for g in 0..grid.t {
            let part = e(e(patches.narrow(0, g * per, per))?.contiguous())?;
            let (o, t) = self.forward_group(&part, single)?;
            outs.push(o);
            for (slot, feat) in t.into_iter().enumerate() {
                while taps.len() <= slot {
                    taps.push(Vec::new());
                }
                taps[slot].push(feat);
            }
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let out = e(Tensor::cat(&refs, 0))?;
        let mut merged_taps = Vec::with_capacity(taps.len());
        for slot in &taps {
            let refs: Vec<&Tensor> = slot.iter().collect();
            merged_taps.push(e(Tensor::cat(&refs, 0))?);
        }
        Ok((out, merged_taps))
    }

    fn forward_group(
        &self,
        patches: &Tensor,
        grid: ImageGrid,
    ) -> Result<(Tensor, Vec<Tensor>), VisionError> {
        let cfg = &self.config;
        let n = grid.patches();
        let hd = cfg.head_dim();
        let nh = cfg.num_heads;
        let eps = cfg.layer_norm_eps;
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };

        let x = patches
            .to_dtype(self.dtype)
            .map_err(|err| VisionError::Forward(err.to_string()))?;
        let mut x = self.patch_embed.forward(&x)?;

        let pos = self.interpolated_pos(grid);
        let pos = e(Tensor::from_vec(pos, vec![n, cfg.hidden_size], self.device))?;
        let pos = e(pos.to_dtype(self.dtype))?;
        x = e(x.add(&pos))?;

        let rope = if self.use_rope {
            let (cos, sin) = self.rope_tables(grid);
            let cos = e(Tensor::from_vec(cos, vec![1, n, hd], self.device))?;
            let sin = e(Tensor::from_vec(sin, vec![1, n, hd], self.device))?;
            Some((e(cos.to_dtype(DType::F32))?, e(sin.to_dtype(DType::F32))?))
        } else {
            None
        };

        let scale = 1.0 / (hd as f32).sqrt();
        let mut taps: Vec<Tensor> = Vec::with_capacity(self.deepstack.len());
        for (bi, blk) in self.blocks.iter().enumerate() {
            let residual = x.clone();
            let h = blk.norm1.forward(&x, eps)?;
            let qkv = blk.qkv.forward(&h)?;
            let qkv = e(qkv.reshape(vec![n, 3, nh, hd]))?;
            let take = |i: usize| -> Result<Tensor, VisionError> {
                let t = e(e(qkv.narrow(1, i, 1))?.contiguous())?;
                let t = e(t.reshape(vec![n, nh, hd]))?;
                let t = e(t.permute(vec![1, 0, 2]))?;
                e(t.contiguous())
            };
            let (mut q, mut k, v) = (take(0)?, take(1)?, take(2)?);
            if let Some((cos, sin)) = &rope {
                let qf = e(q.to_dtype(DType::F32))?;
                let kf = e(k.to_dtype(DType::F32))?;
                q = e(self.apply_rope(&qf, cos, sin)?.to_dtype(self.dtype))?;
                k = e(self.apply_rope(&kf, cos, sin)?.to_dtype(self.dtype))?;
            }
            let attn = attention_chunked(&q, &k, &v, scale, ATTN_Q_CHUNK)?;
            let attn = e(attn.permute(vec![1, 0, 2]))?;
            let attn = e(e(attn.contiguous())?.reshape(vec![n, cfg.hidden_size]))?;
            let attn = blk.proj.forward(&attn)?;
            x = e(residual.add(&attn))?;

            let residual = x.clone();
            let h = blk.norm2.forward(&x, eps)?;
            let h = blk.fc1.forward(&h)?;
            let h = e(h.gelu_tanh())?;
            let h = blk.fc2.forward(&h)?;
            x = e(residual.add(&h))?;

            if let Some(slot) = cfg.deepstack_visual_indexes.iter().position(|d| *d == bi) {
                let t = e(e(x.contiguous())?.reshape(vec![n / cfg.merge_unit(), cfg.merged_dim()]))?;
                let feat = self.deepstack[slot].forward(&t, eps)?;
                while taps.len() <= slot {
                    taps.push(feat.clone());
                }
                taps[slot] = feat;
            }
        }

        let x = self.merger_norm.forward(&x, eps)?;
        let x = e(x.contiguous())?;
        let merged = e(x.reshape(vec![n / cfg.merge_unit(), cfg.merged_dim()]))?;
        let h = self.merger_fc1.forward(&merged)?;
        let h = e(h.gelu_tanh())?;
        Ok((self.merger_fc2.forward(&h)?, taps))
    }
}

/// Сколько query-строк внимания считать за один вызов `scaled_dot_attention`.
///
/// Тот держит матрицу score'ов `[heads, Nq, Nk]` в F32 целиком: на картинке
/// в ~1000 vision-токенов (≈4000 патчей) это ≈1 ГБ на score'ы и столько же на
/// softmax — поверх весов LLM такой цельный кусок из фрагментированного
/// пула не выкраивается (`alloc_uninit(999571456) … OOM` при 4 ГБ
/// «свободной» VRAM). Softmax построчный, поэтому разрез по запросам
/// результата не меняет, а пик падает пропорционально чанку.
pub const ATTN_Q_CHUNK: usize = 1024;

/// Внимание `[heads, N, hd]` с разрезом по строкам запросов (см.
/// [`ATTN_Q_CHUNK`]). При `chunk >= N` — один вызов без копий.
pub fn attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    chunk: usize,
) -> Result<Tensor, VisionError> {
    let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
        r.map_err(|err| VisionError::Forward(err.to_string()))
    };
    let n = q.dims()[1];
    let chunk = chunk.max(1);
    if n <= chunk {
        return e(scaled_dot_attention(q, k, v, scale, None));
    }
    let mut parts: Vec<Tensor> = Vec::with_capacity(n.div_ceil(chunk));
    let mut off = 0usize;
    while off < n {
        let len = chunk.min(n - off);
        let qc = e(e(q.narrow(1, off, len))?.contiguous())?;
        parts.push(e(scaled_dot_attention(&qc, k, v, scale, None))?);
        off += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    e(Tensor::cat(&refs, 1))
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

    /// Разрез по запросам не меняет результат: softmax построчный.
    #[test]
    fn chunked_attention_matches_single_shot() {
        synaptix_kernels_cpu::ensure_registered();
        let (nh, n, hd) = (2usize, 37usize, 8usize);
        let mut seed = 12345u32;
        let mut rnd = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };
        let mk = |data: Vec<f32>| Tensor::from_vec(data, vec![nh, n, hd], Device::Cpu).unwrap();
        let (q, k, v) = (mk(rnd(nh * n * hd)), mk(rnd(nh * n * hd)), mk(rnd(nh * n * hd)));
        let scale = 1.0 / (hd as f32).sqrt();
        let full = attention_chunked(&q, &k, &v, scale, usize::MAX).unwrap();
        let chunked = attention_chunked(&q, &k, &v, scale, 7).unwrap();
        assert_eq!(full.dims(), chunked.dims());
        let a = full.to_vec3::<f32>().unwrap();
        let b = chunked.to_vec3::<f32>().unwrap();
        for (ra, rb) in a.iter().flatten().zip(b.iter().flatten()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-5, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn token_coords_follow_merge_blocks() {
        let cfg = VisionConfig::default();
        let tower = VisionTower {
            config: cfg.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            patch_embed: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            pos_embed: vec![0.0; cfg.num_position_embeddings * cfg.hidden_size],
            blocks: Vec::new(),
            merger_norm: Norm {
                weight: Tensor::zeros(vec![1], DType::F32, Device::Cpu).unwrap(),
                bias: Tensor::zeros(vec![1], DType::F32, Device::Cpu).unwrap(),
            },
            merger_fc1: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            merger_fc2: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            deepstack: Vec::new(),
            use_rope: false,
        };
        let grid = ImageGrid { t: 1, h: 4, w: 4 };
        let c = tower.token_coords(grid);
        assert_eq!(c.len(), 16);
        assert_eq!(c[0], (0, 0));
        assert_eq!(c[1], (0, 1));
        assert_eq!(c[2], (1, 0));
        assert_eq!(c[3], (1, 1));
        assert_eq!(c[4], (0, 2));
        assert_eq!(c[15], (3, 3));
    }

    #[test]
    fn rope_tables_duplicate_halves() {
        let cfg = VisionConfig::default();
        let tower = VisionTower {
            config: cfg.clone(),
            device: Device::Cpu,
            dtype: DType::F32,
            patch_embed: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            pos_embed: vec![0.0; cfg.num_position_embeddings * cfg.hidden_size],
            blocks: Vec::new(),
            merger_norm: Norm {
                weight: Tensor::zeros(vec![1], DType::F32, Device::Cpu).unwrap(),
                bias: Tensor::zeros(vec![1], DType::F32, Device::Cpu).unwrap(),
            },
            merger_fc1: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            merger_fc2: Lin { wt: Tensor::zeros(vec![1, 1], DType::F32, Device::Cpu).unwrap(), bias: None },
            deepstack: Vec::new(),
            use_rope: true,
        };
        let grid = ImageGrid { t: 1, h: 2, w: 2 };
        let (cos, sin) = tower.rope_tables(grid);
        let hd = cfg.head_dim();
        let half = hd / 2;
        assert_eq!(cos.len(), 4 * hd);
        for i in 0..4 {
            for d in 0..half {
                assert_eq!(cos[i * hd + d], cos[i * hd + half + d]);
                assert_eq!(sin[i * hd + d], sin[i * hd + half + d]);
            }
        }
        assert_eq!(cos[0], 1.0);
        assert_eq!(sin[0], 0.0);
    }
}
