use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::config::VisionConfig;
use crate::preprocess::ImageGrid;

pub const VIS_PREFIX: &str = "model.vision_tower";

pub trait VisionWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError>;
}

pub struct BundleVisionWeights {
    loader: synaptix_io::weights::syn_bundle::SynBundleLoader,
}

impl BundleVisionWeights {
    pub fn open(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self, VisionError> {
        let loader = synaptix_io::weights::syn_bundle::SynBundleLoader::open(path)
            .map_err(|e| VisionError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader })
    }

    pub fn has(&self, key: &str) -> bool {
        use synaptix_io::weights::WeightLoader;
        self.loader.names().iter().any(|n| *n == key)
    }
}

impl VisionWeights for BundleVisionWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError> {
        use synaptix_io::weights::WeightLoader;
        self.loader
            .load_to(key, device, dtype)
            .map_err(|e| VisionError::Load(format!("{key}: {e}")))
    }
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
    fn load(w: &dyn VisionWeights, key: &str, device: Device, dtype: DType) -> Result<Self, VisionError> {
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
    q_proj: Lin,
    k_proj: Lin,
    v_proj: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
    full: bool,
}

pub struct VisionTower {
    pub config: VisionConfig,
    pub device: Device,
    pub dtype: DType,
    rms_eps: f32,
    patch_embed: Lin,
    pos_table: Vec<f32>,
    ln_pre: Norm,
    blocks: Vec<Block>,
    ln_post: Norm,
    adapter_fc1: Lin,
    adapter_fc2: Lin,
    projection: Lin,
    ones_hidden: Tensor,
}

pub struct WindowPlan {
    pub index: Vec<u32>,
    pub cu_windows: Vec<usize>,
}

pub fn window_plan(grid: ImageGrid, window: usize) -> WindowPlan {
    let (gh, gw) = (grid.h, grid.w);
    let mut index = Vec::with_capacity(grid.patches());
    let mut cu = vec![0usize];
    let nwh = gh.div_ceil(window);
    let nww = gw.div_ceil(window);
    for t in 0..grid.t {
        let off = (t * gh * gw) as u32;
        for wh in 0..nwh {
            for ww in 0..nww {
                let r0 = wh * window;
                let c0 = ww * window;
                let r1 = (r0 + window).min(gh);
                let c1 = (c0 + window).min(gw);
                let mut count = 0usize;
                for r in r0..r1 {
                    for c in c0..c1 {
                        index.push(off + (r * gw + c) as u32);
                        count += 1;
                    }
                }
                if count > 0 {
                    cu.push(cu.last().unwrap() + count);
                }
            }
        }
    }
    WindowPlan { index, cu_windows: cu }
}

fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16)
}

fn interp_axis(i: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = (i as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = src.floor();
    let mut out = [(0usize, 0f32); 2];
    for (k, off) in [0f32, 1f32].iter().enumerate() {
        let raw = floor + off;
        let dist = (src - raw).abs();
        let mut wgt = (1.0 - dist).max(0.0);
        if raw < 0.0 || raw > (side - 1) as f32 {
            wgt = 0.0;
        }
        let tap = raw.clamp(0.0, (side - 1) as f32) as usize;
        out[k] = (tap, wgt);
    }
    out
}

impl VisionTower {
    pub fn build(
        config: VisionConfig,
        rms_eps: f32,
        weights: &dyn VisionWeights,
        device: Device,
        dtype: DType,
    ) -> Result<Self, VisionError> {
        let p = |s: &str| format!("{VIS_PREFIX}.{s}");
        let patch_embed = Lin::load(weights, &p("patch_embedder.patch_embedding"), device, dtype, false)?;
        let pos = weights.tensor(
            &p("patch_embedder.position_embedding_table.weight"),
            Device::Cpu,
            DType::F32,
        )?;
        let pos_table = pos
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| VisionError::Load(e.to_string()))?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let bp = p(&format!("layers.{i}"));
            blocks.push(Block {
                norm1: Norm::load(weights, &format!("{bp}.norm1"), device, dtype)?,
                norm2: Norm::load(weights, &format!("{bp}.norm2"), device, dtype)?,
                q_proj: Lin::load(weights, &format!("{bp}.attn.q_proj"), device, dtype, true)?,
                k_proj: Lin::load(weights, &format!("{bp}.attn.k_proj"), device, dtype, true)?,
                v_proj: Lin::load(weights, &format!("{bp}.attn.v_proj"), device, dtype, true)?,
                proj: Lin::load(weights, &format!("{bp}.attn.proj"), device, dtype, true)?,
                fc1: Lin::load(weights, &format!("{bp}.mlp.fc1"), device, dtype, true)?,
                fc2: Lin::load(weights, &format!("{bp}.mlp.fc2"), device, dtype, true)?,
                full: config.full_layers[i],
            });
        }

        let projection = Lin::load(weights, "model.vision_projection", device, dtype, false)?;
        let out_dim = projection.wt.dims()[1];
        let ones_hidden = Tensor::from_vec(vec![1.0_f32; out_dim], vec![out_dim], device)
            .and_then(|t| t.to_dtype(dtype))
            .map_err(|e| VisionError::Load(e.to_string()))?;

        Ok(Self {
            ln_pre: Norm::load(weights, &p("ln_pre"), device, dtype)?,
            ln_post: Norm::load(weights, &p("ln_post"), device, dtype)?,
            adapter_fc1: Lin::load(weights, "model.vision_adapter.fc1", device, dtype, false)?,
            adapter_fc2: Lin::load(weights, "model.vision_adapter.fc2", device, dtype, false)?,
            projection,
            config,
            device,
            dtype,
            rms_eps,
            patch_embed,
            pos_table,
            blocks,
            ones_hidden,
        })
    }

    fn interpolated_pos(&self, grid: ImageGrid) -> Vec<f32> {
        let side = self.config.pos_emb_side;
        let dim = self.config.hidden_size;
        let (gh, gw) = (grid.h, grid.w);
        let n = grid.patches();
        let mut out = vec![0f32; n * dim];
        let rows: Vec<[(usize, f32); 2]> = (0..gh).map(|r| interp_axis(r, gh, side)).collect();
        let cols: Vec<[(usize, f32); 2]> = (0..gw).map(|c| interp_axis(c, gw, side)).collect();
        for t in 0..grid.t {
            for r in 0..gh {
                for c in 0..gw {
                    let base = ((t * gh + r) * gw + c) * dim;
                    for (ty, wy) in rows[r] {
                        if wy == 0.0 {
                            continue;
                        }
                        for (tx, wx) in cols[c] {
                            let wgt = wy * wx;
                            if wgt == 0.0 {
                                continue;
                            }
                            let src = (ty * side + tx) * dim;
                            for d in 0..dim {
                                out[base + d] += wgt * self.pos_table[src + d];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn rope_tables(&self, grid: ImageGrid, order: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.config.head_dim();
        let quarter = hd / 4;
        let spatial = hd / 2;
        let theta = self.config.rope_theta;
        let quant = |x: f32| if self.dtype == DType::BF16 { bf16_round(x) } else { x };
        let inv: Vec<f32> = (0..quarter)
            .map(|i| quant(1.0 / theta.powf(2.0 * i as f32 / spatial as f32)))
            .collect();
        let n = order.len();
        let gw = grid.w;
        let frame = grid.h * gw;
        let mut cos = vec![0f32; n * hd];
        let mut sin = vec![0f32; n * hd];
        for (i, tok) in order.iter().enumerate() {
            let within = (*tok as usize) % frame;
            let h_id = (within / gw + 1) as f32;
            let w_id = (within % gw + 1) as f32;
            let base = i * hd;
            for j in 0..quarter {
                let fw = w_id * inv[j];
                let fh = h_id * inv[j];
                for (k, v) in [
                    (j, fw),
                    (quarter + j, fh),
                    (spatial + j, fw),
                    (spatial + quarter + j, fh),
                ] {
                    cos[base + k] = quant(v.cos());
                    sin[base + k] = quant(v.sin());
                }
            }
        }
        (cos, sin)
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, VisionError> {
        let dims = x.dims().to_vec();
        let last = dims.len() - 1;
        let hd = dims[last];
        let half = hd / 2;
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };
        let x1 = e(x.narrow(last, 0, half).and_then(|t| t.contiguous()))?;
        let x2 = e(x.narrow(last, half, half).and_then(|t| t.contiguous()))?;
        let neg = e(x2.mul_scalar(-1.0))?;
        let rot = e(Tensor::cat(&[&neg, &x1], last))?;
        let a = e(x.broadcast_mul(cos))?;
        let b = e(rot.broadcast_mul(sin))?;
        e(a.add(&b))
    }

    fn attention(
        &self,
        blk: &Block,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu: &[usize],
    ) -> Result<Tensor, VisionError> {
        let n = x.dims()[0];
        let nh = self.config.num_attention_heads;
        let hd = self.config.head_dim();
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };
        let split = |t: Tensor| -> Result<Tensor, VisionError> {
            e(t.reshape(vec![n, nh, hd]))
        };
        let q = split(blk.q_proj.forward(x)?)?;
        let k = split(blk.k_proj.forward(x)?)?;
        let v = split(blk.v_proj.forward(x)?)?;
        let qf = e(q.to_dtype(DType::F32))?;
        let kf = e(k.to_dtype(DType::F32))?;
        let q = e(self.apply_rope(&qf, cos, sin)?.to_dtype(self.dtype))?;
        let k = e(self.apply_rope(&kf, cos, sin)?.to_dtype(self.dtype))?;
        let heads_first = |t: &Tensor| -> Result<Tensor, VisionError> {
            e(t.permute(vec![1, 0, 2]).and_then(|t| t.contiguous()))
        };
        let (q, k, v) = (heads_first(&q)?, heads_first(&k)?, heads_first(&v)?);
        let scale = 1.0 / (hd as f32).sqrt();
        let q_chunk = 1024usize;
        let mut outs = Vec::with_capacity(cu.len().saturating_sub(1));
        for wi in 0..cu.len() - 1 {
            let start = cu[wi];
            let len = cu[wi + 1] - start;
            let ks = e(k.narrow(1, start, len).and_then(|t| t.contiguous()))?;
            let vs = e(v.narrow(1, start, len).and_then(|t| t.contiguous()))?;
            let mut off = 0usize;
            while off < len {
                let step = q_chunk.min(len - off);
                let qs = e(q.narrow(1, start + off, step).and_then(|t| t.contiguous()))?;
                let a = scaled_dot_attention(&qs, &ks, &vs, scale, None)
                    .map_err(|e| VisionError::Forward(e.to_string()))?;
                outs.push(a);
                off += step;
            }
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let attn = e(Tensor::cat(&refs, 1))?;
        let attn = e(attn.permute(vec![1, 0, 2]).and_then(|t| t.contiguous()))?;
        let attn = e(attn.reshape(vec![n, nh * hd]))?;
        blk.proj.forward(&attn)
    }

    pub fn forward(&self, patches: &Tensor, grid: ImageGrid) -> Result<Tensor, VisionError> {
        let merged = self.forward_tower(patches, grid)?;
        self.project(&merged)
    }

    pub fn project(&self, merged: &Tensor) -> Result<Tensor, VisionError> {
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };
        let x = self.adapter_fc1.forward(merged)?;
        let x = e(x.gelu_exact())?;
        let x = self.adapter_fc2.forward(&x)?;
        let x = e(x.gelu_exact())?;
        let x = self.projection.forward(&x)?;
        rms_norm(&x, &self.ones_hidden, self.rms_eps)
            .map_err(|e| VisionError::Forward(e.to_string()))
    }

    pub fn forward_tower(&self, patches: &Tensor, grid: ImageGrid) -> Result<Tensor, VisionError> {
        self.forward_tower_probed(patches, grid, None)
    }

    pub fn forward_tower_probed(
        &self,
        patches: &Tensor,
        grid: ImageGrid,
        mut probe: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor, VisionError> {
        let cfg = &self.config;
        let n = grid.patches();
        let eps = cfg.layer_norm_eps;
        let e = |r: Result<Tensor, synaptix_core::error::SynaptixError>| {
            r.map_err(|e| VisionError::Forward(e.to_string()))
        };

        let x = e(patches.to_dtype(self.dtype))?;
        let mut x = self.patch_embed.forward(&x)?;
        let pos = self.interpolated_pos(grid);
        let pos = e(Tensor::from_vec(pos, vec![n, cfg.hidden_size], self.device))?;
        x = e(x.add(&e(pos.to_dtype(self.dtype))?))?;
        x = self.ln_pre.forward(&x, eps)?;

        let plan = window_plan(grid, cfg.window_patches());
        let idx = e(Tensor::from_vec(plan.index.clone(), vec![n], self.device))?;
        x = e(x.index_select(0, &idx))?;

        let frame = grid.h * grid.w;
        let cu_full: Vec<usize> = (0..=grid.t).map(|t| t * frame).collect();
        let (cos, sin) = self.rope_tables(grid, &plan.index);
        let hd = cfg.head_dim();
        let cos = e(Tensor::from_vec(cos, vec![n, 1, hd], self.device))?;
        let sin = e(Tensor::from_vec(sin, vec![n, 1, hd], self.device))?;
        if let Some(p) = probe.as_deref_mut() {
            p.push(("block_input".into(), x.clone()));
            p.push(("rope_cos".into(), cos.clone()));
            p.push(("rope_sin".into(), sin.clone()));
        }

        for (li, blk) in self.blocks.iter().enumerate() {
            let cu = if blk.full { &cu_full } else { &plan.cu_windows };
            let residual = x.clone();
            let h = blk.norm1.forward(&x, eps)?;
            let attn = self.attention(blk, &h, &cos, &sin, cu)?;
            x = e(residual.add(&attn))?;

            let residual = x.clone();
            let h = blk.norm2.forward(&x, eps)?;
            let h = blk.fc1.forward(&h)?;
            let h = e(h.gelu_exact())?;
            let h = blk.fc2.forward(&h)?;
            x = e(residual.add(&h))?;
            if let Some(p) = probe.as_deref_mut() {
                p.push((format!("hidden_{li}"), x.clone()));
            }
        }

        let mut inverse = vec![0u32; n];
        for (j, tok) in plan.index.iter().enumerate() {
            inverse[*tok as usize] = j as u32;
        }
        let inv = e(Tensor::from_vec(inverse, vec![n], self.device))?;
        x = e(x.index_select(0, &inv))?;
        x = self.ln_post.forward(&x, eps)?;

        let m = cfg.merge_size;
        let (gh, gw) = (grid.h, grid.w);
        let mut shuffle = Vec::with_capacity(n);
        for t in 0..grid.t {
            let off = (t * gh * gw) as u32;
            for bh in 0..gh / m {
                for bw in 0..gw / m {
                    for mh in 0..m {
                        for mw in 0..m {
                            shuffle.push(off + ((bh * m + mh) * gw + bw * m + mw) as u32);
                        }
                    }
                }
            }
        }
        let sh = e(Tensor::from_vec(shuffle, vec![n], self.device))?;
        let x = e(x.index_select(0, &sh))?;
        let unit = cfg.merge_unit();
        let x = e(x.reshape(vec![n / unit, unit, cfg.hidden_size]))?;
        let x = e(x.permute(vec![0, 2, 1]).and_then(|t| t.contiguous()))?;
        e(x.reshape(vec![n / unit, cfg.hidden_size * unit]))
    }
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

    #[test]
    fn window_plan_partitions_grid() {
        let plan = window_plan(ImageGrid { t: 1, h: 40, w: 40 }, 32);
        assert_eq!(plan.index.len(), 1600);
        assert_eq!(*plan.cu_windows.last().unwrap(), 1600);
        assert_eq!(plan.cu_windows.len(), 5);
        assert_eq!(plan.cu_windows[1], 32 * 32);
        assert_eq!(plan.cu_windows[2], 32 * 32 + 32 * 8);
        let mut seen = vec![false; 1600];
        for i in &plan.index {
            assert!(!seen[*i as usize]);
            seen[*i as usize] = true;
        }
        assert_eq!(plan.index[0], 0);
        assert_eq!(plan.index[1], 1);
        assert_eq!(plan.index[32], 40);
    }

    #[test]
    fn interp_axis_matches_grid_sample_zeros() {
        let taps = interp_axis(0, 64, 32);
        let total: f32 = taps.iter().map(|(_, w)| w).sum();
        assert!(total < 1.0);
        let taps_mid = interp_axis(31, 64, 32);
        let total_mid: f32 = taps_mid.iter().map(|(_, w)| w).sum();
        assert!((total_mid - 1.0).abs() < 1e-5);
        let same = interp_axis(5, 32, 32);
        assert_eq!(same[0].0, 5);
        assert!((same[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(same[1].1, 0.0);
    }
}
