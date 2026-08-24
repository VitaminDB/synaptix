use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::config::DiffusionHeadConfig;
use crate::loader::WeightSource;
use crate::{err, Result};

const FREQ_EMBED_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10_000.0;

fn rms_norm_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let orig = x.dtype();
    let xf = x.to_dtype(DType::F32).map_err(err)?;
    let last = xf.rank() - 1;
    let ms = xf.sqr().and_then(|t| t.mean_keepdim(last)).map_err(err)?;
    let inv = ms.affine(1.0, eps).and_then(|t| t.powf(-0.5)).map_err(err)?;
    xf.broadcast_mul(&inv)
        .and_then(|t| t.to_dtype(orig))
        .map_err(err)
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let s = scale.affine(1.0, 1.0).map_err(err)?;
    x.broadcast_mul(&s)
        .and_then(|t| t.broadcast_add(shift))
        .map_err(err)
}

fn chunk_last(x: &Tensor, parts: usize) -> Result<Vec<Tensor>> {
    let last = x.rank() - 1;
    let width = x.dims()[last] / parts;
    let mut out = Vec::with_capacity(parts);
    for i in 0..parts {
        out.push(
            x.narrow(last, i * width, width)
                .and_then(|t| t.contiguous())
                .map_err(err)?,
        );
    }
    Ok(out)
}

struct HeadLayer {
    norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    adaln: Tensor,
}

impl HeadLayer {
    fn load(src: &dyn WeightSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: src.get(&format!("{prefix}.norm.weight"))?,
            gate_proj: src.get(&format!("{prefix}.ffn.gate_proj.weight"))?,
            up_proj: src.get(&format!("{prefix}.ffn.up_proj.weight"))?,
            down_proj: src.get(&format!("{prefix}.ffn.down_proj.weight"))?,
            adaln: src.get(&format!("{prefix}.adaLN_modulation.1.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor, c_act: &Tensor, eps: f32) -> Result<Tensor> {
        let m = c_act.linear(&self.adaln).map_err(err)?;
        let parts = chunk_last(&m, 3)?;
        let h = rms_norm(x, &self.norm, eps).map_err(err)?;
        let h = modulate(&h, &parts[0], &parts[1])?;
        let gate = h.linear(&self.gate_proj).and_then(|t| t.silu()).map_err(err)?;
        let up = h.linear(&self.up_proj).map_err(err)?;
        let ffn = gate
            .mul(&up)
            .and_then(|t| t.linear(&self.down_proj))
            .map_err(err)?;
        x.add(&parts[2].mul(&ffn).map_err(err)?).map_err(err)
    }
}

pub struct DiffusionHead {
    noisy_proj: Tensor,
    cond_proj: Tensor,
    t_mlp0: Tensor,
    t_mlp2: Tensor,
    layers: Vec<HeadLayer>,
    final_linear: Tensor,
    final_adaln: Tensor,
    eps: f32,
    pub latent_size: usize,
    pub hidden_size: usize,
}

impl DiffusionHead {
    pub fn load(src: &dyn WeightSource, cfg: &DiffusionHeadConfig, prefix: &str) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            layers.push(HeadLayer::load(src, &format!("{prefix}.layers.{i}"))?);
        }
        Ok(Self {
            noisy_proj: src.get(&format!("{prefix}.noisy_images_proj.weight"))?,
            cond_proj: src.get(&format!("{prefix}.cond_proj.weight"))?,
            t_mlp0: src.get(&format!("{prefix}.t_embedder.mlp.0.weight"))?,
            t_mlp2: src.get(&format!("{prefix}.t_embedder.mlp.2.weight"))?,
            layers,
            final_linear: src.get(&format!("{prefix}.final_layer.linear.weight"))?,
            final_adaln: src.get(&format!("{prefix}.final_layer.adaLN_modulation.1.weight"))?,
            eps: cfg.rms_norm_eps,
            latent_size: cfg.latent_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn timestep_embedding(&self, timesteps: &[f32], device: synaptix_core::device::Device, dtype: DType) -> Result<Tensor> {
        let half = FREQ_EMBED_SIZE / 2;
        let mut data = vec![0f32; timesteps.len() * FREQ_EMBED_SIZE];
        for (bi, t) in timesteps.iter().enumerate() {
            for i in 0..half {
                let freq = (-MAX_PERIOD.ln() * (i as f64) / (half as f64)).exp();
                let arg = (*t as f64) * freq;
                data[bi * FREQ_EMBED_SIZE + i] = arg.cos() as f32;
                data[bi * FREQ_EMBED_SIZE + half + i] = arg.sin() as f32;
            }
        }
        Tensor::from_vec(data, vec![timesteps.len(), FREQ_EMBED_SIZE], device)
            .map_err(err)?
            .to_dtype(dtype)
            .map_err(err)
    }

    pub fn forward(&self, noisy: &Tensor, timesteps: &[f32], condition: &Tensor) -> Result<Tensor> {
        let dtype = self.noisy_proj.dtype();
        let device = self.noisy_proj.device();
        let x = noisy
            .to_dtype(dtype)
            .and_then(|t| t.linear(&self.noisy_proj))
            .map_err(err)?;
        let tfreq = self.timestep_embedding(timesteps, device, dtype)?;
        let temb = tfreq
            .linear(&self.t_mlp0)
            .and_then(|t| t.silu())
            .and_then(|t| t.linear(&self.t_mlp2))
            .map_err(err)?;
        let cond = condition
            .to_dtype(dtype)
            .and_then(|t| t.linear(&self.cond_proj))
            .map_err(err)?;
        let c = cond.broadcast_add(&temb).map_err(err)?;
        let c_act = c.silu().map_err(err)?;

        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h, &c_act, self.eps)?;
        }
        let m = c_act.linear(&self.final_adaln).map_err(err)?;
        let parts = chunk_last(&m, 2)?;
        let normed = rms_norm_plain(&h, self.eps)?;
        let modulated = modulate(&normed, &parts[0], &parts[1])?;
        modulated.linear(&self.final_linear).map_err(err)
    }
}
