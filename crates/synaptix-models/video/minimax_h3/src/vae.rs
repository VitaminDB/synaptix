use std::f64::consts::PI;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::config::{VaeConfig, VitDecoderConfig};
use crate::loader::ComponentLoader;
use crate::rope::RopeTables;
use crate::H3Error;

type R<T> = Result<T, SynaptixError>;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const GROUP_NORM_GROUPS: usize = 32;
const GROUP_NORM_EPS: f32 = 1e-6;

fn pad_reflect_last2(x: &Tensor, ph: usize, pw: usize) -> R<Tensor> {
    let mut out = x.clone();
    if pw > 0 {
        out = pad_reflect_dim(&out, 4, pw, pw)?;
    }
    if ph > 0 {
        out = pad_reflect_dim(&out, 3, ph, ph)?;
    }
    Ok(out)
}

fn pad_reflect_dim(x: &Tensor, dim: usize, before: usize, after: usize) -> R<Tensor> {
    if before == 0 && after == 0 {
        return Ok(x.clone());
    }
    let n = x.dims()[dim];
    if n < 2 {
        let mut parts: Vec<Tensor> = Vec::new();
        for _ in 0..before {
            parts.push(x.clone());
        }
        parts.push(x.clone());
        for _ in 0..after {
            parts.push(x.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        return Tensor::cat(&refs, dim);
    }
    let mut parts: Vec<Tensor> = Vec::with_capacity(before + 1 + after);
    for i in (1..=before).rev() {
        let idx = i.min(n - 1);
        parts.push(x.narrow(dim, idx, 1)?.contiguous()?);
    }
    parts.push(x.contiguous()?);
    for i in 1..=after {
        let idx = n.saturating_sub(1 + i).max(0);
        parts.push(x.narrow(dim, idx, 1)?.contiguous()?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, dim)
}

fn pad_zeros_front_time(x: &Tensor, n: usize) -> R<Tensor> {
    if n == 0 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let pad = Tensor::zeros(vec![d[0], d[1], n, d[3], d[4]], x.dtype(), x.device())?;
    Tensor::cat(&[&pad, x], 2)
}

pub struct CausalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    pad: (usize, usize, usize),
    stride: (usize, usize, usize),
}

impl CausalConv3d {
    pub fn load(
        w: &ComponentLoader,
        prefix: &str,
        pad: (usize, usize, usize),
        stride: (usize, usize, usize),
        dtype: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            weight: w.get_as(&format!("{prefix}.weight"), dtype)?,
            bias: w.opt(&format!("{prefix}.bias"), dtype)?,
            pad,
            stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        if self.pad.0 == 0 && self.pad.1 == 0 && self.pad.2 == 0 {
            return x.conv3d(&self.weight, self.bias.as_ref(), self.stride, (0, 0, 0));
        }
        let x = pad_reflect_last2(x, self.pad.1, self.pad.2)?;
        if x.dims()[2] == 1 && self.pad.0 > 0 {
            let kd = self.weight.dims()[2];
            let w = self.weight.narrow(2, kd - 1, 1)?.contiguous()?;
            let b = self.bias.as_ref();
            return x.conv3d(&w, b, self.stride, (0, 0, 0));
        }
        let x = pad_zeros_front_time(&x, self.pad.0 * 2)?;
        x.conv3d(&self.weight, self.bias.as_ref(), self.stride, (0, 0, 0))
    }
}

pub struct TemporalGroupNorm {
    weight: Tensor,
    bias: Tensor,
}

impl TemporalGroupNorm {
    pub fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        Ok(Self {
            weight: w.get_as(&format!("{prefix}.weight"), dtype)?,
            bias: w.get_as(&format!("{prefix}.bias"), dtype)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let d = x.dims().to_vec();
        let (b, c, t, h, wd) = (d[0], d[1], d[2], d[3], d[4]);
        let merged = x
            .permute([0, 2, 1, 3, 4])?
            .contiguous()?
            .reshape(vec![b * t, c, h, wd])?;
        let y = synaptix_ops::norm::group_norm::group_norm(
            &merged,
            Some(&self.weight),
            Some(&self.bias),
            GROUP_NORM_GROUPS,
            GROUP_NORM_EPS,
        )?;
        y.reshape(vec![b, t, c, h, wd])?
            .permute([0, 2, 1, 3, 4])?
            .contiguous()
    }
}

pub struct ResnetBlock3D {
    norm1: TemporalGroupNorm,
    norm2: TemporalGroupNorm,
    conv1: CausalConv3d,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResnetBlock3D {
    pub fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        let shortcut = if w.contains(&format!("{prefix}.nin_shortcut.weight")) {
            Some(CausalConv3d::load(
                w,
                &format!("{prefix}.nin_shortcut"),
                (0, 0, 0),
                (1, 1, 1),
                dtype,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: TemporalGroupNorm::load(w, &format!("{prefix}.norm1"), dtype)?,
            norm2: TemporalGroupNorm::load(w, &format!("{prefix}.norm2"), dtype)?,
            conv1: CausalConv3d::load(w, &format!("{prefix}.conv1"), (1, 1, 1), (1, 1, 1), dtype)?,
            conv2: CausalConv3d::load(w, &format!("{prefix}.conv2"), (1, 1, 1), (1, 1, 1), dtype)?,
            shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        match &self.shortcut {
            Some(sc) => h.add(&sc.forward(x)?),
            None => h.add(x),
        }
    }
}

pub struct Downsample3D {
    conv: CausalConv3d,
    space_stride: usize,
}

impl Downsample3D {
    pub fn load(
        w: &ComponentLoader,
        prefix: &str,
        time_stride: usize,
        space_stride: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            conv: CausalConv3d::load(
                w,
                &format!("{prefix}.conv"),
                (1, 0, 0),
                (time_stride, space_stride, space_stride),
                dtype,
            )?,
            space_stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let x = if self.space_stride == 2 {
            let x = pad_reflect_dim(x, 4, 0, 1)?;
            pad_reflect_dim(&x, 3, 0, 1)?
        } else {
            x.clone()
        };
        self.conv.forward(&x)
    }
}

struct DownLevel {
    blocks: Vec<ResnetBlock3D>,
    downsample: Option<Downsample3D>,
}

pub struct VaeEncoder {
    conv_in: CausalConv3d,
    levels: Vec<DownLevel>,
    norm_out: TemporalGroupNorm,
    conv_out: CausalConv3d,
    quant_conv: CausalConv3d,
    pub cfg: VaeConfig,
    device: Device,
    dtype: DType,
}

impl VaeEncoder {
    pub fn load(
        w: &ComponentLoader,
        cfg: VaeConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut levels = Vec::with_capacity(cfg.num_stages());
        for i in 0..cfg.num_stages() {
            let mut blocks = Vec::with_capacity(cfg.num_res_blocks);
            for j in 0..cfg.num_res_blocks {
                blocks.push(ResnetBlock3D::load(
                    w,
                    &format!("encoder.down.{i}.block.{j}"),
                    dtype,
                )?);
            }
            let downsample = if cfg.space_down[i] * cfg.time_down[i] > 1 {
                Some(Downsample3D::load(
                    w,
                    &format!("encoder.down.{i}.downsample"),
                    cfg.time_down[i],
                    cfg.space_down[i],
                    dtype,
                )?)
            } else {
                None
            };
            levels.push(DownLevel { blocks, downsample });
        }
        Ok(Self {
            conv_in: CausalConv3d::load(w, "encoder.conv_in", (1, 1, 1), (1, 1, 1), dtype)?,
            levels,
            norm_out: TemporalGroupNorm::load(w, "encoder.norm_out", dtype)?,
            conv_out: CausalConv3d::load(w, "encoder.conv_out", (1, 1, 1), (1, 1, 1), dtype)?,
            quant_conv: CausalConv3d::load(w, "quant_conv", (0, 0, 0), (1, 1, 1), dtype)?,
            cfg,
            device,
            dtype,
        })
    }

    pub fn normalize_pixels(&self, x: &Tensor) -> R<Tensor> {
        let mean = Tensor::from_vec(IMAGENET_MEAN.to_vec(), vec![1, 3, 1, 1, 1], self.device)?
            .to_dtype(x.dtype())?;
        let std = Tensor::from_vec(IMAGENET_STD.to_vec(), vec![1, 3, 1, 1, 1], self.device)?
            .to_dtype(x.dtype())?;
        x.add_scalar(1.0)?
            .mul_scalar(0.5)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
    }

    fn moments(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for level in &self.levels {
            for b in &level.blocks {
                h = b.forward(&h)?;
            }
            if let Some(d) = &level.downsample {
                h = d.forward(&h)?;
            }
        }
        let h = self.norm_out.forward(&h)?.silu()?;
        let h = self.conv_out.forward(&h)?;
        self.quant_conv.forward(&h)
    }

    pub fn encode(&self, frames: &Tensor) -> Result<Tensor, H3Error> {
        let x = if frames.rank() == 4 { frames.reshape(shape_insert_t(frames))? } else { frames.clone() };
        let t = x.dims()[2];
        let moments = if t == 1 {
            let m = self.moments(&self.normalize_pixels(&x.to_dtype(self.dtype)?)?)?;
            let mt = m.dims()[2];
            m.narrow(2, mt - 1, 1)?.contiguous()?
        } else {
            self.encode_temporal(&x)?
        };
        let z = moments.to_dtype(DType::F32)?;
        let ch = z.dims()[1] / 2;
        let mean = z.narrow(1, 0, ch)?.contiguous()?;
        let (m, s) = self.latent_stats()?;
        Ok(mean.broadcast_sub(&m)?.broadcast_div(&s)?)
    }

    fn encode_temporal(&self, x: &Tensor) -> Result<Tensor, H3Error> {
        let clip = self.cfg.clip_length;
        let total = x.dims()[2];
        let nclips = total.div_ceil(clip);
        let mut parts = Vec::with_capacity(nclips);
        for i in 0..nclips {
            let start = i * clip;
            let len = clip.min(total - start);
            let mut c = x.narrow(2, start, len)?.contiguous()?;
            if len < clip {
                let last = c.narrow(2, len - 1, 1)?.contiguous()?;
                let mut pads: Vec<&Tensor> = vec![&c];
                let reps: Vec<Tensor> = (0..clip - len).map(|_| last.clone()).collect();
                pads.extend(reps.iter());
                c = Tensor::cat(&pads, 2)?;
            }
            let norm = self.normalize_pixels(&c.to_dtype(self.dtype)?)?;
            parts.push(self.moments(&norm)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let z = Tensor::cat(&refs, 2)?;
        if self.cfg.token_drop > 0 {
            let n = z.dims()[2];
            Ok(z.narrow(2, 0, n - self.cfg.token_drop)?.contiguous()?)
        } else {
            Ok(z)
        }
    }

    fn latent_stats(&self) -> R<(Tensor, Tensor)> {
        let c = self.cfg.z_channels;
        let m = Tensor::from_vec(self.cfg.latents_mean.clone(), vec![1, c, 1, 1, 1], self.device)?;
        let s = Tensor::from_vec(self.cfg.latents_std.clone(), vec![1, c, 1, 1, 1], self.device)?;
        Ok((m, s))
    }
}

fn shape_insert_t(x: &Tensor) -> Vec<usize> {
    let d = x.dims();
    vec![d[0], d[1], 1, d[2], d[3]]
}

pub fn token_ids(patch_dims: [usize; 3]) -> Vec<[f32; 3]> {
    let axes: Vec<Vec<f32>> = patch_dims
        .iter()
        .map(|&n| {
            (0..n)
                .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(patch_dims.iter().product());
    for &t in &axes[0] {
        for &h in &axes[1] {
            for &w in &axes[2] {
                out.push([t, h, w]);
            }
        }
    }
    out
}

pub fn vit_rope_angles(ids: &[[f32; 3]], rope_dim: usize, theta: f32, suffix: usize) -> (Vec<f32>, usize) {
    let n_dim = 3usize;
    let step = 2.0 * n_dim as f32 / rope_dim as f32;
    let mut inv = Vec::new();
    let mut e = 0.0f32;
    while e < 1.0 {
        inv.push(1.0 / theta.powf(e));
        e += step;
    }
    let half = inv.len() * n_dim;
    let scale = 2.0 * PI as f32;
    let total = ids.len() + suffix;
    let mut angles = vec![0f32; total * half];
    for (s, id) in ids.iter().enumerate() {
        for axis in 0..n_dim {
            for (i, f) in inv.iter().enumerate() {
                angles[s * half + axis * inv.len() + i] = scale * id[axis] * f;
            }
        }
    }
    (angles, half)
}

struct VitAttention {
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    heads: usize,
    dim_head: usize,
    eps: f32,
    scale: f32,
}

impl VitAttention {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        cfg: &VitDecoderConfig,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            qkv_w: w.get_as(&format!("{prefix}.to_qkv.weight"), dtype)?,
            qkv_b: w.get_as(&format!("{prefix}.to_qkv.bias"), dtype)?,
            out_w: w.get_as(&format!("{prefix}.to_out.weight"), dtype)?,
            out_b: w.get_as(&format!("{prefix}.to_out.bias"), dtype)?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
            eps: cfg.eps,
            scale: 1.0 / (cfg.dim_head as f32).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeTables) -> R<Tensor> {
        let s = x.dims()[0];
        let qkv = x
            .matmul(&self.qkv_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.qkv_b)?
            .reshape(vec![s, self.heads, 3 * self.dim_head])?;
        let q = qkv.narrow(2, 0, self.dim_head)?.contiguous()?;
        let k = qkv.narrow(2, self.dim_head, self.dim_head)?.contiguous()?;
        let v = qkv.narrow(2, 2 * self.dim_head, self.dim_head)?.contiguous()?;
        drop(qkv);

        let q = rms_no_gain(&q, self.eps)?;
        let k = rms_no_gain(&k, self.eps)?;
        let q = rope
            .apply(&q.transpose(0, 1)?.contiguous()?)
            .map_err(err_tensor)?;
        let k = rope
            .apply(&k.transpose(0, 1)?.contiguous()?)
            .map_err(err_tensor)?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let q = q.reshape(vec![1, self.heads, s, self.dim_head])?;
        let k = k.reshape(vec![1, self.heads, s, self.dim_head])?;
        let v = v.reshape(vec![1, self.heads, s, self.dim_head])?;
        let attn = match q.dtype() {
            DType::BF16 | DType::F16 => q
                .flash_attention(&k, &v, self.scale, false)
                .or_else(|_| scaled_dot_attention(&q, &k, &v, self.scale, None))?,
            _ => scaled_dot_attention(&q, &k, &v, self.scale, None)?,
        };
        let attn = attn
            .reshape(vec![self.heads, s, self.dim_head])?
            .transpose(0, 1)?
            .contiguous()?
            .reshape(vec![s, self.heads * self.dim_head])?;
        attn.matmul(&self.out_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.out_b)
    }
}

fn err_tensor(e: H3Error) -> SynaptixError {
    match e {
        H3Error::Tensor(t) => t,
        other => SynaptixError::Other(other.to_string()),
    }
}

fn rms_no_gain(x: &Tensor, eps: f32) -> R<Tensor> {
    let d = *x.dims().last().unwrap();
    let ones = Tensor::ones(vec![d], x.dtype(), x.device())?;
    if let Ok(y) = x.rms_norm_fused(&ones, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, &ones, eps)
}

struct VitFeedForward {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    inner: usize,
}

impl VitFeedForward {
    fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        let w1 = w.get_as(&format!("{prefix}.w1.weight"), dtype)?;
        let inner = w1.dims()[0] / 2;
        Ok(Self {
            w1,
            b1: w.get_as(&format!("{prefix}.w1.bias"), dtype)?,
            w2: w.get_as(&format!("{prefix}.w2.weight"), dtype)?,
            b2: w.get_as(&format!("{prefix}.w2.bias"), dtype)?,
            inner,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = x
            .matmul(&self.w1.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.b1)?;
        let gate = h.narrow(1, 0, self.inner)?.contiguous()?;
        let up = h.narrow(1, self.inner, self.inner)?.contiguous()?;
        drop(h);
        let act = gate.silu_and_mul(&up)?;
        act.matmul(&self.w2.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.b2)
    }
}

struct VitBlock {
    norm1: Tensor,
    norm2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    attn: VitAttention,
    ff: VitFeedForward,
    eps: f32,
}

impl VitBlock {
    fn load(
        w: &ComponentLoader,
        idx: usize,
        cfg: &VitDecoderConfig,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let p = format!("decoder.transformer_blocks.{idx}");
        Ok(Self {
            norm1: w.get_as(&format!("{p}.norm1.weight"), dtype)?,
            norm2: w.get_as(&format!("{p}.norm2.weight"), dtype)?,
            scale1: w.get_as(&format!("{p}.scale1"), dtype)?,
            scale2: w.get_as(&format!("{p}.scale2"), dtype)?,
            attn: VitAttention::load(w, &format!("{p}.attn"), cfg, dtype)?,
            ff: VitFeedForward::load(w, &format!("{p}.ff"), dtype)?,
            eps: cfg.eps,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeTables) -> R<Tensor> {
        let h = rms_gain(x, &self.norm1, self.eps)?;
        let a = self.attn.forward(&h, rope)?;
        let x = match x.fused_gate_residual(&a, &self.scale1) {
            Ok(y) => y,
            Err(_) => x.add(&a.broadcast_mul(&self.scale1)?)?,
        };
        let h = rms_gain(&x, &self.norm2, self.eps)?;
        let f = self.ff.forward(&h)?;
        match x.fused_gate_residual(&f, &self.scale2) {
            Ok(y) => Ok(y),
            Err(_) => x.add(&f.broadcast_mul(&self.scale2)?),
        }
    }
}

fn rms_gain(x: &Tensor, w: &Tensor, eps: f32) -> R<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
}

pub struct VaeDecoder {
    post_quant: CausalConv3d,
    x_embed_w: Tensor,
    x_embed_b: Tensor,
    register_tokens: Tensor,
    blocks: Vec<VitBlock>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    pub cfg: VaeConfig,
    vit: VitDecoderConfig,
    device: Device,
    dtype: DType,
}

impl VaeDecoder {
    pub fn load(
        w: &ComponentLoader,
        cfg: VaeConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let vit = cfg.vit_decoder_kwargs.clone();
        let mut blocks = Vec::with_capacity(vit.num_layers);
        for i in 0..vit.num_layers {
            blocks.push(VitBlock::load(w, i, &vit, dtype)?);
        }
        Ok(Self {
            post_quant: CausalConv3d::load(w, "post_quant_conv", (0, 0, 0), (1, 1, 1), dtype)?,
            x_embed_w: w.get_as("decoder.x_embedder.weight", dtype)?,
            x_embed_b: w.get_as("decoder.x_embedder.bias", dtype)?,
            register_tokens: w.get_as("decoder.register_tokens", dtype)?,
            blocks,
            norm_out_w: w.get_as("decoder.norm_out.weight", dtype)?,
            norm_out_b: w.get_as("decoder.norm_out.bias", dtype)?,
            proj_out_w: w.get_as("decoder.proj_out.weight", dtype)?,
            proj_out_b: w.get_as("decoder.proj_out.bias", dtype)?,
            cfg,
            vit,
            device,
            dtype,
        })
    }

    fn latent_stats(&self) -> R<(Tensor, Tensor)> {
        let c = self.cfg.z_channels;
        let m = Tensor::from_vec(self.cfg.latents_mean.clone(), vec![1, c, 1, 1, 1], self.device)?;
        let s = Tensor::from_vec(self.cfg.latents_std.clone(), vec![1, c, 1, 1, 1], self.device)?;
        Ok((m, s))
    }

    fn finalize_pixels(&self, x: &Tensor) -> R<Tensor> {
        let mean = Tensor::from_vec(IMAGENET_MEAN.to_vec(), vec![1, 3, 1, 1, 1], x.device())?;
        let std = Tensor::from_vec(IMAGENET_STD.to_vec(), vec![1, 3, 1, 1, 1], x.device())?;
        let y = x.to_dtype(DType::F32)?;
        y.broadcast_mul(&std)?.broadcast_add(&mean)?.clamp(0.0, 1.0)
    }

    fn vit_forward(&self, z: &Tensor) -> R<Tensor> {
        let d = z.dims().to_vec();
        let (_, c, t, h, w) = (d[0], d[1], d[2], d[3], d[4]);
        let s = t * h * w;
        let tokens = z
            .reshape(vec![c, s])?
            .transpose(0, 1)?
            .contiguous()?;
        let mut x = tokens
            .matmul(&self.x_embed_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.x_embed_b)?;
        crate::pipeline::dump_tensor("vit_embed", &x);

        let nreg = self.vit.num_register_tokens;
        let dim = self.vit.dim();
        let reg = self.register_tokens.reshape(vec![nreg, dim])?;
        let zero = Tensor::zeros(vec![1, dim], x.dtype(), x.device())?;
        x = Tensor::cat(&[&x, &reg, &zero], 0)?;

        let ids = token_ids([t, h, w]);
        let (angles, half) =
            vit_rope_angles(&ids, self.vit.rope_dim(), self.vit.rope_theta, nreg + 1);
        let rope = RopeTables::from_angles(angles, s + nreg + 1, half, x.device())
            .map_err(err_tensor)?;

        for (bi, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x, &rope)?;
            if bi == 0 {
                crate::pipeline::dump_tensor("vit_blk0", &x);
            }
        }
        crate::pipeline::dump_tensor("vit_last", &x);

        let x = x.narrow(0, 0, s)?.contiguous()?;
        let x = x.layer_norm_fused(&self.norm_out_w, Some(&self.norm_out_b), self.vit.eps)?;
        let out = x
            .matmul(&self.proj_out_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.proj_out_b)?;

        let oc = self.cfg.out_ch;
        let pt = self.vit.patch_size_t;
        let ps = self.vit.patch_size;
        out.reshape(vec![1, t, h, w, oc, pt, ps, ps])?
            .permute([0, 4, 1, 5, 2, 6, 3, 7])?
            .contiguous()?
            .reshape(vec![1, oc, t * pt, h * ps, w * ps])
    }

    fn decode_pixels(&self, z: &Tensor) -> R<Tensor> {
        self.vit_forward(&self.post_quant.forward(z)?)
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, H3Error> {
        let (m, s) = self.latent_stats()?;
        let z = latent
            .to_dtype(DType::F32)?
            .broadcast_mul(&s)?
            .broadcast_add(&m)?
            .to_dtype(self.dtype)?;
        crate::pipeline::dump_tensor("final_latent", latent);
        if crate::runtime::h3_vae_prof() {
            eprintln!(
                "[h3-vae] {} · {}",
                crate::pipeline::tensor_stats("латент", latent),
                crate::pipeline::tensor_stats("денорм", &z)
            );
        }
        let out = if z.dims()[2] == 1 {
            let dec = self.decode_pixels(&z)?;
            let n = dec.dims()[2];
            self.finalize_pixels(&dec.narrow(2, n - 1, 1)?.contiguous()?)?
        } else {
            self.decode_temporal(&z)?
        };
        if crate::runtime::h3_vae_prof() {
            eprintln!("[h3-vae] {}", crate::pipeline::tensor_stats("rgb", &out));
        }
        Ok(out)
    }

    fn decode_temporal(&self, z: &Tensor) -> Result<Tensor, H3Error> {
        let tcs = self.cfg.tokens_chunk_size();
        let ratio_t = self.cfg.vae_ratio_t;
        let frame_pre_padding = (ratio_t - self.cfg.clip_length % ratio_t) % ratio_t;
        let token_overlap = if self.cfg.token_drop == 0 {
            0
        } else {
            (tcs - self.cfg.token_drop % tcs) % tcs
        };
        let frame_overlap = (token_overlap * ratio_t).saturating_sub(frame_pre_padding);
        let chunk_dec = tcs * ratio_t;
        let split_count = usize::from(self.cfg.token_drop > 0) + 1;

        let z_len = z.dims()[2];
        let pseudo = z_len + self.cfg.token_drop;
        let mut pad_tokens = (tcs - pseudo % tcs) % tcs;
        let mut num_chunks =
            (pseudo + pad_tokens) / tcs - usize::from(self.cfg.token_drop > 0);
        if num_chunks < 1 {
            pad_tokens += tcs;
            num_chunks += 1;
        }

        let mut zc = z.clone();
        if pad_tokens > 0 {
            let last = z.narrow(2, z_len - 1, 1)?.contiguous()?;
            let reps: Vec<Tensor> = (0..pad_tokens).map(|_| last.clone()).collect();
            let mut parts: Vec<&Tensor> = vec![z];
            parts.extend(reps.iter());
            zc = Tensor::cat(&parts, 2)?;
        }
        let padded_len = zc.dims()[2];

        let mut out_parts: Vec<Tensor> = Vec::new();
        let mut overlap: Option<Tensor> = None;

        for i in 0..num_chunks {
            let t_start = i * tcs;
            let t_end = (t_start + tcs + token_overlap).min(padded_len);
            if t_start >= t_end {
                continue;
            }
            let clip = zc.narrow(2, t_start, t_end - t_start)?.contiguous()?;
            let dec = self.decode_pixels(&clip)?;
            if i == 0 {
                crate::pipeline::dump_tensor("chunk_z", &clip);
                crate::pipeline::dump_tensor("chunk_raw", &dec);
            }
            let dec_frames = dec.dims()[2];

            for j in 0..split_count {
                let f_start = j * chunk_dec;
                if f_start >= dec_frames {
                    continue;
                }
                let f_end = (f_start + chunk_dec).min(dec_frames);
                let mut part = dec.narrow(2, f_start, f_end - f_start)?.contiguous()?;
                let pf = part.dims()[2];
                if frame_pre_padding >= pf {
                    continue;
                }
                part = part
                    .narrow(2, frame_pre_padding, pf - frame_pre_padding)?
                    .contiguous()?;
                if j == 0 {
                    if let Some(prev) = overlap.take() {
                        part = blend_time(&prev, &part, frame_overlap)?;
                    }
                    out_parts.push(self.finalize_pixels(&part)?);
                } else {
                    overlap = Some(part);
                }
            }
            if i == num_chunks - 1 {
                if let Some(prev) = overlap.take() {
                    out_parts.push(self.finalize_pixels(&prev)?);
                }
            }
        }

        if out_parts.is_empty() {
            return Err(H3Error::Layout("VAE decode: пустой результат".into()));
        }
        let refs: Vec<&Tensor> = out_parts.iter().collect();
        let full = Tensor::cat(&refs, 2)?;
        let want = self.decode_frame_count(z_len, padded_len, num_chunks, pad_tokens);
        let have = full.dims()[2];
        if want > 0 && have > want {
            return Ok(full.narrow(2, 0, want)?.contiguous()?);
        }
        Ok(full)
    }

    fn decode_frame_count(
        &self,
        z_len: usize,
        padded_len: usize,
        num_chunks: usize,
        pad_tokens: usize,
    ) -> usize {
        let tcs = self.cfg.tokens_chunk_size();
        let ratio_t = self.cfg.vae_ratio_t;
        let pre_pad = (ratio_t - self.cfg.clip_length % ratio_t) % ratio_t;
        let token_overlap = if self.cfg.token_drop == 0 {
            0
        } else {
            (tcs - self.cfg.token_drop % tcs) % tcs
        };
        let chunk_dec = tcs * ratio_t;
        let split_count = usize::from(self.cfg.token_drop > 0) + 1;

        let mut total = 0usize;
        let mut final_overlap = 0usize;
        for i in 0..num_chunks {
            let t_start = i * tcs;
            let t_end = t_start + tcs + token_overlap;
            let clip_tokens = t_end.min(padded_len).saturating_sub(t_start.min(padded_len));
            let clip_frames = clip_tokens * ratio_t;
            for j in 0..split_count {
                let f_start = j * chunk_dec;
                let f_end = (f_start + chunk_dec).min(clip_frames);
                let frames = f_end.saturating_sub(f_start).saturating_sub(pre_pad);
                if j == 0 {
                    total += frames;
                } else {
                    final_overlap = frames;
                }
            }
        }
        total += final_overlap;
        total.saturating_sub(self.decode_pad_frames(padded_len, pad_tokens, z_len))
    }

    fn decode_pad_frames(&self, _padded_len: usize, pad_tokens: usize, z_len: usize) -> usize {
        if pad_tokens == 0 {
            return 0;
        }
        let ratio_t = self.cfg.vae_ratio_t;
        let tcs = self.cfg.tokens_chunk_size();
        let intra_tail = self.cfg.clip_length % ratio_t;
        if intra_tail == 0 {
            return pad_tokens * ratio_t;
        }
        (0..pad_tokens)
            .map(|k| if (z_len + k) % tcs == 0 { intra_tail } else { ratio_t })
            .sum()
    }
}

fn blend_time(a: &Tensor, b: &Tensor, extent: usize) -> R<Tensor> {
    let extent = extent.min(a.dims()[2]).min(b.dims()[2]);
    if extent == 0 {
        return Ok(b.clone());
    }
    let n_a = a.dims()[2];
    let tail = a.narrow(2, n_a - extent, extent)?.contiguous()?;
    let head = b.narrow(2, 0, extent)?.contiguous()?;
    let wa: Vec<f32> = (0..extent).map(|i| 1.0 - i as f32 / extent as f32).collect();
    let wb: Vec<f32> = (0..extent).map(|i| i as f32 / extent as f32).collect();
    let wa = Tensor::from_vec(wa, vec![1, 1, extent, 1, 1], a.device())?.to_dtype(a.dtype())?;
    let wb = Tensor::from_vec(wb, vec![1, 1, extent, 1, 1], b.device())?.to_dtype(b.dtype())?;
    let blended = tail.broadcast_mul(&wa)?.add(&head.broadcast_mul(&wb)?)?;
    let rest = b.dims()[2] - extent;
    if rest == 0 {
        return Ok(blended);
    }
    let tailb = b.narrow(2, extent, rest)?.contiguous()?;
    Tensor::cat(&[&blended, &tailb], 2)
}

pub fn rgb_to_frames(rgb: &Tensor) -> Result<Vec<Tensor>, H3Error> {
    let d = rgb.dims().to_vec();
    let frames = d[2];
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        out.push(
            rgb.narrow(2, f, 1)?
                .contiguous()?
                .reshape(vec![d[1], d[3], d[4]])?,
        );
    }
    Ok(out)
}
