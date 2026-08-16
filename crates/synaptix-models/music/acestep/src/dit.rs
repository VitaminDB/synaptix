
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv_transpose1d;

use crate::config::DitConfig;
use crate::encoder::{apply_rope, repeat_kv, rms_norm, rope_tables};
use crate::loader::CompLoader;
use crate::AceError;

type R<T> = Result<T, AceError>;

fn lin(ck: &CompLoader, name: &str, bias: bool, dt: DType) -> R<Linear> {
    let w = ck.get(&format!("{name}.weight"), dt)?;
    let b = if bias { Some(ck.get(&format!("{name}.bias"), dt)?) } else { None };
    Linear::new(w, b).map_err(AceError::Tensor)
}

fn qlin(ck: &CompLoader, name: &str, bias: bool, qdt: DType, compute: DType) -> R<QuantLinear> {
    let w = ck.get(&format!("{name}.weight"), compute)?;
    let b = if bias { Some(ck.get(&format!("{name}.bias"), compute)?) } else { None };
    QuantLinear::build(w, b, qdt, compute).map_err(AceError::Tensor)
}

fn bidir_sliding_mask(s: usize, window: usize, device: Device) -> R<Tensor> {
    let mut data = vec![0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if i.abs_diff(j) > window {
                data[i * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, vec![s, s], device)?)
}

struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    time_proj: Linear,
    hidden: usize,
    dt: DType,
}

impl TimeEmbed {
    fn load(ck: &CompLoader, prefix: &str, hidden: usize, dt: DType) -> R<Self> {
        Ok(Self {
            linear_1: lin(ck, &format!("{prefix}.linear_1"), true, dt)?,
            linear_2: lin(ck, &format!("{prefix}.linear_2"), true, dt)?,
            time_proj: lin(ck, &format!("{prefix}.time_proj"), true, dt)?,
            hidden,
            dt,
        })
    }

    fn sinusoidal(t: f32, device: Device) -> R<Tensor> {
        let half = 128usize;
        let mut e = vec![0f32; 256];
        let ln = (10000f32).ln();
        for i in 0..half {
            let freq = (-ln * i as f32 / half as f32).exp();
            let arg = t * 1000.0 * freq;
            e[i] = arg.cos();
            e[half + i] = arg.sin();
        }
        Ok(Tensor::from_vec(e, vec![1usize, 256usize], device)?)
    }

    fn forward(&self, t: f32, device: Device) -> R<(Tensor, Tensor)> {
        let f = Self::sinusoidal(t, device)?.to_dtype(self.dt)?;
        let temb = self.linear_2.forward(&self.linear_1.forward(&f).map_err(AceError::Tensor)?.silu()?).map_err(AceError::Tensor)?;
        let proj = self.time_proj.forward(&temb.silu()?).map_err(AceError::Tensor)?;
        let proj = proj.reshape(vec![1usize, 6usize, self.hidden])?;
        Ok((temb, proj))
    }
}

struct Attn {
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
}

impl Attn {
    #[allow(clippy::too_many_arguments)]
    fn load(ck: &CompLoader, prefix: &str, nh: usize, nkv: usize, hd: usize, eps: f32, dt: DType, qdt: DType) -> R<Self> {
        Ok(Self {
            q_proj: qlin(ck, &format!("{prefix}.q_proj"), false, qdt, dt)?,
            k_proj: qlin(ck, &format!("{prefix}.k_proj"), false, qdt, dt)?,
            v_proj: qlin(ck, &format!("{prefix}.v_proj"), false, qdt, dt)?,
            o_proj: qlin(ck, &format!("{prefix}.o_proj"), false, qdt, dt)?,
            q_norm: ck.get(&format!("{prefix}.q_norm.weight"), dt)?,
            k_norm: ck.get(&format!("{prefix}.k_norm.weight"), dt)?,
            nh,
            nkv,
            hd,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, kv: &Tensor, rope: Option<(&Tensor, &Tensor)>, mask: Option<&Tensor>, window: usize) -> R<Tensor> {
        let dx = x.dims().to_vec();
        let (n, sx) = (dx[0], dx[1]);
        let sk = kv.dims()[1];
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);

        let q = self.q_proj.forward(x).map_err(AceError::Tensor)?.contiguous()?.reshape(vec![n, sx, nh, hd])?;
        let q = rms_norm(&q, &self.q_norm, self.eps)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(kv).map_err(AceError::Tensor)?.contiguous()?.reshape(vec![n, sk, nkv, hd])?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(kv).map_err(AceError::Tensor)?.contiguous()?.reshape(vec![n, sk, nkv, hd])?.transpose(1, 2)?.contiguous()?;

        let (q, k) = match rope {
            Some((cos, sin)) => (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?),
            None => (q, k),
        };
        let scale = 1.0 / (hd as f32).sqrt();
        let qd = q.dtype();
        let rep = |t: &Tensor| -> R<Tensor> { repeat_kv(t, nh / nkv).map_err(Into::into) };
        let attn = match mask {
            Some(m) if window > 0 && hd == 128 && matches!(q.device(), Device::Cuda(_)) => {
                let (qb, kb, vb) = if matches!(qd, DType::BF16 | DType::F16) {
                    (q.clone(), k.clone(), v.clone())
                } else {
                    (q.to_dtype(DType::BF16)?, k.to_dtype(DType::BF16)?, v.to_dtype(DType::BF16)?)
                };
                match qb.flash_attention_window(&kb, &vb, scale, window as i32, false) {
                    Ok(a) => a.to_dtype(qd)?,
                    Err(_) => scaled_dot_attention(&q, &rep(&k)?, &rep(&v)?, scale, Some(m))?,
                }
            }
            Some(m) => scaled_dot_attention(&q, &rep(&k)?, &rep(&v)?, scale, Some(m))?,
            None if matches!(q.device(), Device::Cuda(_)) => {
                let (qb, kb, vb) = if matches!(qd, DType::BF16 | DType::F16) {
                    (q.clone(), k.clone(), v.clone())
                } else {
                    (q.to_dtype(DType::BF16)?, k.to_dtype(DType::BF16)?, v.to_dtype(DType::BF16)?)
                };
                match qb.flash_attention(&kb, &vb, scale, false) {
                    Ok(a) => a.to_dtype(qd)?,
                    Err(_) => scaled_dot_attention(&q, &rep(&k)?, &rep(&v)?, scale, None)?,
                }
            }
            None => scaled_dot_attention(&q, &rep(&k)?, &rep(&v)?, scale, None)?,
        };
        let attn = attn.transpose(1, 2)?.contiguous()?.reshape(vec![n, sx, nh * hd])?;
        self.o_proj.forward(&attn).map_err(AceError::Tensor)
    }
}

struct DitLayer {
    self_attn_norm: Tensor,
    self_attn: Attn,
    cross_attn_norm: Tensor,
    cross_attn: Attn,
    mlp_norm: Tensor,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
    scale_shift: Tensor,
    eps: f32,
    is_sliding: bool,
    sliding_window: usize,
}

impl DitLayer {
    fn load(ck: &CompLoader, prefix: &str, cfg: &DitConfig, dt: DType, is_sliding: bool, qdt: DType) -> R<Self> {
        let (nh, nkv, hd, eps) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rms_norm_eps as f32,
        );
        Ok(Self {
            self_attn_norm: ck.get(&format!("{prefix}.self_attn_norm.weight"), dt)?,
            self_attn: Attn::load(ck, &format!("{prefix}.self_attn"), nh, nkv, hd, eps, dt, qdt)?,
            cross_attn_norm: ck.get(&format!("{prefix}.cross_attn_norm.weight"), dt)?,
            cross_attn: Attn::load(ck, &format!("{prefix}.cross_attn"), nh, nkv, hd, eps, dt, qdt)?,
            mlp_norm: ck.get(&format!("{prefix}.mlp_norm.weight"), dt)?,
            gate: qlin(ck, &format!("{prefix}.mlp.gate_proj"), false, qdt, dt)?,
            up: qlin(ck, &format!("{prefix}.mlp.up_proj"), false, qdt, dt)?,
            down: qlin(ck, &format!("{prefix}.mlp.down_proj"), false, qdt, dt)?,
            scale_shift: ck.get(&format!("{prefix}.scale_shift_table"), dt)?,
            eps,
            is_sliding,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, tproj: &Tensor, enc: &Tensor, sliding_mask: Option<&Tensor>) -> R<Tensor> {
        let mod6 = self.scale_shift.broadcast_add(tproj)?;
        let m = |i: usize| -> R<Tensor> { Ok(mod6.narrow(1, i, 1)?.contiguous()?) };
        let (shift_msa, scale_msa, gate_msa) = (m(0)?, m(1)?, m(2)?);
        let (c_shift, c_scale, c_gate) = (m(3)?, m(4)?, m(5)?);

        let nh = rms_norm(x, &self.self_attn_norm, self.eps)?
            .broadcast_mul(&scale_msa.affine(1.0, 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let smask = if self.is_sliding { sliding_mask } else { None };
        let win = if self.is_sliding { self.sliding_window } else { 0 };
        let attn = self.self_attn.forward(&nh, &nh, Some((cos, sin)), smask, win)?;
        let x = x.broadcast_add(&attn.broadcast_mul(&gate_msa)?)?;

        let nh = rms_norm(&x, &self.cross_attn_norm, self.eps)?;
        let attn = self.cross_attn.forward(&nh, enc, None, None, 0)?;
        let x = x.broadcast_add(&attn)?;

        let nh = rms_norm(&x, &self.mlp_norm, self.eps)?
            .broadcast_mul(&c_scale.affine(1.0, 1.0)?)?
            .broadcast_add(&c_shift)?;
        let g = self.gate.forward(&nh).map_err(AceError::Tensor)?;
        let u = self.up.forward(&nh).map_err(AceError::Tensor)?;
        let ff = self.down.forward(&g.silu()?.broadcast_mul(&u)?).map_err(AceError::Tensor)?;
        Ok(x.broadcast_add(&ff.broadcast_mul(&c_gate)?)?)
    }
}

pub struct Dit {
    proj_in_w: Tensor,
    proj_in_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    time_embed: TimeEmbed,
    time_embed_r: TimeEmbed,
    condition_embedder: Linear,
    layers: Vec<DitLayer>,
    norm_out: Tensor,
    scale_shift_out: Tensor,
    cfg: DitConfig,
    dt: DType,
    quantized: bool,
}

impl Dit {
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    pub fn load(ck: &CompLoader, cfg: &DitConfig, dt: DType, qdt: DType) -> R<Self> {
        let h = cfg.hidden_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DitLayer::load(ck, &format!("decoder.layers.{i}"), cfg, dt, i % 2 == 0, qdt)?);
        }
        Ok(Self {
            proj_in_w: ck.get("decoder.proj_in.1.weight", dt)?,
            proj_in_b: ck.get("decoder.proj_in.1.bias", dt)?,
            proj_out_w: ck.get("decoder.proj_out.1.weight", dt)?,
            proj_out_b: ck.get("decoder.proj_out.1.bias", dt)?,
            time_embed: TimeEmbed::load(ck, "decoder.time_embed", h, dt)?,
            time_embed_r: TimeEmbed::load(ck, "decoder.time_embed_r", h, dt)?,
            condition_embedder: lin(ck, "decoder.condition_embedder", true, dt)?,
            layers,
            norm_out: ck.get("decoder.norm_out.weight", dt)?,
            scale_shift_out: ck.get("decoder.scale_shift_table", dt)?,
            cfg: cfg.clone(),
            dt,
            quantized: qdt.is_quantized(),
        })
    }

    pub fn compute_temb(&self, timestep: f32, timestep_r: f32, device: Device) -> R<(Tensor, Tensor)> {
        let (temb_t, proj_t) = self.time_embed.forward(timestep, device)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(timestep - timestep_r, device)?;
        Ok((temb_t.broadcast_add(&temb_r)?, proj_t.broadcast_add(&proj_r)?))
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        timestep: f32,
        timestep_r: f32,
        context_latents: &Tensor,
        encoder_hidden: &Tensor,
    ) -> R<Tensor> {
        let (temb, tproj) = self.compute_temb(timestep, timestep_r, hidden.device())?;
        self.forward_with_temb(hidden, &temb, &tproj, context_latents, encoder_hidden)
    }

    pub fn forward_with_temb(
        &self,
        hidden: &Tensor,
        temb: &Tensor,
        tproj: &Tensor,
        context_latents: &Tensor,
        encoder_hidden: &Tensor,
    ) -> R<Tensor> {
        let (h, orig_len) = self.proj_in_h(hidden, context_latents)?;
        let s = h.dims()[1];
        let (enc, cos, sin, sliding_mask) = self.layer_inputs(encoder_hidden, s, hidden.device())?;
        let h = self.forward_layers(&h, temb, tproj, &enc, &cos, &sin, &sliding_mask)?;
        self.proj_out_v(&h, orig_len)
    }

    pub fn proj_in_h(&self, hidden: &Tensor, context_latents: &Tensor) -> R<(Tensor, usize)> {
        let device = hidden.device();
        let patch = self.cfg.patch_size;
        let mut x = Tensor::cat(&[context_latents, hidden], 2)?.to_dtype(self.dt)?;
        let orig_len = x.dims()[1];
        let pad = (patch - orig_len % patch) % patch;
        if pad > 0 {
            let d = x.dims().to_vec();
            let z = Tensor::zeros(vec![d[0], pad, d[2]], x.dtype(), device)?;
            x = Tensor::cat(&[&x, &z], 1)?;
        }
        let (bsz, lpad, cin) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let lp = lpad / patch;
        let cout = self.proj_in_w.dims()[0];
        let w2 = self
            .proj_in_w
            .permute(vec![0, 2, 1])?
            .contiguous()?
            .reshape(vec![cout, patch * cin])?
            .transpose(0, 1)?
            .contiguous()?;
        let xp = x.contiguous()?.reshape(vec![bsz * lp, patch * cin])?;
        let out = xp
            .matmul(&w2)?
            .reshape(vec![bsz, lp, cout])?
            .broadcast_add(&self.proj_in_b.reshape(vec![1, 1, cout])?)?;
        Ok((out, orig_len))
    }

    pub fn layer_inputs(
        &self,
        encoder_hidden: &Tensor,
        s: usize,
        device: Device,
    ) -> R<(Tensor, Tensor, Tensor, Tensor)> {
        let enc = self.condition_embedder.forward(&encoder_hidden.to_dtype(self.dt)?).map_err(AceError::Tensor)?;
        let (cos, sin) = rope_tables(self.cfg.head_dim, s, self.cfg.rope_theta as f32, device)?;
        let (cos, sin) = (cos.to_dtype(self.dt)?, sin.to_dtype(self.dt)?);
        let sliding_mask = bidir_sliding_mask(s, self.cfg.sliding_window, device)?;
        Ok((enc, cos, sin, sliding_mask))
    }

    pub fn forward_layers(
        &self,
        h: &Tensor,
        temb: &Tensor,
        tproj: &Tensor,
        enc: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        sliding_mask: &Tensor,
    ) -> R<Tensor> {
        let mut h = h.clone();
        for layer in &self.layers {
            h = layer.forward(&h, cos, sin, tproj, enc, Some(sliding_mask))?;
        }
        let mod2 = self.scale_shift_out.broadcast_add(&temb.reshape(vec![temb.dims()[0], 1usize, temb.dims()[1]])?)?;
        let shift = mod2.narrow(1, 0, 1)?.contiguous()?;
        let scale = mod2.narrow(1, 1, 1)?.contiguous()?;
        Ok(rms_norm(&h, &self.norm_out, self.cfg.rms_norm_eps as f32)?
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?)
    }

    pub fn proj_out_v(&self, h_normed: &Tensor, orig_len: usize) -> R<Tensor> {
        let patch = self.cfg.patch_size;
        let ht = h_normed.transpose(1, 2)?.contiguous()?;
        let ht = conv_transpose1d(&ht, &self.proj_out_w, Some(&self.proj_out_b), patch, 0, 0, 1, 1)?;
        let out = ht.transpose(1, 2)?.contiguous()?;
        Ok(out.narrow(1, 0, orig_len)?.contiguous()?.to_dtype(DType::F32)?)
    }
}
