
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::config::DitConfig;
use crate::encoder::{rms_norm, rope_tables, EncoderLayer};
use crate::loader::CompLoader;
use crate::AceError;

pub struct Detokenizer {
    embed: Linear,
    special: Tensor,
    layers: Vec<EncoderLayer>,
    norm: Tensor,
    proj_out: Linear,
    cos: Tensor,
    sin: Tensor,
    pool: usize,
    eps: f32,
    out_dim: usize,
}

impl Detokenizer {
    pub fn load(ck: &CompLoader, cfg: &DitConfig) -> Result<Self, AceError> {
        let prefix = "detokenizer";
        let embed = Linear::new(
            ck.f32(&format!("{prefix}.embed_tokens.weight"))?,
            Some(ck.f32(&format!("{prefix}.embed_tokens.bias"))?),
        )
        .map_err(AceError::Tensor)?;
        let special = ck.f32(&format!("{prefix}.special_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(EncoderLayer::load(
                ck,
                &format!("{prefix}.layers.{i}"),
                cfg.encoder_num_attention_heads,
                cfg.encoder_num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps as f32,
            )?);
        }
        let norm = ck.f32(&format!("{prefix}.norm.weight"))?;
        let proj_w = ck.f32(&format!("{prefix}.proj_out.weight"))?;
        let out_dim = proj_w.dims()[0];
        let proj_out = Linear::new(proj_w, Some(ck.f32(&format!("{prefix}.proj_out.bias"))?))
            .map_err(AceError::Tensor)?;
        let (cos, sin) = rope_tables(
            cfg.head_dim,
            cfg.pool_window_size,
            cfg.rope_theta as f32,
            ck.device(),
        )?;
        Ok(Self {
            embed,
            special,
            layers,
            norm,
            proj_out,
            cos,
            sin,
            pool: cfg.pool_window_size,
            eps: cfg.rms_norm_eps as f32,
            out_dim,
        })
    }

    pub fn forward(&self, codes_5hz: &Tensor) -> Result<Tensor, AceError> {
        let codes = codes_5hz.to_dtype(DType::F32)?;
        let d = codes.dims().to_vec();
        let (b, t, hid) = (d[0], d[1], d[2]);
        let p = self.pool;

        let x = self.embed.forward(&codes).map_err(AceError::Tensor)?;
        let x = x.reshape(vec![b, t, 1usize, hid])?.broadcast_add(&self.special)?.contiguous()?;
        let mut h = x.reshape(vec![b * t, p, hid])?;

        // Full attention: the detok window (P=5) is far below the sliding
        // window, so windowing is a no-op here.
        for layer in &self.layers {
            h = layer.forward(&h, &self.cos, &self.sin, None)?;
        }
        let h = rms_norm(&h, &self.norm, self.eps)?;
        let h = self.proj_out.forward(&h).map_err(AceError::Tensor)?;
        Ok(h.contiguous()?.reshape(vec![b, t * p, self.out_dim])?)
    }

    pub fn device(&self) -> Device {
        self.cos.device()
    }
}
