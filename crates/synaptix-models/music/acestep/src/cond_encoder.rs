
use synaptix_core::tensor::Tensor;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::config::DitConfig;
use crate::encoder::{build_sliding_mask, rms_norm, rope_tables, EncoderLayer};
use crate::loader::CompLoader;
use crate::AceError;

struct LyricEncoder {
    embed: Linear,
    layers: Vec<EncoderLayer>,
    norm: Tensor,
    head_dim: usize,
    rope_theta: f32,
    eps: f32,
    window: usize,
}

impl LyricEncoder {
    fn load(ck: &CompLoader, cfg: &DitConfig) -> Result<Self, AceError> {
        let prefix = "encoder.lyric_encoder";
        let embed = Linear::new(
            ck.f32(&format!("{prefix}.embed_tokens.weight"))?,
            Some(ck.f32(&format!("{prefix}.embed_tokens.bias"))?),
        )
        .map_err(AceError::Tensor)?;
        let mut layers = Vec::with_capacity(cfg.num_lyric_encoder_hidden_layers);
        for i in 0..cfg.num_lyric_encoder_hidden_layers {
            layers.push(EncoderLayer::load(
                ck,
                &format!("{prefix}.layers.{i}"),
                cfg.encoder_num_attention_heads,
                cfg.encoder_num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps as f32,
            )?);
        }
        Ok(Self {
            embed,
            layers,
            norm: ck.f32(&format!("{prefix}.norm.weight"))?,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta as f32,
            eps: cfg.rms_norm_eps as f32,
            window: cfg.sliding_window,
        })
    }

    fn forward(&self, lyric_emb: &Tensor) -> Result<Tensor, AceError> {
        let l = lyric_emb.dims()[1];
        let mut h = self.embed.forward(lyric_emb).map_err(AceError::Tensor)?;
        let (cos, sin) = rope_tables(self.head_dim, l, self.rope_theta, h.device())?;
        // Even-indexed layers use bidirectional sliding-window attention (the
        // mask is a no-op when l <= window, so only build it for longer lyrics).
        let mask = if l > self.window {
            Some(build_sliding_mask(l, self.window, h.device())?)
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let m = if i % 2 == 0 { mask.as_ref() } else { None };
            h = layer.forward(&h, &cos, &sin, m)?;
        }
        rms_norm(&h, &self.norm, self.eps)
    }
}

struct TimbreEncoder {
    embed: Linear,
    special: Tensor,
    layers: Vec<EncoderLayer>,
    norm: Tensor,
    head_dim: usize,
    rope_theta: f32,
    eps: f32,
    window: usize,
}

impl TimbreEncoder {
    fn load(ck: &CompLoader, cfg: &DitConfig) -> Result<Self, AceError> {
        let prefix = "encoder.timbre_encoder";
        let embed = Linear::new(
            ck.f32(&format!("{prefix}.embed_tokens.weight"))?,
            Some(ck.f32(&format!("{prefix}.embed_tokens.bias"))?),
        )
        .map_err(AceError::Tensor)?;
        let mut layers = Vec::with_capacity(cfg.num_timbre_encoder_hidden_layers);
        for i in 0..cfg.num_timbre_encoder_hidden_layers {
            layers.push(EncoderLayer::load(
                ck,
                &format!("{prefix}.layers.{i}"),
                cfg.encoder_num_attention_heads,
                cfg.encoder_num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps as f32,
            )?);
        }
        Ok(Self {
            embed,
            special: ck.f32(&format!("{prefix}.special_token"))?,
            layers,
            norm: ck.f32(&format!("{prefix}.norm.weight"))?,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta as f32,
            eps: cfg.rms_norm_eps as f32,
            window: cfg.sliding_window,
        })
    }

    fn forward(&self, ref_latent: &Tensor) -> Result<Tensor, AceError> {
        let x = self.embed.forward(ref_latent).map_err(AceError::Tensor)?;
        let mut h = Tensor::cat(&[&self.special, &x], 1)?.contiguous()?;
        let s = h.dims()[1];
        let (cos, sin) = rope_tables(self.head_dim, s, self.rope_theta, h.device())?;
        // Timbre always sees 751 frames (> window 128) → even layers 0,2 must use
        // the sliding mask so the pooled CLS timbre vector matches training.
        let mask = if s > self.window {
            Some(build_sliding_mask(s, self.window, h.device())?)
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let m = if i % 2 == 0 { mask.as_ref() } else { None };
            h = layer.forward(&h, &cos, &sin, m)?;
        }
        let h = rms_norm(&h, &self.norm, self.eps)?;
        Ok(h.narrow(1, 0, 1)?.contiguous()?)
    }
}

pub struct ConditionEncoder {
    text_projector: Linear,
    lyric_encoder: LyricEncoder,
    timbre_encoder: TimbreEncoder,
}

impl ConditionEncoder {
    pub fn load(ck: &CompLoader, cfg: &DitConfig) -> Result<Self, AceError> {
        let text_projector = Linear::new(ck.f32("encoder.text_projector.weight")?, None)
            .map_err(AceError::Tensor)?;
        Ok(Self {
            text_projector,
            lyric_encoder: LyricEncoder::load(ck, cfg)?,
            timbre_encoder: TimbreEncoder::load(ck, cfg)?,
        })
    }

    pub fn timbre_emb(&self, ref_latent: &Tensor) -> Result<Tensor, AceError> {
        self.timbre_encoder.forward(ref_latent)
    }

    pub fn forward_full(&self, text_hidden: &Tensor, lyric_hidden: &Tensor, timbre_ref: &Tensor) -> Result<Tensor, AceError> {
        let text = self.text_project(text_hidden)?;
        let lyric = self.lyric_encode(lyric_hidden)?;
        let timbre = self.timbre_emb(timbre_ref)?;
        Ok(Tensor::cat(&[&lyric, &timbre, &text], 1)?)
    }

    pub fn forward(&self, text_hidden: &Tensor, lyric_hidden: &Tensor) -> Result<Tensor, AceError> {
        let text = self.text_project(text_hidden)?;
        let lyric = self.lyric_encode(lyric_hidden)?;
        Ok(Tensor::cat(&[&lyric, &text], 1)?)
    }

    pub fn text_project(&self, text_hidden: &Tensor) -> Result<Tensor, AceError> {
        self.text_projector.forward(text_hidden).map_err(AceError::Tensor)
    }

    pub fn lyric_encode(&self, lyric_hidden: &Tensor) -> Result<Tensor, AceError> {
        self.lyric_encoder.forward(lyric_hidden)
    }
}
