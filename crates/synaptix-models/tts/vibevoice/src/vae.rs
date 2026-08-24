use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::config::{parse_depths, AcousticTokenizerConfig, SemanticTokenizerConfig};
use crate::conv::{ConvIds, SConv1d, SConvTranspose1d, StreamingCache};
use crate::loader::WeightSource;
use crate::{err, Result, VibeVoiceError};

fn conv_rms(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let t = x
        .transpose(1, 2)
        .and_then(|t| t.contiguous())
        .map_err(err)?;
    let n = rms_norm(&t, weight, eps).map_err(err)?;
    n.transpose(1, 2).and_then(|t| t.contiguous()).map_err(err)
}

fn scale_channels(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let c = gamma.dims()[0];
    let g = gamma.reshape(vec![1usize, c, 1usize]).map_err(err)?;
    x.broadcast_mul(&g).map_err(err)
}

struct Ffn {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl Ffn {
    fn load(src: &dyn WeightSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: src.get(&format!("{prefix}.linear1.weight"))?,
            b1: src.get(&format!("{prefix}.linear1.bias"))?,
            w2: src.get(&format!("{prefix}.linear2.weight"))?,
            b2: src.get(&format!("{prefix}.linear2.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x
            .linear(&self.w1)
            .and_then(|t| t.broadcast_add(&self.b1))
            .and_then(|t| t.gelu_exact())
            .map_err(err)?;
        h.linear(&self.w2)
            .and_then(|t| t.broadcast_add(&self.b2))
            .map_err(err)
    }
}

struct Block {
    norm: Tensor,
    ffn_norm: Tensor,
    gamma: Option<Tensor>,
    ffn_gamma: Option<Tensor>,
    mixer: SConv1d,
    ffn: Ffn,
    eps: f32,
}

impl Block {
    fn load(
        src: &dyn WeightSource,
        prefix: &str,
        dim: usize,
        depthwise: bool,
        causal: bool,
        eps: f32,
        layer_scale: bool,
        ids: &mut ConvIds,
    ) -> Result<Self> {
        let w = src.get(&format!("{prefix}.mixer.conv.conv.conv.weight"))?;
        let b = src.opt(&format!("{prefix}.mixer.conv.conv.conv.bias"))?;
        let groups = if depthwise { dim } else { 1 };
        let mixer = SConv1d::new(w, b, 1, 1, groups, causal, ids.take());
        Ok(Self {
            norm: src.get(&format!("{prefix}.norm.weight"))?,
            ffn_norm: src.get(&format!("{prefix}.ffn_norm.weight"))?,
            gamma: if layer_scale {
                Some(src.get(&format!("{prefix}.gamma"))?)
            } else {
                None
            },
            ffn_gamma: if layer_scale {
                Some(src.get(&format!("{prefix}.ffn_gamma"))?)
            } else {
                None
            },
            mixer,
            ffn: Ffn::load(src, &format!("{prefix}.ffn"))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let residual = x.clone();
        let h = conv_rms(x, &self.norm, self.eps)?;
        let h = match cache {
            Some(c) => self.mixer.forward_streaming(&h, c)?,
            None => self.mixer.forward(&h)?,
        };
        let h = match &self.gamma {
            Some(g) => scale_channels(&h, g)?,
            None => h,
        };
        let x = residual.add(&h).map_err(err)?;

        let residual = x.clone();
        let h = conv_rms(&x, &self.ffn_norm, self.eps)?;
        let h = h
            .permute(vec![0usize, 2, 1])
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let h = self.ffn.forward(&h)?;
        let h = h
            .permute(vec![0usize, 2, 1])
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let h = match &self.ffn_gamma {
            Some(g) => scale_channels(&h, g)?,
            None => h,
        };
        residual.add(&h).map_err(err)
    }
}

pub struct TokenizerEncoder {
    downsample: Vec<SConv1d>,
    stages: Vec<Vec<Block>>,
    head: SConv1d,
}

impl TokenizerEncoder {
    fn load(
        src: &dyn WeightSource,
        prefix: &str,
        n_filters: usize,
        ratios: &[usize],
        depths: &[usize],
        causal: bool,
        eps: f32,
        depthwise: bool,
        layer_scale: bool,
        ids: &mut ConvIds,
    ) -> Result<Self> {
        let reversed: Vec<usize> = ratios.iter().rev().copied().collect();
        let mut downsample = Vec::with_capacity(depths.len());
        for i in 0..depths.len() {
            let w = src.get(&format!("{prefix}.downsample_layers.{i}.0.conv.conv.weight"))?;
            let b = src.opt(&format!("{prefix}.downsample_layers.{i}.0.conv.conv.bias"))?;
            let stride = if i == 0 { 1 } else { reversed[i - 1] };
            downsample.push(SConv1d::new(w, b, stride, 1, 1, causal, ids.take()));
        }
        let mut stages = Vec::with_capacity(depths.len());
        for (i, depth) in depths.iter().enumerate() {
            let dim = n_filters * (1usize << i);
            let mut blocks = Vec::with_capacity(*depth);
            for j in 0..*depth {
                blocks.push(Block::load(
                    src,
                    &format!("{prefix}.stages.{i}.{j}"),
                    dim,
                    depthwise,
                    causal,
                    eps,
                    layer_scale,
                    ids,
                )?);
            }
            stages.push(blocks);
        }
        let hw = src.get(&format!("{prefix}.head.conv.conv.weight"))?;
        let hb = src.opt(&format!("{prefix}.head.conv.conv.bias"))?;
        let head = SConv1d::new(hw, hb, 1, 1, 1, causal, ids.take());
        Ok(Self {
            downsample,
            stages,
            head,
        })
    }

    pub fn forward(&self, x: &Tensor, mut cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, down) in self.downsample.iter().enumerate() {
            h = match cache.as_deref_mut() {
                Some(c) => down.forward_streaming(&h, c)?,
                None => down.forward(&h)?,
            };
            for block in &self.stages[i] {
                h = block.forward(&h, cache.as_deref_mut())?;
            }
        }
        match cache.as_deref_mut() {
            Some(c) => self.head.forward_streaming(&h, c),
            None => self.head.forward(&h),
        }
    }
}

enum UpLayer {
    Stem(SConv1d),
    Trans(SConvTranspose1d),
}

pub struct TokenizerDecoder {
    upsample: Vec<UpLayer>,
    stages: Vec<Vec<Block>>,
    head: SConv1d,
}

impl TokenizerDecoder {
    #[allow(clippy::too_many_arguments)]
    fn load(
        src: &dyn WeightSource,
        prefix: &str,
        n_filters: usize,
        ratios: &[usize],
        depths: &[usize],
        causal: bool,
        eps: f32,
        depthwise: bool,
        layer_scale: bool,
        ids: &mut ConvIds,
    ) -> Result<Self> {
        let mut upsample = Vec::with_capacity(depths.len());
        let sw = src.get(&format!("{prefix}.upsample_layers.0.0.conv.conv.weight"))?;
        let sb = src.opt(&format!("{prefix}.upsample_layers.0.0.conv.conv.bias"))?;
        upsample.push(UpLayer::Stem(SConv1d::new(sw, sb, 1, 1, 1, causal, ids.take())));
        for i in 1..depths.len() {
            let w = src.get(&format!("{prefix}.upsample_layers.{i}.0.convtr.convtr.weight"))?;
            let b = src.opt(&format!("{prefix}.upsample_layers.{i}.0.convtr.convtr.bias"))?;
            upsample.push(UpLayer::Trans(SConvTranspose1d::new(
                w,
                b,
                ratios[i - 1],
                causal,
                ids.take(),
            )));
        }
        let levels = depths.len();
        let mut stages = Vec::with_capacity(levels);
        for (i, depth) in depths.iter().enumerate() {
            let dim = n_filters * (1usize << (levels - 1 - i));
            let mut blocks = Vec::with_capacity(*depth);
            for j in 0..*depth {
                blocks.push(Block::load(
                    src,
                    &format!("{prefix}.stages.{i}.{j}"),
                    dim,
                    depthwise,
                    causal,
                    eps,
                    layer_scale,
                    ids,
                )?);
            }
            stages.push(blocks);
        }
        let hw = src.get(&format!("{prefix}.head.conv.conv.weight"))?;
        let hb = src.opt(&format!("{prefix}.head.conv.conv.bias"))?;
        let head = SConv1d::new(hw, hb, 1, 1, 1, causal, ids.take());
        Ok(Self {
            upsample,
            stages,
            head,
        })
    }

    pub fn forward(&self, x: &Tensor, mut cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, up) in self.upsample.iter().enumerate() {
            h = match (up, cache.as_deref_mut()) {
                (UpLayer::Stem(c), Some(cc)) => c.forward_streaming(&h, cc)?,
                (UpLayer::Stem(c), None) => c.forward(&h)?,
                (UpLayer::Trans(c), Some(cc)) => c.forward_streaming(&h, cc)?,
                (UpLayer::Trans(c), None) => c.forward(&h)?,
            };
            for block in &self.stages[i] {
                h = block.forward(&h, cache.as_deref_mut())?;
            }
        }
        match cache.as_deref_mut() {
            Some(c) => self.head.forward_streaming(&h, c),
            None => self.head.forward(&h),
        }
    }
}

pub struct AcousticTokenizer {
    encoder: TokenizerEncoder,
    decoder: TokenizerDecoder,
    pub fix_std: f32,
    pub std_dist_type: String,
    pub vae_dim: usize,
    conv_slots: usize,
}

impl AcousticTokenizer {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &AcousticTokenizerConfig,
        prefix: &str,
    ) -> Result<Self> {
        if cfg.layernorm != "RMSNorm" {
            return Err(VibeVoiceError::Config(format!(
                "acoustic layernorm '{}' unsupported",
                cfg.layernorm
            )));
        }
        let enc_depths = parse_depths(&cfg.encoder_depths)?;
        let dec_depths = match &cfg.decoder_depths {
            Some(s) => parse_depths(s)?,
            None => enc_depths.iter().rev().copied().collect(),
        };
        let dec_ratios = cfg
            .decoder_ratios
            .clone()
            .unwrap_or_else(|| cfg.encoder_ratios.clone());
        let depthwise = cfg.mixer_layer == "depthwise_conv";
        let layer_scale = cfg.layer_scale_init_value > 0.0;
        let mut ids = ConvIds::new();
        let encoder = TokenizerEncoder::load(
            src,
            &format!("{prefix}.encoder"),
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            &enc_depths,
            cfg.causal,
            cfg.layernorm_eps,
            depthwise,
            layer_scale,
            &mut ids,
        )?;
        let decoder = TokenizerDecoder::load(
            src,
            &format!("{prefix}.decoder"),
            cfg.decoder_n_filters,
            &dec_ratios,
            &dec_depths,
            cfg.causal,
            cfg.layernorm_eps,
            depthwise,
            layer_scale,
            &mut ids,
        )?;
        Ok(Self {
            encoder,
            decoder,
            fix_std: cfg.fix_std,
            std_dist_type: cfg.std_dist_type.clone(),
            vae_dim: cfg.vae_dim,
            conv_slots: ids.count(),
        })
    }

    pub fn new_cache(&self) -> StreamingCache {
        StreamingCache::new(self.conv_slots)
    }

    pub fn encode(&self, audio: &Tensor, cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let latents = self.encoder.forward(audio, cache)?;
        latents
            .permute(vec![0usize, 2, 1])
            .and_then(|t| t.contiguous())
            .map_err(err)
    }

    pub fn decode(&self, latents: &Tensor, cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let x = if latents.dims()[1] == self.vae_dim {
            latents.clone()
        } else {
            latents
                .permute(vec![0usize, 2, 1])
                .and_then(|t| t.contiguous())
                .map_err(err)?
        };
        self.decoder.forward(&x, cache)
    }

    pub fn sample_gaussian(&self, mean: &Tensor, per_batch_std: &[f32]) -> Result<Tensor> {
        let dims = mean.dims().to_vec();
        let bsz = dims[0];
        if per_batch_std.len() != bsz {
            return Err(VibeVoiceError::Inference(format!(
                "sample_gaussian: std len {} != batch {bsz}",
                per_batch_std.len()
            )));
        }
        let mut scale = vec![0f32; bsz];
        scale[..bsz].copy_from_slice(&per_batch_std[..bsz]);
        let std_t = Tensor::from_vec(scale, vec![bsz, 1usize, 1usize], mean.device())
            .map_err(err)?
            .to_dtype(mean.dtype())
            .map_err(err)?;
        let noise = Tensor::randn(dims, mean.device())
            .map_err(err)?
            .to_dtype(mean.dtype())
            .map_err(err)?;
        mean.broadcast_add(&noise.broadcast_mul(&std_t).map_err(err)?)
            .map_err(err)
    }

    pub fn sample_with_noise(
        &self,
        mean: &Tensor,
        per_batch_std: &[f32],
        noise: &Tensor,
    ) -> Result<Tensor> {
        let bsz = mean.dims()[0];
        let std_t = Tensor::from_vec(per_batch_std.to_vec(), vec![bsz, 1usize, 1usize], mean.device())
            .map_err(err)?
            .to_dtype(mean.dtype())
            .map_err(err)?;
        mean.broadcast_add(&noise.broadcast_mul(&std_t).map_err(err)?)
            .map_err(err)
    }
}

pub struct SemanticTokenizer {
    encoder: TokenizerEncoder,
    conv_slots: usize,
}

impl SemanticTokenizer {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &SemanticTokenizerConfig,
        prefix: &str,
    ) -> Result<Self> {
        let depths = parse_depths(&cfg.encoder_depths)?;
        let depthwise = cfg.mixer_layer == "depthwise_conv";
        let layer_scale = cfg.layer_scale_init_value > 0.0;
        let mut ids = ConvIds::new();
        let encoder = TokenizerEncoder::load(
            src,
            &format!("{prefix}.encoder"),
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            &depths,
            cfg.causal,
            cfg.layernorm_eps,
            depthwise,
            layer_scale,
            &mut ids,
        )?;
        Ok(Self {
            encoder,
            conv_slots: ids.count(),
        })
    }

    pub fn new_cache(&self) -> StreamingCache {
        StreamingCache::new(self.conv_slots)
    }

    pub fn encode(&self, audio: &Tensor, cache: Option<&mut StreamingCache>) -> Result<Tensor> {
        let latents = self.encoder.forward(audio, cache)?;
        latents
            .permute(vec![0usize, 2, 1])
            .and_then(|t| t.contiguous())
            .map_err(err)
    }
}

pub struct SpeechConnector {
    fc1_w: Tensor,
    fc1_b: Tensor,
    norm: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    eps: f32,
}

impl SpeechConnector {
    pub fn load(src: &dyn WeightSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: src.get(&format!("{prefix}.fc1.weight"))?,
            fc1_b: src.get(&format!("{prefix}.fc1.bias"))?,
            norm: src.get(&format!("{prefix}.norm.weight"))?,
            fc2_w: src.get(&format!("{prefix}.fc2.weight"))?,
            fc2_b: src.get(&format!("{prefix}.fc2.bias"))?,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x
            .linear(&self.fc1_w)
            .and_then(|t| t.broadcast_add(&self.fc1_b))
            .map_err(err)?;
        let h = rms_norm(&h, &self.norm, self.eps).map_err(err)?;
        h.linear(&self.fc2_w)
            .and_then(|t| t.broadcast_add(&self.fc2_b))
            .map_err(err)
    }
}

pub fn scalar_tensor(value: f32, dtype: DType, device: synaptix_core::device::Device) -> Result<Tensor> {
    Tensor::from_vec(vec![value], vec![1usize], device)
        .map_err(err)?
        .to_dtype(dtype)
        .map_err(err)
}
