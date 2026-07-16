//! Encode-путь нейро-кодека HiggsAudioV2 (ref-аудио 24 кГц → коды `[n_q, T]`).
//!
//! Порт `HiggsAudioV2TokenizerModel.encode` + `_extract_semantic_features` из HF
//! transformers (`models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`),
//! HuBERT-semantic (`models/hubert/modeling_hubert.py`) и DAC acoustic encoder
//! (`models/dac/modeling_dac.py`). Нужен для voice-cloning.
//!
//! Пайплайн encode(input[1,1,N_24k]):
//!   1. e_semantic_input = _extract_semantic_features:
//!        resample 24k→16k (torchaudio sinc-hann) → [:,0,:] → F.pad(160,160)
//!        → HuBERT(output_hidden_states) → stack 13 состояний → mean(dim=1)
//!        → ::semantic_downsample_factor → [1, T_s, 768]
//!   2. e_semantic = encoder_semantic(e_semantic_input.transpose(1,2)) → [1,768,T]
//!   3. e_acoustic = acoustic_encoder(input | pad(hop/2)) → [1,256,T]
//!   4. embeddings = fc(cat([e_acoustic, e_semantic],1).T).T → [1,1024,T]
//!   5. codes = quantizer.encode(embeddings, bandwidth=last) → [n_q,T]
//!
//! Snake DAC (как в decode), ELU для encoder_semantic/HuBERT, GELU(exact) для
//! HuBERT feature-extractor/FFN. RVQ: L2-nearest (argmax по −dist) + residual-loop.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::activation::elu::elu;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d_dilated;
use synaptix_ops::norm::group_norm::group_norm;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::config::{HiggsAudioConfig, SemanticConfig};
use crate::loader::OmniVoiceCodecWeights;
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Inference(e.to_string())
}

// ── torchaudio resample (sinc_interp_hann, lowpass_width=6, rolloff=0.99) ──────

/// Полный bit-faithful порт `torchaudio.functional.resample` для целых частот.
/// Ядро строится в f64 (как torch при dtype=f32 → idx_dtype=f64), затем conv
/// stride=orig_freq, crop до ceil(new*len/orig). Возвращает f32-волну.
pub fn resample(input: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    if orig_freq == new_freq || input.is_empty() {
        return input.to_vec();
    }
    let lowpass_filter_width: i64 = 6;
    let rolloff = 0.99_f64;
    let g = gcd(orig_freq, new_freq);
    let orig = (orig_freq / g) as i64;
    let new = (new_freq / g) as i64;
    let base_freq = (orig.min(new) as f64) * rolloff;
    let width = ((lowpass_filter_width as f64) * (orig as f64) / base_freq).ceil() as i64;

    // idx = arange(-width, width+orig)/orig  →  длина (2*width + orig).
    let klen = (2 * width + orig) as usize;
    // kernel[j, kk] для j in 0..new.
    let mut kernel = vec![0.0_f64; (new as usize) * klen];
    let pi = std::f64::consts::PI;
    let scale = base_freq / (orig as f64);
    for j in 0..new {
        let tj = (-(j as f64)) / (new as f64);
        for (kk, kslot) in (0..klen).enumerate() {
            let idx = ((kslot as i64 - width) as f64) / (orig as f64);
            let mut t = (tj + idx) * base_freq;
            // clamp(-lpw, lpw)
            if t > lowpass_filter_width as f64 {
                t = lowpass_filter_width as f64;
            } else if t < -(lowpass_filter_width as f64) {
                t = -(lowpass_filter_width as f64);
            }
            // hann window: cos(t*pi/lpw/2)^2
            let w = (t * pi / (lowpass_filter_width as f64) / 2.0).cos();
            let window = w * w;
            let tp = t * pi;
            let sinc = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kernel[j as usize * klen + kk] = sinc * window * scale;
        }
    }

    // pad (width, width+orig), conv1d stride=orig, transpose+reshape.
    let length = input.len();
    let pad_left = width as usize;
    let pad_right = (width + orig) as usize;
    let padded_len = pad_left + length + pad_right;
    let mut padded = vec![0.0_f64; padded_len];
    for (i, &v) in input.iter().enumerate() {
        padded[pad_left + i] = v as f64;
    }

    // conv1d: out positions p = 0,1,...; in-start = p*orig; out[p, j] = Σ_kk pad[p*orig+kk]·kernel[j,kk].
    let n_out_pos = if padded_len >= klen {
        (padded_len - klen) / (orig as usize) + 1
    } else {
        0
    };
    // resampled[p*new + j] = out[p, j]; затем crop до target_length.
    let target_length = ((new as f64) * (length as f64) / (orig as f64)).ceil() as usize;
    let mut out = Vec::with_capacity(n_out_pos * new as usize);
    for p in 0..n_out_pos {
        let base = p * orig as usize;
        for j in 0..new as usize {
            let krow = j * klen;
            let mut acc = 0.0_f64;
            for kk in 0..klen {
                acc += padded[base + kk] * kernel[krow + kk];
            }
            out.push(acc as f32);
        }
    }
    out.truncate(target_length);
    out
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

// ── shared conv / snake helpers ───────────────────────────────────────────────

struct Conv {
    w: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    pad: usize,
    dilation: usize,
}

impl Conv {
    fn load(
        w: &OmniVoiceCodecWeights,
        prefix: &str,
        stride: usize,
        pad: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        let bias = if bias {
            Some(w.get(&format!("{prefix}.bias"))?.clone())
        } else {
            None
        };
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"))?.clone(),
            bias,
            stride,
            pad,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv1d_dilated(x, &self.w, self.bias.as_ref(), self.stride, self.pad, self.dilation)
            .map_err(err)
    }
}

/// DAC Snake1d: `y = x + (alpha + 1e-9).reciprocal() · sin(alpha·x)²`.
struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str) -> Result<Self> {
        Ok(Self { alpha: w.get(&format!("{prefix}.alpha"))?.clone() })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha).map_err(err)?;
        let s = ax.sin().map_err(err)?.sqr().map_err(err)?;
        let inv = self.alpha.affine(1.0, 1e-9).map_err(err)?.recip().map_err(err)?;
        x.broadcast_add(&s.broadcast_mul(&inv).map_err(err)?).map_err(err)
    }
}

fn center_crop_add(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let lx = x.dims()[2];
    let ly = y.dims()[2];
    let pad = (lx - ly) / 2;
    let xc = if pad > 0 {
        x.narrow(2, pad, lx - 2 * pad).map_err(err)?
    } else {
        x.clone()
    };
    xc.broadcast_add(y).map_err(err)
}

// ── DAC acoustic encoder ──────────────────────────────────────────────────────

/// DacResidualUnit: snake1 → conv1(k7,dil) → snake2 → conv2(k1) → center-crop add.
struct DacResUnit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

impl DacResUnit {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, dilation: usize) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1: Snake::load(w, &format!("{prefix}.snake1"))?,
            conv1: Conv::load(w, &format!("{prefix}.conv1"), 1, pad, dilation, true)?,
            snake2: Snake::load(w, &format!("{prefix}.snake2"))?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), 1, 0, 1, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.snake1.forward(x)?)?;
        let y = self.conv2.forward(&self.snake2.forward(&y)?)?;
        center_crop_add(x, &y)
    }
}

/// DacEncoderBlock: res_unit1/2/3 (dil 1/3/9) → snake1 → conv1 (strided down).
struct DacEncoderBlock {
    res_units: Vec<DacResUnit>,
    snake1: Snake,
    conv1: Conv,
}

impl DacEncoderBlock {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, stride: usize) -> Result<Self> {
        let res_units = vec![
            DacResUnit::load(w, &format!("{prefix}.res_unit1"), 1)?,
            DacResUnit::load(w, &format!("{prefix}.res_unit2"), 3)?,
            DacResUnit::load(w, &format!("{prefix}.res_unit3"), 9)?,
        ];
        // conv1: Conv1d(half, dimension, kernel=2*stride, stride, padding=ceil(stride/2)).
        let pad = stride.div_ceil(2);
        let conv1 = Conv::load(w, &format!("{prefix}.conv1"), stride, pad, 1, true)?;
        Ok(Self {
            res_units,
            snake1: Snake::load(w, &format!("{prefix}.snake1"))?,
            conv1,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.res_units[0].forward(x)?;
        h = self.res_units[1].forward(&h)?;
        h = self.res_units[2].forward(&h)?;
        h = self.snake1.forward(&h)?;
        self.conv1.forward(&h)
    }
}

/// DAC acoustic_encoder: conv1 → 5× block → snake1 → conv2 → [1,256,T].
struct AcousticEncoder {
    conv1: Conv,
    blocks: Vec<DacEncoderBlock>,
    snake1: Snake,
    conv2: Conv,
}

impl AcousticEncoder {
    fn load(w: &OmniVoiceCodecWeights, cfg: &HiggsAudioConfig) -> Result<Self> {
        let strides = &cfg.acoustic_model_config.downsampling_ratios; // [8,5,4,2,3]
        // conv1: Conv1d(1, enc_hidden, k7, padding=3).
        let conv1 = Conv::load(w, "acoustic_encoder.conv1", 1, 3, 1, true)?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &s) in strides.iter().enumerate() {
            blocks.push(DacEncoderBlock::load(w, &format!("acoustic_encoder.block.{i}"), s)?);
        }
        let snake1 = Snake::load(w, "acoustic_encoder.snake1")?;
        // conv2: Conv1d(d_model, hidden_size=256, k3, padding=1).
        let conv2 = Conv::load(w, "acoustic_encoder.conv2", 1, 1, 1, true)?;
        Ok(Self { conv1, blocks, snake1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.snake1.forward(&h)?;
        self.conv2.forward(&h)
    }
}

// ── encoder_semantic (SemanticEncoder, ELU residual units) ────────────────────

/// HiggsAudioV2TokenizerResidualUnit: ELU → conv1(k3,dil,bias=false) → ELU →
/// conv2(k1,bias=false) → residual add (без crop, padding сохраняет длину).
struct SemResUnit {
    conv1: Conv,
    conv2: Conv,
}

impl SemResUnit {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, dilation: usize, unit_k: usize) -> Result<Self> {
        let pad = ((unit_k - 1) / 2) * dilation;
        Ok(Self {
            conv1: Conv::load(w, &format!("{prefix}.conv1"), 1, pad, dilation, false)?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), 1, 0, 1, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = elu(x, 1.0).map_err(err)?;
        let y = self.conv1.forward(&y)?;
        let y = elu(&y, 1.0).map_err(err)?;
        let y = self.conv2.forward(&y)?;
        x.add(&y).map_err(err)
    }
}

/// HiggsAudioV2TokenizerSemanticEncoderBlock: res_units → conv (stride==1 → k3).
struct SemEncBlock {
    res_units: Vec<SemResUnit>,
    conv: Conv,
}

impl SemEncBlock {
    fn load(
        w: &OmniVoiceCodecWeights,
        prefix: &str,
        dilations: &[usize],
        unit_k: usize,
        stride: usize,
    ) -> Result<Self> {
        let mut res_units = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            res_units.push(SemResUnit::load(
                w,
                &format!("{prefix}.res_units.{i}"),
                d,
                unit_k,
            )?);
        }
        // kernel = 3 if stride==1 else 2*stride; padding = (kernel-1)//2; bias=true.
        let kernel = if stride == 1 { 3 } else { 2 * stride };
        let pad = (kernel - 1) / 2;
        let conv = Conv::load(w, &format!("{prefix}.conv"), stride, pad, 1, true)?;
        Ok(Self { res_units, conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for u in &self.res_units {
            h = u.forward(&h)?;
        }
        self.conv.forward(&h)
    }
}

/// SemanticEncoder: conv(k3,bias=false) → conv_blocks (strides [1,1]).
struct SemanticEncoder {
    conv: Conv,
    blocks: Vec<SemEncBlock>,
}

impl SemanticEncoder {
    fn load(w: &OmniVoiceCodecWeights) -> Result<Self> {
        // strides=[1,1], block_dilations=[1,1], unit_kernel_size=3, kernel_size=3
        // (HiggsAudioV2TokenizerConfig defaults — фиксированы для этой архитектуры).
        let strides = [1usize, 1];
        let dilations = [1usize, 1];
        let unit_k = 3usize;
        let kernel_size = 3usize;
        // conv: Conv1d(sem_hidden, sem_hidden, k3, padding=k//2=1, bias=false).
        let conv = Conv::load(w, "encoder_semantic.conv", 1, kernel_size / 2, 1, false)?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &s) in strides.iter().enumerate() {
            blocks.push(SemEncBlock::load(
                w,
                &format!("encoder_semantic.conv_blocks.{i}"),
                &dilations,
                unit_k,
                s,
            )?);
        }
        Ok(Self { conv, blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        Ok(h)
    }
}

// ── HuBERT semantic model ─────────────────────────────────────────────────────

/// Один conv-слой feature_extractor. Слой 0 = GroupNorm(num_groups=out_dim);
/// 1..6 — без нормы (NoLayerNormConvLayer). GELU(exact) активация.
struct HubertConvLayer {
    conv: Conv,
    gn_weight: Option<Tensor>,
    gn_bias: Option<Tensor>,
    out_dim: usize,
}

impl HubertConvLayer {
    fn load(w: &OmniVoiceCodecWeights, idx: usize, scfg: &SemanticConfig) -> Result<Self> {
        let s = scfg.conv_stride[idx];
        let prefix = format!("semantic_model.feature_extractor.conv_layers.{idx}");
        let conv = Conv::load(w, &format!("{prefix}.conv"), s, 0, 1, scfg.conv_bias)?;
        let out_dim = scfg.conv_dim[idx];
        let (gn_weight, gn_bias) = if idx == 0 {
            (
                Some(w.get(&format!("{prefix}.layer_norm.weight"))?.clone()),
                Some(w.get(&format!("{prefix}.layer_norm.bias"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self { conv, gn_weight, gn_bias, out_dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv.forward(x)?;
        if let (Some(gw), Some(gb)) = (&self.gn_weight, &self.gn_bias) {
            // GroupNorm(num_groups=out_dim, num_channels=out_dim) → per-channel norm.
            h = group_norm(&h, Some(gw), Some(gb), self.out_dim, 1e-5).map_err(err)?;
        }
        h.gelu_exact().map_err(err)
    }
}

/// HuBERT positional conv-embed (weight-norm параметризация, groups=16, k=128).
struct HubertPosConv {
    weight: Tensor, // восстановленный [768,48,128]
    bias: Tensor,
    groups: usize,
    pad: usize,
    num_pad_remove: usize,
}

impl HubertPosConv {
    fn load(w: &OmniVoiceCodecWeights, scfg: &SemanticConfig) -> Result<Self> {
        // weight_norm dim=2: w = g * v / ||v||_{over dims 0,1, keepdim per k-slot}.
        let g = w.get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0")?; // [1,1,128]
        let v = w.get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1")?; // [768,48,128]
        let bias = w.get("semantic_model.encoder.pos_conv_embed.conv.bias")?.clone();
        // norm over (0,1) keepdim → [1,1,128].
        let v_sq = v.sqr().map_err(err)?;
        let norm = v_sq.sum([0usize, 1]).map_err(err)?.sqrt().map_err(err)?; // [128]
        let norm = norm.reshape(vec![1, 1, v.dims()[2]]).map_err(err)?;
        let inv = norm.recip().map_err(err)?;
        let weight = v.broadcast_mul(&g.broadcast_mul(&inv).map_err(err)?).map_err(err)?;
        let k = scfg.num_conv_pos_embeddings;
        Ok(Self {
            weight,
            bias,
            groups: scfg.num_conv_pos_embedding_groups,
            pad: k / 2,
            num_pad_remove: if k % 2 == 0 { 1 } else { 0 },
        })
    }

    fn forward(&self, x_bld: &Tensor) -> Result<Tensor> {
        // x_bld: [B, L, C] → conv expects [B, C, L].
        let x = x_bld.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;
        let mut h = grouped_conv1d(&x, &self.weight, Some(&self.bias), 1, self.pad, self.groups)?;
        if self.num_pad_remove > 0 {
            let l = h.dims()[2];
            h = h.narrow(2, 0, l - self.num_pad_remove).map_err(err)?;
        }
        h = h.gelu_exact().map_err(err)?;
        h.transpose(1, 2).map_err(err)?.contiguous().map_err(err)
    }
}

/// Grouped Conv1d (через per-group matmul-conv). weight `[Cout, Cin/groups, K]`.
fn grouped_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    pad: usize,
    groups: usize,
) -> Result<Tensor> {
    if groups == 1 {
        return conv1d_dilated(x, weight, bias, stride, pad, 1).map_err(err);
    }
    let c_in = x.dims()[1];
    let c_out = weight.dims()[0];
    let cin_g = c_in / groups;
    let cout_g = c_out / groups;
    let mut parts: Vec<Tensor> = Vec::with_capacity(groups);
    for gi in 0..groups {
        let xg = x.narrow(1, gi * cin_g, cin_g).map_err(err)?.contiguous().map_err(err)?;
        let wg = weight.narrow(0, gi * cout_g, cout_g).map_err(err)?.contiguous().map_err(err)?;
        let bg = match bias {
            Some(b) => Some(b.narrow(0, gi * cout_g, cout_g).map_err(err)?),
            None => None,
        };
        parts.push(conv1d_dilated(&xg, &wg, bg.as_ref(), stride, pad, 1).map_err(err)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 1).map_err(err)
}

/// Linear (`weight [out,in]`, `+bias`). x: [..., in] → [..., out].
struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, bias: bool) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"))?.clone(),
            b: if bias {
                Some(w.get(&format!("{prefix}.bias"))?.clone())
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.linear(&self.w).map_err(err)?;
        match &self.b {
            Some(b) => y.broadcast_add(b).map_err(err),
            None => Ok(y),
        }
    }
}

struct HubertLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln_w: Tensor,
    ln_b: Tensor,
    fc1: Linear,
    fc2: Linear,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
}

impl HubertLayer {
    fn load(w: &OmniVoiceCodecWeights, idx: usize) -> Result<Self> {
        let p = format!("semantic_model.encoder.layers.{idx}");
        Ok(Self {
            q: Linear::load(w, &format!("{p}.attention.q_proj"), true)?,
            k: Linear::load(w, &format!("{p}.attention.k_proj"), true)?,
            v: Linear::load(w, &format!("{p}.attention.v_proj"), true)?,
            o: Linear::load(w, &format!("{p}.attention.out_proj"), true)?,
            ln_w: w.get(&format!("{p}.layer_norm.weight"))?.clone(),
            ln_b: w.get(&format!("{p}.layer_norm.bias"))?.clone(),
            fc1: Linear::load(w, &format!("{p}.feed_forward.intermediate_dense"), true)?,
            fc2: Linear::load(w, &format!("{p}.feed_forward.output_dense"), true)?,
            final_ln_w: w.get(&format!("{p}.final_layer_norm.weight"))?.clone(),
            final_ln_b: w.get(&format!("{p}.final_layer_norm.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Tensor, n_heads: usize, eps: f32) -> Result<Tensor> {
        let d = x.dims();
        let (b, s, c) = (d[0], d[1], d[2]);
        let hd = c / n_heads;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // post-norm attention block.
        let q = self
            .q
            .forward(x)?
            .reshape(vec![b, s, n_heads, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let k = self
            .k
            .forward(x)?
            .reshape(vec![b, s, n_heads, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let v = self
            .v
            .forward(x)?
            .reshape(vec![b, s, n_heads, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let attn = scaled_dot_attention(&q, &k, &v, scale, None).map_err(err)?;
        let attn = attn
            .permute(vec![0, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![b, s, c]))
            .map_err(err)?;
        let attn = self.o.forward(&attn)?;

        let mut h = x.add(&attn).map_err(err)?;
        h = layer_norm(&h, Some(&self.ln_w), Some(&self.ln_b), eps).map_err(err)?;

        // feed_forward (GELU exact) + residual + final_layer_norm.
        let ff = self.fc1.forward(&h)?;
        let ff = ff.gelu_exact().map_err(err)?;
        let ff = self.fc2.forward(&ff)?;
        h = h.add(&ff).map_err(err)?;
        layer_norm(&h, Some(&self.final_ln_w), Some(&self.final_ln_b), eps).map_err(err)
    }
}

/// HuBERT semantic model (group-norm feature-extractor, post-norm encoder).
struct Hubert {
    conv_layers: Vec<HubertConvLayer>,
    fp_ln_w: Tensor,
    fp_ln_b: Tensor,
    fp_proj: Linear,
    pos_conv: HubertPosConv,
    enc_ln_w: Tensor,
    enc_ln_b: Tensor,
    layers: Vec<HubertLayer>,
    n_heads: usize,
    eps: f32,
}

impl Hubert {
    fn load(w: &OmniVoiceCodecWeights, scfg: &SemanticConfig) -> Result<Self> {
        let mut conv_layers = Vec::with_capacity(scfg.conv_kernel.len());
        for i in 0..scfg.conv_kernel.len() {
            conv_layers.push(HubertConvLayer::load(w, i, scfg)?);
        }
        let mut layers = Vec::with_capacity(scfg.num_hidden_layers);
        for i in 0..scfg.num_hidden_layers {
            layers.push(HubertLayer::load(w, i)?);
        }
        Ok(Self {
            conv_layers,
            fp_ln_w: w.get("semantic_model.feature_projection.layer_norm.weight")?.clone(),
            fp_ln_b: w.get("semantic_model.feature_projection.layer_norm.bias")?.clone(),
            fp_proj: Linear::load(w, "semantic_model.feature_projection.projection", true)?,
            pos_conv: HubertPosConv::load(w, scfg)?,
            enc_ln_w: w.get("semantic_model.encoder.layer_norm.weight")?.clone(),
            enc_ln_b: w.get("semantic_model.encoder.layer_norm.bias")?.clone(),
            layers,
            n_heads: scfg.num_attention_heads,
            eps: scfg.layer_norm_eps as f32,
        })
    }

    /// `input_values [B, N]` → vec из (num_layers+1) скрытых состояний `[B,S,768]`
    /// (embeddings + после каждого encoder-слоя; output_hidden_states семантика HF).
    fn forward_hidden_states(&self, input_values: &Tensor) -> Result<Vec<Tensor>> {
        // feature_extractor: [B,1,N] → conv-stack → [B,512,S].
        let mut h = input_values.unsqueeze(1).map_err(err)?; // [B,1,N]
        for layer in &self.conv_layers {
            h = layer.forward(&h)?;
        }
        // transpose(1,2) → [B,S,512].
        let extract = h.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;

        // feature_projection: LayerNorm(512) → Linear 512→768.
        let proj = layer_norm(&extract, Some(&self.fp_ln_w), Some(&self.fp_ln_b), self.eps).map_err(err)?;
        let hidden = self.fp_proj.forward(&proj)?; // [B,S,768]
        // _mask_hidden_states: mask_time_prob=0 + not training → identity.

        // encoder: pos_conv add → layer_norm → layers.
        let pos = self.pos_conv.forward(&hidden)?;
        let mut hs = hidden.add(&pos).map_err(err)?;
        hs = layer_norm(&hs, Some(&self.enc_ln_w), Some(&self.enc_ln_b), self.eps).map_err(err)?;

        // output_hidden_states: первый элемент — вход в первый слой (после ln+pos+dropout),
        // затем после каждого слоя.
        let mut states: Vec<Tensor> = Vec::with_capacity(self.layers.len() + 1);
        states.push(hs.clone());
        for layer in &self.layers {
            hs = layer.forward(&hs, self.n_heads, self.eps)?;
            states.push(hs.clone());
        }
        Ok(states)
    }
}

// ── RVQ encode ────────────────────────────────────────────────────────────────

struct RvqQuantizer {
    project_in: Linear,
    embed: Tensor,       // [codebook_size, codebook_dim]
    embed_t: Tensor,     // [codebook_dim, codebook_size]
    embed_sq: Tensor,    // [1, codebook_size]  (Σ embed² по dim)
    project_out: Linear,
}

impl RvqQuantizer {
    fn load(w: &OmniVoiceCodecWeights, i: usize) -> Result<Self> {
        let p = format!("quantizer.quantizers.{i}");
        let embed = w.get(&format!("{p}.codebook.embed"))?.clone(); // [N, D]
        let embed_t = embed.transpose(0, 1).map_err(err)?.contiguous().map_err(err)?; // [D, N]
        let embed_sq = embed
            .sqr()
            .map_err(err)?
            .sum([1usize])
            .map_err(err)?
            .reshape(vec![1, embed.dims()[0]])
            .map_err(err)?; // [1, N]
        Ok(Self {
            project_in: Linear::load(w, &format!("{p}.project_in"), true)?,
            embed,
            embed_t,
            embed_sq,
            project_out: Linear::load(w, &format!("{p}.project_out"), true)?,
        })
    }

    /// encode: residual `[1, hidden, T]` → indices `[T]` (i64).
    /// `hidden = project_in(residual.permute(0,2,1))` → [1,T,D];
    /// `dist = -(h² - 2 h@E^T + E²); idx = argmax(dist)` (= L2-nearest).
    fn encode(&self, residual: &Tensor) -> Result<Tensor> {
        let h = residual.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?; // [1,T,hidden]
        let h = self.project_in.forward(&h)?; // [1,T,D]
        let t = h.dims()[1];
        let d = h.dims()[2];
        let h2 = h.reshape(vec![t, d]).map_err(err)?; // [T,D]
        let scaled = h2.sqr().map_err(err)?.sum([1usize]).map_err(err)?.reshape(vec![t, 1]).map_err(err)?; // [T,1]
        let cross = h2.matmul(&self.embed_t).map_err(err)?; // [T,N]
        // dist = -(scaled - 2*cross + embed_sq) = -scaled + 2*cross - embed_sq.
        let two_cross = cross.affine(2.0, 0.0).map_err(err)?;
        let dist = two_cross
            .broadcast_sub(&scaled).map_err(err)?
            .broadcast_sub(&self.embed_sq).map_err(err)?; // [T,N]
        let idx = dist.argmax(1).map_err(err)?; // [T]
        idx.to_dtype(DType::I64).map_err(err)
    }

    /// decode: indices `[T]` → quantized `[1, hidden, T]`.
    fn decode(&self, idx: &Tensor) -> Result<Tensor> {
        let q = self.embed.index_select(0, idx).map_err(err)?; // [T,D]
        let out = self.project_out.forward(&q)?; // [T,hidden]
        out.transpose(0, 1)
            .map_err(err)?
            .contiguous()
            .map_err(err)?
            .unsqueeze(0)
            .map_err(err) // [1,hidden,T]
    }
}

// ── top-level encoder ─────────────────────────────────────────────────────────

/// Промежуточные тензоры encode (для гейта/дебага).
pub struct EncodeStages {
    pub semantic_features: Tensor, // [1, T_s, 768]
    pub e_semantic: Tensor,        // [1, 768, T]
    pub e_acoustic: Tensor,        // [1, 256, T]
    pub embeddings: Tensor,        // [1, 1024, T]
    pub codes: Tensor,             // [n_q, T] i64
}

/// Encode-путь нейро-кодека HiggsAudioV2: ref-волна 24 кГц → коды `[n_q, T]`.
pub struct CodecEncoder {
    hubert: Hubert,
    encoder_semantic: SemanticEncoder,
    acoustic_encoder: AcousticEncoder,
    fc: Linear,
    quantizers: Vec<RvqQuantizer>,
    n_q: usize,
    sample_rate: usize,
    semantic_sample_rate: usize,
    semantic_pad: usize,
    semantic_downsample_factor: usize,
    hop_length: usize,
    down_ratios: Vec<usize>,
    device: Device,
}

impl CodecEncoder {
    /// Собрать encoder из codec-весов + конфига. `n_q` берётся из bandwidth
    /// (`config.target_bandwidths[-1]`). Acoustic-path = первые n_q квантизаторов.
    pub fn build(cfg: &HiggsAudioConfig, w: &OmniVoiceCodecWeights) -> Result<Self> {
        let scfg = &cfg.semantic_model_config;
        let n_q = cfg.num_quantizers_for_encode();
        let mut quantizers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            quantizers.push(RvqQuantizer::load(w, i)?);
        }
        Ok(Self {
            hubert: Hubert::load(w, scfg)?,
            encoder_semantic: SemanticEncoder::load(w)?,
            acoustic_encoder: AcousticEncoder::load(w, cfg)?,
            fc: Linear::load(w, "fc", true)?,
            quantizers,
            n_q,
            sample_rate: cfg.sample_rate,
            semantic_sample_rate: cfg.semantic_sample_rate,
            // _extract_semantic_features жёстко F.pad(x,(160,160)) (см. исходник).
            semantic_pad: 160,
            semantic_downsample_factor: cfg.semantic_downsample_factor(),
            hop_length: cfg.hop_length(),
            down_ratios: cfg.acoustic_model_config.downsampling_ratios.clone(),
            device: w.device,
        })
    }

    /// `_extract_semantic_features`: 24k-волна `[N]` → `[1, T_s, 768]`.
    fn extract_semantic_features(&self, input_24k: &[f32]) -> Result<Tensor> {
        // resample 24k→16k.
        let rs = resample(input_24k, self.sample_rate, self.semantic_sample_rate);
        // [:,0,:] (mono) + F.pad(160,160).
        let n = rs.len();
        let mut padded = vec![0.0f32; n + 2 * self.semantic_pad];
        padded[self.semantic_pad..self.semantic_pad + n].copy_from_slice(&rs);
        let x = Tensor::from_vec(padded, vec![1, n + 2 * self.semantic_pad], self.device).map_err(err)?;

        let states = self.hubert.forward_hidden_states(&x)?; // (13) × [1,S,768]
        // stack dim=1 → mean(dim=1): среднее по всем состояниям.
        let mut acc: Option<Tensor> = None;
        for s in &states {
            acc = Some(match acc {
                None => s.clone(),
                Some(a) => a.add(s).map_err(err)?,
            });
        }
        let acc = acc.ok_or_else(|| OmniVoiceError::Inference("hubert: no hidden_states".into()))?;
        let mean = acc.affine(1.0 / states.len() as f32, 0.0).map_err(err)?; // [1,S,768]

        // downsample ::factor по оси времени (dim=1).
        if self.semantic_downsample_factor > 1 {
            let s = mean.dims()[1];
            let f = self.semantic_downsample_factor;
            let take = s.div_ceil(f);
            let idx: Vec<u32> = (0..take).map(|i| (i * f) as u32).collect();
            let idx = Tensor::from_vec(idx, (take,), self.device).map_err(err)?;
            mean.index_select(1, &idx).map_err(err)
        } else {
            Ok(mean)
        }
    }

    /// encode(`input_24k [N]` mono f32) → коды `[n_q, T]` (i64).
    pub fn encode(&self, input_24k: &Tensor) -> Result<Tensor> {
        self.encode_stages(input_24k).map(|s| s.codes)
    }

    /// encode + промежуточные тензоры (для послойного гейта/дебага).
    pub fn encode_stages(&self, input_24k: &Tensor) -> Result<EncodeStages> {
        let flat = input_24k.flatten_all().map_err(err)?;
        let samples: Vec<f32> = flat.to_dtype(DType::F32).map_err(err)?.to_vec1::<f32>().map_err(err)?;

        // 1. semantic features → [1,T_s,768].
        let semantic_features = self.extract_semantic_features(&samples)?;
        // 2. encoder_semantic(transpose(1,2)) → [1,768,T].
        let e_sem_in_t = semantic_features.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;
        let e_semantic = self.encoder_semantic.forward(&e_sem_in_t)?;

        // 3. acoustic_encoder(input | pad(hop/2)) — pad если длины conv-выходов != e_semantic.
        let n = samples.len();
        let input_t = Tensor::from_vec(samples.clone(), vec![1, 1, n], self.device).map_err(err)?;
        let t_sem = e_semantic.dims()[2];
        let acoustic_len = self.acoustic_conv_out_len(n);
        let e_acoustic = if acoustic_len != t_sem {
            let pad = self.hop_length / 2;
            let mut padded = vec![0.0f32; n + 2 * pad];
            padded[pad..pad + n].copy_from_slice(&samples);
            let xp = Tensor::from_vec(padded, vec![1, 1, n + 2 * pad], self.device).map_err(err)?;
            self.acoustic_encoder.forward(&xp)?
        } else {
            self.acoustic_encoder.forward(&input_t)?
        };

        // 4. embeddings = fc(cat([e_acoustic, e_semantic],1).transpose(1,2)).transpose(1,2).
        let emb = Tensor::cat(&[&e_acoustic, &e_semantic], 1).map_err(err)?; // [1,1024,T]
        let emb_t = emb.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?; // [1,T,1024]
        let emb_t = self.fc.forward(&emb_t)?; // [1,T,1024]
        let embeddings = emb_t.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?; // [1,1024,T]

        // 5. RVQ encode (residual-loop, n_q квантизаторов).
        let mut residual = embeddings.clone();
        let t = residual.dims()[2];
        let mut all_codes: Vec<Tensor> = Vec::with_capacity(self.n_q);
        for q in &self.quantizers {
            let idx = q.encode(&residual)?; // [T]
            let quantized = q.decode(&idx)?; // [1,hidden,T]
            residual = residual.sub(&quantized).map_err(err)?;
            all_codes.push(idx.reshape(vec![1, t]).map_err(err)?);
        }
        let refs: Vec<&Tensor> = all_codes.iter().collect();
        let codes = Tensor::cat(&refs, 0).map_err(err)?; // [n_q, T]
        Ok(EncodeStages {
            semantic_features,
            e_semantic,
            e_acoustic,
            embeddings,
            codes,
        })
    }

    /// Длина выхода всех Conv1d acoustic_encoder для входа длины `n`
    /// (как `_get_conv1d_output_lengths`: последовательное применение формулы
    /// conv1d_output_length по ВСЕМ Conv1d в порядке обхода модулей).
    fn acoustic_conv_out_len(&self, n: usize) -> usize {
        // Порядок обхода nn.Module.modules() для DacEncoder:
        //   conv1(k7,s1,p3) → block[i]{ res_unit1{conv1 k7 dil1 p3, conv2 k1},
        //   res_unit2{conv1 k7 dil3 p18, conv2 k1}, res_unit3{conv1 k7 dil9 p54,
        //   conv2 k1}, conv1(k=2s, stride s, p=ceil(s/2)) } → conv2(k3,s1,p1).
        let out = |l: usize, k: usize, s: usize, p: usize, d: usize| -> usize {
            let eff = d * (k - 1) + 1;
            (l + 2 * p).saturating_sub(eff) / s + 1
        };
        let mut l = n;
        l = out(l, 7, 1, 3, 1); // conv1
        for &s in &self.down_ratios {
            // res units (k7 dil 1/3/9 + matching padding → length preserved) + k1.
            for &d in &[1usize, 3, 9] {
                l = out(l, 7, 1, ((7 - 1) * d) / 2, d);
                l = out(l, 1, 1, 0, 1);
            }
            // strided downsample conv (k=2s, stride s, pad=ceil(s/2)).
            l = out(l, 2 * s, s, s.div_ceil(2), 1);
        }
        l = out(l, 3, 1, 1, 1); // conv2
        l
    }

    pub fn n_q(&self) -> usize {
        self.n_q
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
