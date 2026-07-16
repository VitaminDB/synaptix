//! Decode-путь нейро-кодека HiggsAudioV2 (коды `[n_q, T]` → волна 24 кГц).
//!
//! Порт `HiggsAudioV2TokenizerModel.decode` + `DacDecoder` из HF transformers
//! (`models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`,
//! `models/dac/modeling_dac.py`):
//!
//!   codes[n_q,T] → RVQ.decode (Σ_i project_out(embed_lookup(codes_i))) → [1024,T]
//!   → fc2 (Linear 1024→256) → [256,T]
//!   → acoustic_decoder (DAC): conv1 → 5× DacDecoderBlock(snake1, conv_t1,
//!     res_unit1/2/3) → snake1 → conv2 → [1,samples].
//!
//! OmniVoice кладёт 8 кодбуков, codec имеет 9 квантизаторов — `decode` итерирует
//! `for i, indices in enumerate(codes)`, т.е. использует первые `n_q` (=8)
//! квантизаторов. semantic/fc/fc1/encoder в decode НЕ участвуют (encode-only).
//!
//! Snake-активация DAC: `x + (alpha + 1e-9).reciprocal()·sin(alpha·x)²` — alpha
//! применяется напрямую (БЕЗ exp, в отличие от ACE-Step Snake), single-alpha
//! shape `[1,C,1]`. ConvTranspose1d: `_adjust_dac_decoder` ставит
//! `output_padding = stride % 2` и снимает финальный Tanh (→ Identity).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::{conv1d_dilated, conv_transpose1d};

use crate::config::HiggsAudioConfig;
use crate::loader::OmniVoiceCodecWeights;
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Inference(e.to_string())
}

/// DAC Snake1d: `y = x + (alpha + eps).reciprocal() · sin(alpha·x)²`.
/// `alpha` хранится как `[1,C,1]` и бродкастится по batch/time.
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

/// Обычная Conv1d (`weight [Cout,Cin,K]`), weight-norm уже слит в веса.
struct Conv {
    w: Tensor,
    bias: Tensor,
    stride: usize,
    pad: usize,
    dilation: usize,
}

impl Conv {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, stride: usize, pad: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.get(&format!("{prefix}.bias"))?.clone(),
            stride,
            pad,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv1d_dilated(x, &self.w, Some(&self.bias), self.stride, self.pad, self.dilation).map_err(err)
    }
}

/// ConvTranspose1d (`weight [Cin,Cout,K]`) — DAC upsample.
struct ConvT {
    w: Tensor,
    bias: Tensor,
    stride: usize,
    pad: usize,
    output_pad: usize,
}

impl ConvT {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, stride: usize, pad: usize, output_pad: usize) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"))?.clone(),
            bias: w.get(&format!("{prefix}.bias"))?.clone(),
            stride,
            pad,
            output_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv_transpose1d(x, &self.w, Some(&self.bias), self.stride, self.pad, self.output_pad, 1, 1)
            .map_err(err)
    }
}

/// DacResidualUnit: snake1 → conv1(k7,dil) → snake2 → conv2(k1) → residual add
/// (с центральным кропом hidden_state при необходимости).
struct ResidualUnit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, dilation: usize) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1: Snake::load(w, &format!("{prefix}.snake1"))?,
            conv1: Conv::load(w, &format!("{prefix}.conv1"), 1, pad, dilation)?,
            snake2: Snake::load(w, &format!("{prefix}.snake2"))?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), 1, 0, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.snake1.forward(x)?)?;
        let y = self.conv2.forward(&self.snake2.forward(&y)?)?;
        let lx = x.dims()[2];
        let ly = y.dims()[2];
        let pad = (lx - ly) / 2;
        let xc = if pad > 0 {
            x.narrow(2, pad, lx - 2 * pad).map_err(err)?
        } else {
            x.clone()
        };
        xc.broadcast_add(&y).map_err(err)
    }
}

/// DacDecoderBlock: snake1 → conv_t1 (upsample) → res_unit1/2/3 (dil 1/3/9).
struct DecoderBlock {
    snake1: Snake,
    conv_t1: ConvT,
    res_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(w: &OmniVoiceCodecWeights, prefix: &str, stride: usize) -> Result<Self> {
        // ConvTranspose1d(k=2*stride, stride, padding=ceil(stride/2),
        // output_padding=stride%2). math.ceil(stride/2) для нечётного=(s+1)/2.
        let pad = stride.div_ceil(2);
        let output_pad = stride % 2;
        let res_units = vec![
            ResidualUnit::load(w, &format!("{prefix}.res_unit1"), 1)?,
            ResidualUnit::load(w, &format!("{prefix}.res_unit2"), 3)?,
            ResidualUnit::load(w, &format!("{prefix}.res_unit3"), 9)?,
        ];
        Ok(Self {
            snake1: Snake::load(w, &format!("{prefix}.snake1"))?,
            conv_t1: ConvT::load(w, &format!("{prefix}.conv_t1"), stride, pad, output_pad)?,
            res_units,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.snake1.forward(x)?;
        h = self.conv_t1.forward(&h)?;
        for u in &self.res_units {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// DAC acoustic_decoder: conv1 → 5× DecoderBlock → snake1 → conv2 → [1,samples].
struct AcousticDecoder {
    conv1: Conv,
    blocks: Vec<DecoderBlock>,
    snake1: Snake,
    conv2: Conv,
}

impl AcousticDecoder {
    fn load(w: &OmniVoiceCodecWeights, strides: &[usize]) -> Result<Self> {
        let conv1 = Conv::load(w, "acoustic_decoder.conv1", 1, 3, 1)?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &s) in strides.iter().enumerate() {
            blocks.push(DecoderBlock::load(w, &format!("acoustic_decoder.block.{i}"), s)?);
        }
        let snake1 = Snake::load(w, "acoustic_decoder.snake1")?;
        let conv2 = Conv::load(w, "acoustic_decoder.conv2", 1, 3, 1)?;
        Ok(Self { conv1, blocks, snake1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.snake1.forward(&h)?;
        // tanh снят (_adjust_dac_decoder → Identity).
        self.conv2.forward(&h)
    }
}

/// Один RVQ-квантизатор decode: `embed_lookup(codes) → project_out (Linear)`.
struct Quantizer {
    embed: Tensor,
    project_out_w: Tensor,
    project_out_b: Tensor,
}

impl Quantizer {
    fn load(w: &OmniVoiceCodecWeights, i: usize) -> Result<Self> {
        let p = format!("quantizer.quantizers.{i}");
        Ok(Self {
            embed: w.get(&format!("{p}.codebook.embed"))?.clone(),
            project_out_w: w.get(&format!("{p}.project_out.weight"))?.clone(),
            project_out_b: w.get(&format!("{p}.project_out.bias"))?.clone(),
        })
    }

    /// `codes_i [T] (i64) → [T, hidden]`.
    fn decode(&self, codes_i: &Tensor) -> Result<Tensor> {
        let q = self.embed.index_select(0, codes_i).map_err(err)?; // [T,64]
        let out = q.linear(&self.project_out_w).map_err(err)?; // [T,1024]
        out.broadcast_add(&self.project_out_b.unsqueeze(0).map_err(err)?).map_err(err)
    }
}

/// Decode-путь нейро-кодека HiggsAudioV2: коды `[n_q, T]` → волна 24 кГц.
pub struct CodecDecoder {
    quantizers: Vec<Quantizer>,
    fc2_w: Tensor,
    fc2_b: Tensor,
    decoder: AcousticDecoder,
    device: Device,
}

impl CodecDecoder {
    /// Собрать decoder из codec-весов и конфига HiggsAudioV2. `n_q` — число
    /// квантизаторов, которое реально подаётся в decode (OmniVoice = 8).
    pub fn build(cfg: &HiggsAudioConfig, w: &OmniVoiceCodecWeights, n_q: usize) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            quantizers.push(Quantizer::load(w, i)?);
        }
        let fc2_w = w.get("fc2.weight")?.clone();
        let fc2_b = w.get("fc2.bias")?.clone();
        let decoder = AcousticDecoder::load(w, &cfg.acoustic_model_config.upsampling_ratios)?;
        Ok(Self {
            quantizers,
            fc2_w,
            fc2_b,
            decoder,
            device: w.device,
        })
    }

    /// `codes [n_q, T] (i64)` → волна `[samples] (f32)`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        if codes.rank() != 2 {
            return Err(OmniVoiceError::Inference(format!(
                "codec.decode: expect codes [n_q,T], got {:?}",
                codes.dims()
            )));
        }
        let n_q = codes.dims()[0];
        let t = codes.dims()[1];
        if n_q > self.quantizers.len() {
            return Err(OmniVoiceError::Inference(format!(
                "codec.decode: codes have {n_q} codebooks > {} quantizers",
                self.quantizers.len()
            )));
        }
        // codes → host (I64); codes_i строим контигуозно через from_vec (narrow/squeeze
        // дают non-contiguous → contiguous() на int-CUDA = "cuda unary: dtype").
        let codes_host = codes
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(err)?; // [n_q*T]

        // RVQ.decode: Σ_i project_out(embed_lookup(codes_i)). Σ в [T,hidden].
        let mut quantized: Option<Tensor> = None;
        for i in 0..n_q {
            let codes_i =
                Tensor::from_vec(codes_host[i * t..(i + 1) * t].to_vec(), vec![t], self.device)
                    .map_err(err)?; // [T]
            let q = self.quantizers[i].decode(&codes_i)?; // [T,1024]
            quantized = Some(match quantized {
                None => q,
                Some(acc) => acc.broadcast_add(&q).map_err(err)?,
            });
        }
        let quantized = quantized.ok_or_else(|| {
            OmniVoiceError::Inference("codec.decode: zero codebooks".into())
        })?; // [T,1024]

        // fc2: Linear 1024→256, на [T,1024] → [T,256], затем channel-first [1,256,T].
        let acoustic = quantized.linear(&self.fc2_w).map_err(err)?; // [T,256]
        let acoustic = acoustic
            .broadcast_add(&self.fc2_b.unsqueeze(0).map_err(err)?)
            .map_err(err)?;
        let acoustic = acoustic
            .transpose(0, 1)
            .map_err(err)?
            .contiguous()
            .map_err(err)?
            .reshape(vec![1, acoustic.dims()[1], t])
            .map_err(err)?; // [1,256,T]

        // DAC acoustic_decoder → [1,1,samples].
        let audio = self.decoder.forward(&acoustic)?;
        let samples = audio.dims()[audio.rank() - 1];
        audio.reshape(vec![samples]).map_err(err)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
