//! Vocos (charactr/vocos-mel-24khz): мел 100 → свёртка k7 → LN → 8 ConvNeXt (k7, 1536, GELU, γ) →
//! LN → Linear 1026 → |X| = min(eˣ, 100), фаза → ISTFT (n_fft 1024, hop 256, центр).

use synaptix_audio::stft::{istft, PadMode, StftConfig};
use synaptix_audio::window::WindowKind;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::conv::{conv1d, depthwise_conv};
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::dit::W;
use crate::{F5Error, Result, HOP};

fn t12(x: &Tensor) -> std::result::Result<Tensor, SynaptixError> {
    x.transpose(1, 2)?.contiguous()
}

struct Block {
    dw: (Tensor, Tensor),
    ln: (Tensor, Tensor),
    pw1: QuantLinear,
    pw2: QuantLinear,
    gamma: Tensor,
}

pub struct Vocos {
    embed: (Tensor, Tensor),
    norm: (Tensor, Tensor),
    blocks: Vec<Block>,
    final_norm: (Tensor, Tensor),
    out: QuantLinear,
    device: Device,
    dtype: DType,
}

impl Vocos {
    pub fn load(w: &W) -> Result<Self> {
        let pair = |p: &str| -> Result<(Tensor, Tensor)> { Ok((w.get(&format!("{p}.weight"))?, w.get(&format!("{p}.bias"))?)) };
        let lin = |p: &str| -> Result<QuantLinear> { Ok(QuantLinear::dense(w.get(&format!("{p}.weight"))?, Some(w.get(&format!("{p}.bias"))?))?) };
        let mut blocks = Vec::new();
        for i in 0.. {
            let p = format!("backbone.convnext.{i}");
            if w.l.load_to(&format!("{}{p}.gamma", w.prefix), Device::Cpu, DType::F32).is_err() {
                break;
            }
            let g = w.get(&format!("{p}.gamma"))?;
            let d = g.dims()[0];
            blocks.push(Block {
                dw: pair(&format!("{p}.dwconv"))?,
                ln: pair(&format!("{p}.norm"))?,
                pw1: lin(&format!("{p}.pwconv1"))?,
                pw2: lin(&format!("{p}.pwconv2"))?,
                gamma: g.reshape(vec![1usize, 1, d])?,
            });
        }
        if blocks.is_empty() {
            return Err(F5Error::Load("vocoder: нет блоков backbone.convnext.*".into()));
        }
        Ok(Self {
            embed: pair("backbone.embed")?,
            norm: pair("backbone.norm")?,
            blocks,
            final_norm: pair("backbone.final_layer_norm")?,
            out: lin("head.out")?,
            device: w.device,
            dtype: w.dtype,
        })
    }

    /// Мел `[кадр][100]` (хост) → звук 24 кГц длиной `(кадров − 1)·256`.
    pub fn decode(&self, mel: &[Vec<f32>]) -> Result<Vec<f32>> {
        let t = mel.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        let c = mel[0].len();
        let flat: Vec<f32> = mel.iter().flatten().copied().collect();
        let x = Tensor::from_vec(flat, vec![1usize, t, c], Device::Cpu)?.to_dtype(self.dtype)?.to_device(self.device)?;
        let x = conv1d(&t12(&x)?, &self.embed.0, Some(&self.embed.1), 1, 3)?; // [1, D, T]
        let mut h = layer_norm(&t12(&x)?, Some(&self.norm.0), Some(&self.norm.1), 1e-6)?; // [1, T, D]
        for b in &self.blocks {
            let d = h.dims()[2];
            let y = t12(&depthwise_conv(&t12(&h)?, &b.dw.0, Some(&b.dw.1), 1, 3, d)?)?;
            let y = layer_norm(&y, Some(&b.ln.0), Some(&b.ln.1), 1e-6)?;
            let y = b.pw2.forward(&b.pw1.forward(&y)?.gelu_exact()?)?;
            h = h.add(&y.broadcast_mul(&b.gamma)?)?;
        }
        let h = layer_norm(&h, Some(&self.final_norm.0), Some(&self.final_norm.1), 1e-6)?;
        let o = self.out.forward(&h)?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?; // [T, 1026]
        let nf = o.len() / t / 2;
        let spec: Vec<Vec<synaptix_audio::stft::Complex32>> = (0..t)
            .map(|i| {
                let r = &o[i * 2 * nf..(i + 1) * 2 * nf];
                (0..nf)
                    .map(|k| {
                        let m = r[k].exp().min(100.0);
                        let p = r[nf + k];
                        synaptix_audio::stft::Complex32::new(m * p.cos(), m * p.sin())
                    })
                    .collect()
            })
            .collect();
        let n_fft = (nf - 1) * 2;
        let cfg = StftConfig { n_fft, hop_length: HOP, win_length: n_fft, window: WindowKind::Hann, center: true, pad_mode: PadMode::Reflect };
        let mut y = istft(&spec, &cfg).map_err(|e| F5Error::Other(format!("istft: {e}")))?;
        y.truncate((t - 1) * HOP);
        Ok(y)
    }
}
