//! LTX-2.3 Audio VAE decode: аудио-латент `[B,8,F,16]` → стерео log-mel
//! `[B,2,T,64]` (T=4·F−3 causal). 2D mel-VAE. Конфиг: pixel_norm, causality_axis
//! =height (causal по времени), z 8, mel 64, ch 128 ch_mult [1,2,4], 2 res-блока,
//! без attention. CausalConv2d (asymmetric pad), Upsample (nearest×2 + drop-first).

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::conv::conv2d::conv2d;

use crate::loader::{LtxCheckpoint, AUDIO_VAE_PREFIX};
use crate::LtxError;

type R<T> = Result<T, SynaptixError>;
const PIXEL_EPS: f64 = 1e-6; // build_normalization_layer(PIXEL) eps
const DSF: usize = 4; // LATENT_DOWNSAMPLE_FACTOR

fn pixel_norm(x: &Tensor) -> R<Tensor> {
    let ms = x.sqr()?.mean_keepdim(1)?;
    x.broadcast_div(&ms.add_scalar(PIXEL_EPS as f32)?.sqrt()?)?.contiguous()
}

/// CausalConv2d (causality_axis=height, k3): pad time(H) сверху 2/снизу 0,
/// freq(W) симметрично 1/1; conv2d valid. `k1` (nin_shortcut) — без паддинга.
fn cconv(x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
    let kh = w.dims()[2];
    if kh == 1 {
        return conv2d(x, w, Some(b), (1, 1), (0, 0), (1, 1));
    }
    let (bs, c, h, wd) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
    let dt = x.dtype();
    // pad time сверху на kh-1=2 (causal), freq симметрично 1/1.
    let ztop = Tensor::zeros(vec![bs, c, kh - 1, wd], dt, x.device())?;
    let xp = Tensor::cat(&[&ztop, x], 2)?; // H+2
    let h2 = h + kh - 1;
    let zl = Tensor::zeros(vec![bs, c, h2, 1], dt, x.device())?;
    let xp = Tensor::cat(&[&zl, &xp, &zl], 3)?.contiguous()?; // W+2
    conv2d(&xp, w, Some(b), (1, 1), (0, 0), (1, 1))
}

struct Conv {
    w: Tensor,
    b: Tensor,
}
impl Conv {
    fn load(ckpt: &LtxCheckpoint, p: &str, device: Device) -> Result<Self, LtxError> {
        Ok(Self {
            w: ckpt.get(&format!("{p}.conv.weight"))?.to_device(device)?.to_dtype(DType::F32)?,
            b: ckpt.get(&format!("{p}.conv.bias"))?.to_device(device)?.to_dtype(DType::F32)?,
        })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        cconv(x, &self.w, &self.b)
    }
}

/// ResnetBlock2D: PixelNorm→silu→conv1→PixelNorm→silu→conv2 (+nin_shortcut если
/// in≠out).
struct Resnet {
    conv1: Conv,
    conv2: Conv,
    nin: Option<Conv>,
}
impl Resnet {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = self.conv1.fwd(&pixel_norm(x)?.silu()?)?;
        let h = self.conv2.fwd(&pixel_norm(&h)?.silu()?)?;
        let skip = match &self.nin {
            Some(n) => n.fwd(x)?,
            None => x.clone(),
        };
        skip.add(&h)
    }
}

struct Stage {
    blocks: Vec<Resnet>,
    upsample: Option<Conv>,
}

pub struct AudioVaeDecoder {
    mean: Tensor, // [1,8,1,16]
    std: Tensor,
    conv_in: Conv,
    mid: Vec<Resnet>, // block_1, block_2
    up: Vec<Stage>,   // [level0, level1, level2]
    conv_out: Conv,
    device: Device,
}

impl AudioVaeDecoder {
    pub fn load(ckpt: &LtxCheckpoint, device: Device) -> Result<Self, LtxError> {
        let dec = format!("{AUDIO_VAE_PREFIX}.decoder");
        let stat = |n: &str| -> Result<Tensor, LtxError> {
            // stats [128] = (c=8 × f=16), c-major → reshape [1,8,1,16]
            Ok(ckpt.get(&format!("{AUDIO_VAE_PREFIX}.per_channel_statistics.{n}"))?
                .to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1, 8, 1, 16])?)
        };
        let conv = |p: &str| Conv::load(ckpt, p, device);
        let resnet = |p: &str, nin: bool| -> Result<Resnet, LtxError> {
            Ok(Resnet {
                conv1: conv(&format!("{p}.conv1"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                nin: if nin { Some(conv(&format!("{p}.nin_shortcut"))?) } else { None },
            })
        };
        // up: 3 уровня; block_out = ch*ch_mult[level] (ch=128, mult [1,2,4]).
        // Уровень l, первый блок: in≠out (nin_shortcut) если block_in≠block_out.
        // block_in: l2 in=512, l1 in=512, l0 in=256.
        let mut up = Vec::new();
        for level in 0..3 {
            let p = format!("{dec}.up.{level}");
            let mut blocks = Vec::new();
            for j in 0..3 {
                // nin только на первом блоке уровней 0 и 1 (где in≠out)
                let nin = j == 0 && (level == 0 || level == 1);
                blocks.push(resnet(&format!("{p}.block.{j}"), nin)?);
            }
            let upsample = if level != 0 { Some(conv(&format!("{p}.upsample.conv"))?) } else { None };
            up.push(Stage { blocks, upsample });
        }
        Ok(Self {
            mean: stat("mean-of-means")?,
            std: stat("std-of-means")?,
            conv_in: conv(&format!("{dec}.conv_in"))?,
            mid: vec![
                resnet(&format!("{dec}.mid.block_1"), false)?,
                resnet(&format!("{dec}.mid.block_2"), false)?,
            ],
            up,
            conv_out: conv(&format!("{dec}.conv_out"))?,
            device,
        })
    }

    /// Upsample: nearest×2 (H,W) → cconv → drop первой строки времени (height).
    fn upsample(&self, x: &Tensor, conv: &Conv) -> R<Tensor> {
        let up = x.upsample_nearest2x()?;
        let c = conv.fwd(&up)?;
        let h = c.dims()[2];
        c.narrow(2, 1, h - 1)?.contiguous() // drop first time row
    }

    /// Декод латента `[B,8,F,16]` → log-mel `[B,2,4·F−3,64]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, LtxError> {
        let x = latent.to_device(self.device)?.to_dtype(DType::F32)?;
        // un_normalize: x*std + mean (per channel,mel)
        let mut x = x.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        x = self.conv_in.fwd(&x)?;
        for r in &self.mid {
            x = r.forward(&x)?;
        }
        // up path: уровни 2,1,0 (reversed)
        for level in (0..3).rev() {
            let stage = &self.up[level];
            for blk in &stage.blocks {
                x = blk.forward(&x)?;
            }
            if let Some(uc) = &stage.upsample {
                x = self.upsample(&x, uc)?;
            }
        }
        x = pixel_norm(&x)?.silu()?;
        let out = self.conv_out.fwd(&x)?;
        let _ = DSF;
        Ok(out)
    }
}

/// log-mel для LTX audio-VAE: волна 16kHz → STFT (n_fft=1024, hop=160, hann,
/// center/reflect) → MAGNITUDE (power=1.0) → mel-64 (Slaney scale+norm, 0..8kHz)
/// → log(clamp 1e-5). Каналы стерео (mono дублируется). → `[1,2,T,64]` f32.
pub fn ltx_log_mel(channels: &[Vec<f32>], device: Device) -> Result<Tensor, LtxError> {
    use synaptix_audio::mel::{apply_mel_filterbank, mel_filterbank, MelConfig, MelNorm, MelScale};
    use synaptix_audio::stft::{stft, PadMode, StftConfig};
    use synaptix_audio::WindowKind;
    let scfg = StftConfig {
        n_fft: 1024,
        hop_length: 160,
        win_length: 1024,
        window: WindowKind::Hann,
        center: true,
        pad_mode: PadMode::Reflect,
    };
    let mcfg = MelConfig {
        n_mels: 64,
        f_min: 0.0,
        f_max: 8000.0,
        n_fft: 1024,
        sample_rate: 16000,
        mel_scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
    };
    let fb = mel_filterbank(&mcfg);
    let chans: Vec<&Vec<f32>> = if channels.len() >= 2 {
        vec![&channels[0], &channels[1]]
    } else {
        vec![&channels[0], &channels[0]] // mono → дублируем в стерео
    };
    let mut t_frames = 0usize;
    let mut data: Vec<f32> = Vec::new();
    let mut per_chan: Vec<Vec<Vec<f32>>> = Vec::with_capacity(2);
    for ch in &chans {
        let spec = stft(ch, &scfg).map_err(|e| LtxError::Load(format!("stft: {e}")))?;
        // magnitude (power=1.0): |X|
        let mag: Vec<Vec<f32>> = spec
            .iter()
            .map(|fr| fr.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect())
            .collect();
        let mel = apply_mel_filterbank(&mag, &fb); // [T][64]
        t_frames = mel.len();
        per_chan.push(mel);
    }
    for mel in &per_chan {
        for row in mel {
            for &v in row {
                data.push(v.max(1e-5).ln());
            }
        }
    }
    Ok(Tensor::from_vec(data, vec![1, 2, t_frames, 64], device)?)
}

/// Downsample (causality_axis=height): pad time(H) сверху 2 / freq(W) справа 1
/// (нулями) → conv 3×3 stride 2 без паддинга.
fn downsample(x: &Tensor, conv: &Conv) -> R<Tensor> {
    let (bs, c, h, wd) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
    let dt = x.dtype();
    let ztop = Tensor::zeros(vec![bs, c, 2, wd], dt, x.device())?;
    let xp = Tensor::cat(&[&ztop, x], 2)?; // H+2
    let zr = Tensor::zeros(vec![bs, c, h + 2, 1], dt, x.device())?;
    let xp = Tensor::cat(&[&xp, &zr], 3)?.contiguous()?; // W+1 (справа)
    conv2d(&xp, &conv.w, Some(&conv.b), (2, 2), (0, 0), (1, 1))
}

/// Audio-VAE ЭНКОДЕР: log-mel `[1,2,T,64]` → латент-токены `[1,Fa,128]`
/// (нормализованные means). Зеркало декодера: conv_in(2→128) → 3 уровня
/// (2 res-блока, ch×[1,2,4], downsample после уровней 0,1) → mid(block_1,block_2)
/// → pixel_norm→silu→conv_out(512→16, double_z) → means[:8] → patchify (c f) →
/// (x−mean)/std. Время ÷4 (T→Fa), мел 64→16.
pub struct AudioVaeEncoder {
    mean: Tensor, // [1,1,128] (по токенам)
    std: Tensor,
    conv_in: Conv,
    down: Vec<Stage2>, // 3 уровня: blocks + опц. downsample
    mid: Vec<Resnet>,
    conv_out: Conv,
    device: Device,
}

struct Stage2 {
    blocks: Vec<Resnet>,
    down: Option<Conv>,
}

impl AudioVaeEncoder {
    pub fn load(ckpt: &LtxCheckpoint, device: Device) -> Result<Self, LtxError> {
        let enc = format!("{AUDIO_VAE_PREFIX}.encoder");
        let stat = |n: &str| -> Result<Tensor, LtxError> {
            Ok(ckpt.get(&format!("{AUDIO_VAE_PREFIX}.per_channel_statistics.{n}"))?
                .to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1, 1, 128])?)
        };
        let conv = |p: &str| Conv::load(ckpt, p, device);
        let resnet = |p: &str, nin: bool| -> Result<Resnet, LtxError> {
            Ok(Resnet {
                conv1: conv(&format!("{p}.conv1"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                nin: if nin { Some(conv(&format!("{p}.nin_shortcut"))?) } else { None },
            })
        };
        // ch=128, ch_mult [1,2,4]: in_ch уровня = 128·in_mult[l] (in_mult = 1,1,2),
        // out = 128·mult[l] → nin на первом блоке уровней 1,2 (in≠out).
        let mut down = Vec::new();
        for level in 0..3 {
            let p = format!("{enc}.down.{level}");
            let mut blocks = Vec::new();
            for j in 0..2 {
                let nin = j == 0 && (level == 1 || level == 2);
                blocks.push(resnet(&format!("{p}.block.{j}"), nin)?);
            }
            let ds = if level != 2 { Some(conv(&format!("{p}.downsample"))?) } else { None };
            down.push(Stage2 { blocks, down: ds });
        }
        Ok(Self {
            mean: stat("mean-of-means")?,
            std: stat("std-of-means")?,
            conv_in: conv(&format!("{enc}.conv_in"))?,
            down,
            mid: vec![
                resnet(&format!("{enc}.mid.block_1"), false)?,
                resnet(&format!("{enc}.mid.block_2"), false)?,
            ],
            conv_out: conv(&format!("{enc}.conv_out"))?,
            device,
        })
    }

    /// log-mel `[1,2,T,64]` → нормализованные аудио-токены `[1,Fa,128]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor, LtxError> {
        let mut x = mel.to_device(self.device)?.to_dtype(DType::F32)?;
        x = self.conv_in.fwd(&x)?;
        for st in self.down.iter() {
            for b in &st.blocks {
                x = b.forward(&x)?;
            }
            if let Some(ds) = &st.down {
                x = downsample(&x, ds)?;
            }
        }
        for b in &self.mid {
            x = b.forward(&x)?;
        }
        let x = pixel_norm(&x)?.silu()?;
        let x = self.conv_out.fwd(&x)?; // [1,16,Fa,16] (double_z)
        let means = x.narrow(1, 0, 8)?.contiguous()?; // [1,8,Fa,16]
        let fa = means.dims()[2];
        // patchify b c t f → b t (c f): [1,Fa,128]
        let tok = means.transpose(1, 2)?.contiguous()?.reshape(vec![1, fa, 128])?;
        Ok(tok.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
