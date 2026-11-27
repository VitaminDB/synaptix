//! Вектор диктора: kaldi fbank (80 полос, 16 кГц, без дизеринга) − среднее по
//! времени → CAM++ (`modules/campplus/DTDNN.py`, 3D-Speaker): FCM (свёртки 2D
//! с понижением по частоте 80→10) → TDNN → три плотных блока D-TDNN с
//! контекстной маской (CAM) → статистический пулинг → 192.
//!
//! BatchNorm свёрнут при загрузке в масштаб и сдвиг на канал.

use synaptix_audio::kaldi::{subtract_mean, Fbank, FbankConfig};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::conv::conv1d::{conv1d, conv1d_dilated};
use synaptix_ops::conv::conv2d::conv2d;

use crate::loader::Comp;
use crate::nn::host;
use crate::{SeedVcError, R};

pub const FBANK_BINS: usize = 80;
pub const EMBED: usize = 192;
const BN_EPS: f32 = 1e-5;
/// Длина сегмента контекстного пулинга CAM, кадров.
const SEG_LEN: usize = 100;

/// Признаки эталона: `kaldi.fbank(num_mel_bins=80, dither=0)` по волне в [−1, 1]
/// минус среднее по времени. `[frames × 80]`.
pub fn fbank(wave_16k: &[f32]) -> (Vec<f32>, usize) {
    let fb = Fbank::new(FbankConfig::torchaudio(FBANK_BINS, 16_000.0));
    let mut f = fb.log_mel(wave_16k);
    subtract_mean(&mut f, FBANK_BINS);
    let frames = f.len() / FBANK_BINS;
    (f, frames)
}

/// Свёрнутый BatchNorm: `x · scale + shift`, формы `[1, C, 1(, 1)]`.
struct Bn {
    scale: Tensor,
    shift: Tensor,
}

impl Bn {
    fn load(c: &Comp, p: &str, rank: usize) -> R<Self> {
        let (mean, _) = c.host(&format!("{p}.running_mean"))?;
        let (var, _) = c.host(&format!("{p}.running_var"))?;
        let n = mean.len();
        let (w, b) = if c.has(&format!("{p}.weight")) { (c.host(&format!("{p}.weight"))?.0, c.host(&format!("{p}.bias"))?.0) } else { (vec![1f32; n], vec![0f32; n]) };
        let scale: Vec<f32> = (0..n).map(|i| w[i] / (var[i] + BN_EPS).sqrt()).collect();
        let shift: Vec<f32> = (0..n).map(|i| b[i] - mean[i] * scale[i]).collect();
        let mut shape = vec![1usize; rank];
        shape[1] = n;
        Ok(Self { scale: Tensor::from_vec(scale, shape.clone(), c.device())?, shift: Tensor::from_vec(shift, shape, c.device())? })
    }

    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

struct ResBlock {
    conv1: Tensor,
    bn1: Bn,
    conv2: Tensor,
    bn2: Bn,
    shortcut: Option<(Tensor, Bn)>,
    stride: usize,
}

impl ResBlock {
    fn load(c: &Comp, p: &str, stride: usize) -> R<Self> {
        let shortcut = if c.has(&format!("{p}.shortcut.0.weight")) { Some((c.get(&format!("{p}.shortcut.0.weight"), DType::F32)?, Bn::load(c, &format!("{p}.shortcut.1"), 4)?)) } else { None };
        Ok(Self {
            conv1: c.get(&format!("{p}.conv1.weight"), DType::F32)?,
            bn1: Bn::load(c, &format!("{p}.bn1"), 4)?,
            conv2: c.get(&format!("{p}.conv2.weight"), DType::F32)?,
            bn2: Bn::load(c, &format!("{p}.bn2"), 4)?,
            shortcut,
            stride,
        })
    }

    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let o = self.bn1.fwd(&conv2d(x, &self.conv1, None, (self.stride, 1), (1, 1), (1, 1))?)?.relu()?;
        let o = self.bn2.fwd(&conv2d(&o, &self.conv2, None, (1, 1), (1, 1), (1, 1))?)?;
        let sc = match &self.shortcut {
            Some((w, bn)) => bn.fwd(&conv2d(x, w, None, (self.stride, 1), (0, 0), (1, 1))?)?,
            None => x.clone(),
        };
        Ok(o.add(&sc)?.relu()?)
    }
}

struct DenseLayer {
    bn1: Bn,
    linear1: Tensor,
    bn2: Bn,
    local: Tensor,
    cam1: (Tensor, Tensor),
    cam2: (Tensor, Tensor),
    dilation: usize,
}

/// Среднее по сегментам по 100 кадров (последний — по фактической длине),
/// растянутое обратно до длины входа: `avg_pool1d(ceil_mode=True)` + expand.
fn seg_pooling(x: &Tensor) -> R<Tensor> {
    let (b, c, t) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let mut parts: Vec<Tensor> = Vec::new();
    let mut start = 0;
    while start < t {
        let len = SEG_LEN.min(t - start);
        let m = x.narrow(2, start, len)?.contiguous()?.mean_keepdim(2)?;
        parts.push(m.broadcast_as(vec![b, c, len])?.contiguous()?);
        start += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 2)?.contiguous()?)
}

impl DenseLayer {
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let y = conv1d(&self.bn1.fwd(x)?.relu()?, &self.linear1, None, 1, 0)?;
        let y = self.bn2.fwd(&y)?.relu()?;
        let local = conv1d_dilated(&y, &self.local, None, 1, self.dilation, self.dilation)?;
        let context = seg_pooling(&y)?.broadcast_add(&y.mean_keepdim(2)?)?;
        let context = conv1d(&context, &self.cam1.0, Some(&self.cam1.1), 1, 0)?.relu()?;
        let m = conv1d(&context, &self.cam2.0, Some(&self.cam2.1), 1, 0)?.sigmoid()?;
        Ok(local.mul(&m)?)
    }
}

struct Transit {
    bn: Bn,
    linear: Tensor,
}

pub struct CamPlus {
    conv1: Tensor,
    bn1: Bn,
    layer1: Vec<ResBlock>,
    layer2: Vec<ResBlock>,
    conv2: Tensor,
    bn2: Bn,
    tdnn: Tensor,
    tdnn_bn: Bn,
    blocks: Vec<(Vec<DenseLayer>, Transit)>,
    out_bn: Bn,
    /// `[192 × 1024]` на хосте и свёрнутый BatchNorm без аффинной части.
    dense: Vec<f32>,
    dense_scale: Vec<f32>,
    dense_shift: Vec<f32>,
    device: Device,
}

impl CamPlus {
    pub fn load(c: &Comp) -> R<Self> {
        let w = |n: &str| c.get(n, DType::F32);
        let res = |p: &str| -> R<Vec<ResBlock>> { Ok(vec![ResBlock::load(c, &format!("{p}.0"), 2)?, ResBlock::load(c, &format!("{p}.1"), 1)?]) };
        let mut blocks = Vec::with_capacity(3);
        for (bi, (layers, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)].into_iter().enumerate() {
            let mut ls = Vec::with_capacity(layers);
            for li in 0..layers {
                let p = format!("xvector.block{}.tdnnd{}", bi + 1, li + 1);
                ls.push(DenseLayer {
                    bn1: Bn::load(c, &format!("{p}.nonlinear1.batchnorm"), 3)?,
                    linear1: w(&format!("{p}.linear1.weight"))?,
                    bn2: Bn::load(c, &format!("{p}.nonlinear2.batchnorm"), 3)?,
                    local: w(&format!("{p}.cam_layer.linear_local.weight"))?,
                    cam1: (w(&format!("{p}.cam_layer.linear1.weight"))?, w(&format!("{p}.cam_layer.linear1.bias"))?),
                    cam2: (w(&format!("{p}.cam_layer.linear2.weight"))?, w(&format!("{p}.cam_layer.linear2.bias"))?),
                    dilation,
                });
            }
            let p = format!("xvector.transit{}", bi + 1);
            blocks.push((ls, Transit { bn: Bn::load(c, &format!("{p}.nonlinear.batchnorm"), 3)?, linear: w(&format!("{p}.linear.weight"))? }));
        }
        let (dense, dd) = c.host("xvector.dense.linear.weight")?;
        if dd[0] != EMBED {
            return Err(SeedVcError::Load(format!("campplus: размер вектора {} вместо {EMBED}", dd[0])));
        }
        let (mean, _) = c.host("xvector.dense.nonlinear.batchnorm.running_mean")?;
        let (var, _) = c.host("xvector.dense.nonlinear.batchnorm.running_var")?;
        let dense_scale: Vec<f32> = var.iter().map(|v| 1.0 / (v + BN_EPS).sqrt()).collect();
        let dense_shift: Vec<f32> = mean.iter().zip(&dense_scale).map(|(m, s)| -m * s).collect();
        Ok(Self {
            conv1: w("head.conv1.weight")?,
            bn1: Bn::load(c, "head.bn1", 4)?,
            layer1: res("head.layer1")?,
            layer2: res("head.layer2")?,
            conv2: w("head.conv2.weight")?,
            bn2: Bn::load(c, "head.bn2", 4)?,
            tdnn: w("xvector.tdnn.linear.weight")?,
            tdnn_bn: Bn::load(c, "xvector.tdnn.nonlinear.batchnorm", 3)?,
            blocks,
            out_bn: Bn::load(c, "xvector.out_nonlinear.batchnorm", 3)?,
            dense,
            dense_scale,
            dense_shift,
            device: c.device(),
        })
    }

    /// Признаки `[frames × 80]` → вектор диктора (192). `masked` — пулинг по первым
    /// `frames / 2` кадрам, как `compute_style` v2 (иначе по всем, как в v1).
    pub fn embed(&self, feats: &[f32], frames: usize, masked: bool) -> R<Vec<f32>> {
        if frames < 8 {
            return Err(SeedVcError::Input("образец голоса слишком короткий".into()));
        }
        // [frames, 80] → [1, 1, 80, frames].
        let mut x = vec![0f32; FBANK_BINS * frames];
        for t in 0..frames {
            for f in 0..FBANK_BINS {
                x[f * frames + t] = feats[t * FBANK_BINS + f];
            }
        }
        let x = Tensor::from_vec(x, vec![1, 1, FBANK_BINS, frames], self.device)?;
        let mut h = self.bn1.fwd(&conv2d(&x, &self.conv1, None, (1, 1), (1, 1), (1, 1))?)?.relu()?;
        for b in self.layer1.iter().chain(&self.layer2) {
            h = b.fwd(&h)?;
        }
        let h = self.bn2.fwd(&conv2d(&h, &self.conv2, None, (2, 1), (1, 1), (1, 1))?)?.relu()?;
        let d = h.dims().to_vec();
        let h = h.contiguous()?.reshape(vec![1, d[1] * d[2], d[3]])?;
        let mut h = self.tdnn_bn.fwd(&conv1d(&h, &self.tdnn, None, 2, 2)?)?.relu()?;
        for (layers, transit) in &self.blocks {
            for l in layers {
                h = Tensor::cat(&[&h, &l.fwd(&h)?], 1)?.contiguous()?;
            }
            h = conv1d(&transit.bn.fwd(&h)?.relu()?, &transit.linear, None, 1, 0)?;
        }
        let h = self.out_bn.fwd(&h)?.relu()?;
        let (ch, t) = (h.dims()[1], h.dims()[2]);
        let v = host(&h)?;
        let n = if masked { (frames / 2).min(t) } else { t };
        // Среднее и несмещённое стандартное отклонение по времени.
        let mut stats = vec![0f32; 2 * ch];
        for c in 0..ch {
            let row = &v[c * t..c * t + n];
            let mean = row.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            let var = row.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
            stats[c] = mean as f32;
            stats[ch + c] = var.sqrt() as f32;
        }
        let k = 2 * ch;
        Ok((0..EMBED)
            .map(|o| {
                let acc: f64 = self.dense[o * k..(o + 1) * k].iter().zip(&stats).map(|(w, s)| *w as f64 * *s as f64).sum();
                acc as f32 * self.dense_scale[o] + self.dense_shift[o]
            })
            .collect())
    }
}
