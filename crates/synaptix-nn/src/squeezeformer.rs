use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// Squeezeformer minimal block: stride-2 subsample → Linear projection →
/// repeat-interleave upsample обратно к исходной длине.
///
/// Полная схема (Kim et al., 2022) включает temporal U-Net через conv1d
/// stride-2 down/up + временную свёртку; здесь stub (без conv1d) реализует
/// downsample-upsample через индексные операции, чтобы зацементировать
/// публичный API. Реальный conv1d-stride можно подключить позже.
///
/// `forward(x: [B, T, in_channels])` → `[B, T, hidden_size]`.
pub struct Squeezeformer {
    pub proj: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl Squeezeformer {
    pub fn new(in_channels: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            in_channels,
            hidden_size,
        })
    }

    pub fn from_weights(proj_w: Tensor, proj_b: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(proj_w, proj_b)?;
        Ok(Self {
            in_channels: proj.in_features(),
            hidden_size: proj.out_features(),
            proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.in_channels {
            return Err(SynaptixError::Unsupported("Squeezeformer: expects x [B, T, in_channels]"));
        }
        let t = x.dims()[1];
        if t < 2 {
            // Слишком короткая последовательность — просто проектируем без resampling'а.
            return self.proj.forward(x);
        }
        let half = t / 2;
        let sub = subsample_stride2(x, 2 * half)?;
        let projected = self.proj.forward(&sub)?;
        let upsampled = projected.repeat_interleave(1, 2)?;
        // Возможно мы потеряли последний элемент при нечётном T → дополняем
        // повтором последнего фрейма (минимальный padding) до исходной длины.
        let cur_t = upsampled.dims()[1];
        if cur_t == t {
            Ok(upsampled)
        } else if cur_t < t {
            let last = upsampled.narrow(1, cur_t - 1, 1)?.contiguous()?;
            let upsampled_c = upsampled.contiguous()?;
            let pads: Vec<Tensor> = (0..(t - cur_t))
                .map(|_| last.clone())
                .collect();
            let mut parts: Vec<&Tensor> = vec![&upsampled_c];
            parts.extend(pads.iter());
            Tensor::cat(&parts, 1)
        } else {
            upsampled.narrow(1, 0, t)
        }
    }
}

fn subsample_stride2(x: &Tensor, even_len: usize) -> Result<Tensor> {
    // Берём чётные индексы 0, 2, 4, ... до even_len-2; результат имеет seq = even_len / 2.
    let half = even_len / 2;
    let mut chunks: Vec<Tensor> = Vec::with_capacity(half);
    for k in 0..half {
        chunks.push(x.narrow(1, 2 * k, 1)?.contiguous()?);
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, 1)
}
