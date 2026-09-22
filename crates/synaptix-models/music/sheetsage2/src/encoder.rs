//! Энкодер: MERT2 (ConvNeXt-сабсэмплер + 24 конформер-блока с RoPE) →
//! взвешенная сумма всех слоёв → проекция в ширину декодера.
//!
//! LoRA-адаптеры внимания релиза влиты в веса при упаковке (`W += B·A·α/r` в
//! F32), так что здесь это обычный MERT2. Энкодер всегда видит полное окно
//! 300 с — хвост дополнен тишиной, как в релизе; маски нет, внимание по всем
//! 7 500 кадрам.
//!
//! Типы повторяют автокаст релиза (см. [`crate::nn`]): у первой ступени
//! сабсэмплера остаток в F32 (вход — мел в F32), у второй и третьей — в BF16
//! (после страйд-свёртки), у конформер-блоков — снова F32 (финальный
//! LayerNorm каждого блока отдаёт F32).

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::{Mert2Config, SheetSage2Config};
use crate::loader::Weights;
use crate::mel::MelFrontend;
use crate::nn::{self, add, attention, depthwise_conv, gelu, merge_heads, Linear, Norm, R};

struct ConvNextLayer {
    dw_weight: Tensor,
    dw_bias: Tensor,
    norm: Norm,
    fc1: Linear,
    grn_weight: Tensor,
    grn_bias: Tensor,
    fc2: Linear,
}

impl ConvNextLayer {
    fn load(w: &Weights, p: &str, eps: f32, compute: DType) -> R<Self> {
        Ok(Self {
            dw_weight: w.get(&format!("{p}.depthwise_block.1.weight"), compute)?,
            dw_bias: w.get(&format!("{p}.depthwise_block.1.bias"), compute)?,
            norm: Norm::load(w, &format!("{p}.pointwise_block.0"), eps)?,
            fc1: Linear::load(w, &format!("{p}.pointwise_block.1"), compute, true)?,
            grn_weight: w.f32(&format!("{p}.pointwise_block.3.weight"))?,
            grn_bias: w.f32(&format!("{p}.pointwise_block.3.bias"))?,
            fc2: Linear::load(w, &format!("{p}.pointwise_block.4"), compute, true)?,
        })
    }

    /// `[1, T, C]` → `[1, T, C]`; тип остатка — по правилам продвижения.
    fn forward(&self, h: &Tensor) -> R<Tensor> {
        let compute = self.dw_weight.dtype();
        let x = h.to_dtype(compute)?.transpose(1, 2)?.contiguous()?;
        let x = depthwise_conv(&x, &self.dw_weight, Some(&self.dw_bias), 3)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let y = gelu(&self.fc1.forward(&self.norm.forward(&x)?)?)?;
        // Global Response Norm: L2 по времени (всё окно), нормированная средним по каналам.
        let y32 = y.to_dtype(DType::F32)?;
        let magnitude = y32.sqr()?.sum_keepdim(1)?.sqrt()?;
        let scale = magnitude.broadcast_div(&magnitude.mean_keepdim(2)?.add_scalar(1e-6)?)?;
        let grn = y32
            .broadcast_mul(&scale)?
            .broadcast_mul(&self.grn_weight)?
            .broadcast_add(&self.grn_bias)?
            .add(&y32)?;
        let y = self.fc2.forward(&grn)?;
        add(h, &y)
    }
}

struct Resampler {
    norm: Norm,
    /// Свёртка k=2, stride=2 как linear по склеенным парам кадров: `[Cout, 2·Cin]`.
    proj: Linear,
}

impl Resampler {
    fn load(w: &Weights, p: &str, eps: f32, compute: DType) -> R<Self> {
        let weight = w.get(&format!("{p}.2.weight"), compute)?; // [Cout, Cin, 2]
        let (cout, cin, k) = (weight.dims()[0], weight.dims()[1], weight.dims()[2]);
        let weight = weight.permute(vec![0, 2, 1])?.contiguous()?.reshape(vec![cout, k * cin])?;
        Ok(Self {
            norm: Norm::load(w, &format!("{p}.0"), eps)?,
            proj: Linear::from_parts(weight, Some(w.get(&format!("{p}.2.bias"), compute)?)),
        })
    }

    fn forward(&self, h: &Tensor) -> R<Tensor> {
        let x = self.norm.forward(h)?;
        let (b, t, c) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let t_out = t / 2;
        let x = x.narrow(1, 0, t_out * 2)?.contiguous()?.reshape(vec![b, t_out, 2 * c])?;
        self.proj.forward(&x)
    }
}

struct Stage {
    resampler: Option<Resampler>,
    layers: Vec<ConvNextLayer>,
}

struct ConformerBlock {
    ffn1_norm: Norm,
    ffn1_in: Linear,
    ffn1_out: Linear,
    attn_norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    conv_norm: Norm,
    /// Точечная свёртка 1×1 → GLU: `[2C, C]`.
    pw1: Linear,
    dw: Tensor,
    conv_inner_norm: Norm,
    pw2: Linear,
    ffn2_norm: Norm,
    ffn2_in: Linear,
    ffn2_out: Linear,
    final_norm: Norm,
}

impl ConformerBlock {
    fn load(w: &Weights, p: &str, cfg: &Mert2Config, compute: DType) -> R<Self> {
        let eps = cfg.layer_norm_eps;
        let pointwise = |name: &str| -> R<Linear> {
            let weight = w.get(&format!("{p}.conv_module.conv_block.{name}.weight"), compute)?;
            let (o, i) = (weight.dims()[0], weight.dims()[1]);
            Ok(Linear::from_parts(weight.reshape(vec![o, i])?, None))
        };
        Ok(Self {
            ffn1_norm: Norm::load(w, &format!("{p}.ffn1_layer_norm"), eps)?,
            ffn1_in: Linear::load(w, &format!("{p}.ffn1.w_1"), compute, true)?,
            ffn1_out: Linear::load(w, &format!("{p}.ffn1.w_2"), compute, true)?,
            attn_norm: Norm::load(w, &format!("{p}.attn_layer_norm"), eps)?,
            q: Linear::load(w, &format!("{p}.attn.query_proj"), compute, true)?,
            k: Linear::load(w, &format!("{p}.attn.key_proj"), compute, true)?,
            v: Linear::load(w, &format!("{p}.attn.value_proj"), compute, true)?,
            o: Linear::load(w, &format!("{p}.attn.out_proj"), compute, true)?,
            conv_norm: Norm::load(w, &format!("{p}.conv_module.layer_norm"), eps)?,
            pw1: pointwise("1")?,
            dw: w.get(&format!("{p}.conv_module.conv_block.3.weight"), compute)?,
            conv_inner_norm: Norm::load(w, &format!("{p}.conv_module.conv_block.4.1"), eps)?,
            pw2: pointwise("6")?,
            ffn2_norm: Norm::load(w, &format!("{p}.ffn2_layer_norm"), eps)?,
            ffn2_in: Linear::load(w, &format!("{p}.ffn2.w_1"), compute, true)?,
            ffn2_out: Linear::load(w, &format!("{p}.ffn2.w_2"), compute, true)?,
            final_norm: Norm::load(w, &format!("{p}.final_layer_norm"), eps)?,
        })
    }

    fn ffn(norm: &Norm, fc1: &Linear, fc2: &Linear, h: &Tensor) -> R<Tensor> {
        let y = fc2.forward(&gelu(&fc1.forward(&norm.forward(h)?)?)?)?;
        let y = y.affine(0.5, 0.0)?;
        add(h, &y)
    }

    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> R<Tensor> {
        let d = x.dims()[3];
        let half = d / 2;
        let first = x.narrow(3, 0, half)?.contiguous()?;
        let second = x.narrow(3, half, half)?.contiguous()?.neg()?;
        let rotated = Tensor::cat(&[&second, &first], 3)?;
        Ok(x.broadcast_mul(cos)?.add(&rotated.broadcast_mul(sin)?)?)
    }

    fn forward(&self, h: &Tensor, cos: &Tensor, sin: &Tensor, heads: usize, kernel: usize) -> R<Tensor> {
        let h = Self::ffn(&self.ffn1_norm, &self.ffn1_in, &self.ffn1_out, h)?;

        // Внимание с RoPE (половинное вращение), без маски.
        let x = self.attn_norm.forward(&h)?;
        let (b, t, width) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let hd = width / heads;
        let shape = vec![b, t, heads, hd];
        let q = Self::rope(&self.q.forward(&x)?.reshape(shape.clone())?, cos, sin)?;
        let k = Self::rope(&self.k.forward(&x)?.reshape(shape.clone())?, cos, sin)?;
        let v = self.v.forward(&x)?.reshape(shape)?;
        let perm = |z: Tensor| -> R<Tensor> { Ok(z.permute(vec![0, 2, 1, 3])?.contiguous()?) };
        let attn = attention(&perm(q)?, &perm(k)?, &perm(v)?, 1.0 / (hd as f32).sqrt(), false)?;
        let attn = self.o.forward(&merge_heads(&attn)?)?;
        let h = add(&attn, &h)?;

        // Свёрточный модуль: 1×1 → GLU → depthwise k31 → LN → GELU → 1×1.
        let x = self.pw1.forward(&self.conv_norm.forward(&h)?)?; // [1, T, 2C]
        let c = x.dims()[2] / 2;
        let a = x.narrow(2, 0, c)?.contiguous()?.to_dtype(DType::F32)?;
        let g = x.narrow(2, c, c)?.contiguous()?.to_dtype(DType::F32)?.sigmoid()?;
        let x = a.mul(&g)?.to_dtype(x.dtype())?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = depthwise_conv(&x, &self.dw, None, (kernel - 1) / 2)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = gelu(&self.conv_inner_norm.forward(&x)?)?;
        let x = self.pw2.forward(&x)?;
        let h = add(&x, &h)?;

        let h = Self::ffn(&self.ffn2_norm, &self.ffn2_in, &self.ffn2_out, &h)?;
        self.final_norm.forward(&h)
    }
}

/// Вход энкодера — нормированный лог-мел; выход — память декодера `[1, кадры, 512]`.
pub struct Encoder {
    pub mel: MelFrontend,
    stages: Vec<Stage>,
    blocks: Vec<ConformerBlock>,
    /// softmax весов слоёв (F32 на хосте): `[вход, слой 1, …, слой 24]`.
    layer_weights: Vec<f32>,
    projection: Linear,
    config: Mert2Config,
    compute: DType,
    device: Device,
}

impl Encoder {
    pub fn load(w: &Weights, cfg: &SheetSage2Config, compute: DType) -> R<Self> {
        let mc = &cfg.backbone;
        let device = w.device();
        let window: Vec<f32> = w
            .get_on("encoder.feature_extractor.spectrogram.window", Device::Cpu, DType::F32)?
            .to_vec1()?;
        let mel = MelFrontend::new(
            mc,
            window,
            w.f32("encoder.feature_extractor.mel_scale.fb")?,
            w.f32("encoder.feature_extractor.mel_mean")?,
            w.f32("encoder.feature_extractor.mel_std")?,
            device,
        )?;
        let eps = mc.subsampling_layer_norm_eps;
        let mut channels = vec![mc.num_mel_bins];
        channels.extend(&mc.subsampling_channels);
        let strides = [1usize, 2, 2];
        let mut stages = Vec::with_capacity(3);
        for i in 0..3 {
            let p = format!("encoder.subsampling_module.{i}");
            let resampler = if channels[i] != channels[i + 1] || strides[i] > 1 {
                Some(Resampler::load(w, &format!("{p}.resampling_layer"), eps, compute)?)
            } else {
                None
            };
            let layers = (0..mc.subsampling_depths[i])
                .map(|j| ConvNextLayer::load(w, &format!("{p}.convnext_layers.{j}"), eps, compute))
                .collect::<R<Vec<_>>>()?;
            stages.push(Stage { resampler, layers });
        }
        let blocks = (0..mc.num_hidden_layers)
            .map(|n| ConformerBlock::load(w, &format!("encoder.layers.{n}"), mc, compute))
            .collect::<R<Vec<_>>>()?;
        let raw: Vec<f32> = w.get_on("layer_weight", Device::Cpu, DType::F32)?.to_vec1()?;
        if raw.len() != mc.num_hidden_layers + 1 {
            return Err(crate::SheetError::Load("layer_weight: число весов ≠ число слоёв + 1".into()));
        }
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let layer_weights = exps.iter().map(|e| e / sum).collect();
        Ok(Self {
            mel,
            stages,
            blocks,
            layer_weights,
            projection: Linear::load(w, "encoder_projection", compute, true)?,
            config: mc.clone(),
            compute,
            device,
        })
    }

    /// Таблицы RoPE `[1, T, 1, head_dim]` (половины продублированы) в типе `dtype`.
    fn rope_tables(&self, t: usize, dtype: DType) -> R<(Tensor, Tensor)> {
        let hd = self.config.head_dim();
        let half = hd / 2;
        let base = self.config.rotary_embedding_base;
        let inv: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / hd as f32)).collect();
        let mut cos = vec![0f32; t * hd];
        let mut sin = vec![0f32; t * hd];
        for p in 0..t {
            for (i, &f) in inv.iter().enumerate() {
                let a = p as f32 * f;
                let (s, c) = a.sin_cos();
                cos[p * hd + i] = c;
                cos[p * hd + half + i] = c;
                sin[p * hd + i] = s;
                sin[p * hd + half + i] = s;
            }
        }
        let mk = |v: Vec<f32>| -> R<Tensor> {
            Ok(Tensor::from_vec(v, vec![1usize, t, 1, hd], Device::Cpu)?.to_device(self.device)?.to_dtype(dtype)?)
        };
        Ok((mk(cos)?, mk(sin)?))
    }

    /// Окно звука (24 кГц моно, уже дополненное до окна модели) → память
    /// декодера `[1, кадры, 512]` в вычислительном типе.
    pub fn forward(&self, audio: &[f32], cancel: &dyn Fn() -> bool) -> R<Tensor> {
        let mel = self.mel.forward(audio)?; // [1, T, 128] F32
        let mut h = mel;
        for stage in &self.stages {
            if let Some(r) = &stage.resampler {
                h = r.forward(&h)?;
            }
            for layer in &stage.layers {
                h = layer.forward(&h)?;
            }
        }
        if cancel() {
            return Err(crate::SheetError::Cancelled("энкодер"));
        }
        let t = h.dims()[1];
        // Таблицы RoPE — в типе выхода сабсэмплера (у релиза `.to(hidden.dtype)`).
        let (cos, sin) = self.rope_tables(t, h.dtype())?;
        // Первый член смеси: 0-мерный вес F32 × тензор BF16 даёт BF16 (вес округлён).
        let w0 = if h.dtype() == DType::BF16 {
            half::bf16::from_f32(self.layer_weights[0]).to_f32()
        } else {
            self.layer_weights[0]
        };
        let mut mixed = h.affine(w0, 0.0)?;
        let heads = self.config.num_attention_heads;
        let kernel = self.config.conv_depthwise_kernel_size;
        for (i, block) in self.blocks.iter().enumerate() {
            if cancel() {
                return Err(crate::SheetError::Cancelled("энкодер"));
            }
            h = block.forward(&h, &cos, &sin, heads, kernel)?;
            mixed = nn::add(&mixed, &h.affine(self.layer_weights[i + 1], 0.0)?)?;
        }
        let memory = self.projection.forward(&mixed)?;
        Ok(memory.to_dtype(self.compute)?)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
