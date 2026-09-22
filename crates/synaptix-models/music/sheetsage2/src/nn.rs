//! Слои с точностью «как у автокаста» релиза.
//!
//! Эталонный режим SheetSage2 — `torch.autocast(bfloat16)` поверх весов F32:
//! матричные операции (linear, свёртки, внимание) идут в BF16, а LayerNorm,
//! нормы и softmax — в F32; сложение BF16 с F32 даёт F32. Отсюда правило:
//! остаточный поток держит тот тип, который получился бы в торче, а слои
//! возвращают тип, который вернул бы автокаст. При `compute = F32` всё
//! сводится к обычному F32.

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::loader::Weights;
use crate::SheetError;

pub type R<T> = Result<T, SheetError>;

/// Сложение с продвижением типа: одинаковые — как есть, разные — в F32.
pub fn add(a: &Tensor, b: &Tensor) -> R<Tensor> {
    if a.dtype() == b.dtype() {
        Ok(a.add(b)?)
    } else {
        Ok(a.to_dtype(DType::F32)?.add(&b.to_dtype(DType::F32)?)?)
    }
}

/// Linear в вычислительном типе (вес и смещение уже в нём).
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str, compute: DType, bias: bool) -> R<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"), compute)?,
            bias: if bias { Some(w.get(&format!("{prefix}.bias"), compute)?) } else { None },
        })
    }

    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let x = x.to_dtype(self.weight.dtype())?;
        Ok(x.linear_bias_residual(&self.weight, self.bias.as_ref(), None)?)
    }
}

/// LayerNorm: вход любого типа, выход F32 (как у автокаста).
pub struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    pub fn load(w: &Weights, prefix: &str, eps: f32) -> R<Self> {
        Ok(Self {
            weight: w.f32(&format!("{prefix}.weight"))?,
            bias: w.f32(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        Ok(layer_norm(&x, Some(&self.weight), Some(&self.bias), self.eps)?)
    }
}

/// GELU (точная, через erf) — в типе входа, считается в F32 и округляется раз.
pub fn gelu(x: &Tensor) -> R<Tensor> {
    let dtype = x.dtype();
    if dtype == DType::F32 {
        return Ok(x.gelu_exact()?);
    }
    Ok(x.to_dtype(DType::F32)?.gelu_exact()?.to_dtype(dtype)?)
}

/// Внимание `[B, H, Tq, D] × [B, H, Tk, D]`. На CUDA в BF16/F16 — flash-ядро;
/// иначе — явный softmax по блокам запросов (память `H·блок·Tk`, а не `H·Tq·Tk`).
/// `causal` — запрос `i` видит ключи до `i + (Tk − Tq)` включительно.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> R<Tensor> {
    let low = matches!(q.dtype(), DType::BF16 | DType::F16);
    if low && matches!(q.device(), Device::Cuda(_)) {
        match q.flash_attention(k, v, scale, causal) {
            Ok(out) => return Ok(out),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e.into()),
        }
    }
    let (b, h, tq, d) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let tk = k.dims()[2];
    let dtype = q.dtype();
    let q3 = q.contiguous()?.reshape(vec![b * h, tq, d])?;
    let kt = k.contiguous()?.reshape(vec![b * h, tk, d])?.transpose(1, 2)?.contiguous()?;
    let v3 = v.contiguous()?.reshape(vec![b * h, tk, d])?;
    let offset = tk as isize - tq as isize;
    const BLOCK: usize = 1024;
    let mut outs = Vec::with_capacity(tq.div_ceil(BLOCK));
    let mut start = 0usize;
    while start < tq {
        let len = BLOCK.min(tq - start);
        let qb = q3.narrow(1, start, len)?.contiguous()?;
        let mut scores = qb.matmul(&kt)?.to_dtype(DType::F32)?.affine(scale, 0.0)?;
        if causal {
            let mut mask = vec![0f32; len * tk];
            for i in 0..len {
                let last = start as isize + i as isize + offset;
                for j in 0..tk {
                    if j as isize > last {
                        mask[i * tk + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec(mask, vec![1usize, len, tk], Device::Cpu)?.to_device(q.device())?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = softmax_dim(&scores, 2)?.to_dtype(dtype)?;
        outs.push(probs.matmul(&v3)?);
        start += len;
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    let out = if refs.len() == 1 { outs[0].clone() } else { Tensor::cat(&refs, 1)? };
    Ok(out.reshape(vec![b, h, tq, d])?)
}

/// `[B, T, H·D]` → `[B, H, T, D]`.
pub fn split_heads(x: &Tensor, heads: usize) -> R<Tensor> {
    let (b, t, width) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    Ok(x.reshape(vec![b, t, heads, width / heads])?.permute(vec![0, 2, 1, 3])?.contiguous()?)
}

/// `[B, H, T, D]` → `[B, T, H·D]`.
pub fn merge_heads(x: &Tensor) -> R<Tensor> {
    let (b, h, t, d) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
    Ok(x.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, t, h * d])?)
}

/// Depthwise-свёртка `[B, C, T]`, вес `[C, 1, K]`, с нулевым паддингом `pad`.
/// На CUDA — ядро движка; на CPU — прямой цикл (разложение движка по каналам
/// дало бы тысячу мелких свёрток на слой).
pub fn depthwise_conv(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, pad: usize) -> R<Tensor> {
    let x = x.contiguous()?;
    match x.dwconv1d(weight, bias, 1, pad, false) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e.into()),
    }
    let (b, c, t) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let k = weight.dims()[2];
    let dtype = x.dtype();
    let device = x.device();
    let xs: Vec<f32> = x.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1()?;
    let ws: Vec<f32> = weight.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1()?;
    let bs: Option<Vec<f32>> = match bias {
        Some(bt) => Some(bt.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1()?),
        None => None,
    };
    let t_out = t + 2 * pad + 1 - k;
    let mut out = vec![0f32; b * c * t_out];
    std::thread::scope(|scope| {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let rows = b * c;
        let per = rows.div_ceil(threads);
        for (ti, chunk) in out.chunks_mut(per * t_out).enumerate() {
            let (xs, ws, bs) = (&xs, &ws, &bs);
            scope.spawn(move || {
                for (r, row) in chunk.chunks_mut(t_out).enumerate() {
                    let bc = ti * per + r;
                    let ch = bc % c;
                    let src = &xs[bc * t..(bc + 1) * t];
                    let wk = &ws[ch * k..(ch + 1) * k];
                    let b0 = bs.as_ref().map(|v| v[ch]).unwrap_or(0.0);
                    for (o, dst) in row.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for (j, &wv) in wk.iter().enumerate() {
                            let idx = o as isize + j as isize - pad as isize;
                            if idx >= 0 && (idx as usize) < t {
                                acc += src[idx as usize] * wv;
                            }
                        }
                        *dst = acc + b0;
                    }
                }
            });
        }
    });
    Ok(Tensor::from_vec(out, vec![b, c, t_out], Device::Cpu)?.to_device(device)?.to_dtype(dtype)?)
}
