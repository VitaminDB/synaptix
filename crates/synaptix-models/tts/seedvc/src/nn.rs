//! Общие кирпичи: линейный слой, RMSNorm, RoPE по парам соседних каналов.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::loader::Comp;
use crate::R;

pub fn host(t: &Tensor) -> R<Vec<f32>> {
    Ok(t.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

/// `y = x·Wᵀ + b` по последней оси; вес хранится транспонированным `[in, out]`.
pub struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    pub fn load(c: &Comp, name: &str, bias: bool) -> R<Self> {
        let w = c.get(&format!("{name}.weight"), DType::F32)?;
        let d = w.dims().to_vec();
        // Свёртка 1×1 `[out, in, 1]` — тот же линейный слой.
        let w = w.reshape(vec![d[0], d[1]])?.transpose(0, 1)?.contiguous()?;
        Ok(Self { w, b: if bias { Some(c.get(&format!("{name}.bias"), DType::F32)?) } else { None } })
    }

    pub fn out_dim(&self) -> usize {
        self.w.dims()[1]
    }

    pub fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let d = x.dims().to_vec();
        let n: usize = d[..d.len() - 1].iter().product();
        let y = x.contiguous()?.reshape(vec![n, d[d.len() - 1]])?.matmul(&self.w)?;
        let y = match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut s = d[..d.len() - 1].to_vec();
        s.push(self.w.dims()[1]);
        Ok(y.reshape(s)?)
    }
}

/// RMSNorm эталона: `x · rsqrt(mean(x²) + eps) · w` по последней оси.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> R<Tensor> {
    let last = x.rank() - 1;
    let ms = x.sqr()?.mean_keepdim(last)?.add_scalar(eps)?.sqrt()?;
    Ok(x.broadcast_div(&ms)?.broadcast_mul(w)?)
}

/// SwiGLU: `w2(silu(w1·x) · w3·x)`.
pub fn swiglu(x: &Tensor, w1: &Lin, w3: &Lin, w2: &Lin) -> R<Tensor> {
    w2.fwd(&w1.fwd(x)?.silu()?.mul(&w3.fwd(x)?)?)
}

/// Таблица RoPE `[P, half, 2]` (cos, sin) — из чекпойнта (эталон хранит её
/// округлённой до bf16, поэтому берём готовую, а не считаем заново).
pub struct Rope {
    table: Vec<f32>,
    half: usize,
    positions: usize,
}

impl Rope {
    pub fn load(c: &Comp, name: &str) -> R<Self> {
        let (table, d) = c.host(name)?;
        Ok(Self { table, half: d[1], positions: d[0] })
    }

    /// Таблица, посчитанная в f32 (`precompute_freqs_cis` без округления — v1).
    pub fn compute(positions: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..half).map(|j| 1.0 / base.powf(2.0 * j as f32 / head_dim as f32)).collect();
        let mut table = vec![0f32; positions * half * 2];
        for p in 0..positions {
            for (j, f) in freqs.iter().enumerate() {
                let a = p as f32 * f;
                table[(p * half + j) * 2] = a.cos();
                table[(p * half + j) * 2 + 1] = a.sin();
            }
        }
        Self { table, half, positions }
    }

    pub fn max_positions(&self) -> usize {
        self.positions
    }

    /// Множители для позиций: `(cos, ±sin)` формы `[1, S, 1, 2·half]`; у sin знак
    /// минус на чётных каналах (см. [`apply_rope`]).
    pub fn tables(&self, positions: &[usize], device: Device) -> R<(Tensor, Tensor)> {
        let (s, d) = (positions.len(), 2 * self.half);
        let (mut cs, mut sn) = (vec![0f32; s * d], vec![0f32; s * d]);
        for (i, &p) in positions.iter().enumerate() {
            let p = p.min(self.positions - 1);
            for j in 0..self.half {
                let (c, sv) = (self.table[(p * self.half + j) * 2], self.table[(p * self.half + j) * 2 + 1]);
                cs[i * d + 2 * j] = c;
                cs[i * d + 2 * j + 1] = c;
                sn[i * d + 2 * j] = -sv;
                sn[i * d + 2 * j + 1] = sv;
            }
        }
        Ok((Tensor::from_vec(cs, vec![1, s, 1, d], device)?, Tensor::from_vec(sn, vec![1, s, 1, d], device)?))
    }
}

/// Поворот пар `(x₂ᵢ, x₂ᵢ₊₁)`: `x₂ᵢ·cos − x₂ᵢ₊₁·sin`, `x₂ᵢ₊₁·cos + x₂ᵢ·sin`.
/// `x` — `[B, S, H, D]`.
pub fn apply_rope(x: &Tensor, rope: &(Tensor, Tensor)) -> R<Tensor> {
    let d = x.dims().to_vec();
    let pairs = x.contiguous()?.reshape(vec![d[0], d[1], d[2], d[3] / 2, 2])?;
    let (a, b) = (pairs.narrow(4, 0, 1)?.contiguous()?, pairs.narrow(4, 1, 1)?.contiguous()?);
    let swapped = Tensor::cat(&[&b, &a], 4)?.contiguous()?.reshape(d)?;
    Ok(x.broadcast_mul(&rope.0)?.add(&swapped.broadcast_mul(&rope.1)?)?)
}

/// Шум сэмплера токенов: `Exp(1)` как функция (сид, шаг, токен) — хэш
/// SplitMix64. Без состояния между токенами, поэтому тот же шум легко
/// подставить в эталон (`scripts/reference/seedvc_full.py`) и сверить
/// стохастическую генерацию токен в токен.
pub struct Rng {
    seed: u64,
    step: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { seed, step: 0 }
    }

    /// Перейти к следующему шагу генерации.
    pub fn advance(&mut self) {
        self.step += 1;
    }

    /// `−ln(u)`, `u ∈ (0, 1]` — для токена `token` на текущем шаге.
    pub fn exp(&self, token: usize) -> f64 {
        let mut z = self.seed.wrapping_add(self.step.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_add((token as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        -(((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64).ln()
    }
}
