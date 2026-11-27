//! DAC-декодер Flow-VAE: латенты `[1, 128, L]` → стерео `[2, L·512]`, 44,1 кГц.
//! Два канала — два потока по 64 канала латента. Свёртки с weight_norm
//! (`w = g · v / ‖v‖`, норма по всем осям, кроме нулевой) сворачиваются при загрузке.

use synaptix_core::{dtype::DType, tensor::Tensor};
use synaptix_ops::conv::conv1d::conv1d_dilated;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;

use crate::loader::Comp;
use crate::R;

const STRIDES: [usize; 4] = [8, 8, 4, 2];
pub const HOP: usize = 512;
pub const SAMPLE_RATE: u32 = 44_100;

struct Conv {
    w: Tensor,
    b: Tensor,
}

struct ResUnit {
    a1: Tensor,
    c1: Conv,
    a2: Tensor,
    c2: Conv,
    dilation: usize,
}

struct Block {
    alpha: Tensor,
    up: Conv,
    stride: usize,
    units: Vec<ResUnit>,
}

pub struct Vocoder {
    in_proj: Conv,
    conv_in: Conv,
    blocks: Vec<Block>,
    alpha_out: Tensor,
    conv_out: Conv,
}

/// Snake: `x + sin²(α·x) / (α + 1e-9)`, α — `[1, C, 1]`.
fn snake(x: &Tensor, alpha: &Tensor) -> R<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    Ok(x.add(&s.broadcast_div(&alpha.add_scalar(1e-9)?)?)?)
}

fn wn(c: &Comp, p: &str) -> R<Conv> {
    let (g, v) = (c.get(&format!("{p}.weight_g"), DType::F32)?, c.get(&format!("{p}.weight_v"), DType::F32)?);
    let d = v.dims().to_vec();
    let norm = v.sqr()?.reshape(vec![d[0], d[1] * d[2]])?.sum_keepdim(1)?.sqrt()?.reshape(vec![d[0], 1, 1])?;
    Ok(Conv { w: v.broadcast_div(&norm)?.broadcast_mul(&g)?.contiguous()?, b: c.get(&format!("{p}.bias"), DType::F32)? })
}

impl Vocoder {
    pub fn load(c: &Comp) -> R<Self> {
        let mut blocks = Vec::with_capacity(4);
        for (i, &stride) in STRIDES.iter().enumerate() {
            let p = format!("blocks.{i}");
            let mut units = Vec::with_capacity(3);
            for (k, dilation) in [1usize, 3, 9].into_iter().enumerate() {
                let u = format!("{p}.res_unit{}", k + 1);
                units.push(ResUnit {
                    a1: c.get(&format!("{u}.snake1.alpha"), DType::F32)?,
                    c1: wn(c, &format!("{u}.conv1"))?,
                    a2: c.get(&format!("{u}.snake2.alpha"), DType::F32)?,
                    c2: wn(c, &format!("{u}.conv2"))?,
                    dilation,
                });
            }
            blocks.push(Block { alpha: c.get(&format!("{p}.snake1.alpha"), DType::F32)?, up: wn(c, &format!("{p}.conv_t1"))?, stride, units });
        }
        Ok(Self {
            in_proj: Conv { w: c.get("dec_in_proj.weight", DType::F32)?, b: c.get("dec_in_proj.bias", DType::F32)? },
            conv_in: wn(c, "conv_in")?,
            blocks,
            alpha_out: c.get("snake_out.alpha", DType::F32)?,
            conv_out: wn(c, "conv_out")?,
        })
    }

    /// Латенты `[1, 128, L]` → волна `[2, L·512]` в [−1, 1] (F32).
    pub fn decode(&self, latents: &Tensor) -> R<Tensor> {
        let l = latents.dims()[2];
        let x = latents.to_dtype(DType::F32)?.contiguous()?.reshape(vec![2, 64, l])?;
        let x = conv1d_dilated(&x, &self.in_proj.w, Some(&self.in_proj.b), 1, 0, 1)?;
        let mut x = conv1d_dilated(&x, &self.conv_in.w, Some(&self.conv_in.b), 1, 3, 1)?;
        for b in &self.blocks {
            x = conv_transpose1d(&snake(&x, &b.alpha)?, &b.up.w, Some(&b.up.b), b.stride, b.stride.div_ceil(2), 0, 1, 1)?;
            for u in &b.units {
                let h = conv1d_dilated(&snake(&x, &u.a1)?, &u.c1.w, Some(&u.c1.b), 1, 3 * u.dilation, u.dilation)?;
                let h = conv1d_dilated(&snake(&h, &u.a2)?, &u.c2.w, Some(&u.c2.b), 1, 0, 1)?;
                x = x.add(&h)?;
            }
        }
        let y = conv1d_dilated(&snake(&x, &self.alpha_out)?, &self.conv_out.w, Some(&self.conv_out.b), 1, 3, 1)?.tanh()?;
        let n = y.dims()[2];
        Ok(y.reshape(vec![2, n])?)
    }
}
