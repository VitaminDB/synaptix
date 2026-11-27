//! Flow-matching: DiT (`modules/v2/dit_wrapper.py`, `dit_model.py`) и сэмплер
//! Эйлера с CFG по двум условиям (`modules/v2/cfm.py`).
//!
//! Вход DiT на кадр: `[x (80) | мел образца (80) | условие (512)]` → Linear
//! 672→512; перед кадрами — токен времени и токен диктора. 13 блоков: RMSNorm
//! с модуляцией по времени (shift/scale/gate), внимание 8×64 с RoPE,
//! SwiGLU 1536. Выход — скорость `[B, 80, T]`.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::loader::Comp;
use crate::nn::{apply_rope, rms_norm, swiglu, Lin, Rope};
use crate::{SeedVcError, N_MELS, R};

const DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-5;
const FREQ_DIM: usize = 256;
pub const STYLE_DIM: usize = 192;

struct Block {
    mod_lin: Lin,
    attn_norm: Tensor,
    wqkv: Lin,
    wo: Lin,
    ffn_norm: Tensor,
    w1: Lin,
    w3: Lin,
    w2: Lin,
}

pub struct Dit {
    cond_projection: Lin,
    merge: Lin,
    style_in: Lin,
    t1: Lin,
    t2: Lin,
    blocks: Vec<Block>,
    final_mod: Lin,
    final_norm: Tensor,
    out1: Lin,
    out2: Lin,
    rope: Rope,
    device: Device,
}

fn chunk(x: &Tensor, i: usize, n: usize) -> R<Tensor> {
    let last = x.rank() - 1;
    let d = x.dims()[last] / n;
    Ok(x.narrow(last, i * d, d)?.contiguous()?)
}

impl Dit {
    pub fn load(c: &Comp) -> R<Self> {
        let e = "estimator";
        let mut blocks = Vec::new();
        let mut i = 0;
        while c.has(&format!("{e}.transformer.layers.{i}.attention.wqkv.weight")) {
            let p = format!("{e}.transformer.layers.{i}");
            blocks.push(Block {
                mod_lin: Lin::load(c, &format!("{p}.attention_norm.linear"), true)?,
                attn_norm: c.get(&format!("{p}.attention_norm.norm.weight"), DType::F32)?,
                wqkv: Lin::load(c, &format!("{p}.attention.wqkv"), false)?,
                wo: Lin::load(c, &format!("{p}.attention.wo"), false)?,
                ffn_norm: c.get(&format!("{p}.ffn_norm.weight"), DType::F32)?,
                w1: Lin::load(c, &format!("{p}.feed_forward.w1"), false)?,
                w3: Lin::load(c, &format!("{p}.feed_forward.w3"), false)?,
                w2: Lin::load(c, &format!("{p}.feed_forward.w2"), false)?,
            });
            i += 1;
        }
        if blocks.is_empty() {
            return Err(SeedVcError::Load("в компоненте cfm нет блоков трансформера".into()));
        }
        Ok(Self {
            cond_projection: Lin::load(c, &format!("{e}.cond_projection"), true)?,
            merge: Lin::load(c, &format!("{e}.cond_x_merge_linear"), true)?,
            style_in: Lin::load(c, &format!("{e}.style_in"), true)?,
            t1: Lin::load(c, &format!("{e}.t_embedder.mlp.0"), true)?,
            t2: Lin::load(c, &format!("{e}.t_embedder.mlp.2"), true)?,
            blocks,
            final_mod: Lin::load(c, &format!("{e}.transformer.norm.linear"), true)?,
            final_norm: c.get(&format!("{e}.transformer.norm.norm.weight"), DType::F32)?,
            out1: Lin::load(c, &format!("{e}.final_mlp.0"), true)?,
            out2: Lin::load(c, &format!("{e}.final_mlp.2"), true)?,
            rope: Rope::load(c, &format!("{e}.transformer.freqs_cis"))?,
            device: c.device(),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Скорость потока. `x`, `prompt` — `[B, 80, T]`; `style` — `[B, 192]`;
    /// `cond` — `[B, T, 512]`; `t ∈ [0, 1]` общий для пакета. → `[B, 80, T]`.
    pub fn forward(&self, x: &Tensor, prompt: &Tensor, t: f32, style: &Tensor, cond: &Tensor) -> R<Tensor> {
        let (b, frames) = (x.dims()[0], x.dims()[2]);
        let s = frames + 2;
        if s > self.rope.max_positions() {
            return Err(SeedVcError::Input(format!("окно {frames} кадров длиннее предела {}", self.rope.max_positions() - 2)));
        }
        // Время: [cos, sin] от 1000·t·freqs → MLP.
        let half = FREQ_DIM / 2;
        let mut emb = vec![0f32; FREQ_DIM];
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = 1000.0 * t * f;
            emb[i] = a.cos();
            emb[half + i] = a.sin();
        }
        let t1 = self.t2.fwd(&self.t1.fwd(&Tensor::from_vec(emb, vec![1, FREQ_DIM], self.device)?)?.silu()?)?; // [1, 512]
        let t_act = t1.silu()?;

        let xt = x.transpose(1, 2)?.contiguous()?;
        let pt = prompt.transpose(1, 2)?.contiguous()?;
        let cond = self.cond_projection.fwd(cond)?;
        let x_in = self.merge.fwd(&Tensor::cat(&[&xt, &pt, &cond], 2)?.contiguous()?)?; // [B, T, 512]
        let style = self.style_in.fwd(style)?.reshape(vec![b, 1, DIM])?;
        let t_tok = t1.reshape(vec![1, 1, DIM])?.broadcast_as(vec![b, 1, DIM])?.contiguous()?;
        let mut h = Tensor::cat(&[&t_tok, &style, &x_in], 1)?.contiguous()?; // [B, S, 512]

        let positions: Vec<usize> = (0..s).collect();
        let rope = self.rope.tables(&positions, self.device)?;
        let heads = |t: Tensor| -> R<Tensor> { Ok(t.reshape(vec![b, s, HEADS, HEAD_DIM])?) };
        let per_head = |t: Tensor| -> R<Tensor> { Ok(t.permute(vec![0, 2, 1, 3])?.contiguous()?) };
        let one = |v: Tensor| -> R<Tensor> { Ok(v.reshape(vec![1, 1, DIM])?) };
        for blk in &self.blocks {
            let m = blk.mod_lin.fwd(&t_act)?; // [1, 6·512]: shift/scale/gate внимания, затем MLP
            let (shift_a, scale_a, gate_a) = (one(chunk(&m, 0, 6)?)?, one(chunk(&m, 1, 6)?)?, one(chunk(&m, 2, 6)?)?);
            let (shift_m, scale_m, gate_m) = (one(chunk(&m, 3, 6)?)?, one(chunk(&m, 4, 6)?)?, one(chunk(&m, 5, 6)?)?);
            let n = rms_norm(&h, &blk.attn_norm, EPS)?.broadcast_mul(&scale_a.add_scalar(1.0)?)?.broadcast_add(&shift_a)?;
            let qkv = blk.wqkv.fwd(&n)?;
            let q = per_head(apply_rope(&heads(chunk(&qkv, 0, 3)?)?, &rope)?)?;
            let k = per_head(apply_rope(&heads(chunk(&qkv, 1, 3)?)?, &rope)?)?;
            let v = per_head(heads(chunk(&qkv, 2, 3)?)?)?;
            let a = scaled_dot_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None)?;
            let a = blk.wo.fwd(&a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, DIM])?)?;
            h = h.add(&a.broadcast_mul(&gate_a)?)?;
            let n = rms_norm(&h, &blk.ffn_norm, EPS)?.broadcast_mul(&scale_m.add_scalar(1.0)?)?.broadcast_add(&shift_m)?;
            h = h.add(&swiglu(&n, &blk.w1, &blk.w3, &blk.w2)?.broadcast_mul(&gate_m)?)?;
        }
        let m = self.final_mod.fwd(&t_act)?;
        let (scale, shift) = (one(chunk(&m, 0, 2)?)?, one(chunk(&m, 1, 2)?)?);
        let h = rms_norm(&h, &self.final_norm, EPS)?.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
        let h = h.narrow(1, 2, frames)?.contiguous()?;
        let y = self.out2.fwd(&self.out1.fwd(&h)?.silu()?)?; // [B, T, 80]
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

/// Сетка времени эталона: `t_i = 1 − cos(π/2 · i/N)`, `i = 0…N` (арифметика f32, как `torch.linspace`).
pub fn time_grid(steps: usize) -> Vec<f32> {
    let n = steps as f32;
    (0..=steps)
        .map(|i| {
            // torch.linspace: первая половина — от начала, вторая — от конца.
            let lin = if i < (steps + 1) / 2 { i as f32 * (1.0 / n) } else { 1.0 - (steps - i) as f32 * (1.0 / n) };
            lin + (-1.0) * ((std::f32::consts::PI / 2.0 * lin).cos() - 1.0 + lin)
        })
        .collect()
}

pub struct CfmParams {
    pub steps: usize,
    /// CFG по содержанию (`intelligibility_cfg_rate`).
    pub intelligibility: f32,
    /// CFG по голосу (`similarity_cfg_rate`).
    pub similarity: f32,
    /// Обезличивание: без образца и вектора диктора, CFG только по содержанию.
    pub random_voice: bool,
}

/// Сэмплер Эйлера. `mu` — условие `[1, T, 512]` (образец + источник), `prompt` — мел
/// образца `[1, 80, P]`, `style` — `[1, 192]`, `noise` — `[1, 80, T]`.
/// Возвращает мел `[1, 80, T]` (первые `P` кадров обнулены).
pub fn sample(dit: &Dit, mu: &Tensor, prompt: &Tensor, style: &Tensor, noise: &Tensor, p: &CfmParams, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Tensor> {
    let dev = dit.device();
    let frames = mu.dims()[1];
    let plen = prompt.dims()[2].min(frames);
    if noise.dims() != [1, N_MELS, frames] {
        return Err(SeedVcError::Other(format!("шум {:?}, ожидается [1, {N_MELS}, {frames}]", noise.dims())));
    }
    // Маска кадров образца (0) и генерируемых (1); мел образца, дополненный нулями.
    let mask: Vec<f32> = (0..frames).map(|i| if i < plen { 0.0 } else { 1.0 }).collect();
    let mask = Tensor::from_vec(mask, vec![1, 1, frames], dev)?;
    let zeros_tail = Tensor::zeros(vec![1, N_MELS, frames - plen], DType::F32, dev)?;
    let prompt_x = if plen == frames { prompt.narrow(2, 0, plen)?.contiguous()? } else { Tensor::cat(&[&prompt.narrow(2, 0, plen)?.contiguous()?, &zeros_tail], 2)?.contiguous()? };
    let zero_prompt = Tensor::zeros(vec![1, N_MELS, frames], DType::F32, dev)?;
    let zero_style = Tensor::zeros(vec![1, STYLE_DIM], DType::F32, dev)?;
    let zero_mu = Tensor::zeros(vec![1, frames, mu.dims()[2]], DType::F32, dev)?;
    let mu = mu.contiguous()?;
    let (a, bsim) = (p.intelligibility, p.similarity);

    let grid = time_grid(p.steps.max(1));
    let mut x = noise.broadcast_mul(&mask)?;
    let mut t = grid[0];
    let mut dt = grid[1] - grid[0];
    let cat = |v: &[&Tensor], d: usize| -> R<Tensor> { Ok(Tensor::cat(v, d)?.contiguous()?) };
    let row = |v: &Tensor, i: usize| -> R<Tensor> { Ok(v.narrow(0, i, 1)?.contiguous()?) };
    for step in 1..grid.len() {
        let v = if p.random_voice {
            let o = dit.forward(&cat(&[&x, &x], 0)?, &cat(&[&zero_prompt, &zero_prompt], 0)?, t, &cat(&[&zero_style, &zero_style], 0)?, &cat(&[&mu, &zero_mu], 0)?)?;
            row(&o, 0)?.mul_scalar(1.0 + a)?.sub(&row(&o, 1)?.mul_scalar(a)?)?
        } else if a == 0.0 && bsim == 0.0 {
            dit.forward(&x, &prompt_x, t, style, &mu)?
        } else if a == 0.0 {
            let o = dit.forward(&cat(&[&x, &x], 0)?, &cat(&[&prompt_x, &zero_prompt], 0)?, t, &cat(&[style, &zero_style], 0)?, &cat(&[&mu, &mu], 0)?)?;
            row(&o, 0)?.mul_scalar(1.0 + bsim)?.sub(&row(&o, 1)?.mul_scalar(bsim)?)?
        } else if bsim == 0.0 {
            let o = dit.forward(&cat(&[&x, &x], 0)?, &cat(&[&prompt_x, &zero_prompt], 0)?, t, &cat(&[style, &zero_style], 0)?, &cat(&[&mu, &zero_mu], 0)?)?;
            row(&o, 0)?.mul_scalar(1.0 + a)?.sub(&row(&o, 1)?.mul_scalar(a)?)?
        } else {
            // Три ветки: (образец, диктор, содержание) / (—, —, содержание) / (—, —, —).
            let o = dit.forward(&cat(&[&x, &x, &x], 0)?, &cat(&[&prompt_x, &zero_prompt, &zero_prompt], 0)?, t, &cat(&[style, &zero_style, &zero_style], 0)?, &cat(&[&mu, &mu, &zero_mu], 0)?)?;
            row(&o, 0)?.mul_scalar(1.0 + a + bsim)?.sub(&row(&o, 2)?.mul_scalar(a)?)?.sub(&row(&o, 1)?.mul_scalar(bsim)?)?
        };
        x = x.add(&v.mul_scalar(dt)?)?;
        t += dt;
        if step < grid.len() - 1 {
            dt = grid[step + 1] - t;
        }
        x = x.broadcast_mul(&mask)?;
        if !progress(step, grid.len() - 1) {
            return Err(SeedVcError::Cancelled);
        }
    }
    Ok(x)
}
