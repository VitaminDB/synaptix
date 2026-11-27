//! Башня кодбуков `codebook_mtp_shared` (эталон — `decoder_layers.py`,
//! `audio_decoder.py`): 4 общих слоя + 9 по кодбукам. Каждый слой — каузальный
//! блок по оси времени со своим KV-кэшем (позиции те же, что у основы):
//!
//! ```text
//! e      = семантический кадр ? Emb[code] : 0   (+ extra у общего слоя 0)
//! mixed  = eh_proj(cat(enorm(e), hnorm(h)))
//! q,g,k,v = wqkv(attention_norm(mixed))          // [q 4096 | gate 4096 | k 512 | v 512]
//! a      = wo(attn(rope(q_norm q), rope(k_norm k), v) · sigmoid(g))
//! r1     = mixed + a
//! r2     = r1 + w2(silu(w1 n) · w3 n),  n = ffn_norm(r1)
//! out    = final_layernorm(r2)
//! ```
//!
//! Нормы — RMS с весом `1 + w`; RoPE интерливный на первых 64 каналах головы.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::WeightSource;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::{Drama3Error, CODEBOOKS, CODEBOOK_SIZE, R};

pub const DIM: usize = 2048;
const HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROTARY: usize = 64;
const ROPE_BASE: f64 = 1.0e7;
const EPS: f32 = 1e-6;
pub const SHARED: usize = 4;
pub const RESIDUAL: usize = CODEBOOKS - 1;
pub const LAYERS: usize = SHARED + RESIDUAL;

fn le(e: impl std::fmt::Display) -> Drama3Error {
    Drama3Error::Load(e.to_string())
}

/// Веса башни из бандла (имена HF).
pub struct Src<'a> {
    pub w: &'a dyn WeightSource,
    pub device: Device,
    pub dtype: DType,
}

impl Src<'_> {
    fn get(&self, name: &str) -> R<Tensor> {
        self.w.tensor(name, self.device, self.dtype).map_err(|e| Drama3Error::Load(format!("{name}: {e}")))
    }
    /// Вес нормы `1 + w`.
    fn norm(&self, name: &str) -> R<Tensor> {
        Ok(self.w.tensor(name, self.device, DType::F32).map_err(le)?.add_scalar(1.0)?.to_dtype(self.dtype)?)
    }
    /// Линейный слой без смещения: квант из бандла как есть, иначе плотный.
    fn lin(&self, base: &str) -> R<QuantLinear> {
        let name = format!("{base}.weight");
        if let Some(q) = self.w.quant(&name, self.device) {
            return Ok(QuantLinear::Quant { w: q.map_err(le)?, bias: None });
        }
        Ok(QuantLinear::dense(self.get(&name)?, None)?)
    }
}

struct Layer {
    emb: Tensor,
    enorm: Tensor,
    hnorm: Tensor,
    eh: QuantLinear,
    attn_norm: Tensor,
    wqkv: QuantLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    wo: QuantLinear,
    ffn_norm: Tensor,
    w1: QuantLinear,
    w3: QuantLinear,
    w2: QuantLinear,
    final_norm: Tensor,
}

impl Layer {
    fn load(s: &Src, p: &str, emb: &str) -> R<Self> {
        let t = format!("{p}.transformer_layer");
        Ok(Self {
            emb: s.get(&format!("{p}.{emb}.weight"))?,
            enorm: s.norm(&format!("{p}.enorm.weight"))?,
            hnorm: s.norm(&format!("{p}.hnorm.weight"))?,
            eh: s.lin(&format!("{p}.eh_proj"))?,
            attn_norm: s.norm(&format!("{t}.attention_norm.weight"))?,
            wqkv: s.lin(&format!("{t}.attention.wqkv"))?,
            q_norm: s.norm(&format!("{t}.attention.q_norm.weight"))?,
            k_norm: s.norm(&format!("{t}.attention.k_norm.weight"))?,
            wo: s.lin(&format!("{t}.attention.wo"))?,
            ffn_norm: s.norm(&format!("{t}.ffn_norm.weight"))?,
            w1: s.lin(&format!("{t}.feed_forward.w1"))?,
            w3: s.lin(&format!("{t}.feed_forward.w3"))?,
            w2: s.lin(&format!("{t}.feed_forward.w2"))?,
            final_norm: s.norm(&format!("{p}.final_layernorm.weight"))?,
        })
    }
}

/// KV-кэш башни: на слой и kv-голову — `[1, cap, 256]`.
pub struct TowerCache {
    k: Vec<Vec<Tensor>>,
    v: Vec<Vec<Tensor>>,
    pub len: usize,
    cap: usize,
}

pub struct Tower {
    layers: Vec<Layer>,
    heads: Vec<QuantLinear>,
    /// `audio_decoder.codebook_embeddings` `[10·4096, H]`, строка `k·4096 + c_k`.
    pub cb_emb: Tensor,
    cos: Tensor,
    sin: Tensor,
    cap: usize,
    device: Device,
    dtype: DType,
}

fn host(t: &Tensor) -> R<Vec<f32>> {
    Ok(t.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

impl Tower {
    pub fn load(s: &Src, max_pos: usize) -> R<Self> {
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..SHARED {
            layers.push(Layer::load(s, &format!("audio_decoder_mtp_shared.layers.{i}"), "semantic_embedding")?);
        }
        for i in 0..RESIDUAL {
            layers.push(Layer::load(s, &format!("audio_decoder_mtp.layers.{i}"), "codebook_embedding")?);
        }
        let heads = (0..RESIDUAL).map(|i| s.lin(&format!("audio_decoder_mtp.heads.{i}"))).collect::<R<Vec<_>>>()?;
        let half = ROTARY / 2;
        let (mut cos, mut sin) = (vec![0f32; max_pos * half], vec![0f32; max_pos * half]);
        for p in 0..max_pos {
            for i in 0..half {
                // Как в эталоне: частоты и углы в f32.
                let f = (1.0 / ROPE_BASE.powf((2 * i) as f64 / ROTARY as f64)) as f32;
                let a = p as f32 * f;
                cos[p * half + i] = a.cos();
                sin[p * half + i] = a.sin();
            }
        }
        Ok(Self {
            layers,
            heads,
            cb_emb: s.get("audio_decoder.codebook_embeddings.weight")?,
            cos: Tensor::from_vec(cos, vec![max_pos, half], s.device)?,
            sin: Tensor::from_vec(sin, vec![max_pos, half], s.device)?,
            cap: max_pos,
            device: s.device,
            dtype: s.dtype,
        })
    }

    pub fn make_cache(&self, cap: usize) -> R<TowerCache> {
        if cap > self.cap {
            return Err(Drama3Error::Input(format!("контекст {cap} больше предела башни {}", self.cap)));
        }
        let mk = || -> R<Vec<Vec<Tensor>>> {
            (0..LAYERS)
                .map(|_| (0..KV_HEADS).map(|_| Ok(Tensor::zeros(vec![1, cap, HEAD_DIM], self.dtype, self.device)?)).collect::<R<Vec<_>>>())
                .collect()
        };
        Ok(TowerCache { k: mk()?, v: mk()?, len: 0, cap })
    }

    /// Строки таблицы по индексам → `[n, H]`.
    fn rows(&self, table: &Tensor, idx: &[usize]) -> R<Tensor> {
        let i = Tensor::from_vec(idx.iter().map(|&v| v as i64).collect::<Vec<i64>>(), vec![idx.len()], self.device)?;
        Ok(table.index_select(0, &i)?.contiguous()?)
    }

    fn column(&self, v: Vec<f32>) -> R<Tensor> {
        let n = v.len();
        Ok(Tensor::from_vec(v, vec![n, 1], self.device)?.to_dtype(self.dtype)?)
    }

    /// Доп. вход общего слоя 0: сумма эмбеддингов кодов 2…9 входных кадров
    /// (у несемантических — ноль). `frames[i]` — `[токен, c0..c9]`.
    pub fn shared_input(&self, frames: &[crate::Row]) -> R<Tensor> {
        let n = frames.len();
        let mut acc: Option<Tensor> = None;
        for k in 2..CODEBOOKS {
            let idx: Vec<usize> = frames.iter().map(|f| k * CODEBOOK_SIZE + (f[1 + k] as usize).min(CODEBOOK_SIZE - 1)).collect();
            let e = self.rows(&self.cb_emb, &idx)?;
            acc = Some(match acc {
                Some(a) => a.add(&e)?,
                None => e,
            });
        }
        let m = self.column(frames.iter().map(|f| if crate::is_semantic(f[0]) { 1.0 } else { 0.0 }).collect())?;
        let _ = n;
        Ok(acc.expect("кодбуков больше двух").broadcast_mul(&m)?)
    }

    /// Вход основы для кадров: текстовый эмбеддинг, у семантических —
    /// `(emb + E_cb[4096 + c1]) / √2`. `text` — `[n, H]`.
    pub fn backbone_input(&self, frames: &[crate::Row], text: &Tensor) -> R<Tensor> {
        let idx: Vec<usize> = frames.iter().map(|f| CODEBOOK_SIZE + (f[2] as usize).min(CODEBOOK_SIZE - 1)).collect();
        let sem: Vec<bool> = frames.iter().map(|f| crate::is_semantic(f[0])).collect();
        let add = self.rows(&self.cb_emb, &idx)?.broadcast_mul(&self.column(sem.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect())?)?;
        let scale = self.column(sem.iter().map(|&s| if s { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 }).collect())?;
        Ok(text.to_dtype(self.dtype)?.add(&add)?.broadcast_mul(&scale)?)
    }

    /// `x` `[n, heads, 256]` → RoPE на первых 64 каналах → `[1, heads, n, 256]`.
    fn rope(&self, x: &Tensor, pos0: usize) -> R<Tensor> {
        let (n, h) = (x.dims()[0], x.dims()[1]);
        let x = x.permute(vec![1, 0, 2])?.contiguous()?.reshape(vec![1, h, n, HEAD_DIM])?;
        let rot = x.narrow(3, 0, ROTARY)?.contiguous()?;
        let rest = x.narrow(3, ROTARY, HEAD_DIM - ROTARY)?.contiguous()?;
        let cos = self.cos.narrow(0, pos0, n)?.contiguous()?;
        let sin = self.sin.narrow(0, pos0, n)?.contiguous()?;
        let rot = apply_rope_with_cossin(&rot, &cos, &sin, RopeLayout::Interleaved)?;
        Ok(Tensor::cat(&[&rot, &rest], 3)?.contiguous()?)
    }

    /// Слой `li` на строках позиций `pos0 .. pos0 + n`: `h`, `e` — `[n, H]`.
    fn layer(&self, li: usize, h: &Tensor, e: &Tensor, pos0: usize, cache: &mut TowerCache) -> R<Tensor> {
        let l = &self.layers[li];
        let n = h.dims()[0];
        let t = pos0 + n;
        if t > cache.cap {
            return Err(Drama3Error::Other(format!("кэш башни переполнен: {t} > {}", cache.cap)));
        }
        let mixed = l.eh.forward(&Tensor::cat(&[&rms_norm(e, &l.enorm, EPS)?, &rms_norm(h, &l.hnorm, EPS)?], 1)?.contiguous()?)?;
        let qkv = l.wqkv.forward(&rms_norm(&mixed, &l.attn_norm, EPS)?)?;
        let qw = HEADS * HEAD_DIM;
        let kw = KV_HEADS * HEAD_DIM;
        let q = rms_norm(&qkv.narrow(1, 0, qw)?.contiguous()?.reshape(vec![n, HEADS, HEAD_DIM])?, &l.q_norm, EPS)?;
        let g = qkv.narrow(1, qw, qw)?.contiguous()?;
        let k = rms_norm(&qkv.narrow(1, 2 * qw, kw)?.contiguous()?.reshape(vec![n, KV_HEADS, HEAD_DIM])?, &l.k_norm, EPS)?;
        let v = qkv.narrow(1, 2 * qw + kw, kw)?.contiguous()?.reshape(vec![n, KV_HEADS, HEAD_DIM])?;
        let q = self.rope(&q, pos0)?; // [1, 16, n, 256]
        let k = self.rope(&k, pos0)?; // [1, 2, n, 256]
        let v = v.permute(vec![1, 0, 2])?.contiguous()?; // [2, n, 256]
        let per = HEADS / KV_HEADS;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mask = if n > 1 {
            let mut m = vec![0f32; n * t];
            for r in 0..n {
                for c in pos0 + r + 1..t {
                    m[r * t + c] = -1.0e30;
                }
            }
            Some(Tensor::from_vec(m, vec![1, 1, n, t], self.device)?)
        } else {
            None
        };
        let mut outs: Vec<Tensor> = Vec::with_capacity(KV_HEADS);
        for kh in 0..KV_HEADS {
            cache.k[li][kh].copy_rows_from(pos0, &k.narrow(1, kh, 1)?.contiguous()?.reshape(vec![1, n, HEAD_DIM])?)?;
            cache.v[li][kh].copy_rows_from(pos0, &v.narrow(0, kh, 1)?.contiguous()?.reshape(vec![1, n, HEAD_DIM])?)?;
            let kk = cache.k[li][kh].narrow(1, 0, t)?.contiguous()?.reshape(vec![1, 1, t, HEAD_DIM])?;
            let vv = cache.v[li][kh].narrow(1, 0, t)?.contiguous()?.reshape(vec![1, 1, t, HEAD_DIM])?;
            let qh = q.narrow(1, kh * per, per)?.contiguous()?; // [1, per, n, 256]
            let o = if n == 1 {
                // Один запрос: головы группы — «позиции запроса» против T ключей.
                let qh = qh.reshape(vec![1, 1, per, HEAD_DIM])?;
                scaled_dot_attention(&qh, &kk, &vv, scale, None)?.reshape(vec![1, per * HEAD_DIM])?
            } else {
                let kk = kk.broadcast_as(vec![1, per, t, HEAD_DIM])?.contiguous()?;
                let vv = vv.broadcast_as(vec![1, per, t, HEAD_DIM])?.contiguous()?;
                let o = scaled_dot_attention(&qh, &kk, &vv, scale, mask.as_ref())?;
                o.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![n, per * HEAD_DIM])?
            };
            outs.push(o);
        }
        let a = Tensor::cat(&[&outs[0], &outs[1]], 1)?.contiguous()?;
        let a = l.wo.forward(&a.mul(&g.sigmoid()?)?)?;
        let r1 = mixed.add(&a)?;
        let f = rms_norm(&r1, &l.ffn_norm, EPS)?;
        let gate = l.w1.forward(&f)?;
        let up = l.w3.forward(&f)?;
        let act = gate.silu_and_mul(&up).or_else(|_| gate.silu().and_then(|x| x.mul(&up)))?;
        let r2 = r1.add(&l.w2.forward(&act)?)?;
        Ok(rms_norm(&r2, &l.final_norm, EPS)?)
    }

    /// Эмбеддинг метки слоя: строка таблицы слоя у семантических кадров, иначе ноль.
    fn label(&self, li: usize, codes: &[u32], sem: &[bool]) -> R<Tensor> {
        let idx: Vec<usize> = codes.iter().map(|&c| (c as usize).min(CODEBOOK_SIZE - 1)).collect();
        let m = self.column(sem.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect())?;
        Ok(self.rows(&self.layers[li].emb, &idx)?.broadcast_mul(&m)?)
    }

    /// Префилл: строки промпта `input[i]` (позиции `pos0 + i`) с метками из
    /// следующей строки `next[i]`. `h` — состояния основы после финальной
    /// нормы `[n, H]`. `tap` — снять выход слоёв (сверка).
    pub fn prefill(&self, h: &Tensor, input: &[crate::Row], next: &[crate::Row], pos0: usize, cache: &mut TowerCache, mut tap: Option<&mut Vec<Tensor>>) -> R<()> {
        let n = input.len();
        if n == 0 {
            return Ok(());
        }
        let sem: Vec<bool> = next.iter().map(|f| crate::is_semantic(f[0])).collect();
        let extra = self.shared_input(input)?;
        let mut x = h.to_dtype(self.dtype)?.contiguous()?;
        for li in 0..LAYERS {
            let col = if li < SHARED { 1 } else { 1 + (li - SHARED) };
            let codes: Vec<u32> = next.iter().map(|f| f[col]).collect();
            let mut e = self.label(li, &codes, &sem)?;
            if li == 0 {
                e = e.add(&extra)?;
            }
            x = self.layer(li, &x, &e, pos0, cache)?;
            if let Some(t) = tap.as_deref_mut() {
                t.push(x.clone());
            }
        }
        cache.len = pos0 + n;
        Ok(())
    }

    /// Шаг декода на позиции `pos`: по состоянию основы `h` `[1, H]`, входному
    /// кадру и уже выбранному головному токену — коды 1…9. `pick(r, логиты)` —
    /// выбор кода головы `r`.
    pub fn step(&self, h: &Tensor, input: &crate::Row, main: u32, pos: usize, cache: &mut TowerCache, pick: &mut dyn FnMut(usize, &[f32]) -> u32) -> R<[u32; CODEBOOKS]> {
        let sem = [crate::is_semantic(main)];
        let c0 = main.saturating_sub(crate::SEMANTIC_BEGIN).min(CODEBOOK_SIZE as u32 - 1);
        let mut codes = [0u32; CODEBOOKS];
        codes[0] = c0;
        let mut x = h.to_dtype(self.dtype)?.contiguous()?;
        let extra = self.shared_input(std::slice::from_ref(input))?;
        for li in 0..SHARED {
            let mut e = self.label(li, &[c0], &sem)?;
            if li == 0 {
                e = e.add(&extra)?;
            }
            x = self.layer(li, &x, &e, pos, cache)?;
        }
        for r in 0..RESIDUAL {
            let e = self.label(SHARED + r, &[codes[r]], &sem)?;
            x = self.layer(SHARED + r, &x, &e, pos, cache)?;
            codes[r + 1] = pick(r, &host(&self.heads[r].forward(&x)?)?);
        }
        cache.len = pos + 1;
        Ok(codes)
    }

    /// Логиты головы `r` по выходу слоя (сверка).
    pub fn head(&self, r: usize, x: &Tensor) -> R<Vec<f32>> {
        host(&self.heads[r].forward(x)?)
    }
}
