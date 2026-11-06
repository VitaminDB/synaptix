//! DualAR-трансформер Audio8: медленный (семантика по кадрам) и быстрый (кодбуки кадра).
//!
//! RoPE upstream — комплексная форма fish-speech: пары соседних измерений (2j, 2j+1)
//! с частотой j, таблицы cos/sin округлены до BF16. Строки q/k в `wqkv` при загрузке
//! переставляются по головам в раскладку «половинами» (j ↔ 2j, j+hd/2 ↔ 2j+1): скалярное
//! произведение q·k от общей перестановки не меняется, а вращение становится split-RoPE
//! одного слитого ядра. Если `wqkv` в бандле квантован — остаётся interleaved-путь.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::config::Audio8Config;
use crate::loader::{Checkpoint, Lin};
use crate::{Audio8Error, Result};

const NEG_LARGE: f32 = -1.0e30;

/// f32 → ближайшее BF16 (округление к чётному), обратно в f32.
pub fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    let r = ((b >> 16) & 1) + 0x7FFF;
    f32::from_bits((b.wrapping_add(r)) & 0xFFFF_0000)
}

/// Таблицы cos/sin `[len, hd/2]` (F32 со значениями BF16, как `_precompute_rope` upstream).
pub fn rope_tables(len: usize, head_dim: usize, base: f32, device: Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32)).collect();
    let mut cos = vec![0f32; len * half];
    let mut sin = vec![0f32; len * half];
    for t in 0..len {
        for i in 0..half {
            let a = t as f32 * freqs[i];
            cos[t * half + i] = round_bf16(a.cos());
            sin[t * half + i] = round_bf16(a.sin());
        }
    }
    Ok((Tensor::from_vec(cos, (len, half), device)?, Tensor::from_vec(sin, (len, half), device)?))
}

/// Перестановка строк q/k (по головам): interleaved → split.
fn split_perm(n_head: usize, n_kv: usize, hd: usize) -> Vec<i64> {
    let mut idx = Vec::with_capacity((n_head + 2 * n_kv) * hd);
    for h in 0..n_head + n_kv {
        let base = h * hd;
        for j in 0..hd / 2 {
            idx.push((base + 2 * j) as i64);
        }
        for j in 0..hd / 2 {
            idx.push((base + 2 * j + 1) as i64);
        }
    }
    let v0 = (n_head + n_kv) * hd;
    idx.extend((v0..v0 + n_kv * hd).map(|i| i as i64));
    idx
}

struct Block {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    wqkv: Lin,
    wo: Lin,
    w1: Lin,
    w3: Lin,
    w2: Lin,
}

pub struct Kv {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    pub cap: usize,
}

/// Стопка блоков с общим RoPE (медленная или быстрая).
pub struct Stack {
    blocks: Vec<Block>,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    eps: f32,
    cos: Tensor,
    sin: Tensor,
    layout: RopeLayout,
    device: Device,
    dtype: DType,
}

impl Stack {
    #[allow(clippy::too_many_arguments)]
    fn load(
        ck: &Checkpoint,
        prefix: &str,
        n_layer: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        qkv_bias: bool,
        o_bias: bool,
        rope_len: usize,
        quantize: bool,
    ) -> Result<Self> {
        let cfg = &ck.config;
        let dev = ck.device;
        let perm = split_perm(n_head, n_kv, hd);
        let first_wqkv = format!("{prefix}.0.attention.wqkv");
        // Квантованный в бандле wqkv переставить нельзя — тогда interleaved-RoPE.
        let split = std::env::var("SYN_AUDIO8_ROPE").as_deref() != Ok("interleaved")
            && ck.has(&format!("{first_wqkv}.weight"));
        let permute = |w: Tensor, b: Option<Tensor>| -> Result<(Tensor, Option<Tensor>)> {
            let idx = Tensor::from_vec(perm.clone(), vec![perm.len()], w.device())?;
            let w = w.index_select(0, &idx)?.contiguous()?;
            let b = b.map(|b| b.index_select(0, &idx).and_then(|t| t.contiguous())).transpose()?;
            Ok((w, b))
        };
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("{prefix}.{i}.{s}");
            let wqkv = if split {
                ck.linear(&p("attention.wqkv"), qkv_bias, quantize, Some(&permute))?
            } else {
                ck.linear(&p("attention.wqkv"), qkv_bias, quantize, None)?
            };
            blocks.push(Block {
                attn_norm: ck.get(&p("attention_norm.weight"))?,
                ffn_norm: ck.get(&p("ffn_norm.weight"))?,
                wqkv,
                wo: ck.linear(&p("attention.wo"), o_bias, quantize, None)?,
                w1: ck.linear(&p("feed_forward.w1"), false, quantize, None)?,
                w3: ck.linear(&p("feed_forward.w3"), false, quantize, None)?,
                w2: ck.linear(&p("feed_forward.w2"), false, quantize, None)?,
            });
        }
        let (cos, sin) = rope_tables(rope_len, hd, cfg.rope_base, dev)?;
        Ok(Self {
            blocks,
            n_head,
            n_kv,
            hd,
            eps: cfg.norm_eps,
            cos,
            sin,
            layout: if split { RopeLayout::Split } else { RopeLayout::Interleaved },
            device: dev,
            dtype: ck.compute,
        })
    }

    pub fn new_cache(&self, cap: usize) -> Result<Kv> {
        let mut k = Vec::with_capacity(self.blocks.len());
        let mut v = Vec::with_capacity(self.blocks.len());
        for _ in 0..self.blocks.len() {
            k.push(Tensor::zeros(vec![1usize, self.n_kv, cap, self.hd], self.dtype, self.device)?);
            v.push(Tensor::zeros(vec![1usize, self.n_kv, cap, self.hd], self.dtype, self.device)?);
        }
        Ok(Kv { k, v, cap })
    }

    fn heads(&self, t: &Tensor, off: usize, n: usize, s: usize) -> Result<Tensor> {
        Ok(t.narrow(2, off * self.hd, n * self.hd)?
            .contiguous()?
            .reshape(vec![1usize, s, n, self.hd])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?)
    }

    fn attention(&self, b: &Block, li: usize, h: &Tensor, kv: &mut Kv, past: usize, s: usize) -> Result<Tensor> {
        let (nh, nkv, hd) = (self.n_head, self.n_kv, self.hd);
        let qkv = b.wqkv.forward(h)?;
        let q = self.heads(&qkv, 0, nh, s)?;
        let k = self.heads(&qkv, nh, nkv, s)?;
        let v = self.heads(&qkv, nh + nkv, nkv, s)?;
        let cos = self.cos.narrow(0, past, s)?.contiguous()?;
        let sin = self.sin.narrow(0, past, s)?.contiguous()?;
        let q = apply_rope_with_cossin(&q, &cos, &sin, self.layout)?;
        let k = apply_rope_with_cossin(&k, &cos, &sin, self.layout)?;
        kv.k[li].kv_append_inplace(&k, past)?;
        kv.v[li].kv_append_inplace(&v, past)?;
        let total = past + s;
        let k_all = kv.k[li].narrow(2, 0, total)?;
        let v_all = kv.v[li].narrow(2, 0, total)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let o = match q.flash_attention(&k_all, &v_all, scale, true) {
            Ok(o) => o,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let g = nh / nkv;
                let kr = repeat_kv(&k_all, g)?;
                let vr = repeat_kv(&v_all, g)?;
                let bias = if s == 1 { None } else { Some(causal_bias(s, total, self.device)?.to_dtype(q.dtype())?) };
                scaled_dot_attention(&q, &kr, &vr, scale, bias.as_ref())?
            }
            Err(e) => return Err(e.into()),
        };
        let o = o.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![1usize, s, nh * hd])?;
        b.wo.forward(&o)
    }

    /// Проход по слоям: `x` `[1, S, D]`, позиции `past..past+S` (KV пишется туда же).
    pub fn forward(&self, x: &Tensor, kv: &mut Kv, past: usize) -> Result<Tensor> {
        let s = x.dims()[1];
        if past + s > kv.cap {
            return Err(Audio8Error::Other(format!("KV переполнен: {past}+{s} > {}", kv.cap)));
        }
        let mut hidden = x.to_dtype(self.dtype)?;
        for (i, b) in self.blocks.iter().enumerate() {
            let h = rms_norm(&hidden, &b.attn_norm, self.eps)?;
            let a = self.attention(b, i, &h, kv, past, s)?;
            hidden = hidden.add(&a)?;
            let h = rms_norm(&hidden, &b.ffn_norm, self.eps)?;
            let g = b.w1.forward(&h)?;
            let u = b.w3.forward(&h)?;
            let act = match g.silu_and_mul(&u) {
                Ok(t) => t,
                Err(_) => g.silu()?.mul(&u)?,
            };
            hidden = hidden.add(&b.w2.forward(&act)?)?;
        }
        Ok(hidden)
    }
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.contiguous()?);
    }
    let d = x.dims().to_vec();
    let (b, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    Ok(x.unsqueeze(2)?
        .broadcast_as(vec![b, nkv, group, s, hd])?
        .contiguous()?
        .reshape(vec![b, nkv * group, s, hd])?)
}

fn causal_bias(q_len: usize, kv_len: usize, device: Device) -> Result<Tensor> {
    let past = kv_len - q_len;
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in past + i + 1..kv_len {
            data[i * kv_len + j] = NEG_LARGE;
        }
    }
    Ok(Tensor::from_vec(data, vec![1usize, 1, q_len, kv_len], device)?)
}

/// Модель DualAR целиком.
pub struct DualAr {
    pub cfg: Audio8Config,
    /// Таблица токенов (155776 × 896): на карте нужны лишь строки промпта и
    /// семантическая голова — на CUDA таблица живёт в RAM.
    embeddings: Tensor,
    codebook_embeddings: Tensor,
    fast_embeddings: Tensor,
    pub slow: Stack,
    norm: Tensor,
    pub fast: Stack,
    fast_norm: Tensor,
    fast_output: Lin,
    /// Строки выходной (связанной с эмбеддингом) головы, которые пропускает
    /// семантический фильтр upstream: семантические токены подряд, последний — EOS.
    sem_head: Tensor,
    pub device: Device,
    pub dtype: DType,
}

impl DualAr {
    pub fn load(ck: &Checkpoint) -> Result<Self> {
        let c = ck.config.clone();
        let slow = Stack::load(ck, "layers", c.n_layer, c.n_head, c.n_local_heads, c.head_dim, c.attention_qkv_bias, c.attention_o_bias, c.max_seq_len, true)?;
        // Быстрый AR — 4 слоя, на каждый кадр 10 шагов: квант на лету не трогаем
        // (чувствительнее медленного, а по объёму — 1/6 модели).
        let fast_q = std::env::var("SYN_AUDIO8_FAST_QUANT").as_deref() == Ok("1");
        let fast = Stack::load(ck, "fast_layers", c.n_fast_layer, c.fast_n_head, c.fast_n_local_heads, c.fast_head_dim, c.fast_attention_qkv_bias, c.fast_attention_o_bias, c.num_codebooks, fast_q)?;
        let table_dev = if ck.device.is_cuda() { Device::Cpu } else { ck.device };
        let embeddings = ck.get_as("embeddings.weight", table_dev, ck.compute)?;
        let mut rows: Vec<i64> = (c.semantic_begin_id..=c.semantic_end_id).map(|i| i as i64).collect();
        rows.push(c.eos_token_id as i64);
        let idx = Tensor::from_vec(rows.clone(), vec![rows.len()], table_dev)?;
        let sem_head = embeddings.index_select(0, &idx)?.contiguous()?.to_device(ck.device)?;
        Ok(Self {
            embeddings,
            codebook_embeddings: ck.get("codebook_embeddings.weight")?,
            fast_embeddings: ck.get("fast_embeddings.weight")?,
            slow,
            norm: ck.get("norm.weight")?,
            fast,
            fast_norm: ck.get("fast_norm.weight")?,
            fast_output: ck.linear("fast_output", false, false, None)?,
            sem_head,
            device: ck.device,
            dtype: ck.compute,
            cfg: c,
        })
    }

    /// Эмбеддинг столбцов `[1 + num_codebooks, S]` (строка 0 — токен, далее коды):
    /// токен + сумма эмбеддингов кодбуков там, где токен семантический.
    pub fn embed(&self, rows: &[Vec<u32>]) -> Result<Tensor> {
        let c = &self.cfg;
        let s = rows[0].len();
        let tok: Vec<i64> = rows[0].iter().map(|&t| t as i64).collect();
        let idx = Tensor::from_vec(tok, vec![s], self.embeddings.device())?;
        let base = self.embeddings.index_select(0, &idx)?.to_device(self.device)?.reshape(vec![1usize, s, c.dim])?;
        let sem: Vec<bool> = rows[0].iter().map(|&t| t >= c.semantic_begin_id && t <= c.semantic_end_id).collect();
        if !sem.iter().any(|&b| b) {
            return Ok(base);
        }
        let nc = c.num_codebooks;
        let mut cb = Vec::with_capacity(s * nc);
        for t in 0..s {
            for i in 0..nc {
                cb.push((rows[i + 1][t] as usize + i * c.codebook_size) as i64);
            }
        }
        let idx = Tensor::from_vec(cb, vec![s * nc], self.device)?;
        let sum = self
            .codebook_embeddings
            .index_select(0, &idx)?
            .reshape(vec![s, nc, c.dim])?
            .to_dtype(DType::F32)?
            .sum_keepdim(1)?
            .reshape(vec![1usize, s, c.dim])?;
        let mask: Vec<f32> = sem.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, vec![1usize, s, 1], self.device)?;
        let sum = sum.broadcast_mul(&mask)?.to_dtype(self.dtype)?;
        Ok(base.add(&sum)?)
    }

    /// Шаг медленного AR: столбцы `rows` с позиции `past`. Возвращает логиты по
    /// строкам `sem_head` (F32, хост) и вход быстрого AR `[1, 1, D]`.
    pub fn slow_step(&self, rows: &[Vec<u32>], kv: &mut Kv, past: usize) -> Result<(Vec<f32>, Tensor)> {
        let x = self.embed(rows)?;
        let s = x.dims()[1];
        let h = self.slow.forward(&x, kv, past)?;
        let last = h.narrow(1, s - 1, 1)?.contiguous()?;
        let normed = rms_norm(&last, &self.norm, self.cfg.norm_eps)?;
        let logits = normed.linear(&self.sem_head)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let fast_in = if self.cfg.norm_fastlayer_input { normed } else { last };
        Ok((logits, fast_in))
    }

    /// Полные логиты словаря по последней позиции (для сверки с эталоном).
    pub fn full_logits(&self, hidden_normed: &Tensor) -> Result<Vec<f32>> {
        Ok(hidden_normed.linear(&self.embeddings.to_device(self.device)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Шаг быстрого AR на позиции `pos`: вход `[1, 1, D]` → логиты кодбука (F32, хост).
    pub fn fast_step(&self, x: &Tensor, kv: &mut Kv, pos: usize) -> Result<Vec<f32>> {
        let h = self.fast.forward(x, kv, pos)?;
        let h = rms_norm(&h, &self.fast_norm, self.cfg.norm_eps)?;
        Ok(self.fast_output.forward(&h)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Только прогон быстрого AR (заполнение KV), без головы.
    pub fn fast_prime(&self, x: &Tensor, kv: &mut Kv) -> Result<()> {
        self.fast.forward(x, kv, 0)?;
        Ok(())
    }

    /// Эмбеддинг кода для быстрого AR: `[1, 1, D]`.
    pub fn fast_embed(&self, code: u32) -> Result<Tensor> {
        let idx = Tensor::from_vec(vec![code as i64], vec![1usize], self.device)?;
        Ok(self.fast_embeddings.index_select(0, &idx)?.reshape(vec![1usize, 1, self.cfg.dim])?)
    }
}
