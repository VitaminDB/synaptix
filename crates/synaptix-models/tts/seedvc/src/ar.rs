//! Авторегрессия манеры (`modules/v2/ar.py`): по «узким» токенам (фонетика
//! без длительностей) и широким токенам образца дописывает широкие токены
//! источника в манере образца.
//!
//! Трансформер: 12 слоёв, 768, 12 голов запроса × 64 и 2 головы ключа/значения
//! (GQA, каждая на 6 голов запроса), pre-RMSNorm, SwiGLU 2304, RoPE.
//! Последовательность: `[SEP, узкие(образец + источник), SEP, широкие(образец), …]`;
//! позиции RoPE после второго SEP считаются заново с нуля.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::loader::Comp;
use crate::nn::{apply_rope, host, rms_norm, swiglu, Lin, Rng, Rope};
use crate::regulator::Embedding;
use crate::{SeedVcError, R};

const DIM: usize = 768;
const HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-5;
/// Широкий словарь без EOS.
pub const WIDE_VOCAB: usize = 2048;
pub const EOS: u32 = WIDE_VOCAB as u32;
/// Предел шагов генерации эталона.
const MAX_STEPS: usize = 4000;
/// Пока сгенерировано меньше стольких токенов, EOS запрещён.
const MIN_TOKENS: usize = 10;

struct Layer {
    attn_norm: Tensor,
    wqkv: Lin,
    wo: Lin,
    ffn_norm: Tensor,
    w1: Lin,
    w3: Lin,
    w2: Lin,
}

pub struct Ar {
    sep: Vec<f32>,
    /// Эмбеддинги широких токенов (2049 × 768).
    emb: Embedding,
    /// Регулятор длины AR: эмбеддинги узких токенов (32 × 768).
    cond_emb: Embedding,
    layers: Vec<Layer>,
    norm: Tensor,
    output: Lin,
    rope: Rope,
    device: Device,
}

#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

/// Кэш ключей и значений слоя: `[1, 2, S, 64]`.
type Kv = Vec<Option<(Tensor, Tensor)>>;

impl Ar {
    pub fn load(c: &Comp) -> R<Self> {
        let mut layers = Vec::new();
        let mut i = 0;
        while c.has(&format!("model.layers.{i}.attention.wqkv.weight")) {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: c.get(&format!("{p}.attention_norm.weight"), DType::F32)?,
                wqkv: Lin::load(c, &format!("{p}.attention.wqkv"), false)?,
                wo: Lin::load(c, &format!("{p}.attention.wo"), false)?,
                ffn_norm: c.get(&format!("{p}.ffn_norm.weight"), DType::F32)?,
                w1: Lin::load(c, &format!("{p}.feed_forward.w1"), false)?,
                w3: Lin::load(c, &format!("{p}.feed_forward.w3"), false)?,
                w2: Lin::load(c, &format!("{p}.feed_forward.w2"), false)?,
            });
            i += 1;
        }
        if layers.is_empty() {
            return Err(SeedVcError::Load("в компоненте ar нет слоёв".into()));
        }
        Ok(Self {
            sep: c.host("sep_token_emb")?.0,
            emb: Embedding::load(c, "model.embeddings.weight")?,
            cond_emb: Embedding::load(c, "length_regulator.embedding.weight")?,
            layers,
            norm: c.get("model.norm.weight", DType::F32)?,
            output: Lin::load(c, "model.output", false)?,
            rope: Rope::load(c, "model.freqs_cis")?,
            device: c.device(),
        })
    }

    /// Предел длины всей последовательности (условие + образец + генерация).
    pub fn max_positions(&self) -> usize {
        self.rope.max_positions()
    }

    /// Эмбеддинги условия `[n, 768]` на хосте (для сверки с эталоном).
    pub fn cond_embeddings(&self, narrow: &[u32]) -> R<Vec<f32>> {
        let mut v = Vec::with_capacity(narrow.len() * DIM);
        for &t in narrow {
            v.extend_from_slice(self.cond_emb.row(t)?);
        }
        Ok(v)
    }

    /// Проход по `x` `[1, s, 768]` с позициями RoPE `positions`; ключи и значения
    /// дописываются в кэш. Запросы видят весь прежний кэш и себя каузально.
    fn forward(&self, x: &Tensor, positions: &[usize], kv: &mut Kv) -> R<Tensor> {
        let s = x.dims()[1];
        let rope = self.rope.tables(positions, self.device)?;
        let group = HEADS / KV_HEADS;
        let mut h = x.clone();
        for (li, l) in self.layers.iter().enumerate() {
            let n = rms_norm(&h, &l.attn_norm, EPS)?;
            let qkv = l.wqkv.fwd(&n)?; // [1, s, 768 + 128 + 128]
            let kvd = KV_HEADS * HEAD_DIM;
            let q = qkv.narrow(2, 0, DIM)?.contiguous()?.reshape(vec![1, s, HEADS, HEAD_DIM])?;
            let k = qkv.narrow(2, DIM, kvd)?.contiguous()?.reshape(vec![1, s, KV_HEADS, HEAD_DIM])?;
            let v = qkv.narrow(2, DIM + kvd, kvd)?.contiguous()?.reshape(vec![1, s, KV_HEADS, HEAD_DIM])?;
            let q = apply_rope(&q, &rope)?.permute(vec![0, 2, 1, 3])?.contiguous()?; // [1, 12, s, 64]
            let k = apply_rope(&k, &rope)?.permute(vec![0, 2, 1, 3])?.contiguous()?; // [1, 2, s, 64]
            let v = v.permute(vec![0, 2, 1, 3])?.contiguous()?;
            let (k, v) = match kv[li].take() {
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?.contiguous()?, Tensor::cat(&[&pv, &v], 2)?.contiguous()?),
                None => (k, v),
            };
            let total = k.dims()[2];
            let past = total - s;
            // Головы запроса 0…5 делят первую голову ключа, 6…11 — вторую: складываем их в ось строк.
            let qg = q.reshape(vec![1, KV_HEADS, group * s, HEAD_DIM])?;
            let mask = if s > 1 {
                let mut m = vec![0f32; group * s * total];
                for g in 0..group {
                    for i in 0..s {
                        for j in past + i + 1..total {
                            m[(g * s + i) * total + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                Some(Tensor::from_vec(m, vec![1, 1, group * s, total], self.device)?)
            } else {
                None
            };
            let a = scaled_dot_attention(&qg, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), mask.as_ref())?;
            let a = a.reshape(vec![1, HEADS, s, HEAD_DIM])?.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![1, s, DIM])?;
            kv[li] = Some((k, v));
            h = h.add(&l.wo.fwd(&a)?)?;
            h = h.add(&swiglu(&rms_norm(&h, &l.ffn_norm, EPS)?, &l.w1, &l.w3, &l.w2)?)?;
        }
        Ok(h)
    }

    /// Логиты последней позиции (2049) на хосте.
    fn logits(&self, h: &Tensor) -> R<Vec<f32>> {
        let s = h.dims()[1];
        let last = h.narrow(1, s - 1, 1)?.contiguous()?;
        host(&self.output.fwd(&rms_norm(&last, &self.norm, EPS)?)?)
    }

    /// Начало последовательности: `[SEP, условие, SEP, широкие(образец)]` → логиты первого токена.
    fn prefill(&self, narrow: &[u32], prompt_wide: &[u32], kv: &mut Kv) -> R<(Vec<f32>, usize)> {
        let (lc, lp) = (narrow.len(), prompt_wide.len());
        let mut x = Vec::with_capacity((lc + lp + 2) * DIM);
        x.extend_from_slice(&self.sep);
        for &t in narrow {
            x.extend_from_slice(self.cond_emb.row(t)?);
        }
        x.extend_from_slice(&self.sep);
        for &t in prompt_wide {
            x.extend_from_slice(self.emb.row(t)?);
        }
        let mut pos: Vec<usize> = (0..=lc).collect();
        pos.extend(0..=lp);
        let s = pos.len();
        let h = self.forward(&Tensor::from_vec(x, vec![1, s, DIM], self.device)?, &pos, kv)?;
        Ok((self.logits(&h)?, lp))
    }

    fn step(&self, token: u32, pos: usize, kv: &mut Kv) -> R<Vec<f32>> {
        let x = Tensor::from_vec(self.emb.row(token)?.to_vec(), vec![1, 1, DIM], self.device)?;
        self.logits(&self.forward(&x, &[pos], kv)?)
    }

    /// Логиты первых `n` шагов жадной генерации и сами токены — для сверки с эталоном.
    pub fn greedy_trace(&self, narrow: &[u32], prompt_wide: &[u32], keep_logits: usize, max_tokens: usize) -> R<(Vec<Vec<f32>>, Vec<u32>)> {
        let mut kv: Kv = (0..self.layers.len()).map(|_| None).collect();
        let (mut logits, lp) = self.prefill(narrow, prompt_wide, &mut kv)?;
        let mut trace = vec![logits.clone()];
        let mut out = Vec::new();
        for i in 0..max_tokens {
            if out.len() < MIN_TOKENS {
                logits[EOS as usize] = f32::NEG_INFINITY;
            }
            let tok = argmax(&logits) as u32;
            if tok == EOS {
                break;
            }
            out.push(tok);
            logits = self.step(tok, lp + 1 + i, &mut kv)?;
            if trace.len() < keep_logits {
                trace.push(logits.clone());
            }
        }
        Ok((trace, out))
    }

    /// Генерация широких токенов. `narrow` — узкие токены (образец + кусок
    /// источника, без повторов подряд), `prompt_wide` — широкие токены образца
    /// (пусто — обезличивание). `progress(сгенерировано)`; `false` — остановить.
    pub fn generate(&self, narrow: &[u32], prompt_wide: &[u32], sp: Sampling, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> bool) -> R<Vec<u32>> {
        let prefix = narrow.len() + prompt_wide.len() + 2;
        if prefix + MIN_TOKENS >= self.max_positions() {
            return Err(SeedVcError::Input(format!("условие авторегрессии {prefix} позиций, предел {}", self.max_positions())));
        }
        let mut kv: Kv = (0..self.layers.len()).map(|_| None).collect();
        let (mut logits, lp) = self.prefill(narrow, prompt_wide, &mut kv)?;
        let mut out: Vec<u32> = Vec::new();
        // Эталон: запас кэша на 4096 позиций и не больше 4000 шагов после первого токена.
        let budget = (self.max_positions() - prefix).min(MAX_STEPS + 1);
        while out.len() < budget {
            // Штраф за повтор эталон применяет только к первому сгенерированному
            // токену (`previous_tokens[0]` в `decode_one_token_ar`) — повторяем как есть.
            let tok = sample(&mut logits, out.first().copied(), out.len() < MIN_TOKENS, sp, rng);
            if tok == EOS {
                break;
            }
            out.push(tok);
            if !progress(out.len()) {
                return Err(SeedVcError::Cancelled);
            }
            if out.len() >= budget {
                break;
            }
            logits = self.step(tok, lp + out.len(), &mut kv)?;
        }
        Ok(out)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `logits_to_probs` + `multinomial_sample_one_no_sync` эталона: штраф за
/// повтор → запрет EOS → top-p по логитам до температуры (первый по убыванию
/// остаётся всегда) → температура → softmax → `argmax(p / Exp(1))`.
pub fn sample(logits: &mut [f32], previous: Option<u32>, suppress_eos: bool, sp: Sampling, rng: &mut Rng) -> u32 {
    if let Some(p) = previous {
        let s = &mut logits[p as usize];
        *s = if *s < 0.0 { *s * sp.repetition_penalty } else { *s / sp.repetition_penalty };
    }
    if suppress_eos {
        logits[EOS as usize] = f32::NEG_INFINITY;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let max = logits[order[0]];
    let z: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let mut cum = 0f64;
    let mut keep = order.len();
    for (rank, &i) in order.iter().enumerate() {
        cum += ((logits[i] - max) as f64).exp() / z;
        if rank > 0 && cum as f32 > sp.top_p {
            keep = rank;
            break;
        }
    }
    // Эталон убирает позиции, где накопленная вероятность (включая сам токен) выше top_p.
    let temp = sp.temperature.max(1e-5);
    let kept = &order[..keep.max(1)];
    let zt: f64 = kept.iter().map(|&i| (((logits[i] - max) / temp) as f64).exp()).sum();
    let mut best = (kept[0], f64::NEG_INFINITY);
    for &i in kept {
        let p = (((logits[i] - max) / temp) as f64).exp() / zt;
        let score = p / rng.exp(i);
        if score > best.1 {
            best = (i, score);
        }
    }
    rng.advance();
    best.0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_top_p_is_greedy() {
        let mut rng = Rng::new(1);
        let sp = Sampling { top_p: 1e-6, temperature: 1.0, repetition_penalty: 1.0 };
        let mut l = vec![0f32; WIDE_VOCAB + 1];
        l[7] = 3.0;
        l[EOS as usize] = 9.0;
        assert_eq!(sample(&mut l.clone(), None, true, sp, &mut rng), 7);
        assert_eq!(sample(&mut l, None, false, sp, &mut rng), EOS);
    }
}
