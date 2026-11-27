//! Авторегрессия MiniMax Music 3: глобальная LM (Qwen3 8B) покадрово сэмплит
//! семантический код (первый кодбук RVQ) с CFG, локальная LM (глубинный
//! декодер) — остальные семь; скрытые состояния кадра (1 + 7 по 4096) уходят
//! условием в flow-matching. Эталон — diffusers `encoders.py`.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::{DecoderModel, KvCache};
use synaptix_llm_qwen3::config::Qwen3Config;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::loader::{read_file, Comp, LmSource};
use crate::prompt::{AUDIO_END, CODE_OFFSET, SEMANTIC_VOCAB};
use crate::{Music3Error, R};

pub const CODEBOOKS: usize = 8;
pub const AUDIO_VOCAB: usize = 1024;
const CFG_SCALE: f32 = 1.5;
const TOP_K: usize = 50;
const COMPUTE: DType = DType::BF16;

/// `y = x·Wᵀ`; вес хранится транспонированным `[in, out]`.
struct Lin(Tensor);

impl Lin {
    fn load(c: &Comp, name: &str) -> R<Self> {
        Ok(Self(c.get(name, COMPUTE)?.transpose(0, 1)?.contiguous()?))
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let d = x.dims().to_vec();
        let n: usize = d[..d.len() - 1].iter().product();
        let y = x.contiguous()?.reshape(vec![n, d[d.len() - 1]])?.matmul(&self.0)?;
        let mut s = d[..d.len() - 1].to_vec();
        s.push(self.0.dims()[1]);
        Ok(y.reshape(s)?)
    }
}

struct DepthLayer {
    norm1: Tensor,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    norm2: Tensor,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// Локальная LM: 4 слоя, 4096, 16 голов, каузальная, выученные позиции.
pub struct DepthDecoder {
    audio_emb: Tensor,
    projection: Lin,
    pos: Tensor,
    layers: Vec<DepthLayer>,
    norm: Tensor,
    heads: Vec<Lin>,
    device: Device,
}

fn rms(x: &Tensor, w: &Tensor) -> R<Tensor> {
    Ok(x.contiguous()?.rms_norm_fused(w, 1e-6, false)?)
}

impl DepthDecoder {
    pub fn load(c: &Comp, device: Device) -> R<Self> {
        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("layers.{i}");
            layers.push(DepthLayer {
                norm1: c.get(&format!("{p}.input_layernorm.weight"), COMPUTE)?,
                q: Lin::load(c, &format!("{p}.attn.to_q.weight"))?,
                k: Lin::load(c, &format!("{p}.attn.to_k.weight"))?,
                v: Lin::load(c, &format!("{p}.attn.to_v.weight"))?,
                out: Lin::load(c, &format!("{p}.attn.to_out.weight"))?,
                norm2: c.get(&format!("{p}.post_attention_layernorm.weight"), COMPUTE)?,
                gate: Lin::load(c, &format!("{p}.gate_proj.weight"))?,
                up: Lin::load(c, &format!("{p}.up_proj.weight"))?,
                down: Lin::load(c, &format!("{p}.down_proj.weight"))?,
            });
        }
        let mut heads = Vec::with_capacity(CODEBOOKS - 1);
        for i in 0..CODEBOOKS - 1 {
            heads.push(Lin::load(c, &format!("audio_heads.{i}.weight"))?);
        }
        Ok(Self {
            audio_emb: c.get("audio_embeddings.weight", COMPUTE)?,
            projection: Lin::load(c, "projection.weight")?,
            pos: c.get("pos_embedding.weight", COMPUTE)?,
            layers,
            norm: c.get("norm.weight", COMPUTE)?,
            heads,
            device,
        })
    }

    /// Строки таблицы акустических кодов (`code + (книга − 1)·1024`) → `[n, H]`.
    fn embed(&self, rows: &[usize]) -> R<Tensor> {
        let parts: Vec<Tensor> = rows.iter().map(|&r| self.audio_emb.narrow(0, r, 1).and_then(|t| t.contiguous())).collect::<Result<_, _>>()?;
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Tensor::cat(&refs, 0)?)
    }

    /// `seq` `[B, S, H]` (уже через `projection`) → нормированные состояния `[B, S, H]`.
    pub fn forward(&self, seq: &Tensor) -> R<Tensor> {
        let (b, s, h) = (seq.dims()[0], seq.dims()[1], seq.dims()[2]);
        let (nh, hd) = (16usize, h / 16);
        let mut x = seq.broadcast_add(&self.pos.narrow(0, 0, s)?.contiguous()?.reshape(vec![1, s, h])?)?;
        let mut mask = vec![0f32; s * s];
        for i in 0..s {
            for j in i + 1..s {
                mask[i * s + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask, vec![1, 1, s, s], self.device)?.to_dtype(COMPUTE)?;
        let heads = |t: Tensor| -> R<Tensor> { Ok(t.reshape(vec![b, s, nh, hd])?.permute(vec![0, 2, 1, 3])?.contiguous()?) };
        for l in &self.layers {
            let n = rms(&x, &l.norm1)?;
            let a = scaled_dot_attention(&heads(l.q.fwd(&n)?)?, &heads(l.k.fwd(&n)?)?, &heads(l.v.fwd(&n)?)?, 1.0 / (hd as f32).sqrt(), Some(&mask))?;
            let a = a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, h])?;
            x = x.add(&l.out.fwd(&a)?)?;
            let n = rms(&x, &l.norm2)?;
            let g = l.gate.fwd(&n)?;
            let u = l.up.fwd(&n)?;
            let act = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u)))?;
            x = x.add(&l.down.fwd(&act)?)?;
        }
        rms(&x, &self.norm)
    }
}

/// Генератор случайных чисел сэмплера (splitmix64) — воспроизводимость по сиду.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Сэмпл из top-k логитов (softmax, мультиномиально). `−∞` — запрещённые.
fn sample_top_k(logits: &[f32], rng: &mut Rng) -> usize {
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(TOP_K);
    let m = logits[idx[0]];
    let p: Vec<f64> = idx.iter().map(|&i| ((logits[i] - m) as f64).exp()).collect();
    let mut r = rng.next() * p.iter().sum::<f64>();
    for (k, w) in p.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return idx[k];
        }
    }
    *idx.last().unwrap()
}

fn host(t: &Tensor) -> R<Vec<f32>> {
    Ok(t.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

pub struct Ar {
    pub model: DecoderModel,
    pub depth: DepthDecoder,
    device: Device,
    hidden: usize,
}

impl Ar {
    /// `quant_w` — квант весов внимания и MLP глобальной LM (`None` — как в бандле / плотно).
    pub fn open(bundle: &Path, device: Device, quant_w: Option<DType>) -> R<Self> {
        let cfg = Qwen3Config::from_hf_json_slice(&read_file(bundle, "language_model/config.json")?).map_err(|e| Music3Error::Load(format!("конфиг LM: {e}")))?;
        let dcfg = cfg.to_decoder_config();
        let src = LmSource(Comp::open(bundle, "language_model", device)?);
        let w = quant_w.unwrap_or(COMPUTE);
        let cap = crate::prompt::MAX_PROMPT_TOKENS + crate::prompt::MAX_FRAMES + 64;
        let model = DecoderModel::build_auto(&dcfg, &src, device, COMPUTE, w, w, COMPUTE, COMPUTE, cap).map_err(|e| Music3Error::Load(format!("LM: {e}")))?.with_kv_cache_dtype(COMPUTE);
        let depth = DepthDecoder::load(&Comp::open(bundle, "rvq_depth_decoder", device)?, device)?;
        Ok(Self { model, depth, device, hidden: dcfg.hidden_size })
    }

    fn me(e: impl std::fmt::Display) -> Music3Error {
        Music3Error::Other(e.to_string())
    }

    /// Покадровая генерация. `cond`/`uncond` — токены промпта (одной длины).
    /// Возвращает скрытые состояния кадров `[1, F, 8·H]` (BF16).
    /// `progress(кадр, максимум)`; `false` — остановить.
    pub fn generate(&self, cond: &[u32], uncond: &[u32], max_frames: usize, seed: u64, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Tensor> {
        let (dev, h) = (self.device, self.hidden);
        let s = cond.len();
        // Ветки CFG — двумя кэшами по одному ряду (flash-ядро внимания LM
        // рассчитано на пакет 1).
        let mut kvs: Vec<KvCache> = Vec::with_capacity(2);
        let mut rows: Vec<Tensor> = Vec::with_capacity(2);
        for ids in [cond, uncond] {
            let mut kv = self.model.make_kv_cache(1, s + max_frames + 8).map_err(Self::me)?;
            let t = Tensor::from_vec(ids.to_vec(), vec![1usize, s], dev)?;
            let hid = self.model.forward_trunk(&t, &mut kv).map_err(Self::me)?;
            rows.push(self.model.normed_at(&hid, s - 1).map_err(Self::me)?.contiguous()?);
            kvs.push(kv);
        }
        let mut last = Tensor::cat(&[&rows[0], &rows[1]], 0)?.contiguous()?; // [2, H]
        drop(rows);
        let mut rng = Rng::new(seed);
        let mut frames: Vec<Tensor> = Vec::with_capacity(max_frames);
        let embed = |ids: &[u32]| -> R<Tensor> { self.model.embed_rows(&Tensor::from_vec(ids.to_vec(), vec![ids.len()], dev)?).map_err(Self::me) };
        for frame in 0..=max_frames {
            // Семантический код: только диапазон кодов и конец аудио.
            let logits = self.model.lm_head_forward(&last).map_err(Self::me)?; // [2, V]
            let body = host(&logits.narrow(1, CODE_OFFSET as usize, SEMANTIC_VOCAB)?.contiguous()?)?;
            let end = host(&logits.narrow(1, AUDIO_END as usize, 1)?.contiguous()?)?;
            let (c, u) = body.split_at(SEMANTIC_VOCAB);
            let mut cl: Vec<f32> = c.to_vec();
            cl.push(end[0]);
            let mut ul: Vec<f32> = u.to_vec();
            ul.push(end[1]);
            // CFG, затем запрет всего вне top-k условной ветки.
            let mut sorted = cl.clone();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let thr = sorted[TOP_K.min(sorted.len()) - 1];
            let guided: Vec<f32> = cl.iter().zip(&ul).map(|(&c, &u)| if c < thr { f32::NEG_INFINITY } else { u + (c - u) * CFG_SCALE }).collect();
            let pick = sample_top_k(&guided, &mut rng);
            if pick == SEMANTIC_VOCAB {
                break;
            }
            let semantic = pick as u32;
            // Глубинный декодер: семь акустических кодов и их скрытые состояния.
            let sem_emb = embed(&[semantic + CODE_OFFSET])?; // [1, H]
            let sem_p = self.depth.projection.fwd(&sem_emb)?;
            let mut seq: Vec<Tensor> = vec![self.depth.projection.fwd(&last)?.reshape(vec![2, 1, h])?, Tensor::cat(&[&sem_p, &sem_p], 0)?.reshape(vec![2, 1, h])?];
            let mut codes = [0usize; CODEBOOKS - 1];
            let mut parts: Vec<Tensor> = Vec::with_capacity(CODEBOOKS);
            parts.push(last.narrow(0, 0, 1)?.contiguous()?);
            for i in 1..CODEBOOKS {
                let refs: Vec<&Tensor> = seq.iter().collect();
                let x = Tensor::cat(&refs, 1)?.contiguous()?;
                let n = x.dims()[1];
                let hs = self.depth.forward(&x)?.narrow(1, n - 1, 1)?.contiguous()?.reshape(vec![2, h])?;
                parts.push(hs.narrow(0, 0, 1)?.contiguous()?);
                let lg = host(&self.depth.heads[i - 1].fwd(&hs)?)?;
                let (c, u) = lg.split_at(AUDIO_VOCAB);
                let guided: Vec<f32> = c.iter().zip(u).map(|(&c, &u)| u + (c - u) * CFG_SCALE).collect();
                let code = sample_top_k(&guided, &mut rng);
                codes[i - 1] = code;
                if i < CODEBOOKS - 1 {
                    let e = self.depth.projection.fwd(&self.depth.embed(&[code + (i - 1) * AUDIO_VOCAB])?)?;
                    seq.push(Tensor::cat(&[&e, &e], 0)?.reshape(vec![2, 1, h])?);
                }
            }
            // Кадр 0 в условие не идёт (как в эталоне), но его коды — в обратную связь.
            if frame > 0 {
                let refs: Vec<&Tensor> = parts.iter().collect();
                frames.push(Tensor::cat(&refs, 1)?.contiguous()?); // [1, 8·H]
                if !progress(frames.len(), max_frames) {
                    return Err(Music3Error::Cancelled);
                }
                if frames.len() >= max_frames {
                    break;
                }
            }
            // Обратная связь: сумма эмбеддингов восьми кодов · 8^−½.
            let rows: Vec<usize> = codes.iter().enumerate().map(|(j, &c)| c + j * AUDIO_VOCAB).collect();
            let extra = self.depth.embed(&rows)?.contiguous()?.sum_keepdim(0)?.contiguous()?;
            let fb = sem_emb.add(&extra)?.mul_scalar((CODEBOOKS as f32).powf(-0.5))?;
            let fb = fb.reshape(vec![1, 1, h])?.contiguous()?;
            let mut rows: Vec<Tensor> = Vec::with_capacity(2);
            for kv in kvs.iter_mut() {
                let hid = self.model.forward_from_hidden(&fb, kv).map_err(Self::me)?;
                rows.push(self.model.normed_at(&hid, 0).map_err(Self::me)?.contiguous()?);
            }
            last = Tensor::cat(&[&rows[0], &rows[1]], 0)?.contiguous()?;
        }
        if frames.is_empty() {
            return Err(Music3Error::Input("модель не выдала ни одного кадра — промпт сразу закончил генерацию".into()));
        }
        let refs: Vec<&Tensor> = frames.iter().collect();
        let all = Tensor::cat(&refs, 0)?;
        let f = all.dims()[0];
        Ok(all.reshape(vec![1, f, CODEBOOKS * h])?)
    }
}
