//! LLM-энкодер текста FLUX.2: Mistral-Small-3.1 (dev) или Qwen3 (klein) до
//! последнего нужного слоя; выход — склейка `hidden_states` трёх слоёв
//! (`[1, S, 3·hidden]`), как `_get_*_prompt_embeds` пайплайнов diffusers.
//!
//! Проход по слоям одноразовый, поэтому слои, не влезшие в VRAM, читаются из
//! mmap-источника по одному прямо в forward (следующий — на loader-стриме),
//! копии на хосте нет: Mistral-24B (~45 ГБ BF16) проходит и на 7-гигабайтной
//! карте. Таблицу эмбеддингов целиком не грузим — строки токенов читаются
//! из mmap.
//!
//! Паддинг до 512 как у пайплайнов: у Mistral токенайзер паддит СЛЕВА, у
//! Qwen3 — справа. Позиции RoPE — `0..S` по всей строке (HF без
//! `position_ids`). Маска внимания HF: причинная по реальным ключам;
//! строка-паддинг слева не видит ни одного ключа — SDPA torch ≥ 2.5 отдаёт
//! для неё нули; паддинг справа видит все реальные токены.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

use crate::config::{TextEncoderArch, TextEncoderConfig};
use crate::memory;
use crate::source::Weights;
use crate::Flux2Error;

/// Максимальная длина промпта у обоих пайплайнов.
pub const MAX_SEQ: usize = 512;

/// `SYSTEM_MESSAGE` из `diffusers/pipelines/flux2/system_messages.py`.
pub const MISTRAL_SYSTEM: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

/// Токенизированный промпт: ids длины [`MAX_SEQ`] и где в них реальные
/// токены.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTokens {
    pub ids: Vec<u32>,
    pub real: std::ops::Range<usize>,
}

pub struct PromptTokenizer {
    tok: HfTokenizer,
    arch: TextEncoderArch,
    pad: u32,
}

impl PromptTokenizer {
    pub fn new(tokenizer_json: &[u8], arch: TextEncoderArch) -> Result<Self> {
        let tok = HfTokenizer::from_bytes(tokenizer_json)
            .map_err(|e| SynaptixError::Other(format!("tokenizer.json: {e}")))?;
        let pad_str = match arch {
            TextEncoderArch::Mistral3 => "<pad>",
            TextEncoderArch::Qwen3 => "<|endoftext|>",
        };
        let pad = tok
            .token_to_id(pad_str)
            .ok_or_else(|| SynaptixError::Other(format!("в токенайзере нет {pad_str}")))?;
        Ok(Self { tok, arch, pad })
    }

    /// Шаблон чата пайплайна (без генерации ответа у Mistral; у Qwen3 —
    /// `add_generation_prompt` с выключенным мышлением).
    pub fn render(&self, prompt: &str) -> String {
        match self.arch {
            TextEncoderArch::Mistral3 => format!(
                "<s>[SYSTEM_PROMPT]{MISTRAL_SYSTEM}[/SYSTEM_PROMPT][INST]{}[/INST]",
                prompt.replace("[IMG]", "")
            ),
            TextEncoderArch::Qwen3 => format!(
                "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
        }
    }

    pub fn encode(&self, prompt: &str) -> Result<PromptTokens> {
        let text = self.render(prompt);
        let mut ids = self
            .tok
            .encode(&text, self.arch == TextEncoderArch::Qwen3)
            .map_err(|e| SynaptixError::Other(format!("токенизация: {e}")))?
            .ids;
        ids.truncate(MAX_SEQ);
        let n = ids.len();
        let pad = MAX_SEQ - n;
        Ok(match self.arch {
            TextEncoderArch::Mistral3 => {
                let mut out = vec![self.pad; pad];
                out.extend(ids);
                PromptTokens { ids: out, real: pad..MAX_SEQ }
            }
            TextEncoderArch::Qwen3 => {
                ids.resize(MAX_SEQ, self.pad);
                PromptTokens { ids, real: 0..n }
            }
        })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Layer {
    fn load(w: &Weights, cfg: &TextEncoderConfig, idx: usize, dev: Device, dt: DType) -> Result<Self> {
        let p = format!("{}.layers.{idx}", cfg.prefix());
        let t = |n: &str| w.get(&format!("{p}.{n}"), dev, dt);
        let l = |n: &str| QuantLinear::build(t(&format!("{n}.weight"))?, None, dt, dt);
        Ok(Self {
            input_norm: t("input_layernorm.weight")?,
            post_norm: t("post_attention_layernorm.weight")?,
            q: l("self_attn.q_proj")?,
            k: l("self_attn.k_proj")?,
            v: l("self_attn.v_proj")?,
            o: l("self_attn.o_proj")?,
            q_norm: if cfg.qk_norm() { Some(t("self_attn.q_norm.weight")?) } else { None },
            k_norm: if cfg.qk_norm() { Some(t("self_attn.k_norm.weight")?) } else { None },
            gate: l("mlp.gate_proj")?,
            up: l("mlp.up_proj")?,
            down: l("mlp.down_proj")?,
        })
    }
}

pub struct TextEncoder {
    cfg: TextEncoderConfig,
    weights: Weights,
    device: Device,
    dtype: DType,
    /// Резидентный префикс слоёв.
    layers: Vec<Layer>,
    /// Сколько слоёв нужно прогнать (до последнего снимаемого).
    run: usize,
}

impl TextEncoder {
    /// `run` — сколько слоёв считать (последний из `taps`). На карту идут
    /// слои, пока остаётся запас под два стримящихся слоя, активации и
    /// рабочий стол; остальные читаются из `weights` в forward.
    pub fn build(w: Weights, cfg: TextEncoderConfig, run: usize, device: Device, dtype: DType) -> Result<Self> {
        let run = run.min(cfg.num_layers);
        let lb = cfg.layer_bytes();
        let act = 512usize << 20;
        let reserve = 2 * lb + act + memory::DESKTOP_MARGIN;
        let mut layers = Vec::new();
        if device.is_cuda() {
            let _g = memory::weights_guard(device);
            for i in 0..run {
                if memory::free_vram(device) < reserve + lb {
                    memory::release_pools(device);
                    if memory::free_vram(device) < reserve + lb {
                        eprintln!(
                            "[flux2] энкодер текста: на карте {i}/{run} слоёв, остальные читаются из источника"
                        );
                        break;
                    }
                }
                layers.push(Layer::load(&w, &cfg, i, device, dtype)?);
            }
        }
        Ok(Self { cfg, weights: w, device, dtype, layers, run })
    }

    pub fn resident_layers(&self) -> usize {
        self.layers.len()
    }

    fn for_each_layer<F>(&self, mut body: F) -> Result<()>
    where
        F: FnMut(usize, &Layer) -> Result<()>,
    {
        for (i, l) in self.layers.iter().enumerate() {
            body(i, l)?;
        }
        let first = self.layers.len();
        let n = self.run;
        if first >= n {
            return Ok(());
        }
        let (dev, dt) = (self.device, self.dtype);
        let ls = match dev {
            Device::Cuda(ord) => Some(synaptix_core::device::cuda::loader_stream(ord)?),
            _ => None,
        };
        let load = |idx: usize, on_loader: bool| -> Result<Layer> {
            let _g = memory::weights_guard(dev);
            if let (true, Some(ls)) = (on_loader, ls.as_ref()) {
                synaptix_core::device::cuda::set_alloc_stream(Some(ls.clone()));
                let r = Layer::load(&self.weights, &self.cfg, idx, dev, dt);
                let _ = ls.synchronize();
                synaptix_core::device::cuda::set_alloc_stream(None);
                r
            } else {
                Layer::load(&self.weights, &self.cfg, idx, dev, dt)
            }
        };
        let mut staged = Some(load(first, false));
        for idx in first..n {
            let cur = staged.take().expect("staged")?;
            let load = &load;
            let (step, next) = std::thread::scope(|sp| {
                let h = (idx + 1 < n).then(|| sp.spawn(move || load(idx + 1, true)));
                let step = body(idx, &cur);
                let next = h.map(|h| {
                    h.join().unwrap_or_else(|_| Err(SynaptixError::Other("поток префетча слоя упал".into())))
                });
                (step, next)
            });
            if let Device::Cuda(ord) = dev {
                if let Ok(cs) = synaptix_core::device::cuda::default_stream(ord) {
                    let _ = cs.synchronize();
                }
            }
            drop(cur);
            step?;
            staged = next;
        }
        Ok(())
    }

    /// Строки таблицы эмбеддингов прямо из mmap → `[S, hidden]` в `dtype`.
    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let name = format!("{}.embed_tokens.weight", self.cfg.prefix());
        let (bytes, sdt, shape) = self
            .weights
            .raw(&name)
            .ok_or_else(|| SynaptixError::Other(format!("нет тензора {name}")))?;
        let (vocab, hidden) = (shape[0], shape[1]);
        let row = sdt.bytes_for_numel(hidden);
        let mut out = Vec::with_capacity(ids.len() * row);
        for &id in ids {
            let id = id as usize;
            if id >= vocab {
                return Err(SynaptixError::Other(format!("токен {id} вне словаря {vocab}")));
            }
            out.extend_from_slice(&bytes[id * row..(id + 1) * row]);
        }
        Tensor::from_raw_bytes(out, (ids.len(), hidden), sdt, self.device)?.to_dtype(self.dtype)
    }

    /// Прогон промпта; выход — `hidden_states[taps]`, склеенные по
    /// последней оси: `[1, S, taps.len()·hidden]`.
    pub fn encode(&self, tokens: &PromptTokens, taps: &[usize]) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let cfg = &self.cfg;
        let s = tokens.ids.len();
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let real = tokens.real.clone();
        let (r0, r1) = (real.start, real.end);
        if r0 >= r1 {
            return Err(SynaptixError::Other("пустой промпт".into()));
        }
        let (cos, sin) = rope_tables(s, hd, cfg.rope_theta, self.device)?;

        let mut x = self.embed(&tokens.ids)?; // [S, hidden]
        let mut caps: Vec<(usize, Tensor)> = Vec::new();
        if taps.contains(&0) {
            caps.push((0, x.clone()));
        }
        self.for_each_layer(|li, l| {
            let h = rms(&x, &l.input_norm, cfg.rms_eps)?;
            let q = l.q.forward(&h)?.reshape((s, nh, hd))?;
            let k = l.k.forward(&h)?.reshape((s, nkv, hd))?;
            let v = l.v.forward(&h)?.reshape((s, nkv, hd))?;
            drop(h);
            let q = match &l.q_norm {
                Some(w) => rms(&q, w, cfg.rms_eps)?,
                None => q,
            };
            let k = match &l.k_norm {
                Some(w) => rms(&k, w, cfg.rms_eps)?,
                None => k,
            };
            // [H, S, D] + RoPE (split-half, как rotate_half у HF).
            let q = rope(&q.transpose(0, 1)?.contiguous()?, &cos, &sin, hd)?;
            let k = rope(&k.transpose(0, 1)?.contiguous()?, &cos, &sin, hd)?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let k = repeat_kv(&k, group)?;
            let v = repeat_kv(&v, group)?;
            let slice = |t: &Tensor, a: usize, b: usize| -> Result<Tensor> {
                t.narrow(1, a, b - a)?.contiguous()?.reshape((1, nh, b - a, hd))
            };
            let (kr, vr) = (slice(&k, r0, r1)?, slice(&v, r0, r1)?);
            let real_out = attend(&slice(&q, r0, r1)?, &kr, &vr, scale, true)?;
            let mut parts: Vec<Tensor> = Vec::new();
            if r0 > 0 {
                // Паддинг слева: ни одного видимого ключа → нули (torch ≥ 2.5).
                parts.push(Tensor::zeros((1, nh, r0, hd), real_out.dtype(), self.device)?);
            }
            parts.push(real_out);
            if r1 < s {
                // Паддинг справа видит все реальные токены.
                parts.push(attend(&slice(&q, r1, s)?, &kr, &vr, scale, false)?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let attn = Tensor::cat(&refs, 2)?; // [1, H, S, D]
            let attn = attn.reshape((nh, s, hd))?.transpose(0, 1)?.contiguous()?.reshape((s, nh * hd))?;
            x = x.add(&l.o.forward(&attn)?)?;

            let h = rms(&x, &l.post_norm, cfg.rms_eps)?;
            let g = l.gate.forward(&h)?;
            let u = l.up.forward(&h)?;
            let a = match g.silu_and_mul(&u) {
                Ok(a) => a,
                Err(_) => g.silu()?.mul(&u)?,
            };
            x = x.add(&l.down.forward(&a)?)?;
            if taps.contains(&(li + 1)) {
                caps.push((li + 1, x.clone()));
            }
            Ok(())
        })?;
        let mut out = Vec::with_capacity(taps.len());
        for t in taps {
            let c = caps
                .iter()
                .find(|(i, _)| i == t)
                .map(|(_, x)| x)
                .ok_or_else(|| SynaptixError::Other(format!("слой {t} энкодера не посчитан")))?;
            out.push(c);
        }
        let joined = Tensor::cat(&out, 1)?; // [S, taps·hidden]
        joined.reshape((1, s, taps.len() * cfg.hidden))
    }
}

/// SDPA `[1,H,Sq,D]` × `[1,H,Sk,D]`: flash на CUDA, иначе явный softmax.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    if q.device().is_cuda() && matches!(q.dtype(), DType::BF16 | DType::F16) {
        if let Ok(o) = q.flash_attention(k, v, scale, causal) {
            return Ok(o);
        }
    }
    let (sq, sk) = (q.dims()[2], k.dims()[2]);
    let mask = if causal {
        let mut m = vec![0f32; sq * sk];
        for i in 0..sq {
            for j in (i + 1)..sk {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
        Some(Tensor::from_vec(m, (1, 1, sq, sk), q.device())?.to_dtype(q.dtype())?)
    } else {
        None
    };
    synaptix_ops::attention::softmax::scaled_dot_attention(q, k, v, scale, mask.as_ref())
}

/// cos/sin `[S, D/2]` (F32): `inv_freq = 1/θ^(2i/D)` в f32, как у HF.
fn rope_tables(s: usize, hd: usize, theta: f64, device: Device) -> Result<(Tensor, Tensor)> {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half).map(|i| (1.0 / theta.powf(2.0 * i as f64 / hd as f64)) as f32).collect();
    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for p in 0..s {
        for i in 0..half {
            let a = p as f32 * inv[i];
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    Ok((Tensor::from_vec(cos, (s, half), device)?, Tensor::from_vec(sin, (s, half), device)?))
}

/// Split-half RoPE по `[H, S, D]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, hd: usize) -> Result<Tensor> {
    if x.device().is_cuda() {
        if let Ok(y) = x.rope_split_partial_fused(cos, sin, hd) {
            return Ok(y);
        }
    }
    let half = hd / 2;
    let dt = x.dtype();
    let (c, s) = (cos.to_dtype(dt)?, sin.to_dtype(dt)?);
    let x0 = x.narrow(2, 0, half)?.contiguous()?;
    let x1 = x.narrow(2, half, half)?.contiguous()?;
    let o0 = x0.broadcast_mul(&c)?.sub(&x1.broadcast_mul(&s)?)?;
    let o1 = x1.broadcast_mul(&c)?.add(&x0.broadcast_mul(&s)?)?;
    Tensor::cat(&[&o0, &o1], 2)
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let (nkv, s, hd) = (d[0], d[1], d[2]);
    let e = x.reshape((nkv, 1, s, hd))?;
    let parts: Vec<&Tensor> = std::iter::repeat_n(&e, group).collect();
    Tensor::cat(&parts, 1)?.reshape((nkv * group, s, hd))?.contiguous()
}

/// Конфиг энкодера и токенайзер из источника.
pub fn open_tokenizer(tokenizer_json: &[u8], arch: TextEncoderArch) -> std::result::Result<PromptTokenizer, Flux2Error> {
    PromptTokenizer::new(tokenizer_json, arch).map_err(|e| Flux2Error::Tokenizer(e.to_string()))
}
