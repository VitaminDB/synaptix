//! Энкодер Qwen-Image 2.1: языковая модель Qwen3-VL 8B. Промпт оборачивается
//! шаблоном пайплайна (системная инструкция, `<image{k}>` + токены картинок,
//! текст), картинки идут через башню зрения ([`crate::vision`]) и подставляются
//! на место `<|image_pad|>`; после первых слоёв к строкам картинок прибавляются
//! deepstack-отводы башни. Выход — скрытое состояние **последнего слоя до
//! финальной нормы** (так обучался DiT) без первых `drop_idx` токенов
//! системной части.
//!
//! Позиции — M-RoPE Qwen3-VL с чередованием частот по осям (t, h, w): текст
//! идёт подряд по всем осям, токены картинки получают (t, h, w) от текущей
//! позиции, после картинки позиция сдвигается на `max(h, w)`.
//!
//! Слои, не влезшие в VRAM, читаются из mmap-источника по одному прямо в
//! forward (следующий — на loader-стриме); таблица эмбеддингов целиком не
//! грузится — строки токенов берутся из mmap.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_image_qwen::memory;
use synaptix_image_qwen::source::Weights;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

use crate::config::Qwen3VlConfig;

pub const LM: &str = "model.language_model";

/// Системная инструкция пайплайна (одна на t2i и правку).
pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

const IMAGE_PAD: &str = "<|image_pad|>";

/// Токенизированный промпт: ids с развёрнутыми `<|image_pad|>` и сколько
/// первых токенов (системная часть) отбросить из выхода.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTokens {
    pub ids: Vec<u32>,
    pub drop_idx: usize,
}

pub struct PromptTokenizer {
    tok: HfTokenizer,
    drop_idx: usize,
}

impl PromptTokenizer {
    pub fn new(tokenizer_json: &[u8]) -> Result<Self> {
        let tok = HfTokenizer::from_bytes(tokenizer_json)
            .map_err(|e| SynaptixError::Other(format!("tokenizer.json: {e}")))?;
        for t in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", IMAGE_PAD] {
            if tok.token_to_id(t).is_none() {
                return Err(SynaptixError::Other(format!("в токенайзере нет {t}")));
            }
        }
        // Число токенов системного сообщения шаблона чата — как
        // `apply_chat_template([system])` у пайплайна.
        let sys = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n");
        let drop_idx =
            tok.encode(&sys, false).map_err(|e| SynaptixError::Other(format!("токенизация: {e}")))?.ids.len();
        Ok(Self { tok, drop_idx })
    }

    pub fn drop_idx(&self) -> usize {
        self.drop_idx
    }

    pub fn image_token_id(&self) -> u32 {
        self.tok.token_to_id(IMAGE_PAD).unwrap_or(151655)
    }

    /// Текст шаблона: `image_tokens[i]` — сколько токенов LLM даёт картинка i.
    /// Пустой промпт → `" "` (у Qwen нет BOS, пустая строка оставила бы
    /// энкодер без входа).
    pub fn render(&self, prompt: &str, image_tokens: &[usize]) -> String {
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let imgs: String = image_tokens
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                format!(
                    "{}<image{}><|vision_start|>{}<|vision_end|>",
                    if i == 0 { "" } else { " " },
                    i + 1,
                    IMAGE_PAD.repeat(n)
                )
            })
            .collect();
        format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{imgs}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    pub fn encode(&self, prompt: &str, image_tokens: &[usize]) -> Result<PromptTokens> {
        let text = self.render(prompt, image_tokens);
        let ids = self.tok.encode(&text, false).map_err(|e| SynaptixError::Other(format!("токенизация: {e}")))?.ids;
        Ok(PromptTokens { ids, drop_idx: self.drop_idx })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
}

struct Layer {
    /// Нормы в F32: остаточный поток считается в F32 (см. [`TextEncoder::encode`]).
    input_norm: Tensor,
    post_norm: Tensor,
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Layer {
    fn load(w: &Weights, idx: usize, dev: Device, dt: DType) -> Result<Self> {
        let p = format!("{LM}.layers.{idx}");
        let t = |n: &str| w.get(&format!("{p}.{n}"), dev, dt);
        let f = |n: &str| w.get(&format!("{p}.{n}"), dev, DType::F32);
        let l = |n: &str| QuantLinear::build(t(&format!("{n}.weight"))?, None, dt, dt);
        Ok(Self {
            input_norm: f("input_layernorm.weight")?,
            post_norm: f("post_attention_layernorm.weight")?,
            q: l("self_attn.q_proj")?,
            k: l("self_attn.k_proj")?,
            v: l("self_attn.v_proj")?,
            o: l("self_attn.o_proj")?,
            q_norm: t("self_attn.q_norm.weight")?,
            k_norm: t("self_attn.k_norm.weight")?,
            gate: l("mlp.gate_proj")?,
            up: l("mlp.up_proj")?,
            down: l("mlp.down_proj")?,
        })
    }
}

/// Позиции M-RoPE `(t, h, w)`: текст подряд, токены каждой картинки — сеткой
/// `grids[i]` (в токенах LLM) от текущей позиции; после картинки позиция
/// сдвигается на `max(h, w)` (`get_rope_index`).
pub fn mrope_positions(ids: &[u32], image_token: u32, grids: &[(usize, usize)]) -> Result<Vec<[u32; 3]>> {
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0u32;
    let mut i = 0usize;
    let mut img = 0usize;
    while i < ids.len() {
        if ids[i] != image_token {
            out.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let &(gh, gw) =
            grids.get(img).ok_or_else(|| SynaptixError::Other("токенов картинок больше, чем картинок".into()))?;
        let n = gh * gw;
        if i + n > ids.len() || ids[i..i + n].iter().any(|&t| t != image_token) {
            return Err(SynaptixError::Other(format!("картинка {img}: ожидалось {n} токенов подряд")));
        }
        for y in 0..gh {
            for x in 0..gw {
                out.push([pos, pos + y as u32, pos + x as u32]);
            }
        }
        pos += gh.max(gw) as u32;
        i += n;
        img += 1;
    }
    if img != grids.len() {
        return Err(SynaptixError::Other(format!("в промпте {img} картинок из {}", grids.len())));
    }
    Ok(out)
}

/// Эмбеддинг одной картинки для подстановки: токены LLM `[n, hidden]`, отводы
/// deepstack по слотам `[n, hidden]` и сетка в токенах LLM.
pub struct ImageEmbeds {
    pub tokens: Tensor,
    pub deepstack: Vec<Tensor>,
    pub grid: (usize, usize),
}

pub struct TextEncoder {
    cfg: Qwen3VlConfig,
    weights: Weights,
    device: Device,
    dtype: DType,
    layers: Vec<Layer>,
}

impl TextEncoder {
    /// На карту идут слои, пока остаётся запас под два стримящихся слоя,
    /// активации и рабочий стол; остальные читаются из `weights` в forward.
    pub fn build(w: Weights, cfg: Qwen3VlConfig, device: Device, dtype: DType, act_bytes: usize) -> Result<Self> {
        let lb = cfg.layer_bytes(dtype.bytes_for_numel(1).max(1));
        let reserve = 2 * lb + act_bytes + memory::DESKTOP_MARGIN;
        let mut layers = Vec::new();
        if device.is_cuda() {
            let _g = memory::weights_guard(device);
            for i in 0..cfg.num_layers {
                if memory::free_vram(device) < reserve + lb {
                    memory::release_pools(device);
                    if memory::free_vram(device) < reserve + lb {
                        eprintln!(
                            "[qwen-image-2.1] энкодер: на карте {i}/{} слоёв, остальные читаются из источника",
                            cfg.num_layers
                        );
                        break;
                    }
                }
                layers.push(Layer::load(&w, i, device, dtype)?);
            }
        }
        Ok(Self { cfg, weights: w, device, dtype, layers })
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
        let n = self.cfg.num_layers;
        if first >= n {
            return Ok(());
        }
        let (dev, dt) = (self.device, self.dtype);
        let ls = match dev {
            Device::Cuda(ord) => Some(synaptix_core::device::cuda::loader_stream(ord)?),
            _ => None,
        };
        // Поток префетча только копирует (BF16 в файле → тип энкодера) и
        // синкает loader-стрим: ядра из него не запускаются.
        let load = |idx: usize, on_loader: bool| -> Result<Layer> {
            let _g = memory::weights_guard(dev);
            if let (true, Some(ls)) = (on_loader, ls.as_ref()) {
                synaptix_core::device::cuda::set_alloc_stream(Some(ls.clone()));
                let r = Layer::load(&self.weights, idx, dev, dt);
                let _ = ls.synchronize();
                synaptix_core::device::cuda::set_alloc_stream(None);
                r
            } else {
                Layer::load(&self.weights, idx, dev, dt)
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
        let name = format!("{LM}.embed_tokens.weight");
        let (bytes, sdt, shape) =
            self.weights.raw(&name).ok_or_else(|| SynaptixError::Other(format!("нет тензора {name}")))?;
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

    /// cos/sin `[S, D/2]` M-RoPE с чередованием: частота i берёт позицию оси
    /// по правилу `apply_interleaved_mrope` (h — i ≡ 1 mod 3, w — i ≡ 2 mod 3
    /// в пределах `3·section`), иначе — секциями подряд.
    fn rope_tables(&self, pos: &[[u32; 3]]) -> Result<(Tensor, Tensor)> {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let inv: Vec<f32> =
            (0..half).map(|i| (1.0 / self.cfg.rope_theta.powf(2.0 * i as f64 / hd as f64)) as f32).collect();
        let mut axis = vec![0usize; half];
        let sec = &self.cfg.mrope_section;
        if self.cfg.mrope_interleaved {
            for (ax, offset) in [(1usize, 1usize), (2, 2)] {
                let limit = (sec.get(ax).copied().unwrap_or(0) * 3).min(half);
                let mut i = offset;
                while i < limit {
                    axis[i] = ax;
                    i += 3;
                }
            }
        } else {
            let mut off = 0usize;
            for (a, &n) in sec.iter().enumerate() {
                for x in axis.iter_mut().skip(off).take(n) {
                    *x = a.min(2);
                }
                off += n;
            }
        }
        let s = pos.len();
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for (p, xyz) in pos.iter().enumerate() {
            for i in 0..half {
                let a = xyz[axis[i]] as f32 * inv[i];
                cos[p * half + i] = (a as f64).cos() as f32;
                sin[p * half + i] = (a as f64).sin() as f32;
            }
        }
        Ok((Tensor::from_vec(cos, (s, half), self.device)?, Tensor::from_vec(sin, (s, half), self.device)?))
    }

    /// Прогон: `ids` с развёрнутыми `<|image_pad|>`, `images` — эмбеддинги
    /// картинок по порядку. Выход — `[S, hidden]` последнего слоя **до**
    /// финальной нормы (F32).
    pub fn encode(&self, ids: &[u32], images: &[ImageEmbeds]) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let cfg = &self.cfg;
        let s = ids.len();
        if s == 0 {
            return Err(SynaptixError::Other("пустой промпт".into()));
        }
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let group = cfg.group_size();
        let scale = 1.0 / (hd as f32).sqrt();
        let grids: Vec<(usize, usize)> = images.iter().map(|e| e.grid).collect();
        let pos = mrope_positions(ids, cfg.image_token_id, &grids)?;
        let (cos, sin) = self.rope_tables(&pos)?;

        // Эмбеддинги: строки текста из таблицы, отрезки картинок — из башни.
        // Заодно запоминаем строки каждой картинки для deepstack. Остаточный
        // поток и нормы — в F32 (у Qwen3 в нём выбросы до сотен: в BF16
        // косинус выхода к F32 падал до 0,98 на коротких промптах), проекции
        // и внимание — в `dtype`.
        let dt = self.dtype;
        let mut x = self.embed(ids)?.to_dtype(DType::F32)?;
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let mut parts: Vec<Tensor> = Vec::new();
            let mut i = 0usize;
            let mut img = 0usize;
            while i < s {
                if ids[i] == cfg.image_token_id {
                    let e = &images[img];
                    let n = e.grid.0 * e.grid.1;
                    if e.tokens.dims()[0] != n {
                        return Err(SynaptixError::Other(format!(
                            "картинка {img}: {} эмбеддингов на {n} токенов",
                            e.tokens.dims()[0]
                        )));
                    }
                    parts.push(e.tokens.to_device(self.device)?.to_dtype(DType::F32)?);
                    spans.push((i, n));
                    i += n;
                    img += 1;
                } else {
                    let j = (i..s).find(|&j| ids[j] == cfg.image_token_id).unwrap_or(s);
                    parts.push(x.narrow(0, i, j - i)?.contiguous()?);
                    i = j;
                }
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            x = Tensor::cat(&refs, 0)?;
        }
        let deep_slots = images.iter().map(|e| e.deepstack.len()).min().unwrap_or(0);

        self.for_each_layer(|li, l| {
            let h = rms(&x, &l.input_norm, cfg.rms_eps)?.to_dtype(dt)?;
            let q = rms(&l.q.forward(&h)?.reshape((s, nh, hd))?, &l.q_norm, cfg.rms_eps)?;
            let k = rms(&l.k.forward(&h)?.reshape((s, nkv, hd))?, &l.k_norm, cfg.rms_eps)?;
            let v = l.v.forward(&h)?.reshape((s, nkv, hd))?;
            drop(h);
            let q = rope(&q.transpose(0, 1)?.contiguous()?, &cos, &sin, hd)?;
            let k = rope(&k.transpose(0, 1)?.contiguous()?, &cos, &sin, hd)?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let k = repeat_kv(&k, group)?;
            let v = repeat_kv(&v, group)?;
            let attn = attend(
                &q.reshape((1, nh, s, hd))?,
                &k.reshape((1, nh, s, hd))?,
                &v.reshape((1, nh, s, hd))?,
                scale,
            )?;
            let attn = attn.reshape((nh, s, hd))?.transpose(0, 1)?.contiguous()?.reshape((s, nh * hd))?;
            x = x.add(&l.o.forward(&attn)?.to_dtype(DType::F32)?)?;
            let h = rms(&x, &l.post_norm, cfg.rms_eps)?.to_dtype(dt)?;
            let g = l.gate.forward(&h)?;
            let u = l.up.forward(&h)?;
            let a = match g.silu_and_mul(&u) {
                Ok(a) => a,
                Err(_) => g.silu()?.mul(&u)?,
            };
            x = x.add(&l.down.forward(&a)?.to_dtype(DType::F32)?)?;
            // Deepstack: после слоя `li` к строкам каждой картинки прибавляется
            // её отвод `li` (`_deepstack_process`).
            if li < deep_slots {
                let mut parts: Vec<Tensor> = Vec::with_capacity(2 * spans.len() + 1);
                let mut cursor = 0usize;
                for (img, &(start, n)) in spans.iter().enumerate() {
                    if start > cursor {
                        parts.push(x.narrow(0, cursor, start - cursor)?.contiguous()?);
                    }
                    let tap = images[img].deepstack[li].to_device(self.device)?.to_dtype(DType::F32)?;
                    parts.push(x.narrow(0, start, n)?.contiguous()?.add(&tap)?);
                    cursor = start + n;
                }
                if cursor < s {
                    parts.push(x.narrow(0, cursor, s - cursor)?.contiguous()?);
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                x = Tensor::cat(&refs, 0)?;
            }
            Ok(())
        })?;
        Ok(x)
    }
}

/// Причинное SDPA `[1,H,S,D]`: flash на CUDA, иначе явный softmax с маской.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    if q.device().is_cuda() && matches!(q.dtype(), DType::BF16 | DType::F16) {
        if let Ok(o) = q.flash_attention(k, v, scale, true) {
            return Ok(o);
        }
    }
    let (sq, sk) = (q.dims()[2], k.dims()[2]);
    let mut m = vec![0f32; sq * sk];
    for i in 0..sq {
        for j in (i + 1)..sk {
            m[i * sk + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(m, (1, 1, sq, sk), q.device())?.to_dtype(q.dtype())?;
    synaptix_ops::attention::softmax::scaled_dot_attention(q, k, v, scale, Some(&mask))
}

/// Split-half RoPE по `[H, S, D]`, cos/sin `[S, D/2]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, hd: usize) -> Result<Tensor> {
    if x.device().is_cuda() {
        if let Ok(y) = x.rope_split_partial_fused(cos, sin, hd) {
            return Ok(y);
        }
    }
    let half = hd / 2;
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let x0 = xf.narrow(2, 0, half)?.contiguous()?;
    let x1 = xf.narrow(2, half, half)?.contiguous()?;
    let o0 = x0.broadcast_mul(cos)?.sub(&x1.broadcast_mul(sin)?)?;
    let o1 = x1.broadcast_mul(cos)?.add(&x0.broadcast_mul(sin)?)?;
    Tensor::cat(&[&o0, &o1], 2)?.to_dtype(dt)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_positions_like_hf() {
        let img = 9u32;
        let ids = [1, 2, 3, img, img, img, img, img, img, 4, 5];
        let p = mrope_positions(&ids, img, &[(2, 3)]).unwrap();
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[3], [3, 3, 3]);
        assert_eq!(p[5], [3, 3, 5]);
        assert_eq!(p[8], [3, 4, 5]);
        assert_eq!(p[9], [6, 6, 6]);
        assert!(mrope_positions(&ids, img, &[]).is_err());
    }

    #[test]
    fn interleaved_axes_rule() {
        // Правило `apply_interleaved_mrope` для секций [24, 20, 20] и головы 128.
        let half = 64;
        let sec = [24usize, 20, 20];
        let mut axis = vec![0usize; half];
        for (ax, offset) in [(1usize, 1usize), (2, 2)] {
            let limit = (sec[ax] * 3).min(half);
            let mut i = offset;
            while i < limit {
                axis[i] = ax;
                i += 3;
            }
        }
        assert_eq!(axis.iter().filter(|&&a| a == 0).count(), 24);
        assert_eq!(axis.iter().filter(|&&a| a == 1).count(), 20);
        assert_eq!(axis.iter().filter(|&&a| a == 2).count(), 20);
        assert_eq!(&axis[..6], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(&axis[60..], &[0, 0, 0, 0]);
    }
}
