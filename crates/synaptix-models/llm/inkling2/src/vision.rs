//! Башня картинок Inkling-2 (`model.visual.*`, `Inkling2VisionModel` transformers).
//!
//! Патч 20×20 → линейная проекция + выученная сетка позиций 32×32, билинейно
//! растянутая на сетку патчей картинки (align_corners=False, края — клэмп).
//! Перед патчами — CLS-токен (позиция (0,0)), у патчей позиции (строка+1,
//! столбец+1). 23 слоя pre-LN ViT с LayerScale, внимание полное по всей
//! картинке с осевым 2D RoPE на парах соседних каналов: первые `hd/4` частот
//! — столбец, следующие — строка, база 10000. Патчи идут клетками 2×2, так что
//! после башни четвёрка соседних строк склеивается в `4·1024` и MLP (GELU)
//! проецирует её в эмбеддинг текста 2304. Признаки встают на места
//! плейсхолдеров `<|unused_200054|>` после `embed_norm`, сами не нормируются.
//!
//! Остаточный поток держится в F32 (у оригинала BF16), матрицы — BF16,
//! внимание — flash-ядро движка `[1, S, H, D]` без материализации скоров.

use rayon::prelude::*;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;
use synaptix_ops::norm::layer_norm;

use crate::config::VisionConfig;
use crate::kern::{fe, R};
use crate::loader::{Inkling2Weights, LoadError};
use crate::model::{EmbedPatch, Inkling2Model, ModelCache};
use crate::preprocess::PreparedImage;

pub const VIS: &str = "model.visual";
/// Плейсхолдер признака картинки (`image_token` processor_config.json).
pub const IMAGE_TOKEN: &str = "<|unused_200054|>";
pub const IMAGE_TOKEN_ID: u32 = 200054;

fn lerr(e: impl std::fmt::Display) -> LoadError {
    LoadError::Weights(e.to_string())
}

struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    fn load(w: &Inkling2Weights, name: &str, bias: bool, dev: Device) -> Result<Self, LoadError> {
        Ok(Self {
            w: w.tensor(&format!("{name}.weight"), dev, DType::BF16)?,
            b: if bias { Some(w.tensor(&format!("{name}.bias"), dev, DType::BF16)?) } else { None },
        })
    }

    /// `x` BF16 `[S, in]` → BF16 `[S, out]`.
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        x.linear_bias_residual(&self.w, self.b.as_ref(), None).map_err(fe)
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
}

impl Ln {
    fn load(w: &Inkling2Weights, name: &str, dev: Device) -> Result<Self, LoadError> {
        Ok(Self { w: w.tensor(&format!("{name}.weight"), dev, DType::F32)?, b: w.tensor(&format!("{name}.bias"), dev, DType::F32)? })
    }

    fn fwd(&self, x: &Tensor, eps: f32) -> R<Tensor> {
        layer_norm(x, Some(&self.w), Some(&self.b), eps).map_err(fe)
    }
}

struct Layer {
    ln1: Ln,
    ln2: Ln,
    /// q, k, v одной матрицей `[3D, D]`.
    qkv: Lin,
    out: Lin,
    fc1: Lin,
    fc2: Lin,
    ls1: Tensor,
    ls2: Tensor,
}

pub struct VisionTower {
    pub cfg: VisionConfig,
    pub device: Device,
    patch: Lin,
    /// Сетка позиций `[G·G, D]` F32 на хосте (интерполяция под каждую картинку).
    pos_table: Vec<f32>,
    /// `class_embedding + cls_position_embedding`, F32 `[1, D]`.
    cls: Tensor,
    ln_pre: Ln,
    layers: Vec<Layer>,
    proj_in: Lin,
    proj_out: Lin,
}

/// Отводы и веса билинейной интерполяции по оси (как
/// `get_vision_interpolation_indices_and_weights`, bilinear, border).
fn taps(i: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = (i as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let f = src.floor();
    let mut out = [(0usize, 0f32); 2];
    for (k, o) in out.iter_mut().enumerate() {
        let raw = f + k as f32;
        let wgt = (1.0 - (src - raw).abs()).max(0.0);
        *o = (raw.clamp(0.0, (side - 1) as f32) as usize, wgt);
    }
    out
}

impl VisionTower {
    pub fn load(w: &Inkling2Weights, dev: Device) -> Result<Self, LoadError> {
        let cfg = w.config.vision.clone().ok_or_else(|| LoadError::Config("в config.json нет vision_config".into()))?;
        let name = |s: &str| format!("{VIS}.{s}");
        if !w.contains(&name("backbone.embeddings.patch_embedding.weight")) {
            return Err(LoadError::Weights("нет весов model.visual.* (бандл без башни — перепакуйте ink2_pack)".into()));
        }
        let d = cfg.hidden_size;
        let pos_table = w.host_f32(&name("backbone.embeddings.position_embedding"))?;
        if pos_table.len() != cfg.posemb_grid_size * cfg.posemb_grid_size * d {
            return Err(LoadError::Weights(format!("position_embedding: {} чисел", pos_table.len())));
        }
        let ce = w.host_f32(&name("backbone.embeddings.class_embedding"))?;
        let cp = w.host_f32(&name("backbone.embeddings.cls_position_embedding"))?;
        // У оригинала сумма в BF16: (class + cls_pos).to(bf16).
        let cls: Vec<f32> = ce.iter().zip(&cp).map(|(a, b)| half::bf16::from_f32(a + b).to_f32()).collect();
        let cls = Tensor::from_vec(cls, vec![1, d], dev).map_err(lerr)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = name(&format!("backbone.layers.{i}"));
            let a = format!("{p}.self_attn");
            let cat = |part: &str, dt: DType| -> Result<Tensor, LoadError> {
                let ts: Vec<Tensor> =
                    ["q_proj", "k_proj", "v_proj"].iter().map(|m| w.tensor(&format!("{a}.{m}.{part}"), dev, dt)).collect::<Result<_, _>>()?;
                let refs: Vec<&Tensor> = ts.iter().collect();
                Tensor::cat(&refs, 0).map_err(lerr)?.contiguous().map_err(lerr)
            };
            layers.push(Layer {
                ln1: Ln::load(w, &format!("{p}.ln_1"), dev)?,
                ln2: Ln::load(w, &format!("{p}.ln_2"), dev)?,
                qkv: Lin { w: cat("weight", DType::BF16)?, b: Some(cat("bias", DType::BF16)?) },
                out: Lin::load(w, &format!("{a}.out_proj"), true, dev)?,
                fc1: Lin::load(w, &format!("{p}.mlp.c_fc"), true, dev)?,
                fc2: Lin::load(w, &format!("{p}.mlp.c_proj"), true, dev)?,
                ls1: w.tensor(&format!("{p}.ls_1.gamma"), dev, DType::F32)?,
                ls2: w.tensor(&format!("{p}.ls_2.gamma"), dev, DType::F32)?,
            });
        }
        Ok(Self {
            patch: Lin::load(w, &name("backbone.embeddings.patch_embedding"), false, dev)?,
            pos_table,
            cls,
            ln_pre: Ln::load(w, &name("backbone.ln_pre"), dev)?,
            layers,
            proj_in: Lin::load(w, &name("proj_in"), cfg.projector_bias, dev)?,
            proj_out: Lin::load(w, &name("proj_out"), cfg.projector_bias, dev)?,
            cfg,
            device: dev,
        })
    }

    /// Координаты (строка, столбец) патчей в порядке башни: клетки построчно,
    /// внутри клетки 2×2 построчно.
    fn patch_coords(&self, rows: usize, cols: usize) -> Vec<(usize, usize)> {
        let m = self.cfg.adapter_merge_size;
        let mut v = Vec::with_capacity(rows * cols * m * m);
        for r in 0..rows {
            for c in 0..cols {
                for mh in 0..m {
                    for mw in 0..m {
                        v.push((r * m + mh, c * m + mw));
                    }
                }
            }
        }
        v
    }

    /// Позиции патчей: сетка G×G, растянутая на `h×w` патчей, F32 `[n, D]`.
    fn positions(&self, coords: &[(usize, usize)], h: usize, w: usize) -> Vec<f32> {
        let (g, d) = (self.cfg.posemb_grid_size, self.cfg.hidden_size);
        let ty: Vec<_> = (0..h).map(|i| taps(i, h, g)).collect();
        let tx: Vec<_> = (0..w).map(|i| taps(i, w, g)).collect();
        let mut out = vec![0f32; coords.len() * d];
        out.par_chunks_mut(d).zip(coords.par_iter()).for_each(|(o, &(r, c))| {
            for &(y, wy) in &ty[r] {
                for &(x, wx) in &tx[c] {
                    let wgt = wy * wx;
                    if wgt == 0.0 {
                        continue;
                    }
                    let src = &self.pos_table[(y * g + x) * d..(y * g + x + 1) * d];
                    for (a, s) in o.iter_mut().zip(src) {
                        *a += wgt * s;
                    }
                }
            }
        });
        out
    }

    /// cos/sin RoPE `[S, hd]` F32 (полная таблица на пары каналов): CLS — (0,0).
    fn rope(&self, coords: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.head_dim();
        let q = hd / 4;
        let spatial = hd / 2;
        let theta = self.cfg.rope_theta;
        let inv: Vec<f32> = (0..q).map(|i| 1.0 / theta.powf((2 * i) as f32 / spatial as f32)).collect();
        let s = coords.len() + 1;
        let mut cos = vec![0f32; s * hd];
        let mut sin = vec![0f32; s * hd];
        for t in 0..s {
            let (row, col) = if t == 0 { (0.0, 0.0) } else { let (r, c) = coords[t - 1]; ((r + 1) as f32, (c + 1) as f32) };
            for j in 0..spatial {
                // Пара j: первые hd/4 пар — столбец, следующие — строка.
                let ang = if j < q { col * inv[j] } else { row * inv[j - q] };
                let (sn, cs) = ang.sin_cos();
                for k in 0..2 {
                    cos[t * hd + 2 * j + k] = cs;
                    sin[t * hd + 2 * j + k] = sn;
                }
            }
        }
        (cos, sin)
    }

    /// Признаки картинки: F32 `[rows·cols, text_hidden]` на карте.
    pub fn encode(&self, img: &PreparedImage) -> R<Tensor> {
        let cfg = &self.cfg;
        let dev = self.device;
        let (d, nh, hd, m) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim(), cfg.adapter_merge_size);
        let eps = cfg.layer_norm_eps;
        let n = img.rows * img.cols * m * m;
        let pd = cfg.native_patch_size * cfg.native_patch_size * 3;
        if img.patches.len() != n * pd {
            return Err(ModelError::Shape(format!("inkling2 vision: {} чисел патчей, ждали {}", img.patches.len(), n * pd)));
        }
        let coords = self.patch_coords(img.rows, img.cols);
        let (ph, pw) = (img.rows * m, img.cols * m);

        let px = Tensor::from_vec(img.patches.clone(), vec![n, pd], dev).map_err(fe)?.to_dtype(DType::BF16).map_err(fe)?;
        let emb = self.patch.fwd(&px)?.to_dtype(DType::F32).map_err(fe)?;
        drop(px);
        let pos = Tensor::from_vec(self.positions(&coords, ph, pw), vec![n, d], dev).map_err(fe)?;
        // Позиции у оригинала приводятся к BF16 перед сложением.
        let pos = pos.to_dtype(DType::BF16).map_err(fe)?.to_dtype(DType::F32).map_err(fe)?;
        let emb = emb.add(&pos).map_err(fe)?;
        let x = Tensor::cat(&[&self.cls, &emb], 0).map_err(fe)?.contiguous().map_err(fe)?;
        drop((emb, pos));
        let mut x = self.ln_pre.fwd(&x, eps)?;

        let s = n + 1;
        let (cos, sin) = self.rope(&coords);
        let cos = Tensor::from_vec(cos, vec![s, hd], dev).map_err(fe)?;
        let sin = Tensor::from_vec(sin, vec![s, hd], dev).map_err(fe)?;
        let scale = 1.0 / (hd as f32).sqrt();

        for l in &self.layers {
            let h = l.ln1.fwd(&x, eps)?.to_dtype(DType::BF16).map_err(fe)?;
            let qkv = l.qkv.fwd(&h)?.reshape(vec![s, 3, nh, hd]).map_err(fe)?;
            let part = |i: usize| -> R<Tensor> {
                qkv.narrow(1, i, 1).map_err(fe)?.contiguous().map_err(fe)?.reshape(vec![1, s, nh, hd]).map_err(fe)
            };
            let q = part(0)?.rope_interleaved_fused(&cos, &sin).map_err(fe)?;
            let k = part(1)?.rope_interleaved_fused(&cos, &sin).map_err(fe)?;
            let v = part(2)?;
            drop(qkv);
            let a = q.flash_attention_bshd(&k, &v, scale, false).map_err(fe)?;
            drop((q, k, v));
            let a = l.out.fwd(&a.reshape(vec![s, d]).map_err(fe)?)?.to_dtype(DType::F32).map_err(fe)?;
            x = x.add(&a.broadcast_mul(&l.ls1).map_err(fe)?).map_err(fe)?;

            let h = l.ln2.fwd(&x, eps)?.to_dtype(DType::BF16).map_err(fe)?;
            let h = l.fc1.fwd(&h)?.gelu_exact().map_err(fe)?;
            let h = l.fc2.fwd(&h)?.to_dtype(DType::F32).map_err(fe)?;
            x = x.add(&h.broadcast_mul(&l.ls2).map_err(fe)?).map_err(fe)?;
        }

        let tokens = img.rows * img.cols;
        let merged = x
            .narrow(0, 1, n)
            .map_err(fe)?
            .contiguous()
            .map_err(fe)?
            .reshape(vec![tokens, m * m * d])
            .map_err(fe)?
            .to_dtype(DType::BF16)
            .map_err(fe)?;
        let h = self.proj_in.fwd(&merged)?.gelu_exact().map_err(fe)?;
        let out = self.proj_out.fwd(&h)?;
        out.to_dtype(DType::F32).map_err(fe)?.contiguous().map_err(fe)
    }
}

/// Картинка в промпте: первая позиция её плейсхолдеров и признаки `[N, H]`.
pub struct ImageSpan {
    pub start: usize,
    pub embeds: Tensor,
}

/// Развернуть по одному плейсхолдеру на картинку в `counts[i]` копий (как
/// `Inkling2Processor`). Возвращает ids и первую позицию каждой картинки.
pub fn expand_image_tokens(ids: &[u32], image_id: u32, counts: &[usize]) -> Result<(Vec<u32>, Vec<usize>), ModelError> {
    let found = ids.iter().filter(|&&t| t == image_id).count();
    if found != counts.len() {
        return Err(ModelError::Shape(format!("inkling2: плейсхолдеров картинок {found}, картинок {}", counts.len())));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut starts = Vec::with_capacity(counts.len());
    let mut k = 0;
    for &t in ids {
        if t == image_id {
            starts.push(out.len());
            out.extend(std::iter::repeat_n(image_id, counts[k]));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}

/// Вставить картинки в отрендеренный шаблоном промпт: перед последней
/// текстовой репликой пользователя — `<|message_user|><|content_image|>
/// <плейсхолдер><|end_message|>` на картинку. Так шаблон рендерит сообщение
/// с частями `[image…, text]`.
pub fn insert_image_parts(prompt: &str, images: usize) -> String {
    let marker = "<|message_user|><|content_text|>";
    let part = format!("<|message_user|><|content_image|>{IMAGE_TOKEN}<|end_message|>").repeat(images);
    match prompt.rfind(marker) {
        Some(i) => format!("{}{part}{}", &prompt[..i], &prompt[i..]),
        None => format!("{prompt}{part}"),
    }
}

/// Префилл промпта с картинками: строки картинок берут готовые признаки
/// (модель сама режет прогон на куски). Возвращает логиты последнего токена.
pub fn prefill_with_images(model: &Inkling2Model, cache: &mut ModelCache, ids: &[u32], images: &[ImageSpan]) -> R<Tensor> {
    if ids.is_empty() {
        return Err(ModelError::Forward("inkling2: пустой промпт".into()));
    }
    let patches: Vec<EmbedPatch> = images.iter().map(|im| EmbedPatch { row: im.start, embeds: im.embeds.clone() }).collect();
    model.forward_with_embeds(ids, &patches, cache, false)
}
