//! Башня зрения Hy4-flash (`vit.*`, HYV4VL): SigLIP-so400m без pre/post-LN и
//! без 2D RoPE + адаптер `perceive` (RMSNorm → свёртка 2×2 → GELU → 1×1 →
//! столбец `image_newline` в конце каждой строки).
//!
//! Препроцесс: RGB → smart_resize (кратность 32, площадь 4096 … 2048²,
//! банковское округление) → бикубик с антиалиасом по uint8 → `u8/127.5 − 1` →
//! патчи 16×16 растром, вектор `[c][ky][kx]`. Позиции — таблица 128×128 с
//! билинейной интерполяцией по масштабу `(g + 0.1)/128` (`align_corners=False`).
//! Выход `[N, 3072]`, `N = (H/32)·(W/32 + 1)` — по одному на токен 120120.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::WeightSource;
use synaptix_ops::attention::softmax::scaled_dot::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::kern::{self, fe, R};

const DT: DType = DType::BF16;
const PATCH: usize = 16;
const MERGE: usize = 2;
const MIN_PIXELS: usize = 4096;
const MAX_PIXELS: usize = 2048 * 2048;
/// Сетка таблицы позиций (`max_image_size / patch`).
const POS_SIDE: usize = 128;
/// Запросов внимания за раз (оценки `[16, chunk, L]` в памяти).
const Q_CHUNK: usize = 1024;

/// Размер после smart_resize `(h, w)`, кратный 32.
pub fn smart_resize(h: usize, w: usize) -> (usize, usize) {
    smart_resize_px(h, w, MAX_PIXELS)
}

/// smart_resize с верхней границей площади `max_px`.
pub fn smart_resize_px(h: usize, w: usize, max_px: usize) -> (usize, usize) {
    let f = (PATCH * MERGE) as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut rh = (hf / f).round_ties_even() * f;
    let mut rw = (wf / f).round_ties_even() * f;
    if rh * rw > max_px as f64 {
        let beta = (hf * wf / max_px as f64).sqrt();
        rh = f.max((hf / beta / f).floor() * f);
        rw = f.max((wf / beta / f).floor() * f);
    } else if rh * rw < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (hf * wf)).sqrt();
        rh = (hf * beta / f).ceil() * f;
        rw = (wf * beta / f).ceil() * f;
    }
    ((rh as usize).max(32), (rw as usize).max(32))
}

/// Токенов 120120 на картинку `h×w` (исходный размер).
pub fn image_tokens(h: usize, w: usize) -> usize {
    let (rh, rw) = smart_resize(h, w);
    (rh / 32) * (rw / 32 + 1)
}

/// Заполнители картинки в тексте (chat_template Hy4).
pub const IMAGE_START: &str = "<｜hy_image▁start:opensource｜>";
pub const IMAGE_PAD: &str = "<｜hy_image▁pad:opensource｜>";
pub const IMAGE_END: &str = "<｜hy_image▁end:opensource｜>";

/// Подготовленная картинка: патчи `[gh·gw, 768]` F32 (`[-1, 1]`) и сетка.
pub struct Prepared {
    pub patches: Vec<f32>,
    pub gh: usize,
    pub gw: usize,
}

impl Prepared {
    /// Строк сетки и столбцов (с колонкой newline) после слияния 2×2.
    pub fn rows_cols(&self) -> (usize, usize) {
        (self.gh / MERGE, self.gw / MERGE + 1)
    }
    pub fn tokens(&self) -> usize {
        let (r, c) = self.rows_cols();
        r * c
    }
}

/// Файл картинки → патчи.
pub fn prepare(path: &std::path::Path) -> R<Prepared> {
    let img = synaptix_io::image::png::load_image(path, Device::Cpu).map_err(|e| synaptix_llm_common::ModelError::Load(format!("картинка: {e}")))?;
    let d = img.dims().to_vec();
    let src = img.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(fe)?;
    Ok(prepare_chw(&src, d[1], d[2]))
}

/// [`prepare`] с потолком числа токенов: картинка уменьшается (по площади),
/// пока `(H/32)·(W/32+1)` не влезет.
pub fn prepare_limited(path: &std::path::Path, max_tokens: Option<usize>) -> R<Prepared> {
    let img = synaptix_io::image::png::load_image(path, Device::Cpu).map_err(|e| synaptix_llm_common::ModelError::Load(format!("картинка: {e}")))?;
    let d = img.dims().to_vec();
    let src = img.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(fe)?;
    let (h, w) = (d[1], d[2]);
    let mut max_px = MAX_PIXELS;
    if let Some(cap) = max_tokens.filter(|c| *c > 0) {
        while max_px > MIN_PIXELS {
            let (rh, rw) = smart_resize_px(h, w, max_px);
            if (rh / 32) * (rw / 32 + 1) <= cap {
                break;
            }
            max_px = max_px * 9 / 10;
        }
    }
    Ok(prepare_chw_px(&src, h, w, max_px))
}

/// `[3, H, W]` в `0…1` → патчи.
pub fn prepare_chw(src: &[f32], h: usize, w: usize) -> Prepared {
    prepare_chw_px(src, h, w, MAX_PIXELS)
}

fn prepare_chw_px(src: &[f32], h: usize, w: usize, max_px: usize) -> Prepared {
    let (rh, rw) = smart_resize_px(h, w, max_px);
    // Ресайз по uint8 (как torchvision/PIL): вход — сетка 1/255, выход округлён к ней.
    let mut px = synaptix_vlm_minicpm::preprocess::resize_bicubic_aa(src, 3, h, w, rh, rw);
    if (rh, rw) == (h, w) {
        px.iter_mut().for_each(|v| *v = (*v * 255.0).round_ties_even() / 255.0);
    }
    px.iter_mut().for_each(|v| *v = *v * 2.0 - 1.0);
    let (patches, grid) = synaptix_vlm_minicpm::preprocess::patchify(&px, 3, rh, rw, PATCH);
    Prepared { patches, gh: grid.h, gw: grid.w }
}

/// Линейный слой: вес `[in, out]`, bias.
struct Lin {
    wt: Tensor,
    b: Tensor,
}

impl Lin {
    fn new(w: Tensor, b: Tensor) -> R<Self> {
        Ok(Self { wt: kern::contig(&w.transpose(0, 1).map_err(fe)?)?, b })
    }
    fn load(ws: &dyn WeightSource, key: &str, dev: Device) -> R<Self> {
        Self::new(ws.tensor(&format!("{key}.weight"), dev, DT)?, ws.tensor(&format!("{key}.bias"), dev, DT)?)
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        x.matmul(&self.wt).and_then(|y| y.broadcast_add(&self.b)).map_err(fe)
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
}

impl Ln {
    fn load(ws: &dyn WeightSource, key: &str, dev: Device) -> R<Self> {
        Ok(Self { w: ws.tensor(&format!("{key}.weight"), dev, DT)?, b: ws.tensor(&format!("{key}.bias"), dev, DT)? })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        layer_norm(x, Some(&self.w), Some(&self.b), 1e-5).map_err(fe)
    }
}

struct Block {
    ln1: Ln,
    qkv: Lin,
    o: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

pub struct Hy4Vision {
    device: Device,
    patch: Lin,
    /// Таблица позиций `[128·128, 1152]` F32 на хосте (без строки «cls»).
    pos_table: Vec<f32>,
    blocks: Vec<Block>,
    norm: Tensor,
    proj0: Lin,
    proj2: Lin,
    newline: Tensor,
    hidden: usize,
    heads: usize,
}

impl Hy4Vision {
    /// Башня из весов, если она есть (`vit.*`).
    pub fn load(ws: &dyn WeightSource, dev: Device, layers: usize) -> R<Option<Self>> {
        if !ws.contains("vit.embeddings.patch_embedding.weight") {
            return Ok(None);
        }
        let pw = ws.tensor("vit.embeddings.patch_embedding.weight", dev, DT)?;
        let hidden = pw.dims()[0];
        let patch = Lin::new(pw.reshape(vec![hidden, 3 * PATCH * PATCH]).map_err(fe)?, ws.tensor("vit.embeddings.patch_embedding.bias", dev, DT)?)?;
        let pt = ws.tensor("vit.embeddings.position_embedding.weight", Device::Cpu, DType::F32)?;
        let all = pt.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(fe)?;
        let pos_table = all[hidden..hidden * (1 + POS_SIDE * POS_SIDE)].to_vec();
        let mut blocks = Vec::with_capacity(layers);
        for l in 0..layers {
            let p = format!("vit.layers.{l}");
            let a = format!("{p}.self_attn");
            let cat = |n: &str| -> R<Tensor> {
                let parts: Vec<Tensor> = ["q_proj", "k_proj", "v_proj"].iter().map(|k| ws.tensor(&format!("{a}.{k}.{n}"), dev, DT)).collect::<R<_>>()?;
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 0).map_err(fe)
            };
            blocks.push(Block {
                ln1: Ln::load(ws, &format!("{p}.input_layernorm"), dev)?,
                qkv: Lin::new(cat("weight")?, cat("bias")?)?,
                o: Lin::load(ws, &format!("{a}.o_proj"), dev)?,
                ln2: Ln::load(ws, &format!("{p}.post_attention_layernorm"), dev)?,
                fc1: Lin::load(ws, &format!("{p}.mlp.dense_h_to_4h"), dev)?,
                fc2: Lin::load(ws, &format!("{p}.mlp.dense_4h_to_h"), dev)?,
            });
        }
        let p0 = ws.tensor("vit.perceive.proj.0.weight", dev, DT)?;
        let n0 = p0.dims()[0];
        let p2 = ws.tensor("vit.perceive.proj.2.weight", dev, DT)?;
        let n2 = p2.dims()[0];
        Ok(Some(Self {
            device: dev,
            patch,
            pos_table,
            blocks,
            norm: ws.tensor("vit.perceive.norm.weight", dev, DType::F32)?,
            proj0: Lin::new(p0.reshape(vec![n0, hidden * MERGE * MERGE]).map_err(fe)?, ws.tensor("vit.perceive.proj.0.bias", dev, DT)?)?,
            proj2: Lin::new(p2.reshape(vec![n2, n0]).map_err(fe)?, ws.tensor("vit.perceive.proj.2.bias", dev, DT)?)?,
            newline: ws.tensor("vit.perceive.image_newline", dev, DT)?.reshape(vec![1, n2]).map_err(fe)?,
            hidden,
            heads: 16,
        }))
    }

    /// Позиционные эмбеддинги `[gh·gw, C]` (F32): билинейно по таблице
    /// 128×128 с масштабом `(g + 0.1)/128`, `align_corners = False`.
    pub fn pos_embed(&self, gh: usize, gw: usize) -> Vec<f32> {
        let c = self.hidden;
        let axis = |g: usize| -> Vec<(usize, usize, f32)> {
            let s = POS_SIDE as f64 / (g as f64 + 0.1);
            (0..g)
                .map(|i| {
                    let src = ((i as f64 + 0.5) * s - 0.5).max(0.0);
                    let i0 = (src.floor() as usize).min(POS_SIDE - 1);
                    let i1 = (i0 + 1).min(POS_SIDE - 1);
                    (i0, i1, (src - i0 as f64) as f32)
                })
                .collect()
        };
        let (ay, ax) = (axis(gh), axis(gw));
        let t = &self.pos_table;
        let row = |y: usize, x: usize| &t[(y * POS_SIDE + x) * c..(y * POS_SIDE + x + 1) * c];
        let mut out = vec![0f32; gh * gw * c];
        for (py, &(y0, y1, ly)) in ay.iter().enumerate() {
            for (px, &(x0, x1, lx)) in ax.iter().enumerate() {
                let o = &mut out[(py * gw + px) * c..(py * gw + px + 1) * c];
                let (a, b, cc, d) = (row(y0, x0), row(y0, x1), row(y1, x0), row(y1, x1));
                let (wa, wb, wc, wd) = ((1.0 - ly) * (1.0 - lx), (1.0 - ly) * lx, ly * (1.0 - lx), ly * lx);
                for k in 0..c {
                    o[k] = wa * a[k] + wb * b[k] + wc * cc[k] + wd * d[k];
                }
            }
        }
        out
    }

    /// Признаки картинки `[N, 3072]` BF16 (строки по порядку токенов 120120).
    pub fn encode(&self, img: &Prepared) -> R<Tensor> {
        let (gh, gw, c, dev) = (img.gh, img.gw, self.hidden, self.device);
        let n = gh * gw;
        let px = Tensor::from_vec(img.patches.clone(), vec![n, 3 * PATCH * PATCH], dev).and_then(|t| t.to_dtype(DT)).map_err(fe)?;
        let pos = Tensor::from_vec(self.pos_embed(gh, gw), vec![n, c], dev).and_then(|t| t.to_dtype(DT)).map_err(fe)?;
        let mut x = self.patch.fwd(&px)?.add(&pos).map_err(fe)?;
        let (nh, hd) = (self.heads, c / self.heads);
        let scale = 1.0 / (hd as f32).sqrt();
        for b in &self.blocks {
            let h = b.ln1.fwd(&x)?;
            let qkv = b.qkv.fwd(&h)?.reshape(vec![n, 3, nh, hd]).map_err(fe)?;
            let take = |i: usize| -> R<Tensor> {
                let t = kern::contig(&qkv.narrow(1, i, 1).map_err(fe)?)?.reshape(vec![n, nh, hd]).map_err(fe)?;
                kern::contig(&t.permute(vec![1, 0, 2]).map_err(fe)?)
            };
            let (q, k, v) = (take(0)?, take(1)?, take(2)?);
            let mut parts = Vec::with_capacity(n.div_ceil(Q_CHUNK));
            let mut off = 0;
            while off < n {
                let len = Q_CHUNK.min(n - off);
                let qc = kern::contig(&q.narrow(1, off, len).map_err(fe)?)?;
                parts.push(scaled_dot_attention(&qc, &k, &v, scale, None).map_err(fe)?);
                off += len;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let a = Tensor::cat(&refs, 1).map_err(fe)?;
            let a = kern::contig(&a.permute(vec![1, 0, 2]).map_err(fe)?)?.reshape(vec![n, c]).map_err(fe)?;
            x = x.add(&b.o.fwd(&a)?).map_err(fe)?;
            let h = b.ln2.fwd(&x)?;
            let h = b.fc1.fwd(&h)?.gelu_exact().map_err(fe)?;
            x = x.add(&b.fc2.fwd(&h)?).map_err(fe)?;
        }
        // perceive: RMSNorm → 2×2 (канал старший, затем dy, dx) → GELU → 1×1 → newline.
        let y = kern::rms(&kern::contig(&x)?, Some(&self.norm), 1e-5, DT)?;
        let (rows, cols) = (gh / MERGE, gw / MERGE);
        let mut idx = Vec::with_capacity(rows * cols * 4);
        for i in 0..rows {
            for j in 0..cols {
                for dy in 0..MERGE {
                    for dx in 0..MERGE {
                        idx.push(((MERGE * i + dy) * gw + MERGE * j + dx) as u32);
                    }
                }
            }
        }
        let idx = Tensor::from_vec(idx, vec![rows * cols * 4], dev).map_err(fe)?;
        let g = y.index_select(0, &idx).and_then(|t| t.reshape(vec![rows * cols, 4, c])).map_err(fe)?;
        let g = kern::contig(&g.transpose(1, 2).map_err(fe)?)?.reshape(vec![rows * cols, 4 * c]).map_err(fe)?;
        let z = self.proj0.fwd(&g)?.gelu_exact().map_err(fe)?;
        let u = self.proj2.fwd(&z)?; // [rows·cols, 3072]
        let mut pieces: Vec<Tensor> = Vec::with_capacity(rows * 2);
        for i in 0..rows {
            pieces.push(kern::contig(&u.narrow(0, i * cols, cols).map_err(fe)?)?);
            pieces.push(self.newline.clone());
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&refs, 0).map_err(fe)
    }
}
