//! Упаковка HF-чекпойнта Inkling-2 в `.syn`.
//!
//! Эксперты NVFP4 (modelopt) переносятся бит в бит: полубайты и коды масштабов
//! как есть (строки w13 разведены в `[gate; up]`, масштабы — в тайловой
//! раскладке движка), общий множитель — тензор `<имя>_alpha` F32 `[E]`.
//! Эксперты MXFP8 — как есть либо реквант в NVFP4. Плотные матрицы — BF16 либо
//! NVFP4/MXFP8. Реквант в NVFP4 идёт с общим множителем `g = amax/(6·448)` на
//! матрицу: блочные масштабы `W/g` ложатся в нормальную зону E4M3, а не в
//! субнормали (см. `docs/INKLING2-MINI-PORT.md`).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::quant_layout::{MANIFEST_NAME, PACKED_SUFFIX, SCALES_SUFFIX};
use synaptix_bundle::{BundleBuilder, FileTag, QuantEntry, QuantManifest, StDtype, StreamTensor, TensorStream};
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_kernels_cpu::quant::dequant::{nvfp4_scale_bytes, nvfp4_scale_index, nvfp4_sf_inner_dim};
use synaptix_kernels_cpu::quant::encode::{quantize_mxfp8_rows, quantize_nvfp4_rows};

use crate::config::Inkling2Config;
use crate::loader::LLM;

/// Формат упаковки группы весов.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    /// Как в оригинале (BF16 у плотных, NVFP4/MXFP8 у экспертов).
    Keep,
    Nvfp4,
    Mxfp8,
    /// int8 с масштабом на строку (`DType::Q8Row`).
    Q8row,
}

impl Fmt {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "keep" | "bf16" | "none" => Ok(Self::Keep),
            "nvfp4" => Ok(Self::Nvfp4),
            "mxfp8" => Ok(Self::Mxfp8),
            "q8row" => Ok(Self::Q8row),
            o => Err(format!("формат {o}: keep | nvfp4 | mxfp8 | q8row")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PackOptions {
    /// Эксперты, что в оригинале MXFP8 (слои 20–23).
    pub mxfp8_experts: Fmt,
    /// Проекции внимания (q, k, v, o, r, gate).
    pub attn: Fmt,
    /// Голова (unembed).
    pub head: Fmt,
}

const E4M3_MAX: f32 = 448.0;

/// Что пишется в тензор плана.
#[derive(Clone)]
enum Item {
    /// Тензор оригинала как есть.
    Copy { src: String },
    /// Часть кванта стопки экспертов (`packed`/`scales`/`alpha`).
    Expert { src: String, part: Part, interleaved: bool, target: Fmt },
    /// Часть кванта плотной матрицы.
    Dense { src: String, part: Part, target: Fmt },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Packed,
    Scales,
    Alpha,
}

struct Packed {
    packed: Vec<u8>,
    scales: Vec<u8>,
    alpha: Vec<f32>,
}

pub struct Inkling2Stream {
    loader: Arc<SafetensorsLoader>,
    plan: Vec<StreamTensor>,
    items: Vec<Item>,
    cache: HashMap<String, Arc<Packed>>,
}

fn e4m3(b: u8) -> f32 {
    synaptix_core::quant::nvfp4_ct::e4m3_to_f32(b)
}

/// NVFP4 с общим множителем: `[n, k]` F32 → полубайты, масштабы (тайлы), `g`.
fn nvfp4_alpha(w: &[f32], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>, f32), String> {
    let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    let g = if amax > 0.0 { amax / (6.0 * E4M3_MAX) } else { 1.0 };
    let scaled: Vec<f32> = w.iter().map(|v| v / g).collect();
    let (p, s) = quantize_nvfp4_rows(&scaled, n, k).map_err(|e| e.to_string())?;
    Ok((p, s, g))
}

impl Inkling2Stream {
    pub fn new(dir: &Path, cfg: &Inkling2Config, opts: PackOptions, manifest: &mut QuantManifest) -> Result<Self, String> {
        let shards: Vec<_> = scan_shards(dir)
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter(|p| p.file_name().is_some_and(|n| n != "mtp.safetensors"))
            .collect();
        let loader = Arc::new(SafetensorsLoader::open_sharded(&shards).map_err(|e| e.to_string())?);
        let mut names: Vec<(String, synaptix_core::dtype::DType, Vec<usize>)> =
            loader.infos().map(|(n, d, s)| (n.to_string(), d, s.to_vec())).collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut plan = Vec::new();
        let mut items = Vec::new();
        let st = |d: synaptix_core::dtype::DType| -> Result<StDtype, String> {
            use synaptix_core::dtype::DType as D;
            Ok(match d {
                D::F32 => StDtype::F32,
                D::BF16 => StDtype::BF16,
                D::F16 => StDtype::F16,
                D::U8 => StDtype::U8,
                o => return Err(format!("тип {o:?} не ждали")),
            })
        };
        let attn_q = ["wq_du.weight", "wk_dv.weight", "wv_dv.weight", "wo_ud.weight", "wr_du.weight", "gate_attn.weight"];
        for (name, dt, shape) in &names {
            if name.starts_with("model.visual.") {
                // Vision — отдельным компонентом позже; текстовой модели не нужен.
                continue;
            }
            if name.contains(".mlp.experts.") {
                // Стопки собираются по w13_weight/w2_weight; спутники (scale/amax/…) уходят в них.
                let base = if let Some(b) = name.strip_suffix(".w13_weight") {
                    Some((b, "gate_up", true))
                } else {
                    name.strip_suffix(".w2_weight").map(|b| (b, "down", false))
                };
                let Some((base, short, interleaved)) = base else { continue };
                let nvfp4 = loader.tensor_info(&format!("{name}_amax")).is_some();
                let (e, n) = (shape[0], shape[1]);
                let k = if nvfp4 { shape[2] * 2 } else { shape[2] };
                let target = if nvfp4 { Fmt::Nvfp4 } else if opts.mxfp8_experts == Fmt::Nvfp4 { Fmt::Nvfp4 } else { Fmt::Mxfp8 };
                let out = format!("{base}.{short}");
                let fmt_key = if target == Fmt::Nvfp4 { "nvfp4" } else { "mxfp8" };
                manifest.tensors.insert(out.clone(), QuantEntry { format: fmt_key.into(), shape: vec![e, n, k] });
                let (pk, sc) = if target == Fmt::Nvfp4 { (k / 2, nvfp4_scale_bytes(n, k)) } else { (k, n * k / 32) };
                plan.push(StreamTensor { name: format!("{out}{PACKED_SUFFIX}"), dtype: StDtype::U8, shape: vec![e, n, pk] });
                items.push(Item::Expert { src: name.clone(), part: Part::Packed, interleaved, target });
                plan.push(StreamTensor { name: format!("{out}{SCALES_SUFFIX}"), dtype: StDtype::U8, shape: vec![e * sc] });
                items.push(Item::Expert { src: name.clone(), part: Part::Scales, interleaved, target });
                plan.push(StreamTensor { name: format!("{out}_alpha"), dtype: StDtype::F32, shape: vec![e] });
                items.push(Item::Expert { src: name.clone(), part: Part::Alpha, interleaved, target });
                continue;
            }
            let quant = if attn_q.iter().any(|s| name.ends_with(&format!(".attn.{s}"))) {
                opts.attn
            } else if name == &format!("{LLM}.unembed.weight") {
                opts.head
            } else {
                Fmt::Keep
            };
            let fits = shape.len() == 2 && shape[0] % 64 == 0 && shape[1] % 64 == 0;
            if quant != Fmt::Keep && fits {
                let (n, k) = (shape[0], shape[1]);
                let fmt_key = match quant { Fmt::Nvfp4 => "nvfp4", Fmt::Q8row => "q8row", _ => "mxfp8" };
                manifest.tensors.insert(name.clone(), QuantEntry { format: fmt_key.into(), shape: vec![n, k] });
                let (pk, sc) = match quant { Fmt::Nvfp4 => (k / 2, nvfp4_scale_bytes(n, k)), Fmt::Q8row => (k, n * 4), _ => (k, n * k / 32) };
                plan.push(StreamTensor { name: format!("{name}{PACKED_SUFFIX}"), dtype: StDtype::U8, shape: vec![n, pk] });
                items.push(Item::Dense { src: name.clone(), part: Part::Packed, target: quant });
                plan.push(StreamTensor { name: format!("{name}{SCALES_SUFFIX}"), dtype: StDtype::U8, shape: vec![sc] });
                items.push(Item::Dense { src: name.clone(), part: Part::Scales, target: quant });
                plan.push(StreamTensor { name: format!("{name}_alpha"), dtype: StDtype::F32, shape: vec![1] });
                items.push(Item::Dense { src: name.clone(), part: Part::Alpha, target: quant });
                continue;
            }
            plan.push(StreamTensor { name: name.clone(), dtype: st(*dt)?, shape: shape.clone() });
            items.push(Item::Copy { src: name.clone() });
        }
        let _ = cfg;
        Ok(Self { loader, plan, items, cache: HashMap::new() })
    }

    fn raw(&self, name: &str) -> Result<&[u8], String> {
        self.loader.raw_bytes(name).map(|(b, _, _)| b).ok_or_else(|| format!("нет {name}"))
    }

    fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let (b, dt, _) = self.loader.raw_bytes(name).ok_or_else(|| format!("нет {name}"))?;
        use synaptix_core::dtype::DType as D;
        Ok(match dt {
            D::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            D::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            o => return Err(format!("{name}: {o:?} не число")),
        })
    }

    /// Квант стопки экспертов целиком (кэшируется до записи всех трёх частей).
    fn expert(&mut self, src: &str, interleaved: bool, target: Fmt) -> Result<Arc<Packed>, String> {
        if let Some(p) = self.cache.get(src) {
            return Ok(p.clone());
        }
        let (_, _, wsh) = self.loader.raw_bytes(src).ok_or_else(|| format!("нет {src}"))?;
        let wsh = wsh.to_vec();
        let w = self.raw(src)?;
        let s = self.raw(&format!("{src}_scale"))?;
        let nvfp4 = self.loader.tensor_info(&format!("{src}_amax")).is_some();
        let (e, n) = (wsh[0], wsh[1]);
        let k = if nvfp4 { wsh[2] * 2 } else { wsh[2] };
        let order: Vec<usize> = if interleaved {
            (0..n / 2).map(|r| 2 * r).chain((0..n / 2).map(|r| 2 * r + 1)).collect()
        } else {
            (0..n).collect()
        };
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut alpha = Vec::with_capacity(e);
        if nvfp4 {
            let amax = self.f32s(&format!("{src}_amax"))?;
            let den = self.f32s(&format!("{src}_outer_scale_denominator"))?[0];
            let (rb, sb, sf) = (k / 2, k / 16, nvfp4_sf_inner_dim(k));
            for ex in 0..e {
                let wb = &w[ex * n * rb..(ex + 1) * n * rb];
                let sx = &s[ex * n * sb..(ex + 1) * n * sb];
                for &src_r in &order {
                    packed.extend_from_slice(&wb[src_r * rb..(src_r + 1) * rb]);
                }
                let mut sc = vec![0u8; nvfp4_scale_bytes(n, k)];
                for (dst, &src_r) in order.iter().enumerate() {
                    for b in 0..sb {
                        sc[nvfp4_scale_index(dst, b, sf)] = sx[src_r * sb + b];
                    }
                }
                scales.extend_from_slice(&sc);
                alpha.push(amax[ex] / den);
            }
        } else {
            let sb = k / 32;
            for ex in 0..e {
                let wb = &w[ex * n * k..(ex + 1) * n * k];
                let sx = &s[ex * n * sb..(ex + 1) * n * sb];
                if target == Fmt::Mxfp8 {
                    for &src_r in &order {
                        packed.extend_from_slice(&wb[src_r * k..(src_r + 1) * k]);
                    }
                    for &src_r in &order {
                        scales.extend_from_slice(&sx[src_r * sb..(src_r + 1) * sb]);
                    }
                    alpha.push(1.0);
                } else {
                    let mut f = vec![0f32; n * k];
                    for (dst, &src_r) in order.iter().enumerate() {
                        for j in 0..k {
                            let sc = f32::from_bits((sx[src_r * sb + j / 32] as u32) << 23);
                            f[dst * k + j] = e4m3(wb[src_r * k + j]) * sc;
                        }
                    }
                    let (p, sc, g) = nvfp4_alpha(&f, n, k)?;
                    packed.extend_from_slice(&p);
                    scales.extend_from_slice(&sc);
                    alpha.push(g);
                }
            }
        }
        let p = Arc::new(Packed { packed, scales, alpha });
        self.cache.insert(src.to_string(), p.clone());
        Ok(p)
    }

    fn dense(&mut self, src: &str, target: Fmt) -> Result<Arc<Packed>, String> {
        if let Some(p) = self.cache.get(src) {
            return Ok(p.clone());
        }
        let (_, _, sh) = self.loader.raw_bytes(src).ok_or_else(|| format!("нет {src}"))?;
        let (n, k) = (sh[0], sh[1]);
        let w = self.f32s(src)?;
        let p = if target == Fmt::Nvfp4 {
            let (p, s, g) = nvfp4_alpha(&w, n, k)?;
            Packed { packed: p, scales: s, alpha: vec![g] }
        } else if target == Fmt::Q8row {
            let (p, sc) = synaptix_core::quant::q8row::quantize(&w, n, k, None);
            Packed { packed: p, scales: sc.iter().flat_map(|v| v.to_le_bytes()).collect(), alpha: vec![1.0] }
        } else {
            let (p, s) = quantize_mxfp8_rows(&w, n, k).map_err(|e| e.to_string())?;
            Packed { packed: p, scales: s, alpha: vec![1.0] }
        };
        let p = Arc::new(p);
        self.cache.insert(src.to_string(), p.clone());
        Ok(p)
    }
}

fn put(w: &mut dyn Write, p: &Packed, part: Part) -> synaptix_bundle::Result<()> {
    match part {
        Part::Packed => w.write_all(&p.packed)?,
        Part::Scales => w.write_all(&p.scales)?,
        Part::Alpha => {
            for a in &p.alpha {
                w.write_all(&a.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

impl TensorStream for Inkling2Stream {
    fn plan(&self) -> &[StreamTensor] {
        &self.plan
    }

    fn write_tensor(&mut self, index: usize, w: &mut dyn Write) -> synaptix_bundle::Result<()> {
        let berr = |e: String| synaptix_bundle::Error::Io(std::io::Error::other(e));
        match self.items[index].clone() {
            Item::Copy { src } => {
                let b = self.raw(&src).map_err(berr)?;
                w.write_all(b)?;
            }
            Item::Expert { src, part, interleaved, target } => {
                let p = self.expert(&src, interleaved, target).map_err(berr)?;
                put(w, &p, part)?;
                if part == Part::Alpha {
                    self.cache.remove(&src);
                }
            }
            Item::Dense { src, part, target } => {
                let p = self.dense(&src, target).map_err(berr)?;
                put(w, &p, part)?;
                if part == Part::Alpha {
                    self.cache.remove(&src);
                }
            }
        }
        Ok(())
    }
}

/// Собрать бандл из HF-каталога.
pub fn pack(dir: &Path, out: &Path, opts: PackOptions) -> Result<(), String> {
    let cfg = Inkling2Config::from_bytes(&std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let mut manifest = QuantManifest::new();
    let stream = Inkling2Stream::new(dir, &cfg, opts, &mut manifest)?;
    let id = dir.file_name().and_then(|s| s.to_str()).unwrap_or("inkling2").to_string();
    let mut b = BundleBuilder::new(id, "1.0.0")
        .arch(cfg.model_type.clone())
        .purpose("text-generation")
        .require_capability(synaptix_bundle::CAP_QUANT_WEIGHTS)
        .component("main", String::new())
        .add_tensor_stream("main", Box::new(stream));
    for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json"] {
        if let Ok(bytes) = std::fs::read(dir.join(f)) {
            b = b.add_file_bytes(f, bytes, FileTag::Inference).map_err(|e| e.to_string())?;
        }
    }
    let m = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    b = b.add_file_bytes(MANIFEST_NAME, m, FileTag::Inference).map_err(|e| e.to_string())?;
    b.write(out).map_err(|e| format!("{}: {e}", out.display()))
}
