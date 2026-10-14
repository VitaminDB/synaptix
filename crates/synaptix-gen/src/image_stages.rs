//! Картинка по стадиям — для нодовых движков (граф службы synaptixd):
//! модель → условие (описание) → референсы → латент (пустой или картинка) →
//! сэмплер → декодер. Каждая стадия грузит свои веса и отпускает их; между
//! стадиями живут только конфиги модели, условие и латенты. Семейство
//! (SDXL, FLUX.1, FLUX.2, Qwen-Image, Qwen-Image 2.1) — по `model_index.json`.
//! [`crate::image::generate`] — те же стадии подряд.

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::image::{detect, image_size, load_rgb, quant_dtype, Family};

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

enum Inner {
    Sdxl(synaptix_image_sdxl::SdxlCheckpoint),
    Flux(synaptix_image_flux::FluxModel),
    Flux2(synaptix_image_flux2::Flux2Model),
    Qwen(synaptix_image_qwen::QwenImageModel),
    Qwen21(synaptix_image_qwen21::QwenImage21Model),
}

/// Открытая модель: конфиги и токенайзеры, веса — по стадиям.
pub struct ImageModel {
    pub family: Family,
    pub path: PathBuf,
    quant: DType,
    /// Qwen-Image 2.1: разрешение условия (стороны референсов), px.
    resolution: usize,
    /// Трансформер квантован (запросом или в самом файле) — внимание в FP8.
    fp8_attention: bool,
    inner: Inner,
}

/// Картинки в памяти (кэш стадий переживает файлы прогона): RGB `[3,H,W]`
/// или RGBA для Qwen-Image 2.1.
#[derive(Clone)]
pub enum Pics {
    Rgb(Vec<Tensor>),
    Rgba(Vec<synaptix_image_qwen21::RgbaImage>),
}

impl Pics {
    pub fn len(&self) -> usize {
        match self {
            Pics::Rgb(v) => v.len(),
            Pics::Rgba(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn sizes(&self) -> Vec<(usize, usize)> {
        match self {
            Pics::Rgb(v) => v.iter().map(image_size).collect(),
            Pics::Rgba(v) => v.iter().map(|i| (i.width, i.height)).collect(),
        }
    }

    fn rgb(&self) -> R<&[Tensor]> {
        match self {
            Pics::Rgb(v) => Ok(v),
            Pics::Rgba(_) => Err("картинки другой модели".into()),
        }
    }

    fn rgba(&self) -> R<&[synaptix_image_qwen21::RgbaImage]> {
        match self {
            Pics::Rgba(v) => Ok(v),
            Pics::Rgb(_) => Err("картинки другой модели".into()),
        }
    }

    fn join(parts: &[&Pics]) -> R<Pics> {
        match parts.first() {
            Some(Pics::Rgba(_)) => Ok(Pics::Rgba(parts.iter().map(|p| p.rgba().map(<[_]>::to_vec)).collect::<R<Vec<_>>>()?.concat())),
            _ => Ok(Pics::Rgb(parts.iter().map(|p| p.rgb().map(<[_]>::to_vec)).collect::<R<Vec<_>>>()?.concat())),
        }
    }
}

/// Условие: закодированное описание (и картинки, которые видел энкодер Qwen).
pub struct Cond {
    prompt: String,
    pics: Pics,
    inner: CondInner,
}

enum CondInner {
    Sdxl(synaptix_image_sdxl::SdxlConditioning),
    Flux { cond: synaptix_image_flux::FluxConditioning, seq: usize },
    Flux2(synaptix_image_flux2::Flux2Conditioning),
    Qwen(synaptix_image_qwen::QwenConditioning),
    Qwen21(synaptix_image_qwen21::Qwen21Conditioning),
}

/// Референсы для правки (FLUX.2, Qwen-Image Edit, Qwen-Image 2.1).
pub struct Refs {
    /// Исходные картинки — их видит и энкодер Qwen-Image.
    pub pics: Pics,
    /// Размеры исходных картинок — для размера по умолчанию.
    pub sizes: Vec<(usize, usize)>,
    inner: RefsInner,
}

enum RefsInner {
    Flux2(synaptix_image_flux2::Flux2References),
    Qwen(synaptix_image_qwen::QwenReferences),
    Qwen21(synaptix_image_qwen21::Qwen21References),
}

impl Refs {
    /// Склеить референсы цепочки (по порядку).
    pub fn join(parts: &[&Refs]) -> R<Refs> {
        let sizes = parts.iter().flat_map(|r| r.sizes.iter().copied()).collect();
        let pics = Pics::join(&parts.iter().map(|r| &r.pics).collect::<Vec<_>>())?;
        let inner = match parts.first().map(|r| &r.inner) {
            None => return Err("нет референсов".into()),
            Some(RefsInner::Flux2(_)) => RefsInner::Flux2(synaptix_image_flux2::Flux2References::join(&parts.iter().map(|r| match &r.inner {
                RefsInner::Flux2(x) => Ok(x),
                _ => Err("референсы разных моделей".to_string()),
            }).collect::<R<Vec<_>>>()?).map_err(err)?),
            Some(RefsInner::Qwen(_)) => RefsInner::Qwen(synaptix_image_qwen::QwenReferences::join(&parts.iter().map(|r| match &r.inner {
                RefsInner::Qwen(x) => Ok(x),
                _ => Err("референсы разных моделей".to_string()),
            }).collect::<R<Vec<_>>>()?).map_err(err)?),
            Some(RefsInner::Qwen21(_)) => RefsInner::Qwen21(synaptix_image_qwen21::Qwen21References::join(&parts.iter().map(|r| match &r.inner {
                RefsInner::Qwen21(x) => Ok(x),
                _ => Err("референсы разных моделей".to_string()),
            }).collect::<R<Vec<_>>>()?).map_err(err)?),
        };
        Ok(Refs { pics, sizes, inner })
    }
}

/// Латент: размер будущей картинки, закодированная картинка (img2img) или
/// результат сэмплера.
pub enum Latent {
    /// 0 — по модели (или по референсу).
    Size(usize, usize),
    Image { latent: Tensor, width: usize, height: usize },
    Sampled { latent: Tensor, width: usize, height: usize, steps: usize, label: String },
}

impl Latent {
    pub fn size(&self) -> (usize, usize) {
        match self {
            Latent::Size(w, h) => (*w, *h),
            Latent::Image { width, height, .. } | Latent::Sampled { width, height, .. } => (*width, *height),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Latent::Size(0, 0) => "размер по модели".into(),
            Latent::Size(w, h) => format!("{w}×{h}"),
            Latent::Image { width, height, .. } => format!("картинка {width}×{height}"),
            Latent::Sampled { label, .. } => label.clone(),
        }
    }
}

/// Параметры сэмплера; `None` — умолчание модели.
#[derive(Debug, Clone, Default)]
pub struct SampleOpts {
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Сила img2img, 0..1.
    pub denoise: Option<f32>,
}

/// Открыть модель (`quant`: `nvfp4`, `mxfp8`, пусто — плотная).
pub fn open(path: &Path, device: Device, quant: Option<&str>) -> R<ImageModel> {
    if !path.exists() {
        return Err(format!("модели нет: {}", path.display()));
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let quant = quant_dtype(quant)?;
    let family = detect(path)?;
    let inner = match family {
        Family::Sdxl => Inner::Sdxl(synaptix_image_sdxl::SdxlCheckpoint::open(path, device).map_err(err)?),
        Family::Flux => {
            use synaptix_image_flux::{FluxModel, OffloadMode};
            let compute = if !device.is_cuda() {
                DType::F32
            } else if quant.is_quantized() {
                DType::F16
            } else {
                DType::BF16
            };
            let quant = if device.is_cuda() { quant } else { compute };
            Inner::Flux(FluxModel::open(path, device, compute, quant).map_err(err)?.with_offload(OffloadMode::Auto))
        }
        Family::Flux2 => {
            use synaptix_image_flux2::{Flux2Model, MemoryMode};
            Inner::Flux2(Flux2Model::open(path, device, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto))
        }
        Family::QwenImage => {
            use synaptix_image_qwen::{MemoryMode, QwenImageModel};
            Inner::Qwen(QwenImageModel::open(path, device, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto))
        }
        Family::QwenImage21 => {
            use synaptix_image_qwen21::{MemoryMode, QwenImage21Model};
            Inner::Qwen21(QwenImage21Model::open(path, device, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto))
        }
    };
    let fp8_attention = quant.is_quantized() || crate::video::stored_format(path).is_some_and(|q| q.is_quantized());
    Ok(ImageModel { family, path: path.to_path_buf(), quant, resolution: synaptix_image_qwen21::model::DEFAULT_RESOLUTION, fp8_attention, inner })
}

fn size_or(w: usize, h: usize, fallback: (usize, usize), snap: impl Fn(usize) -> usize) -> (usize, usize) {
    (snap(if w > 0 { w } else { fallback.0 }), snap(if h > 0 { h } else { fallback.1 }))
}

fn load_rgba(p: &Path) -> R<synaptix_image_qwen21::RgbaImage> {
    let t = synaptix_io::image::png::load_image_rgba(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
    synaptix_image_qwen21::RgbaImage::from_tensor(&t).map_err(err)
}

impl ImageModel {
    /// Qwen-Image 2.1: разрешение условия (0 — по модели, 1024).
    pub fn with_resolution(mut self, px: usize) -> Self {
        if px > 0 {
            self.resolution = px;
        }
        self
    }

    /// Название семейства (вариант — для FLUX.2 и Qwen-Image).
    pub fn name(&self) -> String {
        match &self.inner {
            Inner::Sdxl(_) => "SDXL".into(),
            Inner::Flux(_) => "FLUX.1".into(),
            Inner::Flux2(m) => format!("FLUX.2 {:?}", m.variant()),
            Inner::Qwen(m) => m.variant().label().into(),
            Inner::Qwen21(_) => "Qwen-Image 2.1".into(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        self.fp8_attention
    }

    /// Картинки с диска в память — RGBA для Qwen-Image 2.1, иначе RGB.
    pub fn load(&self, images: &[PathBuf]) -> R<Pics> {
        Ok(match self.inner {
            Inner::Qwen21(_) => Pics::Rgba(images.iter().map(|p| load_rgba(p)).collect::<R<_>>()?),
            _ => Pics::Rgb(images.iter().map(|p| load_rgb(p)).collect::<R<_>>()?),
        })
    }

    /// Описание в условие. `images` — картинки, которые видит энкодер
    /// Qwen-Image (у других семейств не нужны); `seq_len` — токенов T5 у FLUX.1.
    pub fn encode(&self, prompt: &str, negative: Option<&str>, images: &[PathBuf], seq_len: Option<usize>) -> R<Cond> {
        let pics = self.load(images)?;
        self.encode_pics(prompt, negative, pics, seq_len)
    }

    /// То же с картинками в памяти (например, из референсов).
    pub fn encode_pics(&self, prompt: &str, negative: Option<&str>, pics: Pics, seq_len: Option<usize>) -> R<Cond> {
        let negative = negative.map(str::trim).filter(|n| !n.is_empty());
        let inner = match &self.inner {
            Inner::Sdxl(ck) => CondInner::Sdxl(ck.encode_prompt(prompt, negative.unwrap_or("")).map_err(err)?),
            Inner::Flux(m) => {
                let seq = seq_len.filter(|&s| s > 0).unwrap_or_else(|| m.default_max_seq_len());
                CondInner::Flux { cond: m.encode_prompt(prompt, seq).map_err(err)?, seq }
            }
            Inner::Flux2(m) => {
                let v = m.variant();
                CondInner::Flux2(m.encode_prompt(prompt, v.uses_cfg(v.default_guidance())).map_err(err)?)
            }
            Inner::Qwen(m) => {
                let v = m.variant();
                if pics.len() > v.max_images() {
                    return Err(format!("{} принимает картинок: {}, передано {}", v.label(), v.max_images(), pics.len()));
                }
                if v.is_edit() && pics.is_empty() {
                    return Err(format!("{} правит картинку — нужна исходная картинка", v.label()));
                }
                // Негатив Qwen-Image по умолчанию — пробел (CFG 4 по умолчанию).
                let neg = negative.unwrap_or(" ");
                CondInner::Qwen(m.encode_prompt(prompt, Some(neg), pics.rgb()?).map_err(err)?)
            }
            Inner::Qwen21(m) => CondInner::Qwen21(m.encode_prompt(prompt, negative, pics.rgba()?, self.resolution).map_err(err)?),
        };
        let pics = if matches!(self.family, Family::QwenImage | Family::QwenImage21) { pics } else { Pics::Rgb(Vec::new()) };
        Ok(Cond { prompt: prompt.to_string(), pics, inner })
    }

    /// Референсы для правки.
    pub fn references(&self, images: &[PathBuf]) -> R<Refs> {
        if images.is_empty() {
            return Err("нет картинок-референсов".into());
        }
        if matches!(self.inner, Inner::Flux2(_)) && images.len() > 10 {
            return Err(format!("FLUX.2 принимает до 10 референсов, передано {}", images.len()));
        }
        let pics = self.load(images)?;
        self.references_pics(pics)
    }

    /// Референсы из картинок в памяти.
    pub fn references_pics(&self, pics: Pics) -> R<Refs> {
        let sizes = pics.sizes();
        let inner = match &self.inner {
            Inner::Flux2(m) => {
                use synaptix_image_flux2::model::reference_size;
                if pics.len() > 10 {
                    return Err(format!("FLUX.2 принимает до 10 референсов, передано {}", pics.len()));
                }
                let mut fitted = Vec::with_capacity(pics.len());
                for img in pics.rgb()? {
                    let (w, h) = image_size(img);
                    let (rw, rh) = reference_size(w, h);
                    fitted.push(synaptix_io::image::fit_image(img, rw, rh, true).map_err(err)?);
                }
                RefsInner::Flux2(m.encode_references(&fitted).map_err(err)?)
            }
            Inner::Qwen(m) => RefsInner::Qwen(m.encode_references(pics.rgb()?).map_err(err)?),
            Inner::Qwen21(m) => RefsInner::Qwen21(m.encode_references(pics.rgba()?, self.resolution).map_err(err)?),
            _ => return Err(format!("{}: референсы не поддерживаются (есть у FLUX.2 и Qwen-Image)", self.name())),
        };
        Ok(Refs { pics, sizes, inner })
    }

    /// Исходная картинка для img2img → латент. Размер — `size` (0 — по
    /// картинке), `crop` — покрыть и обрезать, иначе растянуть.
    pub fn image_latent(&self, image: &Path, size: (usize, usize), crop: bool) -> R<Latent> {
        let img = load_rgb(image)?;
        let own = image_size(&img);
        let (width, height, latent) = match &self.inner {
            Inner::Sdxl(ck) => {
                let (w, h) = size_or(size.0, size.1, own, synaptix_image_sdxl::stages::snap_side);
                (w, h, ck.encode_image(&synaptix_io::image::fit_image(&img, w, h, crop).map_err(err)?).map_err(err)?)
            }
            Inner::Flux(m) => {
                let (w, h) = size_or(size.0, size.1, own, synaptix_image_flux::model::snap_side);
                (w, h, m.encode_image(&synaptix_io::image::fit_image(&img, w, h, crop).map_err(err)?).map_err(err)?)
            }
            Inner::Flux2(m) => {
                let (w, h) = size_or(size.0, size.1, own, synaptix_image_flux2::model::snap_side);
                (w, h, m.encode_image(&synaptix_io::image::fit_image(&img, w, h, crop).map_err(err)?).map_err(err)?)
            }
            _ => return Err(format!("{}: img2img не поддерживается — правка идёт через референсы", self.name())),
        };
        Ok(Latent::Image { latent, width, height })
    }

    /// Денойз. `latent` — размер (`Size`) или исходная картинка (`Image`);
    /// `refs` — правка (у Qwen-Image без них — картинки условия).
    /// `progress(шаг, всего) -> продолжать`.
    pub fn sample(&self, cond: &Cond, latent: Option<&Latent>, refs: Option<&Refs>, o: &SampleOpts, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Latent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.fp8_attention);
        let (rw, rh) = latent.map(Latent::size).unwrap_or((0, 0));
        let init = match latent {
            Some(Latent::Image { latent, .. }) => Some(latent),
            Some(Latent::Sampled { .. }) => return Err("на входе сэмплера — результат другого сэмплера; нужен пустой латент или картинка".into()),
            _ => None,
        };
        let steps_or = |d: usize| o.steps.filter(|&s| s > 0).unwrap_or(d);
        if let (Some(r), Inner::Sdxl(_) | Inner::Flux(_)) = (refs, &self.inner) {
            let _ = r;
            return Err(format!("{}: референсы не поддерживаются", self.name()));
        }
        let (latent, width, height, steps, label) = match (&self.inner, &cond.inner) {
            (Inner::Sdxl(ck), CondInner::Sdxl(c)) => {
                use synaptix_image_sdxl::stages::snap_side;
                let (width, height) = size_or(rw, rh, (1024, 1024), snap_side);
                let steps = steps_or(30);
                let guidance = o.guidance.unwrap_or(5.0);
                let denoise = if init.is_some() { o.denoise.unwrap_or(0.6) } else { 1.0 };
                let unet = ck.load_unet(self.quant).map_err(err)?;
                let p = synaptix_image_sdxl::SdxlSampleParams { width, height, steps, guidance, seed: o.seed, denoise };
                let lat = ck.sample(&unet, c, init, &p, &mut |i, n| progress(i, n)).map_err(err)?;
                (lat, width, height, steps, format!("SDXL {width}×{height}, {steps} шагов, cfg {guidance}"))
            }
            (Inner::Flux(m), CondInner::Flux { cond: c, seq }) => {
                use synaptix_image_flux::{model::snap_side, FluxModel, SampleParams};
                let (width, height) = size_or(rw, rh, (1024, 1024), snap_side);
                let steps = steps_or(if m.guidance_distilled() { 28 } else { 4 });
                let guidance = o.guidance.unwrap_or(3.5);
                let denoise = if init.is_some() { o.denoise.unwrap_or(0.75) } else { 1.0 };
                let t = m.load_transformer(FluxModel::tokens_for(width, height, *seq)).map_err(err)?;
                let p = SampleParams { width, height, steps, guidance, seed: o.seed, denoise };
                let lat = m.sample(&t, c, init, &p, &mut |i, n| progress(i, n)).map_err(err)?;
                drop(t);
                (lat, width, height, steps, format!("FLUX.1 {width}×{height}, {steps} шагов, guidance {guidance}"))
            }
            (Inner::Flux2(m), CondInner::Flux2(c)) => {
                use synaptix_image_flux2::{model::snap_side, Flux2Model, SampleParams};
                let variant = m.variant();
                let r = match refs.map(|r| &r.inner) {
                    Some(RefsInner::Flux2(r)) => Some(r),
                    Some(_) => return Err("референсы другой модели".into()),
                    None => None,
                };
                let fallback = init.map(|_| (rw, rh)).or(refs.and_then(|r| r.sizes.first().map(|&(w, h)| synaptix_image_flux2::model::reference_size(w, h)))).unwrap_or((1024, 1024));
                let (width, height) = size_or(rw, rh, fallback, snap_side);
                let steps = steps_or(variant.default_steps());
                let guidance = o.guidance.unwrap_or(variant.default_guidance());
                let denoise = if init.is_some() { o.denoise.unwrap_or(0.75) } else { 1.0 };
                // CFG у klein base: условие без негатива — докодировать.
                let recoded;
                let c = if variant.uses_cfg(guidance) && c.negative.is_none() {
                    recoded = m.encode_prompt(&cond.prompt, true).map_err(err)?;
                    &recoded
                } else {
                    c
                };
                let ref_tokens = r.map(|r| r.num_tokens()).unwrap_or(0);
                let t = m.load_transformer(Flux2Model::tokens_for(width, height, ref_tokens)).map_err(err)?;
                let p = SampleParams { width, height, steps, guidance, seed: o.seed, denoise };
                let lat = m.sample(&t, c, r, init, &p, &mut |i, n| progress(i, n)).map_err(err)?;
                drop(t);
                let n = refs.map(|r| r.sizes.len()).unwrap_or(0);
                (lat, width, height, steps, format!("FLUX.2 {variant:?} {width}×{height}, {steps} шагов, референсов {n}"))
            }
            (Inner::Qwen(m), CondInner::Qwen(c)) => {
                use synaptix_image_qwen::{model::snap_side, QwenImageModel, SampleParams};
                if init.is_some() {
                    return Err(format!("{}: img2img не поддерживается — правка идёт через референсы", self.name()));
                }
                let variant = m.variant();
                let own;
                let r = match refs.map(|r| &r.inner) {
                    Some(RefsInner::Qwen(r)) => Some(r),
                    Some(_) => return Err("референсы другой модели".into()),
                    None if !cond.pics.is_empty() => {
                        own = self.references_pics(cond.pics.clone())?;
                        match &own.inner {
                            RefsInner::Qwen(r) => Some(r),
                            _ => None,
                        }
                    }
                    None => None,
                };
                let sizes: Vec<(usize, usize)> = match refs {
                    Some(r) => r.sizes.clone(),
                    None => cond.pics.sizes(),
                };
                let (width, height) = size_or(rw, rh, m.default_size(&sizes), snap_side);
                let steps = steps_or(variant.default_steps());
                let cfg = o.guidance.unwrap_or(4.0);
                let ref_tokens = r.map(|r| r.num_tokens()).unwrap_or(0);
                let txt = c.embeds.dims()[1].max(c.negative.as_ref().map(|n| n.dims()[1]).unwrap_or(0));
                let t = m.load_transformer(QwenImageModel::tokens_for(width, height, ref_tokens, txt)).map_err(err)?;
                let p = SampleParams { width, height, steps, cfg, seed: o.seed };
                let lat = m.sample(&t, c, r, &p, &mut |i, n| progress(i, n)).map_err(err)?;
                drop(t);
                (lat, width, height, steps, format!("{} {width}×{height}, {steps} шагов", variant.label()))
            }
            (Inner::Qwen21(m), CondInner::Qwen21(c)) => {
                use synaptix_image_qwen21::model::{snap_side, DEFAULT_STEPS};
                use synaptix_image_qwen21::{QwenImage21Model, SampleParams};
                if init.is_some() {
                    return Err("Qwen-Image 2.1: img2img не поддерживается — правка идёт через референсы".into());
                }
                let own;
                let r = match refs.map(|r| &r.inner) {
                    Some(RefsInner::Qwen21(r)) => Some(r),
                    Some(_) => return Err("референсы другой модели".into()),
                    None if !cond.pics.is_empty() => {
                        own = self.references_pics(cond.pics.clone())?;
                        match &own.inner {
                            RefsInner::Qwen21(r) => Some(r),
                            _ => None,
                        }
                    }
                    None => None,
                };
                let sizes: Vec<(usize, usize)> = match refs {
                    Some(r) => r.sizes.clone(),
                    None => cond.pics.sizes(),
                };
                let (width, height) = size_or(rw, rh, QwenImage21Model::default_size(&sizes, self.resolution), snap_side);
                let steps = steps_or(DEFAULT_STEPS);
                let cfg = o.guidance.unwrap_or(1.0);
                let ref_tokens = r.map(|r| r.num_tokens()).unwrap_or(0);
                let txt = c.embeds.dims()[1].max(c.negative.as_ref().map(|n| n.0.dims()[1]).unwrap_or(0));
                let t = m.load_transformer(QwenImage21Model::tokens_for(width, height, ref_tokens, txt)).map_err(err)?;
                let p = SampleParams { width, height, steps, cfg, seed: o.seed, kv_cache: true };
                let lat = m.sample(&t, c, r, &p, &mut |i, n| progress(i, n)).map_err(err)?;
                drop(t);
                let n = sizes.len();
                (lat, width, height, steps, format!("Qwen-Image 2.1 {width}×{height}, {steps} шагов, референсов {n}"))
            }
            _ => return Err("условие закодировано другой моделью".into()),
        };
        Ok(Latent::Sampled { latent, width, height, steps, label })
    }

    /// Латент сэмплера → картинка `[C,H,W]` F32 0..1 (у Qwen-Image 2.1 —
    /// с альфой, если она есть).
    pub fn decode(&self, latent: &Latent) -> R<Tensor> {
        let Latent::Sampled { latent, .. } = latent else {
            return Err("декодеру нужен латент сэмплера".into());
        };
        match &self.inner {
            Inner::Sdxl(ck) => ck.decode(latent).map_err(err),
            Inner::Flux(m) => m.decode(latent).map_err(err),
            Inner::Flux2(m) => m.decode(latent).map_err(err),
            Inner::Qwen(m) => m.decode(latent).map_err(err),
            Inner::Qwen21(m) => {
                let image = m.decode(latent).map_err(err)?;
                let transparent = synaptix_image_qwen21::RgbaImage::from_tensor(&image).map_err(err)?.has_transparency();
                if transparent {
                    Ok(image)
                } else {
                    image.narrow(0, 0, 3).map_err(err)?.contiguous().map_err(err)
                }
            }
        }
    }
}
