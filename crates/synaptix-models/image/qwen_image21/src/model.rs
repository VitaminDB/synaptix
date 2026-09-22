//! Стадии Qwen-Image 2.1 для нодовых пайплайнов: промпт (+ референсы) →
//! кондиционирование, референсы → токены латента (VAE), загрузка DiT,
//! денойз (с KV-кэшем префикса и, при желании, true CFG), VAE decode в RGBA.
//!
//! [`QwenImage21Model`] держит источник, конфиги и токенайзер; веса грузит
//! каждая стадия сама и сама же отпускает (энкодер живёт только на время
//! кодирования, VAE — на время кодирования/декода). DiT возвращается
//! вызывающему: его можно держать между прогонами.
//!
//! Референсы приводятся к размерам пайплайна здесь же: `~output_resolution²`
//! с пропорциями исходника, стороны кратны 32 — один и тот же ресайз
//! (Lanczos PIL) идёт и в башню зрения (композит на белом), и в VAE (RGBA).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_image_qwen::memory;
use synaptix_image_qwen::preprocess::calculate_dimensions;
use synaptix_image_qwen::scheduler::QwenScheduler;
use synaptix_image_qwen::source::{self, QwenImageSource};
use synaptix_ops::rng::Philox4x32;

use crate::config::{check_model_index, ProcessorConfig, Qwen21Config, Qwen21VaeConfig, Qwen3VlConfig, SchedulerConfig};
use crate::image::{vision_patches, RgbaImage};
use crate::text_encoder::{ImageEmbeds, PromptTokenizer, TextEncoder};
use crate::transformer::{build_rope, KvCache, Layout, Placement, QwenImage21Transformer};
use crate::vision::VisionTower;
use crate::{vae, QwenImage21Error};

/// Стороны выхода и референсов кратны 32 (VAE 16 × слот энкодера 2).
pub const SIDE_MULTIPLE: usize = 32;
/// `output_resolution` пайплайна по умолчанию.
pub const DEFAULT_RESOLUTION: usize = 1024;
/// Сколько референсов принимает модель (карточка: до 10).
pub const MAX_IMAGES: usize = 10;
/// Шагов по умолчанию (карточка модели).
pub const DEFAULT_STEPS: usize = 40;

/// Выход энкодера: `[1, S, 4096]` на CPU (BF16) без системной части, маска
/// слотов картинок и, для true CFG, то же для негативного промпта.
#[derive(Clone)]
pub struct Qwen21Conditioning {
    pub embeds: Tensor,
    /// `<|image_pad|>` в `embeds` (по слоту на 2×2 латентных токена).
    pub image_pad_mask: Vec<bool>,
    pub negative: Option<(Tensor, Vec<bool>)>,
    /// Сколько референсов видел энкодер и их размеры `(w, h)` после ресайза.
    pub sizes: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Qwen21Conditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qwen21Conditioning({:?}, neg={:?}, картинок {})",
            self.embeds.dims(),
            self.negative.as_ref().map(|n| n.0.dims().to_vec()),
            self.sizes.len()
        )
    }
}

impl Qwen21Conditioning {
    pub fn images(&self) -> usize {
        self.sizes.len()
    }
}

/// Референсы в латенте: нормированные токены подряд `[1, N, 64]` (F32, CPU)
/// и сетки каждой картинки в латентных токенах (h, w).
#[derive(Clone)]
pub struct Qwen21References {
    pub tokens: Tensor,
    pub grids: Vec<(usize, usize)>,
    /// Размеры в пикселях `(w, h)` после приведения.
    pub sizes: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Qwen21References {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qwen21References({} шт., {} токенов)", self.sizes.len(), self.num_tokens())
    }
}

impl Qwen21References {
    pub fn len(&self) -> usize {
        self.sizes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }
    pub fn num_tokens(&self) -> usize {
        self.grids.iter().map(|(h, w)| h * w).sum()
    }

    /// Склеить по порядку (цепочка нод).
    pub fn join(parts: &[&Qwen21References]) -> Result<Self, QwenImage21Error> {
        let parts: Vec<&Qwen21References> = parts.iter().copied().filter(|p| !p.is_empty()).collect();
        if parts.is_empty() {
            return Err(QwenImage21Error::Config("нет картинок".into()));
        }
        let t: Vec<&Tensor> = parts.iter().map(|p| &p.tokens).collect();
        Ok(Self {
            tokens: Tensor::cat(&t, 1)?,
            grids: parts.iter().flat_map(|p| p.grids.iter().copied()).collect(),
            sizes: parts.iter().flat_map(|p| p.sizes.iter().copied()).collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// `true_cfg_scale`; ≤ 1 или без негатива — без CFG (так и задумано у 2.1).
    pub cfg: f32,
    pub seed: u64,
    /// Кэшировать K/V текста и референсов после первого шага.
    pub kv_cache: bool,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self { width: DEFAULT_RESOLUTION, height: DEFAULT_RESOLUTION, steps: DEFAULT_STEPS, cfg: 1.0, seed: 0, kv_cache: true }
    }
}

/// Снап стороны к кратному [`SIDE_MULTIPLE`] (вниз, не меньше 32).
pub fn snap_side(v: usize) -> usize {
    (v / SIDE_MULTIPLE).max(1) * SIDE_MULTIPLE
}

/// Размер референса и выхода по умолчанию: `~res²` с пропорциями `w:h`,
/// стороны кратны 32 (`calculate_dimensions`). → `(w, h)`.
pub fn cond_size(w: usize, h: usize, res: usize) -> (usize, usize) {
    calculate_dimensions(res * res, w as f64 / h.max(1) as f64)
}

pub struct QwenImage21Model {
    source: QwenImageSource,
    device: Device,
    compute: DType,
    quant: DType,
    memory: Placement,
    config: Qwen21Config,
    te_config: Qwen3VlConfig,
    vae_config: Qwen21VaeConfig,
    proc_config: ProcessorConfig,
    sched_config: SchedulerConfig,
    tokenizer: PromptTokenizer,
}

impl QwenImage21Model {
    /// Открыть модель: `.syn`-бандл или каталог diffusers. Читает конфиги и
    /// токенайзер, веса не трогает. `compute` — активации DiT (BF16 на CUDA),
    /// `quant` — веса DiT: NVFP4/MXFP8 или плотные.
    pub fn open(path: impl AsRef<Path>, device: Device, compute: DType, quant: DType) -> Result<Self, QwenImage21Error> {
        let source = QwenImageSource::open(path)?;
        for c in source::COMPONENTS {
            if !source.has_component(c) {
                return Err(QwenImage21Error::Load(format!(
                    "{}: нет компонента `{c}` — нужна раскладка diffusers (transformer/, text_encoder/, vae/)",
                    source.path().display()
                )));
            }
        }
        check_model_index(source.read_opt("model_index.json").as_deref())?;
        let config = Qwen21Config::from_json(&source.read("transformer/config.json")?)?;
        let te_config = Qwen3VlConfig::from_json(&source.read("text_encoder/config.json")?)?;
        let vae_config = Qwen21VaeConfig::from_json(&source.read("vae/config.json")?)?;
        let proc_config = ProcessorConfig::from_json(source.read_opt("processor/preprocessor_config.json").as_deref())?;
        let sched_config = SchedulerConfig::from_json(source.read_opt("scheduler/scheduler_config.json").as_deref())?;
        if te_config.hidden != config.context_in_dim {
            return Err(QwenImage21Error::Config(format!(
                "энкодер {} ≠ context_in_dim {}",
                te_config.hidden, config.context_in_dim
            )));
        }
        if vae_config.z_dim != config.in_channels {
            return Err(QwenImage21Error::Config(format!(
                "VAE z_dim {} ≠ in_channels DiT {}",
                vae_config.z_dim, config.in_channels
            )));
        }
        let tok_json = source
            .read_opt("processor/tokenizer.json")
            .or_else(|| source.read_opt("tokenizer/tokenizer.json"))
            .ok_or_else(|| QwenImage21Error::Tokenizer("нет processor/tokenizer.json".into()))?;
        let tokenizer = PromptTokenizer::new(&tok_json).map_err(|e| QwenImage21Error::Tokenizer(e.to_string()))?;
        let compute = if device.is_cuda() { compute } else { DType::F32 };
        Ok(Self {
            source,
            device,
            compute,
            quant,
            memory: Placement::Auto,
            config,
            te_config,
            vae_config,
            proc_config,
            sched_config,
            tokenizer,
        })
    }

    pub fn with_memory(mut self, mode: Placement) -> Self {
        self.memory = mode;
        self
    }

    pub fn source(&self) -> &QwenImageSource {
        &self.source
    }

    pub fn config(&self) -> &Qwen21Config {
        &self.config
    }

    pub fn text_encoder_config(&self) -> &Qwen3VlConfig {
        &self.te_config
    }

    pub fn vae_config(&self) -> &Qwen21VaeConfig {
        &self.vae_config
    }

    pub fn processor_config(&self) -> &ProcessorConfig {
        &self.proc_config
    }

    pub fn scheduler_config(&self) -> &SchedulerConfig {
        &self.sched_config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tokenizer(&self) -> &PromptTokenizer {
        &self.tokenizer
    }

    fn encoder_dtype(&self) -> DType {
        if self.device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    /// Размер выхода по умолчанию (`(w, h)`): `~res²` с пропорциями
    /// последнего референса; без референсов — `res×res`.
    pub fn default_size(images: &[(usize, usize)], res: usize) -> (usize, usize) {
        match images.last() {
            Some(&(w, h)) => {
                let (w, h) = cond_size(w, h, res);
                (snap_side(w), snap_side(h))
            }
            None => (snap_side(res), snap_side(res)),
        }
    }

    /// Референс, приведённый к размеру пайплайна (Lanczos, RGBA).
    pub fn fit_reference(image: &RgbaImage, res: usize) -> RgbaImage {
        let (w, h) = cond_size(image.width, image.height, res);
        image.resize_lanczos(w, h)
    }

    /// Промпт (+ референсы) → кондиционирование. `negative` — промпт для true
    /// CFG (кодируется с теми же картинками); `res` — `output_resolution`.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative: Option<&str>,
        images: &[RgbaImage],
        res: usize,
    ) -> Result<Qwen21Conditioning, QwenImage21Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        if images.len() > MAX_IMAGES {
            return Err(QwenImage21Error::Config(format!(
                "Qwen-Image 2.1 принимает до {MAX_IMAGES} картинок, пришло {}",
                images.len()
            )));
        }
        let dev = self.device;
        let dt = self.encoder_dtype();
        memory::release_pools(dev);
        let w = self.source.weights(source::TEXT_ENCODER)?;
        let t0 = std::time::Instant::now();

        // Башня зрения — один раз на все картинки (негатив видит те же), в F32.
        let mut embeds: Vec<ImageEmbeds> = Vec::with_capacity(images.len());
        let mut sizes = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let tower = VisionTower::load(&w, &self.te_config.vision, dev, DType::F32)?;
            let m = self.te_config.vision.merge_size;
            for img in images {
                let fitted = Self::fit_reference(img, res);
                sizes.push((fitted.width, fitted.height));
                let p = vision_patches(&fitted, &self.te_config.vision, &self.proc_config)?;
                let grid = p.llm_grid(m);
                let (tokens, deepstack) = tower.forward(&p)?;
                embeds.push(ImageEmbeds { tokens, deepstack, grid });
            }
            drop(tower);
            memory::release_pools(dev);
        }
        let image_tokens: Vec<usize> = embeds.iter().map(|e| e.grid.0 * e.grid.1).collect();

        let enc = TextEncoder::build(w, self.te_config.clone(), dev, dt, 512 << 20)?;
        let image_token = self.tokenizer.image_token_id();
        let one = |p: &str| -> Result<(Tensor, Vec<bool>), QwenImage21Error> {
            let toks = self.tokenizer.encode(p, &image_tokens)?;
            let h = enc.encode(&toks.ids, &embeds)?; // [S, hidden]
            let s = h.dims()[0];
            let skip = toks.drop_idx;
            if s <= skip {
                return Err(QwenImage21Error::Config(format!("промпт короче системной части шаблона ({s} ≤ {skip})")));
            }
            let h = h.narrow(0, skip, s - skip)?.contiguous()?;
            let n = h.dims()[0];
            let mask: Vec<bool> = toks.ids[skip..].iter().map(|&t| t == image_token).collect();
            Ok((h.reshape((1, n, self.te_config.hidden))?.to_dtype(DType::BF16)?.to_device(Device::Cpu)?, mask))
        };
        let (embeds_t, mask) = one(prompt)?;
        let negative = match negative {
            Some(n) => Some(one(n)?),
            None => None,
        };
        eprintln!(
            "[qwen-image-2.1] промпт закодирован за {:.1} с ({} токенов, картинок {}, слоёв на карте {}/{})",
            t0.elapsed().as_secs_f64(),
            embeds_t.dims()[1],
            images.len(),
            enc.resident_layers(),
            self.te_config.num_layers
        );
        drop(enc);
        memory::release_pools(dev);
        Ok(Qwen21Conditioning { embeds: embeds_t, image_pad_mask: mask, negative, sizes })
    }

    /// Бюджет VRAM под активации VAE после освобождения пулов.
    fn vae_budget(&self) -> usize {
        if !self.device.is_cuda() {
            return usize::MAX;
        }
        memory::free_vram(self.device).saturating_sub(memory::DESKTOP_MARGIN + (512 << 20))
    }

    /// Картинка RGBA (стороны кратны 16) → нормированный латент
    /// `[1, 64, H/16, W/16]` на CPU.
    pub fn encode_image(&self, image: &RgbaImage) -> Result<Tensor, QwenImage21Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        let x = image.to_tensor()?;
        let tiling = vae::Tiling::pick(image.width, image.height, &self.vae_config, self.vae_budget());
        Ok(vae::encode(&w, &self.vae_config, self.device, &x, tiling)?.to_device(Device::Cpu)?)
    }

    /// Референсы (любой размер) → токены латента: каждая приводится к
    /// `~res²` с пропорциями исходника (стороны кратны 32), как энкодер.
    pub fn encode_references(&self, images: &[RgbaImage], res: usize) -> Result<Qwen21References, QwenImage21Error> {
        if images.is_empty() {
            return Err(QwenImage21Error::Config("нет картинок".into()));
        }
        let mut parts = Vec::with_capacity(images.len());
        let mut grids = Vec::new();
        let mut sizes = Vec::new();
        for img in images {
            let fitted = Self::fit_reference(img, res);
            let lat = self.encode_image(&fitted)?; // [1, 64, h, w]
            let ld = lat.dims().to_vec();
            parts.push(pack(&lat)?);
            grids.push((ld[2], ld[3]));
            sizes.push((fitted.width, fitted.height));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Qwen21References { tokens: Tensor::cat(&refs, 1)?, grids, sizes })
    }

    /// Токенов в склейке: текст + референсы + латент.
    pub fn tokens_for(width: usize, height: usize, ref_tokens: usize, text_tokens: usize) -> usize {
        text_tokens + (snap_side(height) / 16) * (snap_side(width) / 16) + ref_tokens
    }

    /// Загрузить DiT под прогон в `tokens` токенов.
    pub fn load_transformer(&self, tokens: usize) -> Result<QwenImage21Transformer, QwenImage21Error> {
        memory::release_pools(self.device);
        let w = self.source.weights(source::TRANSFORMER)?;
        let t0 = std::time::Instant::now();
        let t = QwenImage21Transformer::load(&w, &self.config, self.device, self.compute, self.quant, self.memory, tokens)?;
        let r = t.residency();
        eprintln!(
            "[qwen-image-2.1] DiT {:?}/{:?} загружен за {:.1} с: блоков на карте {}, на хосте {}, из источника {}",
            self.quant,
            self.compute,
            t0.elapsed().as_secs_f64(),
            r.device,
            r.host,
            r.source
        );
        memory::release_pools(self.device);
        Ok(t)
    }

    /// Денойз. `refs` — референсы (те же картинки, что видел энкодер).
    /// Возвращает латент `[1, 64, h, w]` (F32, CPU). `progress(шаг, всего)` →
    /// `false` прерывает.
    pub fn sample(
        &self,
        transformer: &QwenImage21Transformer,
        cond: &Qwen21Conditioning,
        refs: Option<&Qwen21References>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, QwenImage21Error> {
        self.sample_from_noise(transformer, cond, refs, None, p, progress)
    }

    /// Как [`Self::sample`], но с заданным шумом `[1, 64, h, w]` — для сверки
    /// с эталоном diffusers.
    pub fn sample_from_noise(
        &self,
        transformer: &QwenImage21Transformer,
        cond: &Qwen21Conditioning,
        refs: Option<&Qwen21References>,
        noise: Option<&Tensor>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, QwenImage21Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let z = self.config.in_channels;
        let (width, height) = (snap_side(p.width), snap_side(p.height));
        let (h, w) = (height / 16, width / 16);
        let n_target = h * w;
        let refs = refs.filter(|r| !r.is_empty());
        let n_refs = refs.map(|r| r.len()).unwrap_or(0);
        if n_refs != cond.images() {
            return Err(QwenImage21Error::Config(format!(
                "энкодер видел {} картинок, сэмплеру дали {n_refs} — подключите одну и ту же цепочку референсов",
                cond.images()
            )));
        }

        let noise = match noise {
            Some(n) => n.to_device(dev)?.to_dtype(DType::F32)?,
            None => {
                let mut rng = Philox4x32::new(p.seed);
                randn_seeded(&[1, z, h, w], dev, &mut rng)?
            }
        };
        let mut latents = pack(&noise)?; // [1, n_target, 64] F32

        let sched = QwenScheduler::new(p.steps.max(1), n_target, &self.sched_config);
        let n = sched.num_steps();
        let sigmas: Vec<f32> = (0..n).map(|i| sched.sigma(i)).collect();
        let mods = transformer.modulations(&sigmas)?;

        let mut img_shapes: Vec<(usize, usize)> = refs.map(|r| r.grids.clone()).unwrap_or_default();
        img_shapes.push((h, w));
        let ref_tokens = match refs {
            Some(r) => Some(r.tokens.to_device(dev)?.to_dtype(DType::F32)?),
            None => None,
        };
        let layout = Layout::build(&cond.image_pad_mask, &img_shapes)?;
        let (cos, sin) = build_rope(&layout.positions, &self.config.axes_dims, dev)?;
        let txt = transformer.embed_text(&cond.embeds)?;
        let cfg_on = p.cfg > 1.0 && cond.negative.is_some();
        let neg = match (&cond.negative, cfg_on) {
            (Some((nt, nm)), true) => {
                let l = Layout::build(nm, &img_shapes)?;
                let (c, s) = build_rope(&l.positions, &self.config.axes_dims, dev)?;
                Some((transformer.embed_text(nt)?, l, c, s))
            }
            _ => None,
        };

        // KV-кэш префикса: только если влезает рядом с активациями.
        let want_cache = p.kv_cache && self.config.causal_condition && n > 1;
        let cache_bytes = KvCache::bytes_for(&self.config, layout.prefix_len, transformer.compute_dtype())
            * if neg.is_some() { 2 } else { 1 };
        let fits = !dev.is_cuda()
            || memory::free_vram(dev) > cache_bytes + memory::dit_activation_bytes(layout.total(), self.config.inner()) + memory::DESKTOP_MARGIN;
        let use_cache = want_cache && fits;
        if want_cache && !fits {
            eprintln!(
                "[qwen-image-2.1] KV-кэш префикса ({:.1} ГБ) не помещается — префикс считается на каждом шаге",
                cache_bytes as f64 / 1e9
            );
        }
        let mut cache = use_cache.then(|| transformer.new_cache());
        let mut neg_cache = (use_cache && neg.is_some()).then(|| transformer.new_cache());

        let t1 = std::time::Instant::now();
        for i in 0..n {
            let input = match &ref_tokens {
                Some(r) => Tensor::cat(&[r, &latents], 1)?,
                None => latents.clone(),
            };
            let v = transformer.forward(&txt, &input, &layout, &mods, i, &cos, &sin, cache.as_mut())?.to_dtype(DType::F32)?;
            let v = match &neg {
                Some((nt, nl, nc, ns)) => {
                    let vn = transformer.forward(nt, &input, nl, &mods, i, nc, ns, neg_cache.as_mut())?.to_dtype(DType::F32)?;
                    // neg + g·(pos − neg), без перенормировки (как у 2.1).
                    vn.add(&v.sub(&vn)?.mul_scalar(p.cfg)?)?
                }
                None => v,
            };
            drop(input);
            latents = sched.step(&v, i, &latents)?;
            if !progress(i + 1, n) && i + 1 < n {
                return Err(QwenImage21Error::Cancelled);
            }
        }
        eprintln!(
            "[qwen-image-2.1] денойз {n} шагов {width}×{height}{}{}{} за {:.1} с",
            refs.map(|r| format!(" + {} реф.", r.len())).unwrap_or_default(),
            if cfg_on { format!(", CFG {}", p.cfg) } else { String::new() },
            if use_cache { ", KV-кэш" } else { "" },
            t1.elapsed().as_secs_f64(),
        );
        drop((mods, cache, neg_cache));
        let out = unpack(&latents, h, w)?.to_device(Device::Cpu)?;
        drop(latents);
        memory::release_pools(dev);
        Ok(out)
    }

    /// Латент `[1, 64, h, w]` → картинка RGBA `[4, 16h, 16w]` F32 в [0, 1] (CPU).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, QwenImage21Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        let d = latent.dims();
        let f = self.vae_config.scale_factor_spatial;
        let tiling = vae::Tiling::pick(d[3] * f, d[2] * f, &self.vae_config, self.vae_budget());
        Ok(vae::decode(&w, &self.vae_config, self.device, latent, tiling)?)
    }
}

/// `[1, C, h, w]` → `[1, h·w, C]` (`_pack_latents` 2.1: без упаковки 2×2).
pub fn pack(x: &Tensor) -> Result<Tensor, QwenImage21Error> {
    let d = x.dims().to_vec();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    Ok(x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?)
}

/// Обратное к [`pack`]: `[1, h·w, C]` → `[1, C, h, w]`.
pub fn unpack(x: &Tensor, h: usize, w: usize) -> Result<Tensor, QwenImage21Error> {
    let d = x.dims().to_vec();
    let (b, c) = (d[0], d[2]);
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((b, c, h, w))?)
}

/// Вернуть драйверу память всех пулов (для вызывающих между стадиями).
pub fn release_pools(device: Device) {
    memory::release_pools(device);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        synaptix_kernels_cpu::ensure_registered();
        let v: Vec<f32> = (0..(64 * 2 * 3)).map(|i| i as f32).collect();
        let x = Tensor::from_vec(v.clone(), (1, 64, 2, 3), Device::Cpu).unwrap();
        let p = pack(&x).unwrap();
        assert_eq!(p.dims(), &[1, 6, 64]);
        let pv = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Токен 1 (пиксель (0,1)), канал 0 → x[0][0][1] = 1; канал 1 → 6.
        assert_eq!(pv[64], 1.0);
        assert_eq!(pv[65], 7.0);
        let u = unpack(&p, 2, 3).unwrap();
        assert_eq!(u.flatten_all().unwrap().to_vec1::<f32>().unwrap(), v);
    }

    #[test]
    fn sizes() {
        assert_eq!(cond_size(300, 200, 256), (320, 224));
        assert_eq!(cond_size(1024, 1024, 1024), (1024, 1024));
        assert_eq!(QwenImage21Model::default_size(&[(300, 200)], 256), (320, 224));
        assert_eq!(QwenImage21Model::default_size(&[], 1000), (992, 992));
        assert_eq!(snap_side(1030), 1024);
        assert_eq!(QwenImage21Model::tokens_for(1024, 1024, 4096, 100), 100 + 4096 + 4096);
    }
}
