//! Стадии Qwen-Image для нодовых пайплайнов: промпт (+ картинки) →
//! кондиционирование, картинки → токены референсов (VAE), загрузка DiT,
//! денойз с true CFG, VAE decode.
//!
//! [`QwenImageModel`] держит источник, конфиги и токенайзер; веса грузит
//! каждая стадия сама и сама же отпускает (энкодер живёт только на время
//! кодирования, VAE — на время кодирования/декода). DiT возвращается
//! вызывающему: его можно держать между прогонами.
//!
//! Картинки для правки приводятся к размерам пайплайна здесь же:
//! - Qwen-Image-Edit: одна картинка ~1 Мп (стороны кратны 32) — и в
//!   VL-энкодер, и в VAE;
//! - 2509/2511 (Plus): в VL-энкодер — ~384² (`CONDITION_IMAGE_SIZE`), в VAE
//!   — ~1 Мп каждая; размер выхода по умолчанию — с последней картинки.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_ops::rng::Philox4x32;

use crate::config::{
    ProcessorConfig, QwenImageConfig, QwenImageVariant, QwenVaeConfig, SchedulerConfig, TextEncoderConfig,
};
use crate::memory;
use crate::preprocess::{self, calculate_dimensions, CONDITION_AREA, VAE_AREA};
use crate::scheduler::QwenScheduler;
use crate::source::{self, QwenImageSource};
use crate::text_encoder::{drop_idx, PromptTokenizer, TextEncoder};
use crate::transformer::{build_rope, rope_positions, Placement, QwenImageTransformer};
use crate::vision::VisionTower;
use crate::{vae, QwenImageError};

/// VAE ужимает в 8, пакет 2×2 — сторона кратна 16.
pub const SIDE_MULTIPLE: usize = 16;

pub use crate::transformer::Placement as MemoryMode;

/// Выход энкодера: `[1, S, 3584]` на CPU (BF16) и, для true CFG, то же для
/// негативного промпта.
#[derive(Clone)]
pub struct QwenConditioning {
    pub embeds: Tensor,
    pub negative: Option<Tensor>,
    /// Сколько картинок видел энкодер (для сообщений и проверок).
    pub images: usize,
}

impl std::fmt::Debug for QwenConditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QwenConditioning({:?}, neg={:?}, картинок {})",
            self.embeds.dims(),
            self.negative.as_ref().map(|n| n.dims().to_vec()),
            self.images
        )
    }
}

/// Картинки для правки в латенте: упакованные нормированные токены подряд
/// `[1, N, 64]` (F32, CPU) и сетки каждой картинки в токенах (h, w).
#[derive(Clone)]
pub struct QwenReferences {
    pub tokens: Tensor,
    pub grids: Vec<(usize, usize)>,
    /// Размеры в пикселях (после приведения к ~1 Мп).
    pub sizes: Vec<(usize, usize)>,
}

impl std::fmt::Debug for QwenReferences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QwenReferences({} шт., {} токенов)", self.sizes.len(), self.num_tokens())
    }
}

impl QwenReferences {
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
    pub fn join(parts: &[&QwenReferences]) -> Result<Self, QwenImageError> {
        let parts: Vec<&QwenReferences> = parts.iter().copied().filter(|p| !p.is_empty()).collect();
        if parts.is_empty() {
            return Err(QwenImageError::Config("нет картинок".into()));
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
    /// Масштаб true CFG (`true_cfg_scale`); ≤ 1 или без негатива — без CFG.
    pub cfg: f32,
    pub seed: u64,
}

/// Снап стороны к кратному [`SIDE_MULTIPLE`] (вниз, не меньше 16).
pub fn snap_side(v: usize) -> usize {
    (v / SIDE_MULTIPLE).max(1) * SIDE_MULTIPLE
}

/// Размер VAE-картинки для правки: ~1 Мп с пропорциями исходника, стороны
/// кратны 32. → `(w, h)`.
pub fn vae_size(w: usize, h: usize) -> (usize, usize) {
    calculate_dimensions(VAE_AREA, w as f64 / h.max(1) as f64)
}

pub struct QwenImageModel {
    source: QwenImageSource,
    device: Device,
    compute: DType,
    quant: DType,
    memory: Placement,
    config: QwenImageConfig,
    te_config: TextEncoderConfig,
    vae_config: QwenVaeConfig,
    proc_config: ProcessorConfig,
    sched_config: SchedulerConfig,
    variant: QwenImageVariant,
    tokenizer: PromptTokenizer,
}

impl QwenImageModel {
    /// Открыть модель: `.syn`-бандл или каталог diffusers. Читает конфиги и
    /// токенайзер, веса не трогает. `compute` — активации DiT (BF16 на CUDA),
    /// `quant` — веса DiT: NVFP4/MXFP8 или плотные.
    pub fn open(path: impl AsRef<Path>, device: Device, compute: DType, quant: DType) -> Result<Self, QwenImageError> {
        let source = QwenImageSource::open(path)?;
        for c in source::COMPONENTS {
            if !source.has_component(c) {
                return Err(QwenImageError::Load(format!(
                    "{}: нет компонента `{c}` — нужна раскладка diffusers (transformer/, text_encoder/, vae/)",
                    source.path().display()
                )));
            }
        }
        let config = QwenImageConfig::from_json(&source.read("transformer/config.json")?)?;
        let te_config = TextEncoderConfig::from_json(&source.read("text_encoder/config.json")?)?;
        let vae_config = QwenVaeConfig::from_json(&source.read("vae/config.json")?)?;
        let proc_config = ProcessorConfig::from_json(source.read_opt("processor/preprocessor_config.json").as_deref())?;
        let sched_config = SchedulerConfig::from_json(source.read_opt("scheduler/scheduler_config.json").as_deref())?;
        let variant = QwenImageVariant::from_model_index(source.read_opt("model_index.json").as_deref());
        if te_config.hidden != config.joint_attention_dim {
            return Err(QwenImageError::Config(format!(
                "энкодер {} ≠ joint_attention_dim {}",
                te_config.hidden, config.joint_attention_dim
            )));
        }
        let tok_json = source
            .read_opt("processor/tokenizer.json")
            .or_else(|| source.read_opt("tokenizer/tokenizer.json"))
            .ok_or_else(|| QwenImageError::Tokenizer("нет processor/tokenizer.json".into()))?;
        let tokenizer = PromptTokenizer::new(&tok_json, variant).map_err(|e| QwenImageError::Tokenizer(e.to_string()))?;
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
            variant,
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

    pub fn config(&self) -> &QwenImageConfig {
        &self.config
    }

    pub fn text_encoder_config(&self) -> &TextEncoderConfig {
        &self.te_config
    }

    pub fn variant(&self) -> QwenImageVariant {
        self.variant
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

    /// Размер выхода по умолчанию (`(w, h)`, кратно 16): ~1 Мп с
    /// пропорциями картинки (у Plus — последней, у Edit — первой).
    pub fn default_size(&self, images: &[(usize, usize)]) -> (usize, usize) {
        let pick = match self.variant {
            QwenImageVariant::EditPlus => images.last(),
            _ => images.first(),
        };
        match pick {
            Some(&(w, h)) => {
                let (w, h) = vae_size(w, h);
                (snap_side(w), snap_side(h))
            }
            None => (1024, 1024),
        }
    }

    /// Картинка для VL-энкодера (Lanczos, как `VaeImageProcessor.resize`).
    fn condition_image(&self, image: &Tensor) -> Result<Tensor, QwenImageError> {
        let d = image.dims();
        let (h, w) = (d[1], d[2]);
        let area = match self.variant {
            QwenImageVariant::EditPlus => CONDITION_AREA,
            _ => VAE_AREA,
        };
        let (nw, nh) = calculate_dimensions(area, w as f64 / h.max(1) as f64);
        Ok(preprocess::resize_image(image, nw, nh)?)
    }

    /// Промпт (+ картинки для правки, каждая `[3, H, W]` в [0, 1]) →
    /// кондиционирование. `negative` — промпт для true CFG (у карточек
    /// моделей — `" "`), кодируется с теми же картинками.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative: Option<&str>,
        images: &[Tensor],
    ) -> Result<QwenConditioning, QwenImageError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        if images.len() > self.variant.max_images() {
            return Err(QwenImageError::Config(format!(
                "{} принимает картинок: {}, пришло {}",
                self.variant.label(),
                self.variant.max_images(),
                images.len()
            )));
        }
        let dev = self.device;
        let dt = self.encoder_dtype();
        memory::release_pools(dev);
        let w = self.source.weights(source::TEXT_ENCODER)?;
        let t0 = std::time::Instant::now();

        // Башня зрения: один раз на все картинки (негатив видит те же).
        let mut vis: Vec<(Tensor, (usize, usize))> = Vec::with_capacity(images.len());
        if !images.is_empty() {
            // Башня зрения — в F32 (на CPU и так F32): в BF16 её 32 блока дают
            // 7 % ошибки эмбеддингов, а LLM её размножает (косинус выхода к
            // эталону 0,91 против 0,99). Весит башня 2,7 ГБ и живёт до LLM.
            let tower = VisionTower::load(&w, &self.te_config.vision, dev, DType::F32)?;
            let m = self.te_config.vision.merge_size;
            for img in images {
                let cimg = self.condition_image(img)?;
                let p = preprocess::vision_patches(&cimg, &self.te_config.vision, &self.proc_config)?;
                let (_, gh, gw) = p.grid;
                let e = tower.forward(&p)?;
                vis.push((e, (gh / m, gw / m)));
            }
            drop(tower);
            memory::release_pools(dev);
        }
        let image_tokens: Vec<usize> = vis.iter().map(|(_, (h, w))| h * w).collect();

        let enc = TextEncoder::build(w, self.te_config.clone(), dev, dt, 512 << 20)?;
        let skip = drop_idx(self.variant);
        let one = |p: &str| -> Result<Tensor, QwenImageError> {
            let toks = self.tokenizer.encode(p, &image_tokens)?;
            let h = enc.encode(&toks.ids, &vis)?; // [S, hidden]
            let s = h.dims()[0];
            if s <= skip {
                return Err(QwenImageError::Config(format!("промпт короче системной части шаблона ({s} ≤ {skip})")));
            }
            let mut h = h.narrow(0, skip, s - skip)?.contiguous()?;
            if self.variant == QwenImageVariant::TextToImage && h.dims()[0] > 512 {
                // max_sequence_length = 512 у пайплайна картинки по тексту.
                h = h.narrow(0, 0, 512)?.contiguous()?;
            }
            let n = h.dims()[0];
            Ok(h.reshape((1, n, self.te_config.hidden))?.to_dtype(DType::BF16)?.to_device(Device::Cpu)?)
        };
        let embeds = one(prompt)?;
        let negative = match negative {
            Some(n) => Some(one(n)?),
            None => None,
        };
        eprintln!(
            "[qwen-image] промпт закодирован за {:.1} с ({} токенов, картинок {}, слоёв на карте {}/{})",
            t0.elapsed().as_secs_f64(),
            embeds.dims()[1],
            images.len(),
            enc.resident_layers(),
            self.te_config.num_layers
        );
        drop(enc);
        memory::release_pools(dev);
        Ok(QwenConditioning { embeds, negative, images: images.len() })
    }

    /// Картинка `[3, H, W]` в [0, 1] (стороны кратны 8) → нормированный
    /// латент `[1, 16, H/8, W/8]` на CPU.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor, QwenImageError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let d = image.dims().to_vec();
        if d.len() != 3 || d[0] != 3 || d[1] % 8 != 0 || d[2] % 8 != 0 {
            return Err(QwenImageError::Config(format!(
                "картинка для VAE должна быть [3, H, W] со сторонами кратными 8, пришло {d:?}"
            )));
        }
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        Ok(vae::encode(&w, &self.vae_config, self.device, image)?.to_device(Device::Cpu)?)
    }

    /// Картинки для правки (каждая `[3, H, W]` в [0, 1], любой размер) →
    /// токены латента. Размер — ~1 Мп с пропорциями исходника (стороны кратны
    /// 32), как `VaeImageProcessor.preprocess` у пайплайнов.
    pub fn encode_references(&self, images: &[Tensor]) -> Result<QwenReferences, QwenImageError> {
        if images.is_empty() {
            return Err(QwenImageError::Config("нет картинок".into()));
        }
        let mut parts = Vec::with_capacity(images.len());
        let mut grids = Vec::new();
        let mut sizes = Vec::new();
        for img in images {
            let d = img.dims();
            let (vw, vh) = vae_size(d[2], d[1]);
            let fitted = preprocess::resize_image(img, vw, vh)?;
            let lat = self.encode_image(&fitted)?; // [1, 16, h, w]
            let ld = lat.dims().to_vec();
            parts.push(pack(&lat)?);
            grids.push((ld[2] / 2, ld[3] / 2));
            sizes.push((vw, vh));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(QwenReferences { tokens: Tensor::cat(&refs, 1)?, grids, sizes })
    }

    /// Токенов в прогоне: текст + латент + референсы.
    pub fn tokens_for(width: usize, height: usize, ref_tokens: usize, text_tokens: usize) -> usize {
        text_tokens + (snap_side(height) / SIDE_MULTIPLE) * (snap_side(width) / SIDE_MULTIPLE) + ref_tokens
    }

    /// Загрузить DiT под прогон в `tokens` токенов.
    pub fn load_transformer(&self, tokens: usize) -> Result<QwenImageTransformer, QwenImageError> {
        memory::release_pools(self.device);
        let w = self.source.weights(source::TRANSFORMER)?;
        let t0 = std::time::Instant::now();
        let t = QwenImageTransformer::load(&w, &self.config, self.device, self.compute, self.quant, self.memory, tokens)?;
        let r = t.residency();
        eprintln!(
            "[qwen-image] DiT {:?}/{:?} загружен за {:.1} с: блоков на карте {}, на хосте {}, из источника {}",
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

    /// Денойз. `refs` — картинки для правки (обязательны у Edit/Plus).
    /// Возвращает латент `[1, 16, h, w]` (F32, CPU). `progress(шаг, всего)` →
    /// `false` прерывает.
    pub fn sample(
        &self,
        transformer: &QwenImageTransformer,
        cond: &QwenConditioning,
        refs: Option<&QwenReferences>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, QwenImageError> {
        self.sample_from_noise(transformer, cond, refs, None, p, progress)
    }

    /// Как [`Self::sample`], но с заданным шумом `[1, 16, h, w]` — для сверки
    /// с эталоном diffusers.
    pub fn sample_from_noise(
        &self,
        transformer: &QwenImageTransformer,
        cond: &QwenConditioning,
        refs: Option<&QwenReferences>,
        noise: Option<&Tensor>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, QwenImageError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let dt = transformer.compute_dtype();
        let z = self.config.out_channels;
        let (width, height) = (snap_side(p.width), snap_side(p.height));
        let (h, w) = (height / 8, width / 8);
        let n_target = (h / 2) * (w / 2);
        if self.variant.is_edit() && refs.is_none_or(|r| r.is_empty()) {
            return Err(QwenImageError::Config(format!("{}: нужна картинка для правки", self.variant.label())));
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
        let t0 = std::time::Instant::now();
        let mods = transformer.modulations(&sigmas)?;
        let t_mods = t0.elapsed().as_secs_f64();

        let mut grids = vec![(h / 2, w / 2)];
        let ref_tokens = match refs {
            Some(r) if !r.is_empty() => {
                grids.extend_from_slice(&r.grids);
                Some(r.tokens.to_device(dev)?.to_dtype(DType::F32)?)
            }
            _ => None,
        };
        let txt = transformer.embed_text(&cond.embeds)?;
        let cfg_on = p.cfg > 1.0 && cond.negative.is_some();
        let neg = match (&cond.negative, cfg_on) {
            (Some(n), true) => Some(transformer.embed_text(n)?),
            _ => None,
        };
        let rope_for = |st: usize| -> Result<(Tensor, Tensor), QwenImageError> {
            Ok(build_rope(&rope_positions(&grids, st), &self.config.axes_dims, dev)?)
        };
        let (cos, sin) = rope_for(txt.dims()[1])?;
        let neg_rope = match &neg {
            Some(nt) if nt.dims()[1] != txt.dims()[1] => Some(rope_for(nt.dims()[1])?),
            _ => None,
        };

        let t1 = std::time::Instant::now();
        for i in 0..n {
            let input = match &ref_tokens {
                Some(r) => Tensor::cat(&[&latents, r], 1)?,
                None => latents.clone(),
            }
            .to_dtype(dt)?;
            let v = transformer.forward(&input, n_target, &txt, &mods, i, &cos, &sin)?;
            let v = v.narrow(1, 0, n_target)?.contiguous()?.to_dtype(DType::F32)?;
            let v = match &neg {
                Some(nt) => {
                    let (c2, s2) = neg_rope.as_ref().map(|(c, s)| (c, s)).unwrap_or((&cos, &sin));
                    let vn = transformer.forward(&input, n_target, nt, &mods, i, c2, s2)?;
                    let vn = vn.narrow(1, 0, n_target)?.contiguous()?.to_dtype(DType::F32)?;
                    // comb = vn + g·(v − vn); comb ·= ‖v‖ / ‖comb‖ по каналам токена.
                    let comb = vn.add(&v.sub(&vn)?.mul_scalar(p.cfg)?)?;
                    let cn = v.sqr()?.sum_keepdim(2)?.sqrt()?;
                    let nn = comb.sqr()?.sum_keepdim(2)?.sqrt()?;
                    comb.broadcast_mul(&cn.broadcast_div(&nn)?)?
                }
                None => v,
            };
            drop(input);
            latents = sched.step(&v, i, &latents)?;
            if !progress(i + 1, n) && i + 1 < n {
                return Err(QwenImageError::Cancelled);
            }
        }
        eprintln!(
            "[qwen-image] денойз {n} шагов {width}×{height}{}{} за {:.1} с (модуляции {:.1} с)",
            refs.map(|r| format!(" + {} карт.", r.len())).unwrap_or_default(),
            if cfg_on { format!(", CFG {}", p.cfg) } else { String::new() },
            t1.elapsed().as_secs_f64(),
            t_mods
        );
        drop(mods);
        let out = unpack(&latents, h, w)?.to_device(Device::Cpu)?;
        drop(latents);
        memory::release_pools(dev);
        Ok(out)
    }

    /// Латент `[1, 16, h, w]` → картинка `[3, 8h, 8w]` F32 в [0, 1] (CPU).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, QwenImageError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        Ok(vae::decode(&w, &self.vae_config, self.device, latent)?)
    }
}

/// `[1, C, h, w]` → `[1, (h/2)(w/2), 4C]` (`_pack_latents`: канал `c·4 + dy·2 + dx`).
pub fn pack(x: &Tensor) -> Result<Tensor, QwenImageError> {
    let d = x.dims().to_vec();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    Ok(x.reshape(vec![b, c, h / 2, 2, w / 2, 2])?
        .permute([0, 2, 4, 1, 3, 5])?
        .contiguous()?
        .reshape((b, (h / 2) * (w / 2), c * 4))?)
}

/// Обратное к [`pack`]: `[1, (h/2)(w/2), 4C]` → `[1, C, h, w]`.
pub fn unpack(x: &Tensor, h: usize, w: usize) -> Result<Tensor, QwenImageError> {
    let d = x.dims().to_vec();
    let (b, c4) = (d[0], d[2]);
    let c = c4 / 4;
    Ok(x.reshape(vec![b, h / 2, w / 2, c, 2, 2])?.permute([0, 3, 1, 4, 2, 5])?.contiguous()?.reshape((b, c, h, w))?)
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
        let v: Vec<f32> = (0..(16 * 4 * 6)).map(|i| i as f32).collect();
        let x = Tensor::from_vec(v.clone(), (1, 16, 4, 6), Device::Cpu).unwrap();
        let p = pack(&x).unwrap();
        assert_eq!(p.dims(), &[1, 6, 64]);
        // Токен 0: канал 4·c + 2·dy + dx. Индекс 1 — (c 0, dy 0, dx 1) =
        // пиксель (0, 1); индекс 2 — (dy 1, dx 0) = пиксель (1, 0) = 6.
        let pv = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(pv[1], 1.0);
        assert_eq!(pv[2], 6.0);
        let u = unpack(&p, 4, 6).unwrap();
        assert_eq!(u.flatten_all().unwrap().to_vec1::<f32>().unwrap(), v);
    }

    #[test]
    fn sizes() {
        assert_eq!(vae_size(1024, 1024), (1024, 1024));
        assert_eq!(snap_side(1030), 1024);
        assert_eq!(QwenImageModel::tokens_for(1024, 1024, 4096, 100), 100 + 4096 + 4096);
    }
}
