//! Стадии FLUX.2 для нодовых пайплайнов: промпт → кондиционирование,
//! референсы → токены, загрузка DiT, денойз, VAE decode/encode.
//!
//! [`Flux2Model`] держит источник, конфиги и токенайзер; веса грузит каждая
//! стадия сама и сама же отпускает (энкодер живёт только на время
//! кодирования, VAE — на время декода). DiT возвращается вызывающему: его
//! можно держать между прогонами.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_ops::rng::Philox4x32;

use crate::config::{Flux2Config, Flux2Variant, Flux2VaeConfig, TextEncoderConfig};
use crate::memory;
use crate::scheduler::Flux2Scheduler;
use crate::source::{self, Flux2Source};
use crate::text_encoder::{self, PromptTokenizer, TextEncoder, MAX_SEQ};
use crate::transformer::{build_rope, Flux2Transformer, Placement};
use crate::{vae, Flux2Error};

/// VAE ужимает в 8, пакет 2×2 — сторона кратна 16.
pub const SIDE_MULTIPLE: usize = 16;
/// Координата `t` первой референсной картинки (дальше +10 на каждую).
const REF_T_STEP: f64 = 10.0;

pub use crate::transformer::Placement as MemoryMode;

/// Выход энкодера: `[1, 512, joint]` на CPU (BF16) и, для CFG у klein base,
/// эмбеддинги пустого промпта.
#[derive(Clone)]
pub struct Flux2Conditioning {
    pub embeds: Tensor,
    pub negative: Option<Tensor>,
}

impl std::fmt::Debug for Flux2Conditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Flux2Conditioning({:?}, cfg={})", self.embeds.dims(), self.negative.is_some())
    }
}

/// Референсные картинки для правки: упакованные нормированные латенты
/// подряд `[1, N, 128]` (F32, CPU) и их координаты RoPE.
#[derive(Clone)]
pub struct Flux2References {
    pub tokens: Tensor,
    pub ids: Vec<[f64; 4]>,
    /// Размеры референсов в пикселях — для сообщений и ключей кэша.
    pub sizes: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Flux2References {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Flux2References({} шт., {} токенов)", self.sizes.len(), self.ids.len())
    }
}

impl Flux2References {
    pub fn len(&self) -> usize {
        self.sizes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }
    pub fn num_tokens(&self) -> usize {
        self.ids.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// dev — guidance-эмбеддинг; klein base — масштаб CFG; дистиллированный
    /// klein его не использует.
    pub guidance: f32,
    pub seed: u64,
    /// Доля шума для img2img (с `init`): 1.0 — с чистого шума.
    pub denoise: f32,
}

/// Снап стороны к кратному [`SIDE_MULTIPLE`] (вниз, не меньше 16).
pub fn snap_side(v: usize) -> usize {
    (v / SIDE_MULTIPLE).max(1) * SIDE_MULTIPLE
}

pub struct Flux2Model {
    source: Flux2Source,
    device: Device,
    compute: DType,
    quant: DType,
    memory: Placement,
    config: Flux2Config,
    te_config: TextEncoderConfig,
    vae_config: Flux2VaeConfig,
    variant: Flux2Variant,
    tokenizer: PromptTokenizer,
}

impl Flux2Model {
    /// Открыть модель: `.syn`-бандл или каталог diffusers. Читает конфиги и
    /// токенайзер, веса не трогает. `compute` — активации DiT (BF16 на CUDA),
    /// `quant` — веса DiT: NVFP4/MXFP8 или плотные.
    pub fn open(path: impl AsRef<Path>, device: Device, compute: DType, quant: DType) -> Result<Self, Flux2Error> {
        let source = Flux2Source::open(path)?;
        for c in source::COMPONENTS {
            if !source.has_component(c) {
                return Err(Flux2Error::Load(format!(
                    "{}: нет компонента `{c}` — нужна раскладка diffusers (transformer/, text_encoder/, vae/)",
                    source.path().display()
                )));
            }
        }
        let config = Flux2Config::from_json(&source.read("transformer/config.json")?)?;
        let te_config = TextEncoderConfig::from_json(&source.read("text_encoder/config.json")?)?;
        let vae_config = Flux2VaeConfig::from_json(&source.read("vae/config.json")?)?;
        let variant = Flux2Variant::from_model_index(source.read_opt("model_index.json").as_deref(), te_config.arch);
        let need = variant.text_layers().len() * te_config.hidden;
        if need != config.joint_attention_dim {
            return Err(Flux2Error::Config(format!(
                "кондиционирование {need} (3 слоя × {}) не совпадает с joint_attention_dim {}",
                te_config.hidden, config.joint_attention_dim
            )));
        }
        let tokenizer = text_encoder::open_tokenizer(&source.read("tokenizer/tokenizer.json")?, te_config.arch)?;
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
            variant,
            tokenizer,
        })
    }

    pub fn with_memory(mut self, mode: Placement) -> Self {
        self.memory = mode;
        self
    }

    pub fn source(&self) -> &Flux2Source {
        &self.source
    }

    pub fn config(&self) -> &Flux2Config {
        &self.config
    }

    pub fn text_encoder_config(&self) -> &TextEncoderConfig {
        &self.te_config
    }

    pub fn variant(&self) -> Flux2Variant {
        self.variant
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Точность энкодера: BF16 на карте, F32 на CPU.
    fn encoder_dtype(&self) -> DType {
        if self.device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    /// Промпт → кондиционирование. `negative` — пустой промпт для CFG
    /// (нужен только klein base с guidance > 1).
    pub fn encode_prompt(&self, prompt: &str, with_negative: bool) -> Result<Flux2Conditioning, Flux2Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        memory::release_pools(dev);
        let taps = self.variant.text_layers();
        let run = *taps.iter().max().unwrap_or(&0);
        let w = self.source.weights(source::TEXT_ENCODER)?;
        let enc = TextEncoder::build(w, self.te_config.clone(), run, dev, self.encoder_dtype())?;
        let t0 = std::time::Instant::now();
        let one = |p: &str| -> Result<Tensor, Flux2Error> {
            let toks = self.tokenizer.encode(p)?;
            Ok(enc.encode(&toks, &taps)?.to_dtype(DType::BF16)?.to_device(Device::Cpu)?)
        };
        let embeds = one(prompt)?;
        let negative = if with_negative { Some(one("")?) } else { None };
        eprintln!(
            "[flux2] промпт закодирован за {:.1} с ({} из {run} слоёв энкодера на карте)",
            t0.elapsed().as_secs_f64(),
            enc.resident_layers()
        );
        drop(enc);
        memory::release_pools(dev);
        Ok(Flux2Conditioning { embeds, negative })
    }

    /// Токенов в прогоне: текст + латент + референсы.
    pub fn tokens_for(width: usize, height: usize, ref_tokens: usize) -> usize {
        MAX_SEQ + (snap_side(height) / SIDE_MULTIPLE) * (snap_side(width) / SIDE_MULTIPLE) + ref_tokens
    }

    /// Загрузить DiT под прогон в `tokens` токенов.
    pub fn load_transformer(&self, tokens: usize) -> Result<Flux2Transformer, Flux2Error> {
        memory::release_pools(self.device);
        let w = self.source.weights(source::TRANSFORMER)?;
        let t0 = std::time::Instant::now();
        let t = Flux2Transformer::load(&w, &self.config, self.device, self.compute, self.quant, self.memory, tokens)?;
        let r = t.residency();
        eprintln!(
            "[flux2] DiT {:?}/{:?} загружен за {:.1} с: блоков на карте {}, на хосте {}, из источника {}",
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

    /// Картинка `[3, H, W]` в [0, 1] (стороны кратны 16) → нормированный
    /// упакованный латент `[1, 128, H/16, W/16]` на CPU.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor, Flux2Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let d = image.dims().to_vec();
        if d.len() != 3 || d[0] != 3 || d[1] % SIDE_MULTIPLE != 0 || d[2] % SIDE_MULTIPLE != 0 {
            return Err(Flux2Error::Config(format!(
                "картинка для VAE должна быть [3, H, W] со сторонами кратными {SIDE_MULTIPLE}, пришло {d:?}"
            )));
        }
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        Ok(vae::encode(&w, &self.vae_config, self.device, image)?.to_device(Device::Cpu)?)
    }

    /// Референсные картинки (каждая `[3, H, W]`, стороны кратны 16) → токены
    /// для правки. Координата `t` — 10, 20, … по порядку.
    pub fn encode_references(&self, images: &[Tensor]) -> Result<Flux2References, Flux2Error> {
        let mut parts = Vec::with_capacity(images.len());
        let mut ids = Vec::new();
        let mut sizes = Vec::new();
        for (j, img) in images.iter().enumerate() {
            let lat = self.encode_image(img)?; // [1, 128, h, w]
            let d = lat.dims().to_vec();
            let (c, h, w) = (d[1], d[2], d[3]);
            let t = REF_T_STEP + REF_T_STEP * j as f64;
            for y in 0..h {
                for x in 0..w {
                    ids.push([t, y as f64, x as f64, 0.0]);
                }
            }
            parts.push(pack(&lat)?.reshape((h * w, c))?);
            sizes.push((img.dims()[2], img.dims()[1]));
        }
        if parts.is_empty() {
            return Err(Flux2Error::Config("нет референсных картинок".into()));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let joined = Tensor::cat(&refs, 0)?;
        let n = joined.dims()[0];
        let tokens = joined.reshape((1, n, self.config.in_channels))?;
        Ok(Flux2References { tokens, ids, sizes })
    }

    /// Денойз. `init` — нормированный латент `[1, 128, h, w]` для img2img
    /// (с `denoise`), `refs` — картинки для правки. Возвращает латент
    /// `[1, 128, h, w]` (F32, CPU). `progress(шаг, всего)` → `false` прерывает.
    pub fn sample(
        &self,
        transformer: &Flux2Transformer,
        cond: &Flux2Conditioning,
        refs: Option<&Flux2References>,
        init: Option<&Tensor>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, Flux2Error> {
        self.sample_from_noise(transformer, cond, refs, init, None, p, progress)
    }

    /// Как [`Self::sample`], но с заданным шумом `[1, 128, h, w]` вместо
    /// сгенерированного по `seed` — для сверки с эталоном diffusers.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_from_noise(
        &self,
        transformer: &Flux2Transformer,
        cond: &Flux2Conditioning,
        refs: Option<&Flux2References>,
        init: Option<&Tensor>,
        noise: Option<&Tensor>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, Flux2Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let dt = transformer.compute_dtype();
        let ch = self.config.in_channels;
        let (width, height) = (snap_side(p.width), snap_side(p.height));
        let (h, w) = (height / SIDE_MULTIPLE, width / SIDE_MULTIPLE);
        let seq = h * w;

        let noise = match noise {
            Some(n) => n.to_device(dev)?.to_dtype(DType::F32)?,
            None => {
                let mut rng = Philox4x32::new(p.seed);
                randn_seeded(&[1, ch, h, w], dev, &mut rng)?
            }
        };
        let noise = pack(&noise)?; // [1, seq, 128] F32

        let sched = Flux2Scheduler::new(p.steps.max(1), seq);
        let n = sched.num_steps();
        let (start, mut latents) = match init {
            Some(x0) => {
                let d = x0.dims();
                if d != [1, ch, h, w] {
                    return Err(Flux2Error::Config(format!(
                        "латент картинки {d:?} не совпадает с размером генерации {width}×{height} \
                         (ожидается [1, {ch}, {h}, {w}])"
                    )));
                }
                let start = sched.start_index(p.denoise);
                if start >= n {
                    return Ok(x0.to_dtype(DType::F32)?.to_device(Device::Cpu)?);
                }
                let x0 = pack(&x0.to_device(dev)?.to_dtype(DType::F32)?)?;
                (start, sched.scale_noise(&x0, &noise, start)?)
            }
            None => (0, noise),
        };

        // Координаты: текст (0,0,0,l), латент (0,y,x,0), референсы (10j,y,x,0).
        let mut ids: Vec<[f64; 4]> = (0..MAX_SEQ).map(|l| [0.0, 0.0, 0.0, l as f64]).collect();
        for y in 0..h {
            for x in 0..w {
                ids.push([0.0, y as f64, x as f64, 0.0]);
            }
        }
        let ref_tokens = match refs {
            Some(r) if !r.is_empty() => {
                ids.extend_from_slice(&r.ids);
                Some(r.tokens.to_device(dev)?.to_dtype(DType::F32)?)
            }
            _ => None,
        };
        let (cos, sin) = build_rope(&ids, &self.config.axes_dims, self.config.rope_theta, dev)?;

        let txt = cond.embeds.to_device(dev)?.to_dtype(dt)?;
        let cfg = self.variant.uses_cfg(p.guidance);
        let neg = match (&cond.negative, cfg) {
            (Some(n), true) => Some(n.to_device(dev)?.to_dtype(dt)?),
            (None, true) => {
                return Err(Flux2Error::Config(
                    "CFG klein base: кондиционирование без пустого промпта (encode_prompt(…, true))".into(),
                ))
            }
            _ => None,
        };
        let guidance = if self.config.guidance_embeds { Some(p.guidance) } else { None };

        let total = n - start;
        let t0 = std::time::Instant::now();
        for (k, i) in (start..n).enumerate() {
            let sigma = sched.sigma(i);
            let input = match &ref_tokens {
                Some(r) => Tensor::cat(&[&latents, r], 1)?,
                None => latents.clone(),
            }
            .to_dtype(dt)?;
            let mut v = transformer.forward(&input, &txt, sigma, guidance, &cos, &sin)?;
            if let Some(neg) = &neg {
                let vn = transformer.forward(&input, neg, sigma, guidance, &cos, &sin)?;
                // v = vn + g·(v − vn)
                v = vn.add(&v.sub(&vn)?.mul_scalar(p.guidance)?)?;
            }
            drop(input);
            let v = if ref_tokens.is_some() { v.narrow(1, 0, seq)?.contiguous()? } else { v };
            latents = sched.step(&v, i, &latents)?;
            if !progress(k + 1, total) && k + 1 < total {
                return Err(Flux2Error::Cancelled);
            }
        }
        eprintln!(
            "[flux2] денойз {total} шагов {width}×{height}{} за {:.1} с",
            refs.map(|r| format!(" + {} реф.", r.len())).unwrap_or_default(),
            t0.elapsed().as_secs_f64()
        );
        let out = unpack(&latents, h, w)?.to_device(Device::Cpu)?;
        drop(latents);
        memory::release_pools(dev);
        Ok(out)
    }

    /// Латент `[1, 128, h, w]` → картинка `[3, 16h, 16w]` F32 в [0, 1] (CPU).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, Flux2Error> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        memory::release_pools(self.device);
        let w = self.source.weights(source::VAE)?;
        Ok(vae::decode(&w, &self.vae_config, self.device, latent)?)
    }
}

/// `[1, C, h, w]` → `[1, h·w, C]`.
pub fn pack(x: &Tensor) -> Result<Tensor, Flux2Error> {
    let d = x.dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    Ok(x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?)
}

/// `[1, h·w, C]` → `[1, C, h, w]`.
pub fn unpack(x: &Tensor, h: usize, w: usize) -> Result<Tensor, Flux2Error> {
    let d = x.dims();
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
    fn snap_and_tokens() {
        assert_eq!(snap_side(1030), 1024);
        assert_eq!(snap_side(5), 16);
        assert_eq!(Flux2Model::tokens_for(1024, 1024, 0), 512 + 4096);
        assert_eq!(Flux2Model::tokens_for(512, 768, 1024), 512 + 32 * 48 + 1024);
    }
}
