//! Стадии SDXL для нодовых пайплайнов: промпт → кондиционирование (два
//! CLIP), загрузка UNet, денойз (txt2img / img2img с CFG), VAE decode и
//! encode.
//!
//! [`SdxlCheckpoint`] держит источник и токенайзеры; веса грузит каждая
//! стадия сама и сама же отпускает (CLIP — на время кодирования, VAE — на
//! время декода/кодирования). UNet возвращается вызывающему: его можно
//! держать между прогонами. Так вся линейка (UNet 5,1 ГБ в F16) проходит и на
//! карте ~7 ГБ.
//!
//! Как `StableDiffusionXLPipeline` diffusers: CLIP-L и bigG — предпоследние
//! скрытые состояния, pooled bigG через `text_projection`; пустой негатив —
//! нули (`force_zeros_for_empty_prompt`); `add_time_ids` = (h, w, 0, 0, h, w);
//! CFG `uncond + g·(cond − uncond)`; Euler с `steps_offset` из конфига.

use std::path::Path;

use synaptix_core::device::cuda::WeightsAllocGuard;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_nn::linear::Linear;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};
use synaptix_nn::unet::{UNet2DConditionConfig, UNet2DConditionModel};
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder, AutoencoderKlEncoder};
use synaptix_ops::rng::Philox4x32;

use crate::model::{tokenizers, MAX_TOKENS};
use crate::scheduler::{EulerParams, SdxlEuler};
use crate::source::{self, SdxlSource};
use crate::tokenizer::ClipTokenizer;
use crate::SdxlError;

/// Сторона картинки кратна 64: VAE ужимает в 8, а UNet ещё дважды делит
/// латент пополам — так уровни сходятся без подгонки размеров (рекомендуемые
/// разрешения SDXL тоже кратны 64).
pub const SIDE_MULTIPLE: usize = 64;

/// Снап стороны к кратному [`SIDE_MULTIPLE`] (вниз, не меньше 64).
pub fn snap_side(v: usize) -> usize {
    (v / SIDE_MULTIPLE).max(1) * SIDE_MULTIPLE
}

/// Кондиционирование: `[2, 77, 2048]` (uncond, cond) и pooled `[2, 1280]`,
/// на CPU в F32.
#[derive(Clone)]
pub struct SdxlConditioning {
    pub hidden: Tensor,
    pub pooled: Tensor,
}

impl std::fmt::Debug for SdxlConditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SdxlConditioning({:?})", self.hidden.dims())
    }
}

/// Загруженный UNet.
pub struct SdxlUnet {
    unet: UNet2DConditionModel,
    device: Device,
    dtype: DType,
    quant: DType,
}

impl SdxlUnet {
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn quant(&self) -> DType {
        self.quant
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdxlSampleParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    /// Сила img2img (с `init`): доля шагов расписания, 1 — с чистого шума.
    pub denoise: f32,
}

fn release_pools(device: Device) {
    if let Device::Cuda(ord) = device {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
    }
}

pub struct SdxlCheckpoint {
    source: SdxlSource,
    device: Device,
    /// Активации UNet и CLIP: F16 на карте (как у diffusers), F32 на CPU.
    dtype: DType,
    /// VAE-декодер: F16 переполняется — BF16 на карте, F32 на CPU (энкодер
    /// всегда F32).
    vae_dtype: DType,
    tok_l: ClipTokenizer,
    tok_g: ClipTokenizer,
    euler: EulerParams,
    scaling_factor: f32,
    force_zeros_for_empty_prompt: bool,
}

impl SdxlCheckpoint {
    /// Открыть `.syn`-бандл или каталог diffusers; веса не трогает.
    pub fn open(path: impl AsRef<Path>, device: Device) -> Result<Self, SdxlError> {
        let source = SdxlSource::open(path)?;
        for c in source::COMPONENTS {
            if !source.has_component(c) {
                return Err(SdxlError::Load(format!(
                    "{}: нет компонента `{c}` — нужна раскладка diffusers (unet/, text_encoder/, text_encoder_2/, vae/)",
                    source.path().display()
                )));
            }
        }
        let (tok_l, tok_g) = tokenizers(&source)?;
        let euler = EulerParams::from_json(source.read_opt("scheduler/scheduler_config.json").as_deref());
        let vae_cfg: Option<serde_json::Value> =
            source.read_opt("vae/config.json").and_then(|b| serde_json::from_slice(&b).ok());
        let scaling_factor = vae_cfg
            .as_ref()
            .and_then(|v| v.get("scaling_factor"))
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(AutoencoderKlConfig::sdxl().scaling_factor);
        let index: Option<serde_json::Value> =
            source.read_opt("model_index.json").and_then(|b| serde_json::from_slice(&b).ok());
        let force_zeros_for_empty_prompt = index
            .as_ref()
            .and_then(|v| v.get("force_zeros_for_empty_prompt"))
            .and_then(|x| x.as_bool())
            .unwrap_or(true);
        let cuda = device.is_cuda();
        Ok(Self {
            source,
            device,
            dtype: if cuda { DType::F16 } else { DType::F32 },
            vae_dtype: if cuda { DType::BF16 } else { DType::F32 },
            tok_l,
            tok_g,
            euler,
            scaling_factor,
            force_zeros_for_empty_prompt,
        })
    }

    pub fn source(&self) -> &SdxlSource {
        &self.source
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn ids(&self, tok: &ClipTokenizer, text: &str) -> Result<Tensor, SdxlError> {
        Ok(Tensor::from_vec(tok.encode(text, MAX_TOKENS), (1, MAX_TOKENS), self.device)?)
    }

    /// Промпт и негатив → кондиционирование. Пустой негатив — нули, как у
    /// пайплайна (`force_zeros_for_empty_prompt`).
    pub fn encode_prompt(&self, prompt: &str, negative: &str) -> Result<SdxlConditioning, SdxlError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let (dev, dt) = (self.device, self.dtype);
        release_pools(dev);
        let zero_neg = negative.trim().is_empty() && self.force_zeros_for_empty_prompt;
        let t0 = std::time::Instant::now();
        let (hl_c, hl_u) = {
            let w = self.source.weights(source::TEXT_ENCODER)?;
            let enc = {
                let _g = WeightsAllocGuard::for_device(dev);
                ClipTextEncoder::load(&ClipTextConfig::clip_l(), "text_model", &|n| w.get(n, dev, dt))?
            };
            let c = enc.forward(&self.ids(&self.tok_l, prompt)?)?.penultimate_hidden_state().clone();
            let u = if zero_neg { None } else { Some(enc.forward(&self.ids(&self.tok_l, negative)?)?.penultimate_hidden_state().clone()) };
            (c, u)
        };
        release_pools(dev);
        let (hg_c, pg_c, hg_u, pg_u) = {
            let w = self.source.weights(source::TEXT_ENCODER_2)?;
            let get = |n: &str| w.get(n, dev, dt);
            let enc = {
                let _g = WeightsAllocGuard::for_device(dev);
                ClipTextEncoder::load(&ClipTextConfig::clip_bigg(), "text_model", &get)?
                    .with_projection(Linear::new(get("text_projection.weight")?, None)?)
            };
            let c = enc.forward(&self.ids(&self.tok_g, prompt)?)?;
            let (hu, pu) = if zero_neg {
                (None, None)
            } else {
                let u = enc.forward(&self.ids(&self.tok_g, negative)?)?;
                (Some(u.penultimate_hidden_state().clone()), Some(u.pooled_output.clone()))
            };
            (c.penultimate_hidden_state().clone(), c.pooled_output.clone(), hu, pu)
        };
        let cond = Tensor::cat(&[&hl_c, &hg_c], 2)?.contiguous()?.to_dtype(DType::F32)?;
        let pooled_c = pg_c.to_dtype(DType::F32)?;
        let (uncond, pooled_u) = match (hl_u, hg_u, pg_u) {
            (Some(l), Some(g), Some(p)) => (Tensor::cat(&[&l, &g], 2)?.contiguous()?.to_dtype(DType::F32)?, p.to_dtype(DType::F32)?),
            _ => (cond.zeros_like()?, pooled_c.zeros_like()?),
        };
        let hidden = Tensor::cat(&[&uncond, &cond], 0)?.to_device(Device::Cpu)?;
        let pooled = Tensor::cat(&[&pooled_u, &pooled_c], 0)?.to_device(Device::Cpu)?;
        eprintln!("[sdxl] промпт закодирован за {:.1} с", t0.elapsed().as_secs_f64());
        release_pools(dev);
        Ok(SdxlConditioning { hidden, pooled })
    }

    /// Загрузить UNet: `quant` — NVFP4/MXFP8 (линейки внимания и GEGLU) или
    /// плотный.
    pub fn load_unet(&self, quant: DType) -> Result<SdxlUnet, SdxlError> {
        let (dev, dt) = (self.device, self.dtype);
        release_pools(dev);
        let quant = if dev.is_cuda() && quant.is_quantized() { quant } else { dt };
        let w = self.source.weights(source::UNET)?;
        let t0 = std::time::Instant::now();
        let unet = {
            let _g = WeightsAllocGuard::for_device(dev);
            synaptix_nn::unet::unet_2d_condition::set_unet_precision(quant, dt);
            let r = UNet2DConditionModel::load(&UNet2DConditionConfig::sdxl(), &|n| w.get(n, dev, dt));
            synaptix_nn::unet::unet_2d_condition::set_unet_precision(DType::BF16, DType::BF16);
            r?
        };
        eprintln!("[sdxl] UNet {quant:?}/{dt:?} загружен за {:.1} с", t0.elapsed().as_secs_f64());
        release_pools(dev);
        Ok(SdxlUnet { unet, device: dev, dtype: dt, quant })
    }

    /// Денойз. `init` — латент `[1, 4, h, w]` (масштабированный, из
    /// [`Self::encode_image`]) для img2img с `denoise`. Возвращает латент
    /// `[1, 4, h, w]` (F32, CPU). `progress(шаг, всего)` → `false` прерывает.
    pub fn sample(
        &self,
        unet: &SdxlUnet,
        cond: &SdxlConditioning,
        init: Option<&Tensor>,
        p: &SdxlSampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, SdxlError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = unet.device;
        let (width, height) = (snap_side(p.width), snap_side(p.height));
        let (h, w) = (height / 8, width / 8);
        let sched = SdxlEuler::new(p.steps.max(1), &self.euler);
        let n = sched.num_steps();
        let mut rng = Philox4x32::new(p.seed);
        let noise = randn_seeded(&[1, 4, h, w], dev, &mut rng)?;
        let (start, mut latents) = match init {
            Some(x0) => {
                let d = x0.dims();
                if d != [1, 4, h, w] {
                    return Err(SdxlError::Config(format!(
                        "латент картинки {d:?} не совпадает с размером {width}×{height} (ожидается [1, 4, {h}, {w}])"
                    )));
                }
                let start = sched.start_for_strength(p.denoise);
                let x0 = x0.to_device(dev)?.to_dtype(DType::F32)?;
                if start >= n {
                    return Ok(x0.to_device(Device::Cpu)?);
                }
                (start, x0.add(&noise.mul_scalar(sched.sigma(start))?)?)
            }
            None => (0, noise.mul_scalar(sched.init_noise_sigma())?),
        };
        let total = n - start;
        let t0 = std::time::Instant::now();
        for (k, i) in (start..n).enumerate() {
            let eps = self.unet_eps(unet, cond, &latents, sched.timestep(i), sched.sigma(i), width, height)?;
            let eu = eps.narrow(0, 0, 1)?;
            let ec = eps.narrow(0, 1, 1)?;
            let e = eu.add(&ec.sub(&eu)?.mul_scalar(p.guidance)?)?;
            latents = latents.add(&e.mul_scalar(sched.dt(i))?)?;
            if !progress(k + 1, total) && k + 1 < total {
                return Err(SdxlError::Cancelled);
            }
        }
        eprintln!("[sdxl] денойз {total} шагов {width}×{height} за {:.1} с", t0.elapsed().as_secs_f64());
        let out = latents.to_device(Device::Cpu)?;
        drop(latents);
        release_pools(dev);
        Ok(out)
    }

    /// ε UNet на шаге с моментом `t` и сигмой `sigma`: `x_t` `[1, 4, h, w]`
    /// (F32, как в цикле) масштабируется `1/√(σ² + 1)` и идёт батчем
    /// (негатив, промпт). → `[2, 4, h, w]` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn unet_eps(
        &self,
        unet: &SdxlUnet,
        cond: &SdxlConditioning,
        x_t: &Tensor,
        t: f32,
        sigma: f32,
        width: usize,
        height: usize,
    ) -> Result<Tensor, SdxlError> {
        let (dev, dt) = (unet.device, unet.dtype);
        let hidden = cond.hidden.to_device(dev)?.to_dtype(dt)?;
        let pooled = cond.pooled.to_device(dev)?.to_dtype(dt)?;
        let row = [height as f32, width as f32, 0.0, 0.0, height as f32, width as f32];
        let ids: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let time_ids = Tensor::from_vec(ids, (2, 6), dev)?.to_dtype(dt)?;
        let x = x_t.to_device(dev)?.to_dtype(DType::F32)?.mul_scalar(1.0 / (sigma * sigma + 1.0).sqrt())?.to_dtype(dt)?;
        let x2 = Tensor::cat(&[&x, &x], 0)?.contiguous()?;
        let tt = Tensor::from_vec(vec![t, t], (2,), dev)?;
        Ok(unet.unet.forward(&x2, &tt, &hidden, &pooled, &time_ids)?.to_dtype(DType::F32)?)
    }

    /// Латент `[1, 4, h, w]` → картинка `[3, 8h, 8w]` F32 в [0, 1] (CPU).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, SdxlError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let (dev, dt) = (self.device, self.vae_dtype);
        release_pools(dev);
        let w = self.source.weights(source::VAE)?;
        let image = {
            let dec = {
                let _g = WeightsAllocGuard::for_device(dev);
                AutoencoderKlDecoder::load(&AutoencoderKlConfig::sdxl(), &|n| w.get(n, dev, dt))?
            };
            let z = latent.to_device(dev)?.to_dtype(DType::F32)?.mul_scalar(1.0 / self.scaling_factor)?.to_dtype(dt)?;
            dec.decode(&z)?
        };
        let image = image.to_dtype(DType::F32)?.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
        let d = image.dims().to_vec();
        let out = image.reshape((d[1], d[2], d[3]))?.contiguous()?.to_device(Device::Cpu)?;
        drop(image);
        release_pools(dev);
        Ok(out)
    }

    /// Картинка `[3, H, W]` в [0, 1] (стороны кратны 8) → масштабированный
    /// латент `[1, 4, H/8, W/8]` (мода распределения) на CPU.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor, SdxlError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let d = image.dims().to_vec();
        if d.len() != 3 || d[0] != 3 || d[1] % 8 != 0 || d[2] % 8 != 0 {
            return Err(SdxlError::Config(format!(
                "картинка для VAE должна быть [3, H, W] со сторонами кратными 8, пришло {d:?}"
            )));
        }
        // Энкодер — в F32: в BF16 латент расходится с diffusers заметно
        // (косинус 0,9988), а весит энкодер мало.
        let (dev, dt) = (self.device, DType::F32);
        release_pools(dev);
        let w = self.source.weights(source::VAE)?;
        let enc = {
            let _g = WeightsAllocGuard::for_device(dev);
            AutoencoderKlEncoder::load(&AutoencoderKlConfig::sdxl(), &|n| w.get(n, dev, dt))?
        };
        let x = image.to_device(dev)?.to_dtype(DType::F32)?.reshape((1, d[0], d[1], d[2]))?.affine(2.0, -1.0)?.to_dtype(dt)?;
        let moments = enc.encode(&x)?;
        let (mean, _) = enc.split_moments(&moments)?;
        let out = mean.to_dtype(DType::F32)?.mul_scalar(self.scaling_factor)?.contiguous()?.to_device(Device::Cpu)?;
        drop((enc, moments, mean, x));
        release_pools(dev);
        Ok(out)
    }
}
