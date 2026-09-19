//! Стадии FLUX для нодовых пайплайнов: промпт → кондиционирование,
//! загрузка трансформера, денойз, VAE decode/encode.
//!
//! [`FluxModel`] держит только источник, конфиги и токенайзеры — веса
//! грузит каждая стадия сама и сама же отпускает (CLIP и T5 живут только на
//! время кодирования). Трансформер возвращается вызывающему: пайплайн может
//! держать его между прогонами («Держать в памяти»), а может сразу отпустить.
//! [`crate::FluxPipeline::txt2img`] — это те же стадии подряд.

use std::path::Path;

use synaptix_core::device::cuda::WeightsAllocGuard;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder, AutoencoderKlEncoder};
use synaptix_ops::rng::Philox4x32;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

use crate::loader::ComponentWeights;
use crate::scheduler::{FlowMatchScheduler, SchedulerConfig};
use crate::source::{FluxComponent, FluxSource};
use crate::t5::{T5Config, T5Encoder};
use crate::tokenizer::ClipTokenizer;
use crate::transformer::{set_load_precision, FluxConfig, FluxTransformer};
use crate::FluxError;

const CLIP_MAX: usize = 77;
/// Потолок T5 у FLUX.1: dev обучен на 512 токенах, schnell — на 256.
pub const T5_MAX_DEV: usize = 512;
pub const T5_MAX_SCHNELL: usize = 256;
/// VAE ужимает в 8 раз, трансформер пакует 2×2 — сторона кратна 16.
pub const SIDE_MULTIPLE: usize = 16;
const LATENT_CHANNELS: usize = 16;

/// Где держать плотные веса трансформера. Квантованные (NVFP4/MXFP8) малы и
/// всегда резидентны — режим на них не влияет.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OffloadMode {
    /// По свободной VRAM: целиком, если влезает с запасом под активации,
    /// иначе столько блоков, сколько влезло, остальные стримятся с хоста.
    #[default]
    Auto,
    Resident,
    Stream,
}

/// Выход текстовых энкодеров: CLIP pooled `[1, 768]` и T5 `[1, L, 4096]`.
#[derive(Clone)]
pub struct FluxConditioning {
    pub pooled: Tensor,
    pub t5: Tensor,
}

impl std::fmt::Debug for FluxConditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FluxConditioning(pooled {:?}, t5 {:?})", self.pooled.dims(), self.t5.dims())
    }
}

/// Параметры одного денойза.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// dev: guidance-эмбеддинг (не CFG). schnell его игнорирует.
    pub guidance: f32,
    pub seed: u64,
    /// Доля шума для img2img: 1.0 — с чистого шума (txt2img), меньше —
    /// от исходного латента. Без `init` не используется.
    pub denoise: f32,
}

/// Вернуть драйверу свободное во ВСЕХ пулах устройства. Веса без
/// [`WeightsAllocGuard`] и активации живут в пуле активаций, а у него
/// порог освобождения — бесконечность: после T5 там оставалось ~10 ГБ
/// зарезервированных, и ни трансформер, ни следующая видео-модель их не
/// видели. Обычный `trim_cuda_mempool_device` трогает только default-пул.
pub fn release_pools(device: Device) {
    if let Device::Cuda(ord) = device {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
    }
}

/// Веса — в weights-пул (а не вперемешку с активациями), на выходе из
/// области видимости staging-буферы загрузки отдаются драйверу.
fn weights_guard(device: Device) -> WeightsAllocGuard {
    WeightsAllocGuard::for_device(device)
}

/// Снап стороны к кратному [`SIDE_MULTIPLE`] (вниз, но не меньше 16).
pub fn snap_side(v: usize) -> usize {
    (v / SIDE_MULTIPLE).max(1) * SIDE_MULTIPLE
}

pub struct FluxModel {
    source: FluxSource,
    device: Device,
    /// Активации трансформера.
    compute: DType,
    /// Веса трансформера: NVFP4/MXFP8 — квант, прочее — плотные в `compute`.
    quant: DType,
    offload: OffloadMode,
    config: FluxConfig,
    scheduler: SchedulerConfig,
    vae_scaling: f32,
    vae_shift: f32,
    clip_tok: ClipTokenizer,
    t5_tok: HfTokenizer,
}

impl FluxModel {
    /// Открыть модель: `.syn`-бандл или каталог diffusers. Читает конфиги и
    /// токенайзеры, веса не трогает.
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, FluxError> {
        let source = FluxSource::open(path)?;
        for c in FluxComponent::ALL {
            if !source.has_component(c) {
                return Err(FluxError::Load(format!(
                    "{}: нет компонента `{}` — нужна раскладка diffusers (transformer/, \
                     text_encoder/, text_encoder_2/, vae/)",
                    source.path().display(),
                    c.name()
                )));
            }
        }
        let config = FluxConfig::from_json(&source.read("transformer/config.json")?)?;
        let scheduler = match source.read_opt("scheduler/scheduler_config.json") {
            Some(b) => SchedulerConfig::from_json(&b).map_err(FluxError::Config)?,
            None => SchedulerConfig::default(),
        };
        let (vae_scaling, vae_shift) = vae_factors(source.read_opt("vae/config.json").as_deref());
        let clip_tok = ClipTokenizer::from_bytes(
            &source.read("tokenizer/vocab.json")?,
            &source.read("tokenizer/merges.txt")?,
            source.read_opt("tokenizer/tokenizer_config.json").as_deref(),
        )?;
        let t5_tok = HfTokenizer::from_bytes(&source.read("tokenizer_2/tokenizer.json")?)
            .map_err(|e| FluxError::Tokenizer(format!("tokenizer_2/tokenizer.json: {e}")))?;
        Ok(Self {
            source,
            device,
            compute,
            quant,
            offload: OffloadMode::Auto,
            config,
            scheduler,
            vae_scaling,
            vae_shift,
            clip_tok,
            t5_tok,
        })
    }

    pub fn with_offload(mut self, mode: OffloadMode) -> Self {
        self.offload = mode;
        self
    }

    pub fn source(&self) -> &FluxSource {
        &self.source
    }

    pub fn config(&self) -> &FluxConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// dev (guidance-эмбеддинг) против schnell (без guidance).
    pub fn guidance_distilled(&self) -> bool {
        self.config.guidance_embeds
    }

    /// Рекомендованная длина T5 для этой модели.
    pub fn default_max_seq_len(&self) -> usize {
        if self.config.guidance_embeds {
            T5_MAX_DEV
        } else {
            T5_MAX_SCHNELL
        }
    }

    /// Точность текстовых энкодеров. T5-XXL в F16 переполняется в FF-слоях,
    /// поэтому энкодеры на GPU всегда в BF16, независимо от активаций
    /// трансформера (квант требует F16).
    fn encoder_dtype(&self) -> DType {
        if self.device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    /// CLIP pooled + T5. Энкодеры грузятся по очереди и освобождаются сразу
    /// после прохода, чтобы пик VRAM был не больше T5 (≈ 9,5 ГБ в BF16).
    /// `max_seq_len` обрезает и паддит T5-последовательность.
    pub fn encode_prompt(&self, prompt: &str, max_seq_len: usize) -> Result<FluxConditioning, FluxError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let edt = self.encoder_dtype();
        let max_seq_len = max_seq_len.clamp(1, T5_MAX_DEV);

        let clip_ids = self.clip_tok.encode(prompt, CLIP_MAX);
        let pooled = {
            let ids = Tensor::from_vec(clip_ids, (1, CLIP_MAX), dev)?;
            let enc = {
                let _g = weights_guard(dev);
                let w = ComponentWeights::from_source(&self.source, FluxComponent::ClipText, dev, edt)?;
                ClipTextEncoder::load(&ClipTextConfig::clip_l(), "text_model", &|n| w.get(n))?
            };
            enc.forward(&ids)?.pooled_output
        };

        let mut t5_ids: Vec<u32> = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| FluxError::Tokenizer(format!("t5 encode: {e}")))?
            .ids;
        t5_ids.truncate(max_seq_len);
        t5_ids.resize(max_seq_len, 0); // pad_token_id = 0
        let t5 = {
            let ids = Tensor::from_vec(t5_ids, (1, max_seq_len), dev)?;
            let enc = {
                let _g = weights_guard(dev);
                let w = ComponentWeights::from_source(&self.source, FluxComponent::T5Text, dev, edt)?;
                T5Encoder::load(&T5Config::xxl(), &|n| w.get(n))?
            };
            enc.forward(&ids)?
        };
        release_pools(dev);
        Ok(FluxConditioning { pooled, t5 })
    }

    /// Загрузить трансформер. `tokens` — текст + картинка самого крупного
    /// прогона, под который выбирается резидентность плотных весов.
    pub fn load_transformer(&self, tokens: usize) -> Result<FluxTransformer, FluxError> {
        let dev = self.device;
        let dt = self.compute;
        let is_quant = self.quant.is_quantized();
        set_load_precision(if is_quant { self.quant } else { dt }, dt);
        // Всё отпущенное прошлыми стадиями — драйверу, иначе замер свободной
        // VRAM ниже занижен, а квантованным весам негде лечь.
        release_pools(dev);
        let _g = weights_guard(dev);

        if is_quant && dev.is_cuda() {
            // Квант при загрузке: плотный вес читается в compute и сразу
            // квантуется, так что на карте остаётся ~6 ГБ (NVFP4) / ~12 ГБ
            // (MXFP8) — стриминг не нужен даже на 2048².
            eprintln!("[FLUX] quant={:?} compute={dt:?}: квантую трансформер резидентно", self.quant);
            let w = ComponentWeights::from_source(&self.source, FluxComponent::Transformer, dev, dt)?;
            return Ok(FluxTransformer::load(&self.config, &|nm| w.get(nm))?);
        }
        if !dev.is_cuda() {
            let w = ComponentWeights::from_source(&self.source, FluxComponent::Transformer, Device::Cpu, dt)?;
            return Ok(FluxTransformer::load(&self.config, &|nm| w.get(nm))?);
        }

        let tokens = tokens as u64;
        let stream = match self.offload {
            OffloadMode::Resident => false,
            OffloadMode::Stream => true,
            OffloadMode::Auto => {
                let ord = if let Device::Cuda(o) = dev { o } else { 0 };
                let weight_bytes = self.source.component_bytes(FluxComponent::Transformer)?;
                // Запас под активации плюс 1,5 ГБ под рабочий стол: карта одна
                // на модель и композитор, и при сотнях свободных мегабайт KWin
                // начинает ронять atomic commit.
                let reserve = 1_500_000_000u64 + tokens * 96_000;
                match synaptix_core::device::cuda::mem_info(ord) {
                    Ok((free, total)) => {
                        let resident = (free as u64) >= weight_bytes + reserve;
                        eprintln!(
                            "[FLUX] auto-offload: free={:.1}GB total={:.1}GB веса≈{:.1}GB резерв={:.1}GB ({tokens} токенов) → {}",
                            free as f64 / 1e9,
                            total as f64 / 1e9,
                            weight_bytes as f64 / 1e9,
                            reserve as f64 / 1e9,
                            if resident { "РЕЗИДЕНТ" } else { "STREAMING" },
                        );
                        !resident
                    }
                    Err(_) => tokens > 4096,
                }
            }
        };
        if stream {
            // Частичный offload: блоки на GPU, пока свободно больше запаса под
            // пик активаций и транзиент стримящегося блока.
            let min_free = 1_800_000_000u64 + tokens * 220_000;
            let w = ComponentWeights::from_source(&self.source, FluxComponent::Transformer, Device::Cpu, dt)?;
            Ok(FluxTransformer::load(&self.config, &|nm| w.get(nm))?.into_partial_streaming(dev, min_free)?)
        } else {
            let w = ComponentWeights::from_source(&self.source, FluxComponent::Transformer, dev, dt)?;
            Ok(FluxTransformer::load(&self.config, &|nm| w.get(nm))?)
        }
    }

    /// Число токенов прогона: T5 + упакованная картинка.
    pub fn tokens_for(width: usize, height: usize, max_seq_len: usize) -> usize {
        max_seq_len + (height / SIDE_MULTIPLE) * (width / SIDE_MULTIPLE)
    }

    /// Денойз. `init` — латент исходной картинки `[1, 16, h, w]` из
    /// [`Self::encode_image`] для img2img (тогда учитывается `denoise`).
    /// Возвращает нормированный латент `[1, 16, h, w]` (F32) для
    /// [`Self::decode`]. `progress(шаг, всего)` → `false` прерывает денойз.
    pub fn sample(
        &self,
        transformer: &FluxTransformer,
        cond: &FluxConditioning,
        init: Option<&Tensor>,
        p: &SampleParams,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Tensor, FluxError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let dt = self.compute;
        let (width, height) = (snap_side(p.width), snap_side(p.height));
        let (h_lat, w_lat) = (height / 8, width / 8);
        let (ph, pw) = (h_lat / 2, w_lat / 2);

        let mut rng = Philox4x32::new(p.seed);
        let noise = randn_seeded(&[1, LATENT_CHANNELS, h_lat, w_lat], dev, &mut rng)?.to_dtype(dt)?;
        let noise = pack(&noise)?;

        let sched = FlowMatchScheduler::with_config(p.steps.max(1), ph * pw, &self.scheduler);
        let n = sched.num_steps();
        let (start, mut latents) = match init {
            Some(x0) => {
                let d = x0.dims();
                if d != [1, LATENT_CHANNELS, h_lat, w_lat] {
                    return Err(FluxError::Config(format!(
                        "латент картинки {d:?} не совпадает с размером генерации {width}×{height} \
                         (ожидается [1, {LATENT_CHANNELS}, {h_lat}, {w_lat}])"
                    )));
                }
                let start = sched.start_index(p.denoise);
                if start >= n {
                    return Ok(x0.to_dtype(DType::F32)?);
                }
                let x0 = pack(&x0.to_device(dev)?.to_dtype(DType::F32)?)?;
                (start, sched.scale_noise(&x0, &noise, start)?)
            }
            // Латент держится в f32 (накопление), в трансформер идёт копия в compute.
            None => (0, noise.to_dtype(DType::F32)?),
        };

        let guidance = Tensor::from_vec(vec![p.guidance], (1,), dev)?;
        let pooled = cond.pooled.to_device(dev)?.to_dtype(dt)?;
        let t5 = cond.t5.to_device(dev)?.to_dtype(dt)?;
        let total = n - start;
        for (k, i) in (start..n).enumerate() {
            let sigma = Tensor::from_vec(vec![sched.sigma(i)], (1,), dev)?;
            let lat_in = latents.to_dtype(dt)?;
            let v = transformer.forward(&lat_in, &t5, &pooled, &sigma, &guidance, ph, pw)?;
            latents = sched.step(&v, i, &latents)?;
            if !progress(k + 1, total) && k + 1 < total {
                return Err(FluxError::Cancelled);
            }
        }
        crate::transformer::prof_dump();
        Ok(unpack(&latents, h_lat, w_lat, LATENT_CHANNELS)?.to_dtype(DType::F32)?)
    }

    /// Латент → картинка `[3, H, W]` F32 в [0, 1]. VAE — в F32 (в F16/BF16
    /// декодер переполняется) и живёт только на время декода.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, FluxError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let lat = latent
            .to_device(dev)?
            .to_dtype(DType::F32)?
            .mul_scalar(1.0 / self.vae_scaling)?
            .add_scalar(self.vae_shift)?;
        let image = {
            let vae = {
                let _g = weights_guard(dev);
                let w = ComponentWeights::from_source(&self.source, FluxComponent::Vae, dev, DType::F32)?;
                AutoencoderKlDecoder::load(&AutoencoderKlConfig::flux(), &|nm| w.get(nm))?
            };
            vae.decode(&lat)?
        };
        let image = image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
        let d = image.dims().to_vec();
        let chw = image.narrow(0, 0, 1)?.reshape(vec![d[1], d[2], d[3]])?;
        let out = chw.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?;
        drop((image, chw, lat));
        release_pools(dev);
        Ok(out)
    }

    /// Картинка `[3, H, W]` в [0, 1] → нормированный латент `[1, 16, h, w]`
    /// (среднее распределения VAE — детерминированно). Размер картинки уже
    /// должен быть кратен [`SIDE_MULTIPLE`].
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor, FluxError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let d = image.dims().to_vec();
        if d.len() != 3 || d[0] != 3 || d[1] % SIDE_MULTIPLE != 0 || d[2] % SIDE_MULTIPLE != 0 {
            return Err(FluxError::Config(format!(
                "картинка для VAE должна быть [3, H, W] со сторонами кратными {SIDE_MULTIPLE}, пришло {d:?}"
            )));
        }
        let dev = self.device;
        let x = image
            .to_device(dev)?
            .to_dtype(DType::F32)?
            .reshape(vec![1, d[0], d[1], d[2]])?
            .affine(2.0, -1.0)?;
        let enc = {
            let _g = weights_guard(dev);
            let w = ComponentWeights::from_source(&self.source, FluxComponent::Vae, dev, DType::F32)?;
            AutoencoderKlEncoder::load(&AutoencoderKlConfig::flux(), &|nm| w.get(nm))?
        };
        let moments = enc.encode(&x)?;
        let (mean, _logvar) = enc.split_moments(&moments)?;
        let latent = mean.add_scalar(-self.vae_shift)?.mul_scalar(self.vae_scaling)?;
        drop((enc, moments, mean, x));
        release_pools(dev);
        Ok(latent)
    }
}

/// `scaling_factor`/`shift_factor` из `vae/config.json`, иначе значения FLUX.1.
fn vae_factors(config: Option<&[u8]>) -> (f32, f32) {
    let v: Option<serde_json::Value> = config.and_then(|b| serde_json::from_slice(b).ok());
    let get = |k: &str, def: f32| {
        v.as_ref().and_then(|v| v.get(k)).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(def)
    };
    (get("scaling_factor", 0.3611), get("shift_factor", 0.1159))
}

/// pack `[1,16,h,w]` → `[1,(h/2)(w/2),64]` (порядок permute (0,2,4,1,3,5)).
pub(crate) fn pack(latents: &Tensor) -> Result<Tensor, FluxError> {
    let d = latents.dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    Ok(latents
        .reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute([0, 2, 4, 1, 3, 5])?
        .contiguous()?
        .reshape((b, (h / 2) * (w / 2), c * 4))?)
}

/// unpack `[1,seq,64]` → `[1,16,h,w]` (порядок permute (0,3,1,4,2,5)).
pub(crate) fn unpack(latents: &Tensor, h: usize, w: usize, c: usize) -> Result<Tensor, FluxError> {
    let b = latents.dims()[0];
    Ok(latents
        .reshape((b, h / 2, w / 2, c, 2, 2))?
        .permute([0, 3, 1, 4, 2, 5])?
        .contiguous()?
        .reshape((b, c, h, w))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_side_rounds_down_to_16() {
        assert_eq!(snap_side(1024), 1024);
        assert_eq!(snap_side(1030), 1024);
        assert_eq!(snap_side(5), 16);
    }

    #[test]
    fn vae_factors_fallback_and_config() {
        assert_eq!(vae_factors(None), (0.3611, 0.1159));
        let cfg = br#"{"scaling_factor": 0.5, "shift_factor": 0.25}"#;
        assert_eq!(vae_factors(Some(cfg)), (0.5, 0.25));
    }
}
