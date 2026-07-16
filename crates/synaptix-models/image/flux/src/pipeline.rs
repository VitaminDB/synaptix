//! FLUX.1-dev txt2img-пайплайн. guidance-distilled (БЕЗ CFG/negative, один
//! forward на шаг). VRAM 24GB < 34GB (полный FLUX) → компоненты грузятся и
//! ДРОПАЮТСЯ последовательно: CLIP→pooled (drop) → T5→seq (drop) → transformer
//! denoise (drop) → VAE decode (drop). Пик VRAM = transformer (23GB) < 24.

use std::path::{Path, PathBuf};

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::schedulers::randn_seeded;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder};
use synaptix_ops::rng::Philox4x32;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

use crate::config::Txt2ImgParams;
use crate::loader::ComponentWeights;
use crate::scheduler::FlowMatchScheduler;
use crate::t5::{T5Config, T5Encoder};
use crate::tokenizer::ClipTokenizer;
use crate::transformer::{FluxConfig, FluxTransformer};
use crate::FluxError;

const VAE_SCALING: f32 = 0.3611;
const VAE_SHIFT: f32 = 0.1159;
const CLIP_MAX: usize = 77;
const T5_MAX: usize = 512;

/// Режим резидент/layer-streaming для dense-весов трансформера.
/// `Auto` (default) — по свободной VRAM; `Resident` форсит резидент;
/// `Stream` форсит layer-streaming. Квант-веса всегда резидентны.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OffloadMode {
    #[default]
    Auto,
    Resident,
    Stream,
}

static OFFLOAD_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_offload_mode(mode: OffloadMode) {
    let v = match mode {
        OffloadMode::Auto => 0,
        OffloadMode::Resident => 1,
        OffloadMode::Stream => 2,
    };
    OFFLOAD_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn offload_mode() -> OffloadMode {
    match OFFLOAD_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => OffloadMode::Resident,
        2 => OffloadMode::Stream,
        _ => OffloadMode::Auto,
    }
}

/// Суммарный размер `*.safetensors` в директории (≈ footprint весов на GPU при
/// stored-dtype == compute-dtype). Для авто-offload решения по свободной VRAM.
fn dir_safetensors_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}

pub struct FluxPipeline {
    dir: PathBuf,
    device: Device,
    dtype: DType,
    /// Квант весов трансформера: NVFP4/MXFP8 → квантованный (резидентный, малый
    /// footprint); иначе (BF16/F16/F32) → плотный (compute-dtype). compute=`dtype`.
    quant: DType,
    clip_tok: ClipTokenizer,
    t5_tok: HfTokenizer,
}

impl FluxPipeline {
    pub fn from_pretrained(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, FluxError> {
        Self::from_pretrained_quant(dir, device, dtype, DType::BF16)
    }

    /// Как `from_pretrained`, но с явным квантом весов трансформера (`quant`:
    /// NVFP4/MXFP8 → квантованный; прочее → плотный compute-dtype).
    pub fn from_pretrained_quant(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        quant: DType,
    ) -> Result<Self, FluxError> {
        let dir = dir.as_ref().to_path_buf();
        let clip_tok = ClipTokenizer::from_dir(dir.join("tokenizer"))?;
        let t5_tok = HfTokenizer::from_file(dir.join("tokenizer_2").join("tokenizer.json"))
            .map_err(|e| FluxError::Tokenizer(format!("t5 tokenizer.json: {e}")))?;
        Ok(Self { dir, device, dtype, quant, clip_tok, t5_tok })
    }

    /// CLIP pooled `[1,768]` + T5 seq `[1,512,4096]`. Грузит CLIP и T5
    /// последовательно, освобождая VRAM после каждого.
    fn encode_prompt(&self, prompt: &str) -> Result<(Tensor, Tensor), FluxError> {
        let dev = self.device;
        // --- CLIP pooled ---
        let clip_ids = self.clip_tok.encode(prompt, CLIP_MAX);
        let pooled = {
            let ids = Tensor::from_vec(clip_ids, (1, CLIP_MAX), dev)?;
            let w = ComponentWeights::open_dir(self.dir.join("text_encoder"), dev, self.dtype)?;
            let enc = ClipTextEncoder::load(&ClipTextConfig::clip_l(), "text_model", &|n| w.get(n))?;
            enc.forward(&ids)?.pooled_output
        };
        // --- T5 seq ---
        let mut t5_ids: Vec<u32> = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| FluxError::Tokenizer(format!("t5 encode: {e}")))?
            .ids;
        t5_ids.truncate(T5_MAX);
        while t5_ids.len() < T5_MAX {
            t5_ids.push(0); // pad_token_id=0
        }
        let seq = {
            let ids = Tensor::from_vec(t5_ids, (1, T5_MAX), dev)?;
            let w = ComponentWeights::open_dir(self.dir.join("text_encoder_2"), dev, self.dtype)?;
            let enc = T5Encoder::load(&T5Config::xxl(), &|n| w.get(n))?;
            enc.forward(&ids)?
        };
        Ok((pooled, seq))
    }

    /// pack `[1,16,h,w]` → `[1,(h/2)(w/2),64]` (порядок permute (0,2,4,1,3,5)).
    fn pack(latents: &Tensor) -> Result<Tensor, FluxError> {
        let d = latents.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        Ok(latents
            .reshape((b, c, h / 2, 2, w / 2, 2))?
            .permute([0, 2, 4, 1, 3, 5])?
            .contiguous()?
            .reshape((b, (h / 2) * (w / 2), c * 4))?)
    }

    /// unpack `[1,seq,64]` → `[1,16,h,w]` (порядок permute (0,3,1,4,2,5)).
    fn unpack(latents: &Tensor, h: usize, w: usize, c: usize) -> Result<Tensor, FluxError> {
        let b = latents.dims()[0];
        Ok(latents
            .reshape((b, h / 2, w / 2, c, 2, 2))?
            .permute([0, 3, 1, 4, 2, 5])?
            .contiguous()?
            .reshape((b, c, h, w))?)
    }

    /// Полный txt2img. Возвращает CHW `[3,H,W]` (F32, [0,1]).
    pub fn txt2img(
        &self,
        params: &Txt2ImgParams,
        mut callback: impl FnMut(usize, usize),
    ) -> Result<Tensor, FluxError> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.device;
        let dt = self.dtype;

        let h_lat = params.latent_height();
        let w_lat = params.latent_width();
        let (ph, pw) = (h_lat / 2, w_lat / 2);

        let (pooled, t5_seq) = self.encode_prompt(&params.prompt)?;
        let shape = [1usize, 16, h_lat, w_lat];
        let mut rng = Philox4x32::new(params.seed);
        let noise = randn_seeded(&shape, dev, &mut rng)?.to_dtype(dt)?;
        let latents = Self::pack(&noise)?;
        let guidance = Tensor::from_vec(vec![params.guidance_scale], (1,), dev)?;

        let sched = FlowMatchScheduler::new(params.steps, ph * pw);
        let n = sched.num_steps();

        // Точность весов трансформера: quant=NVFP4/MXFP8 → квантуем при load (build),
        // иначе dense в compute (dt). compute активаций = dt.
        let is_quant = self.quant.is_quantized();
        crate::transformer::set_load_precision(if is_quant { self.quant } else { dt }, dt);

        let tokens = (T5_MAX + ph * pw) as u64;
        // АВТО-OFFLOAD (только dense): резидент-vs-layer-streaming по РЕАЛЬНОЙ free VRAM.
        // Резидент если памяти хватает на веса + резерв активаций; иначе частичный
        // offload (блоки [..N] на GPU, остальные стримятся). Квант резидентен всегда.
        // [`set_offload_mode`]: Stream форсит стриминг, Resident форсит резидент.
        let omode = offload_mode();
        let need_stream = if is_quant || omode == OffloadMode::Resident {
            false
        } else if omode == OffloadMode::Stream || !dev.is_cuda() {
            omode == OffloadMode::Stream
        } else {
            let ord = if let Device::Cuda(o) = dev { o } else { 0 };
            // encode_prompt дропнул T5-XXL (9.5GB) в mempool — вернём драйверу, иначе
            // mem_get_info занижает free и мы зря уходим в streaming.
            let _ = synaptix_core::device::cuda::synchronize(ord);
            synaptix_core::memory::cuda_pool::trim_cuda_mempool_device(ord).ok();
            let weight_bytes = dir_safetensors_bytes(&self.dir.join("transformer"));
            let reserve = 300_000_000u64 + tokens * 96_000;
            match synaptix_core::device::cuda::mem_info(ord) {
                Ok((free, total)) => {
                    let resident = (free as u64) >= weight_bytes + reserve;
                    eprintln!(
                        "[FLUX] auto-offload: free={:.1}GB total={:.1}GB веса≈{:.1}GB резерв={:.1}GB ({} токенов) → {}",
                        free as f64 / 1e9, total as f64 / 1e9, weight_bytes as f64 / 1e9,
                        reserve as f64 / 1e9, tokens, if resident { "РЕЗИДЕНТ" } else { "STREAMING" },
                    );
                    !resident
                }
                Err(_) => (ph * pw) > 4096,
            }
        };
        let transformer = if is_quant && dev.is_cuda() {
            // КВАНТ: грузим dense в compute-dtype на GPU, load() квантует повесово
            // (dense-вес освобождается сразу) → малый резидентный footprint (NVFP4
            // ~6GB / MXFP8 ~12GB) → стриминг не нужен даже на 2048². Квант-вес
            // бит-идентичен BF16-загрузке (build кастует в F16 перед квантизацией;
            // bf16→f16 точен), а сырые тензоры (qk-нормы) совпадают по dtype с
            // F16-активациями — иначе rms_norm теряла fused-ядро (×15 на qk_norm).
            eprintln!("[FLUX] quant={:?} compute={dt:?}: квантую трансформер резидентно", self.quant);
            let w = ComponentWeights::open_dir(self.dir.join("transformer"), dev, dt)?;
            FluxTransformer::load(&FluxConfig::dev(), &|nm| w.get(nm))?
        } else if dev.is_cuda() && need_stream {
            // Частичный offload: блоки на GPU пока free>min_free (запас под пик
            // активаций + транзиент стримящегося блока), остальные стримятся.
            let min_free = 1_800_000_000u64 + tokens * 220_000;
            let w = ComponentWeights::open_dir(self.dir.join("transformer"), Device::Cpu, dt)?;
            FluxTransformer::load(&FluxConfig::dev(), &|nm| w.get(nm))?
                .into_partial_streaming(dev, min_free)?
        } else {
            let load_dev = if dev.is_cuda() { dev } else { Device::Cpu };
            let w = ComponentWeights::open_dir(self.dir.join("transformer"), load_dev, dt)?;
            FluxTransformer::load(&FluxConfig::dev(), &|nm| w.get(nm))?
        };
        // латент держится в f32 (накопление); в трансформер подаётся bf16-копия.
        let mut latents = latents.to_dtype(DType::F32)?;
        for i in 0..n {
            let sigma = Tensor::from_vec(vec![sched.sigma(i)], (1,), dev)?;
            let lat_in = latents.to_dtype(dt)?;
            let noise_pred =
                transformer.forward(&lat_in, &t5_seq, &pooled, &sigma, &guidance, ph, pw)?;
            latents = sched.step(&noise_pred, i, &latents)?;
            callback(i + 1, n);
        }
        crate::transformer::prof_dump();
        drop(transformer);

        // unpack + денорм + VAE decode
        let lat = Self::unpack(&latents, h_lat, w_lat, 16)?
            .to_dtype(DType::F32)?
            .mul_scalar(1.0 / VAE_SCALING)?
            .add_scalar(VAE_SHIFT)?;
        let image = {
            let w = ComponentWeights::open_dir(self.dir.join("vae"), dev, DType::F32)?;
            let vae = AutoencoderKlDecoder::load(&AutoencoderKlConfig::flux(), &|nm| w.get(nm))?;
            vae.decode(&lat)?
        };
        // postprocess: (x*0.5+0.5).clamp(0,1) → [3,H,W]
        let image = image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
        let d = image.dims().to_vec();
        let chw = image.narrow(0, 0, 1)?.reshape(vec![d[1], d[2], d[3]])?;
        Ok(chw.contiguous()?.to_dtype(DType::F32)?)
    }
}
