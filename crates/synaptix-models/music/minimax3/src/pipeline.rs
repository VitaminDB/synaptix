//! Пайплайн целиком: промпт → авторегрессия → flow-matching → декодер → волна.
//! Этапы грузятся по очереди (LM 8B и трансформер вместе на карту 24 ГБ не
//! помещаются) и освобождают память перед следующим.

use std::path::{Path, PathBuf};

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::ar::Ar;
use crate::flow::{ConditionEncoder, FlowTransformer};
use crate::loader::{read_file, Comp};
use crate::prompt;
use crate::vocoder::{Vocoder, HOP, SAMPLE_RATE};
use crate::{Music3Error, R};

/// Кадров LM в секунду.
pub const FRAME_RATE: f64 = 25.0;
/// Обрезка окон при склейке, в латентах: слева у не-первых, справа у не-последних.
const CROP_LEFT: usize = 86;
const CROP_RIGHT: usize = 344 - 86;

#[derive(Debug, Clone)]
pub struct Music3Request {
    /// Описание музыки: жанр, темп, вокал, инструменты, аранжировка.
    pub description: String,
    /// Текст песни; теги `[verse]`, `[chorus]` — отдельными строками.
    pub lyrics: String,
    /// Верхняя граница длины, с (модель может закончить раньше; предел — 6 минут).
    pub seconds: f64,
    pub seed: u64,
    /// Шагов flow-matching на окно (эталон — 30).
    pub steps: usize,
}

impl Default for Music3Request {
    fn default() -> Self {
        Self { description: String::new(), lyrics: String::new(), seconds: 60.0, seed: 0, steps: 30 }
    }
}

pub struct Music3Output {
    /// Стерео, перемежённые отсчёты в [−1, 1].
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Сгенерировано кадров LM (25 в секунду).
    pub frames: usize,
    pub ar_seconds: f64,
    pub flow_seconds: f64,
    pub decode_seconds: f64,
}

/// Этап для индикатора: (название, шаг, всего).
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize) -> bool;

pub struct Music3 {
    bundle: PathBuf,
    device: Device,
    tokenizer: HfTokenizer,
    /// Квант весов глобальной LM (`None` — как лежит в бандле).
    pub quant: Option<DType>,
}

fn trim(device: Device) {
    if let Device::Cuda(o) = device {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
    }
}

impl Music3 {
    pub fn open(bundle: impl AsRef<Path>, device: Device) -> R<Self> {
        let bundle = bundle.as_ref().to_path_buf();
        if !device.is_cuda() {
            return Err(Music3Error::Load("MiniMax Music 3 работает только на видеокарте".into()));
        }
        let tokenizer = HfTokenizer::from_bytes(&read_file(&bundle, "tokenizer/tokenizer.json")?).map_err(|e| Music3Error::Load(format!("токенизатор: {e}")))?;
        Ok(Self { bundle, device, tokenizer, quant: None })
    }

    /// Токены промпта: условная и безусловная ветки.
    pub fn prompt_ids(&self, description: &str, lyrics: &str) -> R<(Vec<u32>, Vec<u32>)> {
        if description.trim().is_empty() || lyrics.trim().is_empty() {
            return Err(Music3Error::Input("нужны и описание музыки, и текст песни".into()));
        }
        let ids = self.tokenizer.encode(&prompt::assemble(description, lyrics), false).map_err(|e| Music3Error::Input(format!("токенизация: {e}")))?.ids;
        if ids.len() > prompt::MAX_PROMPT_TOKENS {
            return Err(Music3Error::Input(format!("промпт {} токенов, предел {}", ids.len(), prompt::MAX_PROMPT_TOKENS)));
        }
        if ids.len() < 4 {
            return Err(Music3Error::Input("промпт слишком короткий".into()));
        }
        let un = prompt::unconditional(&ids);
        Ok((ids, un))
    }

    pub fn generate(&self, req: &Music3Request, progress: Progress<'_>) -> R<Music3Output> {
        let (cond, uncond) = self.prompt_ids(&req.description, &req.lyrics)?;
        let max_frames = ((req.seconds * FRAME_RATE) as usize).clamp(1, prompt::MAX_FRAMES);
        let dev = self.device;
        synaptix_core::grad::no_grad(|| -> R<Music3Output> {
            // 1. Авторегрессия.
            let _status = synaptix_core::gen_status::Scope::new("загрузка языковой модели");
            let t0 = std::time::Instant::now();
            let frames = {
                let ar = Ar::open(&self.bundle, dev, self.quant)?;
                synaptix_core::gen_status::phase("сочинение (языковая модель)");
                ar.generate(&cond, &uncond, max_frames, req.seed, &mut |i, n| progress("сочинение", i, n))?
            };
            trim(dev);
            let ar_seconds = t0.elapsed().as_secs_f64();
            let n_frames = frames.dims()[1];
            // 2. Flow-matching по окнам.
            synaptix_core::gen_status::phase("звук (flow-matching)");
            let t1 = std::time::Instant::now();
            let chunks = {
                let enc = ConditionEncoder::load(&Comp::open(&self.bundle, "condition_encoder", dev)?, dev)?;
                let tr = FlowTransformer::load(&Comp::open(&self.bundle, "transformer", dev)?, dev)?;
                crate::flow::sample(&enc, &tr, &frames, req.steps.max(1), req.seed.wrapping_add(1), &mut |i, n| progress("звук", i, n))?
            };
            drop(frames);
            trim(dev);
            let flow_seconds = t1.elapsed().as_secs_f64();
            // 3. Декодер и склейка окон.
            synaptix_core::gen_status::phase("декодер");
            let t2 = std::time::Instant::now();
            let voc = Vocoder::load(&Comp::open(&self.bundle, "vocoder", dev)?)?;
            let (mut left, mut right): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
            let n = chunks.len();
            for (i, lat) in chunks.iter().enumerate() {
                let l = lat.dims()[0];
                let x = lat.transpose(0, 1)?.contiguous()?.reshape(vec![1, crate::flow::LATENT, l])?;
                let wave = voc.decode(&x)?.to_device(Device::Cpu)?; // [2, L·512]
                let total = wave.dims()[1];
                let a = if i == 0 { 0 } else { CROP_LEFT * HOP };
                let b = total.saturating_sub(if i + 1 == n { 0 } else { CROP_RIGHT * HOP });
                if b > a {
                    let v = wave.flatten_all()?.to_vec1::<f32>()?;
                    left.extend_from_slice(&v[a..b]);
                    right.extend_from_slice(&v[total + a..total + b]);
                }
                if !progress("декодер", i + 1, n) {
                    return Err(Music3Error::Cancelled);
                }
            }
            drop(voc);
            trim(dev);
            let pcm: Vec<f32> = left.iter().zip(&right).flat_map(|(l, r)| [l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0)]).collect();
            Ok(Music3Output { pcm, sample_rate: SAMPLE_RATE, channels: 2, frames: n_frames, ar_seconds, flow_seconds, decode_seconds: t2.elapsed().as_secs_f64() })
        })
    }
}

/// WAV 16 бит из перемежённых отсчётов.
pub fn wav_bytes(pcm: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data = pcm.len() as u32 * 2;
    let mut w = Vec::with_capacity(44 + data as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
    w.extend_from_slice(&(channels * 2).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    w
}

#[allow(dead_code)]
fn _unused(_: &Tensor) {}
