//! Пайплайн целиком (`VoiceConversionWrapper.convert_voice_with_streaming`):
//! токены содержания и вектор диктора → [авторегрессия манеры] → flow-matching
//! по окнам → BigVGAN → склейка окон с перекрёстным затуханием.

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::ar::{Ar, Sampling};
use crate::astral::{content_tokens, dedup, Astral, Hubert};
use crate::campplus::{fbank, CamPlus, EMBED};
use crate::cfm::{sample, CfmParams, Dit};
use crate::loader::Comp;
use crate::mel::mel_spectrogram;
use crate::nn::Rng;
use crate::regulator::CfmRegulator;
use crate::vocoder::Vocoder;
use crate::{SeedVcError, HOP, N_MELS, R, SAMPLE_RATE};

/// Предел образца голоса, секунд.
pub const MAX_REFERENCE_S: usize = 25;
/// Окно DiT (образец + кусок источника), секунд.
pub(crate) const DIT_CONTEXT_S: usize = 30;
/// Перекрытие соседних окон, мел-кадров.
pub(crate) const OVERLAP_FRAMES: usize = 16;
/// Предел условия авторегрессии (узкие токены образца + куска источника).
const AR_MAX_CONTENT: usize = 1500;

#[derive(Debug, Clone)]
pub struct SeedVcRequest {
    /// Исходная речь (моно, [−1, 1]) и её частота.
    pub source: Vec<f32>,
    pub source_rate: u32,
    /// Образец голоса (моно; используются первые 25 с) и его частота.
    pub reference: Vec<f32>,
    pub reference_rate: u32,
    /// Шагов flow-matching на окно.
    pub steps: usize,
    /// Растяжение по времени: < 1 быстрее, > 1 медленнее.
    pub length_adjust: f32,
    /// CFG по содержанию (разборчивость), 0…1.
    pub intelligibility_cfg: f32,
    /// CFG по голосу (похожесть на образец), 0…1.
    pub similarity_cfg: f32,
    /// Переносить и манеру (акцент, интонацию, эмоцию) — через авторегрессию.
    pub convert_style: bool,
    /// Только обезличить: образец не используется, голос «усреднённый».
    /// Действует вместе с `convert_style` и без него, как в эталоне.
    pub anonymization_only: bool,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SeedVcRequest {
    fn default() -> Self {
        Self {
            source: Vec::new(),
            source_rate: SAMPLE_RATE,
            reference: Vec::new(),
            reference_rate: SAMPLE_RATE,
            steps: 30,
            length_adjust: 1.0,
            intelligibility_cfg: 0.7,
            similarity_cfg: 0.7,
            convert_style: false,
            anonymization_only: false,
            top_p: 0.9,
            temperature: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

pub struct SeedVcOutput {
    /// Моно, [−1, 1].
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    /// Широкие токены, сгенерированные авторегрессией (пусто без `convert_style`).
    pub ar_tokens: Vec<u32>,
    pub content_seconds: f64,
    pub ar_seconds: f64,
    pub cfm_seconds: f64,
    pub vocoder_seconds: f64,
}

/// Этап для индикатора: (название, шаг, всего); `false` — остановить.
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize) -> bool;

/// Готовые волны: 22,05 кГц (мел) и 16 кГц (содержание, диктор).
pub struct Waves<'a> {
    pub src_22k: &'a [f32],
    pub ref_22k: &'a [f32],
    pub src_16k: &'a [f32],
    pub ref_16k: &'a [f32],
}

pub struct SeedVc {
    bundle: PathBuf,
    device: Device,
    pub hubert: Hubert,
    pub wide: Astral,
    pub narrow: Astral,
    pub campplus: CamPlus,
    pub regulator: CfmRegulator,
    pub dit: Dit,
    pub ar: Ar,
    pub vocoder: Vocoder,
}

pub(crate) fn resample(x: &[f32], from: u32, to: u32) -> R<Vec<f32>> {
    if from == to {
        return Ok(x.to_vec());
    }
    synaptix_audio::resample(x, from, to).map_err(|e| SeedVcError::Input(format!("передискретизация {from}→{to}: {e}")))
}

/// Перекрёстное затухание эталона: начало `next` смешивается с хвостом прошлого окна, веса cos².
fn crossfade(prev: &[f32], next: &mut [f32], overlap: usize) {
    let n = overlap.min(next.len()).min(prev.len());
    for i in 0..n {
        let a = if overlap > 1 { i as f64 / (overlap - 1) as f64 } else { 0.0 } * std::f64::consts::FRAC_PI_2;
        let fade_out = a.cos().powi(2);
        let fade_in = (std::f64::consts::FRAC_PI_2 - a).cos().powi(2);
        next[i] = (next[i] as f64 * fade_in + prev[prev.len() - overlap.min(prev.len()) + i] as f64 * fade_out) as f32;
    }
}

/// Склейка окон: у не-последних хвост в `overlap` сэмплов откладывается и смешивается с началом следующего.
pub(crate) struct Stitcher {
    pub(crate) out: Vec<f32>,
    pub(crate) prev: Option<Vec<f32>>,
    pub(crate) overlap: usize,
}

impl Stitcher {
    pub(crate) fn push(&mut self, mut wave: Vec<f32>, last: bool) {
        if let Some(prev) = self.prev.take() {
            crossfade(&prev, &mut wave, self.overlap);
        }
        if last || wave.len() <= self.overlap {
            self.out.extend_from_slice(&wave);
        } else {
            let cut = wave.len() - self.overlap;
            self.out.extend_from_slice(&wave[..cut]);
            self.prev = Some(wave[cut..].to_vec());
        }
    }
}

impl SeedVc {
    pub fn load(bundle: impl AsRef<Path>, device: Device) -> R<Self> {
        let bundle = bundle.as_ref().to_path_buf();
        let comp = |name: &str| Comp::open(&bundle, name, device);
        let cfm = comp("cfm")?;
        Ok(Self {
            hubert: Hubert::load(&bundle, device)?,
            wide: Astral::load(&comp("astral_wide")?)?,
            narrow: Astral::load(&comp("astral_narrow")?)?,
            campplus: CamPlus::load(&comp("campplus")?)?,
            regulator: CfmRegulator::load(&cfm)?,
            dit: Dit::load(&cfm)?,
            ar: Ar::load(&comp("ar")?)?,
            vocoder: Vocoder::load(&comp("bigvgan")?)?,
            bundle,
            device,
        })
    }

    pub fn bundle(&self) -> &Path {
        &self.bundle
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Вектор диктора `[1, 192]` по волне 16 кГц.
    pub fn style(&self, wave_16k: &[f32]) -> R<Tensor> {
        let (f, frames) = fbank(wave_16k);
        Ok(Tensor::from_vec(self.campplus.embed(&f, frames, true)?, vec![1, EMBED], self.device)?)
    }

    pub fn convert(&self, req: &SeedVcRequest, progress: Progress<'_>) -> R<SeedVcOutput> {
        if req.source.is_empty() || (req.reference.is_empty() && !req.anonymization_only) {
            return Err(SeedVcError::Input("нужны исходная речь и образец голоса".into()));
        }
        let src_22k = resample(&req.source, req.source_rate, SAMPLE_RATE)?;
        let mut ref_22k = resample(&req.reference, req.reference_rate, SAMPLE_RATE)?;
        ref_22k.truncate(SAMPLE_RATE as usize * MAX_REFERENCE_S);
        let src_16k = resample(&src_22k, SAMPLE_RATE, 16_000)?;
        let ref_16k = resample(&ref_22k, SAMPLE_RATE, 16_000)?;
        self.convert_waves(&Waves { src_22k: &src_22k, ref_22k: &ref_22k, src_16k: &src_16k, ref_16k: &ref_16k }, req, None, progress)
    }

    /// То же по готовым волнам. `noise` — шум окон `[80 × T]` вместо генератора
    /// (сверка с эталоном); поля `source`/`reference` запроса не читаются.
    pub fn convert_waves(&self, w: &Waves<'_>, req: &SeedVcRequest, noise: Option<&[Vec<f32>]>, progress: Progress<'_>) -> R<SeedVcOutput> {
        let dev = self.device;
        synaptix_core::grad::no_grad(|| -> R<SeedVcOutput> {
            let t0 = Instant::now();
            let (src_mel, src_frames) = mel_spectrogram(w.src_22k)?;
            drop(src_mel);
            let (ref_mel, ref_frames) = mel_spectrogram(w.ref_22k)?;
            let window = SAMPLE_RATE as usize / HOP * DIT_CONTEXT_S;
            if ref_frames + 2 * OVERLAP_FRAMES >= window {
                return Err(SeedVcError::Input("образец голоса слишком длинный".into()));
            }
            let prompt = Tensor::from_vec(ref_mel, vec![1, N_MELS, ref_frames], dev)?;
            if !progress("содержание", 0, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let quant: Vec<&Astral> = if req.convert_style { vec![&self.wide, &self.narrow] } else { vec![&self.wide] };
            let mut src_tok = content_tokens(&self.hubert, &quant, w.src_16k)?;
            if !progress("содержание", 1, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let mut ref_tok = content_tokens(&self.hubert, &quant, w.ref_16k)?;
            let style = self.style(w.ref_16k)?;
            let prompt_cond = self.regulator.forward(&ref_tok[0], ref_frames)?;
            if !progress("содержание", 2, 2) {
                return Err(SeedVcError::Cancelled);
            }
            let content_seconds = t0.elapsed().as_secs_f64();

            let params = CfmParams { steps: req.steps.max(1), intelligibility: req.intelligibility_cfg, similarity: req.similarity_cfg, random_voice: req.anonymization_only };
            let (mut ar_seconds, mut cfm_seconds, mut vocoder_seconds) = (0f64, 0f64, 0f64);
            let mut ar_tokens: Vec<u32> = Vec::new();
            let mut stitch = Stitcher { out: Vec::new(), prev: None, overlap: OVERLAP_FRAMES * HOP };
            let mut chunk_index = 0usize;
            // Одно окно: условие куска → мел → волна.
            let mut render = |cond: &Tensor, last: bool, stitch: &mut Stitcher, progress: &mut dyn FnMut(&str, usize, usize) -> bool, done: usize, total: usize| -> R<()> {
                let t = Instant::now();
                let mu = Tensor::cat(&[&prompt_cond, cond], 1)?.contiguous()?;
                let frames = mu.dims()[1];
                let z = match noise {
                    Some(n) => {
                        let v = n.get(chunk_index).ok_or_else(|| SeedVcError::Other("не хватает заданного шума".into()))?;
                        Tensor::from_vec(v.clone(), vec![1, N_MELS, frames], dev)?
                    }
                    None => Tensor::randn_seeded(vec![1, N_MELS, frames], req.seed.wrapping_add(chunk_index as u64 * 7919), Device::Cpu)?.to_dtype(DType::F32)?.to_device(dev)?,
                };
                chunk_index += 1;
                let mel = sample(&self.dit, &mu, &prompt, &style, &z, &params, &mut |i, n| progress("звук", done * n + i, total * n))?;
                let mel = mel.narrow(2, ref_frames, frames - ref_frames)?.contiguous()?;
                cfm_seconds += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let wave = self.vocoder.decode(&mel)?;
                vocoder_seconds += t.elapsed().as_secs_f64();
                stitch.push(wave, last);
                Ok(())
            };

            if req.convert_style {
                let (src_narrow, ref_narrow) = (dedup(&src_tok.remove(1)), dedup(&ref_tok.remove(1)));
                let (src_wide, ref_wide) = (&src_tok[0], &ref_tok[0]);
                let prefix = if req.anonymization_only { 0 } else { ref_narrow.len() };
                if prefix + 16 > AR_MAX_CONTENT {
                    return Err(SeedVcError::Input("образец голоса слишком длинный для переноса манеры".into()));
                }
                let max_chunk = AR_MAX_CONTENT - ref_narrow.len().min(AR_MAX_CONTENT - 16);
                let chunks: Vec<&[u32]> = src_narrow.chunks(max_chunk).collect();
                let sp = Sampling { top_p: req.top_p, temperature: req.temperature, repetition_penalty: req.repetition_penalty };
                let mut rng = Rng::new(req.seed);
                for (ci, chunk) in chunks.iter().enumerate() {
                    let t = Instant::now();
                    let (narrow, target): (Vec<u32>, &[u32]) = if req.anonymization_only { (chunk.to_vec(), &[]) } else { ([ref_narrow.as_slice(), chunk].concat(), ref_wide.as_slice()) };
                    let expect = (chunk.len() * src_wide.len()).div_ceil(src_narrow.len().max(1)).max(1);
                    let ar_out = self.ar.generate(&narrow, target, sp, &mut rng, &mut |n| progress("манера", n.min(expect), expect))?;
                    ar_seconds += t.elapsed().as_secs_f64();
                    ar_tokens.extend_from_slice(&ar_out);
                    let frames = (src_frames as f64 / src_wide.len() as f64 * ar_out.len() as f64 * req.length_adjust as f64) as usize;
                    let cond = self.regulator.forward(&ar_out, frames.max(1))?;
                    render(&cond, ci + 1 == chunks.len(), &mut stitch, progress, ci, chunks.len())?;
                }
            } else {
                let frames = ((src_frames as f64 * req.length_adjust as f64) as usize).max(1);
                let cond = self.regulator.forward(&src_tok[0], frames)?;
                let max_source = window - ref_frames;
                let total = if frames <= max_source { 1 } else { 1 + (frames - max_source).div_ceil(max_source - OVERLAP_FRAMES) };
                let (mut processed, mut ci) = (0usize, 0usize);
                while processed < frames {
                    let len = max_source.min(frames - processed);
                    let last = processed + max_source >= frames;
                    render(&cond.narrow(1, processed, len)?.contiguous()?, last, &mut stitch, progress, ci, total.max(ci + 1))?;
                    if last {
                        break;
                    }
                    processed += len - OVERLAP_FRAMES;
                    ci += 1;
                }
            }
            Ok(SeedVcOutput { pcm: stitch.out, sample_rate: SAMPLE_RATE, ar_tokens, content_seconds, ar_seconds, cfm_seconds, vocoder_seconds })
        })
    }
}

/// WAV 16 бит моно.
pub fn wav_bytes(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let data = pcm.len() as u32 * 2;
    let mut w = Vec::with_capacity(44 + data as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_keeps_length() {
        let ov = 4;
        let mut s = Stitcher { out: Vec::new(), prev: None, overlap: ov };
        s.push(vec![1.0; 10], false);
        s.push(vec![1.0; 10], false);
        s.push(vec![1.0; 10], true);
        // Каждое не-последнее окно отдаёт len − overlap, последнее — целиком.
        assert_eq!(s.out.len(), 6 + 6 + 10);
        assert!(s.out.iter().all(|v| (v - 1.0).abs() < 1e-6));
    }
}
