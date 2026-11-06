//! Синтез F5-TTS как `utils_infer.infer_process`: образец (≤ 12 с, громкость до RMS 0.1), текст
//! образца + новый текст → символы словаря; длина = образец + пропорция байтов текста; Эйлер по
//! `t` (sway-сэмплинг, CFG); мел после образца → Vocos; длинный текст — кусками с переходом 0,15 с.

use std::path::Path;
use std::time::Instant;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;

use crate::dit::{Cfg, Dit, W};
use crate::text::{chunk_text, to_tokens, Vocab};
use crate::vocos::Vocos;
use crate::{log_mel, noise, F5Error, Result, HOP, N_MELS, SAMPLE_RATE};

const TARGET_RMS: f32 = 0.1;

#[derive(Debug, Clone)]
pub struct Options {
    /// Шагов Эйлера (upstream — 32; 5/6/7/10/12/16 — расписание EPSS).
    pub nfe: usize,
    pub cfg: f32,
    pub sway: f32,
    pub speed: f32,
    pub seed: u64,
    /// Переход между кусками длинного текста, с.
    pub cross_fade: f32,
    /// Принудительная длительность всей фразы (с образцом), с.
    pub fix_duration: Option<f32>,
}

impl Default for Options {
    fn default() -> Self {
        Self { nfe: 32, cfg: 2.0, sway: -1.0, speed: 1.0, seed: 0, cross_fade: 0.15, fix_duration: None }
    }
}

/// Голос: образец 24 кГц (после нормировки громкости) и его текст.
#[derive(Debug, Clone)]
pub struct VoicePrompt {
    pub audio: Vec<f32>,
    pub text: String,
    /// Исходная RMS (выход возвращается к ней, если она была ниже 0.1).
    pub rms: f32,
}

impl VoicePrompt {
    /// Как `preprocess_ref_audio_text` + начало `infer_batch_process`: не длиннее 12 с, тишина по краям
    /// (−50 дБ) срезается, +50 мс тишины в конце; текст заканчивается на «. »; громкость до RMS 0.1.
    pub fn new(pcm: &[f32], sr: u32, text: &str) -> Result<Self> {
        let mut a = if sr == SAMPLE_RATE {
            pcm.to_vec()
        } else {
            synaptix_audio::resample::resample(pcm, sr, SAMPLE_RATE).map_err(|e| F5Error::Other(format!("ресемпл: {e}")))?
        };
        a.truncate(12 * SAMPLE_RATE as usize);
        trim_silence(&mut a);
        a.extend(std::iter::repeat_n(0.0, SAMPLE_RATE as usize / 20));
        Ok(Self::prepared(a, text))
    }

    /// Без обрезки (как эталон `f5_ref.py`): только текст и громкость.
    pub fn prepared(mut audio: Vec<f32>, text: &str) -> Self {
        let mut t = text.trim_end_matches(' ').to_string();
        if t.is_empty() {
            t = text.to_string();
        }
        if !t.ends_with('。') {
            if t.ends_with('.') {
                t.push(' ');
            } else {
                t.push_str(". ");
            }
        }
        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        if rms < TARGET_RMS && rms > 0.0 {
            audio.iter_mut().for_each(|v| *v *= TARGET_RMS / rms);
        }
        Self { audio, text: t, rms }
    }

    pub fn seconds(&self) -> f32 {
        self.audio.len() as f32 / SAMPLE_RATE as f32
    }
}

/// `remove_silence_edges` pydub: шагами по 10 мс, тихо — ниже −50 дБ полной шкалы.
fn trim_silence(a: &mut Vec<f32>) {
    let step = SAMPLE_RATE as usize / 100;
    let thr = 10f32.powf(-50.0 / 20.0);
    let loud = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt() >= thr;
    let mut start = 0;
    while start + step <= a.len() && !loud(&a[start..start + step]) {
        start += step;
    }
    let mut end = a.len();
    while end >= start + step && !loud(&a[end - step..end]) {
        end -= step;
    }
    if start < end {
        a.truncate(end);
        a.drain(..start);
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub chunks: usize,
    pub audio_s: f64,
    pub dit_s: f64,
    pub vocoder_s: f64,
}

pub struct F5Pipeline {
    dit: Dit,
    vocos: Vocos,
    vocab: Vocab,
    pub device: Device,
    /// Описание версии из `config.json` бандла.
    pub config: serde_json::Value,
    last: std::sync::Mutex<Timings>,
}

/// Расписание времени: EPSS для малых NFE, иначе равномерное; затем sway.
pub fn timesteps(nfe: usize, sway: f32) -> Vec<f32> {
    let epss: &[u32] = match nfe {
        5 => &[0, 2, 4, 8, 16, 32],
        6 => &[0, 2, 4, 6, 8, 16, 32],
        7 => &[0, 2, 4, 6, 8, 16, 24, 32],
        10 => &[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12 => &[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        _ => &[],
    };
    let t: Vec<f32> = if epss.is_empty() { (0..=nfe).map(|i| i as f32 / nfe as f32).collect() } else { epss.iter().map(|&k| k as f32 / 32.0).collect() };
    t.into_iter().map(|t| t + sway * ((std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t)).collect()
}

impl F5Pipeline {
    /// `dtype` — счётный тип DiT (по умолчанию F16 на CUDA, F32 на CPU); вокодер — F32.
    pub fn from_syn(path: &Path, device: Device, dtype: Option<DType>, quant: Option<DType>) -> Result<Self> {
        if !path.exists() {
            return Err(F5Error::Load(format!("нет файла {}", path.display())));
        }
        let b = Bundle::open(path).map_err(|e| F5Error::Load(e.to_string()))?;
        let read = |n: &str| b.read_file(n).map(|c| c.into_owned()).map_err(|e| F5Error::Load(format!("{n}: {e}")));
        let vocab = Vocab::parse(&String::from_utf8_lossy(&read("vocab.txt")?));
        let config: serde_json::Value = read("config.json").ok().and_then(|c| serde_json::from_slice(&c).ok()).unwrap_or_default();
        let open = |c: &str| -> Result<SynBundleLoader> { Ok(SynBundleLoader::open(path).map_err(|e| F5Error::Load(e.to_string()))?.with_component(c).with_device(device)) };
        let dtype = dtype.unwrap_or(if device.is_cuda() { DType::F16 } else { DType::F32 });
        let dl = open("dit")?;
        let dit = Dit::load(&W { l: &dl, prefix: "", device, dtype, quant }, Cfg::default())?;
        let vl = open("vocoder")?;
        let vocos = Vocos::load(&W { l: &vl, prefix: "", device, dtype: DType::F32, quant: None })?;
        Ok(Self { dit, vocos, vocab, device, config, last: Default::default() })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn last_timings(&self) -> Timings {
        self.last.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Индексы символов `ref_text + gen` (как `list_str_to_idx`).
    pub fn text_ids(&self, ref_text: &str, gen: &str) -> Vec<i64> {
        self.vocab.ids(&to_tokens(&format!("{ref_text}{gen}")))
    }

    /// Мел образца `[кадр][100]` (как `MelSpec` vocos).
    pub fn ref_mel(voice: &VoicePrompt) -> Vec<Vec<f32>> {
        log_mel(&voice.audio)
    }

    /// Один кусок → (мел после образца `[кадр][100]`, длительность всего в кадрах).
    pub fn sample(&self, voice: &VoicePrompt, gen: &str, o: &Options) -> Result<(Vec<Vec<f32>>, usize)> {
        let mut ref_text = voice.text.clone();
        if ref_text.chars().last().is_some_and(|c| c.len_utf8() == 1) && !ref_text.ends_with(' ') {
            ref_text.push(' ');
        }
        let speed = o.speed * if gen.len() < 10 { 0.3 } else { 1.0 };
        let ids = self.text_ids(&ref_text, gen);
        let ref_len = voice.audio.len() / HOP;
        let mut duration = match o.fix_duration {
            Some(s) => (s * SAMPLE_RATE as f32 / HOP as f32) as usize,
            None => ref_len + (ref_len as f64 / ref_text.len().max(1) as f64 * gen.len() as f64 / speed as f64) as usize,
        };
        let cond_mel = Self::ref_mel(voice);
        let cond_len = cond_mel.len();
        duration = duration.max(ids.len().max(cond_len) + 1).min(65536);
        let n = duration;
        // cond: мел образца, дальше нули
        let mut cond = vec![0f32; n * N_MELS];
        for (i, r) in cond_mel.iter().enumerate().take(n) {
            cond[i * N_MELS..(i + 1) * N_MELS].copy_from_slice(r);
        }
        let dev = self.device;
        let dt = self.dit.dtype;
        let cond_t = Tensor::from_vec(cond.clone(), vec![1usize, n, N_MELS], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?;
        let pre = self.dit.prepare(&ids, n)?;
        let mut x = Tensor::from_vec(noise(o.seed, n * N_MELS), vec![1usize, n, N_MELS], dev)?;
        let ts = timesteps(o.nfe, o.sway);
        for w in ts.windows(2) {
            let (c, u) = self.dit.forward(&x, &cond_t, &pre, w[0])?;
            let v = if o.cfg < 1e-5 { c } else { c.add(&c.sub(&u)?.affine(o.cfg, 0.0)?)? };
            x = x.add(&v.affine(w[1] - w[0], 0.0)?)?;
        }
        let out = x.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        // кадры образца — исходный мел; отдаётся всё после ref_len (как upstream: с кадра ref_len)
        let mel = (ref_len..n)
            .map(|i| if i < cond_len { cond[i * N_MELS..(i + 1) * N_MELS].to_vec() } else { out[i * N_MELS..(i + 1) * N_MELS].to_vec() })
            .collect();
        Ok((mel, n))
    }

    /// Один кусок → звук 24 кГц.
    pub fn infer(&self, voice: &VoicePrompt, gen: &str, o: &Options) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let (mel, _) = self.sample(voice, gen, o)?;
        let t1 = Instant::now();
        let mut wave = self.vocos.decode(&mel)?;
        if voice.rms < TARGET_RMS {
            wave.iter_mut().for_each(|v| *v *= voice.rms / TARGET_RMS);
        }
        if let Ok(mut t) = self.last.lock() {
            t.dit_s += (t1 - t0).as_secs_f64();
            t.vocoder_s += t1.elapsed().as_secs_f64();
        }
        Ok(wave)
    }

    /// Вокодер отдельно (для сверки).
    pub fn vocode(&self, mel: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.vocos.decode(mel)
    }

    /// Текст любой длины: куски по `max_chars = байт/с образца · (22 − длина образца) · темп`,
    /// склейка с линейным переходом.
    pub fn synthesize(&self, voice: &VoicePrompt, text: &str, o: &Options) -> Result<Vec<f32>> {
        if let Ok(mut t) = self.last.lock() {
            *t = Timings::default();
        }
        let sec = voice.seconds().max(0.1);
        let max_chars = ((voice.text.len() as f32 / sec) * (22.0 - sec).max(1.0) * o.speed) as usize;
        let chunks = chunk_text(text.trim(), max_chars.max(1));
        if chunks.is_empty() {
            return Err(F5Error::Other("пустой текст".into()));
        }
        let mut out: Vec<f32> = Vec::new();
        for c in &chunks {
            let w = self.infer(voice, c, o)?;
            let k = ((o.cross_fade * SAMPLE_RATE as f32) as usize).min(out.len()).min(w.len());
            if k == 0 {
                out.extend(w);
                continue;
            }
            let base = out.len() - k;
            for i in 0..k {
                let f = if k > 1 { i as f32 / (k - 1) as f32 } else { 1.0 };
                out[base + i] = out[base + i] * (1.0 - f) + w[i] * f;
            }
            out.extend_from_slice(&w[k..]);
        }
        if let Ok(mut t) = self.last.lock() {
            t.chunks = chunks.len();
            t.audio_s = out.len() as f64 / SAMPLE_RATE as f64;
        }
        Ok(out)
    }
}
