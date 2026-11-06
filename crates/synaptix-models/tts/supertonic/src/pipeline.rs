use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

use crate::model::{DurationPredictor, TextEncoder, VectorField, Vocoder, Weights};
use crate::{noise, Result, StError, SAMPLE_RATE};

/// Готовый голос (стиль): `style_ttl` `[50, 256]`, `style_dp` `[8, 16]`.
#[derive(Debug, Clone)]
pub struct Voice {
    pub ttl: Vec<f32>,
    pub dp: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Options {
    /// Шаги Эйлера (upstream — 8).
    pub steps: usize,
    /// Темп (длительность делится на него; upstream — 1.05).
    pub speed: f32,
    pub seed: u64,
    /// Пауза между кусками длинного текста, с.
    pub silence: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self { steps: 8, speed: 1.05, seed: 0, silence: 0.3 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub chunks: usize,
    pub audio_s: f64,
    pub dp_te_s: f64,
    pub flow_s: f64,
    pub vocoder_s: f64,
}

pub struct SupertonicPipeline {
    dp: DurationPredictor,
    te: TextEncoder,
    vf: VectorField,
    voc: Vocoder,
    indexer: Vec<i64>,
    vocab: usize,
    voices: BTreeMap<String, Voice>,
    device: Device,
    last: std::sync::Mutex<Timings>,
}

fn parse_voice(v: &serde_json::Value) -> Option<Voice> {
    let flat = |x: &serde_json::Value| -> Vec<f32> {
        fn walk(x: &serde_json::Value, out: &mut Vec<f32>) {
            match x {
                serde_json::Value::Array(a) => a.iter().for_each(|y| walk(y, out)),
                serde_json::Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0) as f32),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(x, &mut out);
        out
    };
    let ttl = flat(v.get("style_ttl")?.get("data")?);
    let dp = flat(v.get("style_dp")?.get("data")?);
    (ttl.len() == 50 * 256 && dp.len() == 8 * 16).then_some(Voice { ttl, dp })
}

impl SupertonicPipeline {
    /// Вычисления — в F32 (модель ~100M; как оригинал в onnxruntime).
    pub fn from_syn(path: &Path, device: Device) -> Result<Self> {
        let b = Bundle::open(path).map_err(|e| StError::Load(e.to_string()))?;
        let read = |n: &str| b.read_file(n).map(|c| c.into_owned()).map_err(|e| StError::Load(format!("{n}: {e}")));
        let indexer: Vec<i64> = serde_json::from_slice(&read("unicode_indexer.json")?).map_err(|e| StError::Load(e.to_string()))?;
        let vj: serde_json::Value = serde_json::from_slice(&read("voices.json")?).map_err(|e| StError::Load(e.to_string()))?;
        let voices: BTreeMap<String, Voice> = vj.as_object().map(|o| o.iter().filter_map(|(k, v)| parse_voice(v).map(|x| (k.clone(), x))).collect()).unwrap_or_default();
        let w = |c: &str| Weights::open(path, c, device);
        let dpw = w("duration_predictor")?;
        let dp = DurationPredictor::load(&dpw)?;
        let tew = w("text_encoder")?;
        let te = TextEncoder::load(&tew)?;
        let vocab = tew.find("text_embedder.char_embedder.weight")?.dims()[0];
        let vf = VectorField::load(&w("vector_estimator")?)?;
        let voc = Vocoder::load(&w("vocoder")?)?;
        Ok(Self { dp, te, vf, voc, indexer, vocab, voices, device, last: Default::default() })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    pub fn voice(&self, name: &str) -> Option<&Voice> {
        self.voices.get(name).or_else(|| self.voices.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v))
    }

    pub fn last_timings(&self) -> Timings {
        self.last.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Индексы символов текста (с предобработкой и тегами языка).
    pub fn text_ids(&self, text: &str, lang: &str) -> Result<Vec<i64>> {
        let t = crate::text::preprocess(text, lang).map_err(StError::Other)?;
        Ok(crate::text::ids(&t, &self.indexer, self.vocab))
    }

    /// Один кусок: (PCM, длительность по DP после деления на темп, латент).
    pub fn infer(&self, text: &str, lang: &str, voice: &Voice, o: &Options) -> Result<(Vec<f32>, f32)> {
        let t0 = Instant::now();
        let ids = self.text_ids(text, lang)?;
        let n = ids.len();
        let ids = Tensor::from_vec(ids, vec![n], self.device)?;
        let dp_style = Tensor::from_vec(voice.dp.clone(), vec![1usize, 8, 16], self.device)?;
        let ttl = Tensor::from_vec(voice.ttl.clone(), vec![1usize, 50, 256], self.device)?;
        let dur = self.dp.forward(&ids, &dp_style)? / o.speed.max(0.05);
        let tdp = t0.elapsed().as_secs_f64();
        let text = self.te.forward(&ids, &ttl)?;
        let _ = text.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<f32>()?;
        let t1 = Instant::now();
        if std::env::var_os("SYN_ST_PROF").is_some() {
            eprintln!("[supertonic] символов {n}: DP {:.3} с, текст {:.3} с", tdp, (t1 - t0).as_secs_f64() - tdp);
        }
        // как upstream: латент = ceil(dur·sr / 3072) (в f32), шум в порядке C
        let wav_len = dur * SAMPLE_RATE as f32;
        let l = (((wav_len + 3072.0 - 1.0) / 3072.0) as usize).max(1);
        let mut x = Tensor::from_vec(noise(o.seed, 144 * l), vec![1usize, 144, l], self.device)?;
        for s in 0..o.steps {
            x = self.vf.step(&x, &text, &ttl, s, o.steps)?;
        }
        let t2 = Instant::now();
        let mut pcm = self.voc.forward(&x)?;
        pcm.truncate((SAMPLE_RATE as f32 * dur) as usize);
        if let Ok(mut t) = self.last.lock() {
            t.dp_te_s += (t1 - t0).as_secs_f64();
            t.flow_s += (t2 - t1).as_secs_f64();
            t.vocoder_s += t2.elapsed().as_secs_f64();
        }
        Ok((pcm, dur))
    }

    /// Отладка: промежуточные результаты одного куска (длительность, text_emb,
    /// латент после каждого шага, PCM).
    pub fn debug_parts(&self, text: &str, lang: &str, voice: &Voice, o: &Options) -> Result<(f32, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
        let ids = self.text_ids(text, lang)?;
        let n = ids.len();
        let ids = Tensor::from_vec(ids, vec![n], self.device)?;
        let dp_style = Tensor::from_vec(voice.dp.clone(), vec![1usize, 8, 16], self.device)?;
        let ttl = Tensor::from_vec(voice.ttl.clone(), vec![1usize, 50, 256], self.device)?;
        let dur = self.dp.forward(&ids, &dp_style)? / o.speed.max(0.05);
        let text = self.te.forward(&ids, &ttl)?;
        let l = (((dur * SAMPLE_RATE as f32 + 3071.0) / 3072.0) as usize).max(1);
        let mut x = Tensor::from_vec(noise(o.seed, 144 * l), vec![1usize, 144, l], self.device)?;
        let mut steps = Vec::new();
        for s in 0..o.steps {
            x = self.vf.step(&x, &text, &ttl, s, o.steps)?;
            steps.push(x.flatten_all()?.to_vec1::<f32>()?);
        }
        let pcm = self.voc.forward(&x)?;
        Ok((dur, text.flatten_all()?.to_vec1::<f32>()?, steps, pcm))
    }

    /// Текст любой длины: куски по предложениям (300 символов, ko/ja — 120),
    /// между ними — тишина.
    pub fn synthesize(&self, text: &str, lang: &str, voice: &Voice, o: &Options) -> Result<Vec<f32>> {
        if let Ok(mut t) = self.last.lock() {
            *t = Timings::default();
        }
        let max_len = if matches!(lang, "ko" | "ja") { 120 } else { 300 };
        let chunks = crate::text::chunks(text, max_len);
        if chunks.is_empty() {
            return Err(StError::Other("пустой текст".into()));
        }
        let mut out = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            if i > 0 {
                out.extend(std::iter::repeat_n(0.0, (o.silence * SAMPLE_RATE as f32) as usize));
            }
            out.extend(self.infer(c, lang, voice, o)?.0);
        }
        if let Ok(mut t) = self.last.lock() {
            t.chunks = chunks.len();
            t.audio_s = out.len() as f64 / SAMPLE_RATE as f64;
        }
        Ok(out)
    }
}
