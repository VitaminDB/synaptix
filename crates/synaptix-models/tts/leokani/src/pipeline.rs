//! Текст → речь: G2P Leokani (куски-клаузы) → id фонем (`[0, ids…, 0]`, куски > 400 id делятся
//! у середины по запятой или пробелу) → акустика → вокодер; куски склеиваются без пауз, как upstream.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::g2p::LeokaniG2P;
use crate::model::{Acoustic, Vocoder, W, SR};
use crate::{model_clusters, LkError, Result};

const MAX_IDS: usize = 400;

struct Voice {
    acoustic: Acoustic,
    vocoder: Vocoder,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub chunks: usize,
    pub audio_s: f64,
    pub g2p_s: f64,
    pub model_s: f64,
}

pub struct LeokaniPipeline {
    g2p: LeokaniG2P,
    vocab: BTreeMap<String, i64>,
    voices: BTreeMap<String, Voice>,
    last: std::sync::Mutex<Timings>,
}

impl LeokaniPipeline {
    /// Голоса — компоненты бандла `dala` (женский) и `pono` (мужской); G2P — на CPU.
    pub fn from_syn(path: &Path, device: Device) -> Result<Self> {
        let b = synaptix_bundle::Bundle::open(path).map_err(|e| LkError::Load(e.to_string()))?;
        let read = |n: &str| b.read_file(n).map(|c| c.into_owned()).map_err(|e| LkError::Load(format!("{n}: {e}")));
        let cfg: serde_json::Value = serde_json::from_slice(&read("config.json")?).map_err(|e| LkError::Load(e.to_string()))?;
        let vocab: BTreeMap<String, i64> = cfg["vocab"].as_object().ok_or_else(|| LkError::Load("config.json без vocab".into()))?.iter().filter_map(|(k, v)| v.as_i64().map(|i| (k.clone(), i))).collect();
        let n_mels = cfg["acoustic"]["n_mels"].as_u64().unwrap_or(100) as usize;
        let blocks = cfg["vocoder"]["blocks"].as_u64().unwrap_or(8) as usize;
        let open = |c: &str, d: Device| -> Result<SynBundleLoader> { Ok(SynBundleLoader::open(path).map_err(|e| LkError::Load(e.to_string()))?.with_component(c).with_device(d)) };
        // Сети G2P — компоненты g2p_nets_{context,enc,dec}; имена весов с префиксом сети.
        let nets: BTreeMap<&str, SynBundleLoader> = [("context", "g2p_nets_context"), ("enc", "g2p_nets_enc"), ("dec", "g2p_nets_dec")]
            .into_iter()
            .map(|(p, c)| Ok((p, open(c, Device::Cpu)?)))
            .collect::<Result<_>>()?;
        let get = |n: &str| -> Result<synaptix_core::tensor::Tensor> {
            let l = n.split('.').next().and_then(|p| nets.get(p)).ok_or_else(|| LkError::Load(format!("{n}: нет сети G2P")))?;
            l.load_to(n, Device::Cpu, DType::F32).map_err(|e| LkError::Load(format!("{n}: {e}")))
        };
        let g2p = LeokaniG2P::new(&read("leokani-g2p.json")?, &get)?;
        let mut voices = BTreeMap::new();
        for name in ["dala", "pono"] {
            let l = open(name, device)?;
            if l.names().is_empty() {
                continue;
            }
            let w = W { l: &l, prefix: String::new(), device };
            voices.insert(name.to_string(), Voice { acoustic: Acoustic::load(&w, n_mels)?, vocoder: Vocoder::load(&w, blocks)? });
        }
        if voices.is_empty() {
            return Err(LkError::Load("в бандле нет голосов (dala/pono)".into()));
        }
        Ok(Self { g2p, vocab, voices, last: Default::default() })
    }

    pub fn sample_rate(&self) -> u32 {
        SR
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    pub fn last_timings(&self) -> Timings {
        self.last.lock().map(|t| t.clone()).unwrap_or_default()
    }

    pub fn g2p(&self) -> &LeokaniG2P {
        &self.g2p
    }

    pub fn ids(&self, phonemes: &str) -> Vec<i64> {
        model_clusters(phonemes).iter().filter_map(|c| self.vocab.get(c).copied()).collect()
    }

    /// Куски длиннее 400 id — у середины, по запятой, иначе по пробелу (`Leokani.split`).
    pub fn split(&self, ids: &[i64]) -> Vec<Vec<i64>> {
        if ids.is_empty() {
            return Vec::new();
        }
        if ids.len() <= MAX_IDS {
            return vec![ids.to_vec()];
        }
        let mid = ids.len() / 2;
        for mark in [",", " "] {
            let Some(&m) = self.vocab.get(mark) else { continue };
            if let Some(i) = ids.iter().enumerate().filter(|(_, &x)| x == m).map(|(i, _)| i).min_by_key(|&i| (i as i64 - mid as i64).abs()) {
                let k = i + 1;
                if k > 0 && k < ids.len() {
                    let mut out = self.split(&ids[..k]);
                    out.extend(self.split(&ids[k..]));
                    return out;
                }
            }
        }
        let mut out = self.split(&ids[..mid]);
        out.extend(self.split(&ids[mid..]));
        out
    }

    /// Готовые фонемы Kokoro → PCM.
    pub fn synthesize_phonemes(&self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let v = self.voice(voice)?;
        let mut out = Vec::new();
        for part in self.split(&self.ids(phonemes)) {
            out.extend(self.run(v, &part, speed)?);
        }
        Ok(out)
    }

    fn voice(&self, name: &str) -> Result<&Voice> {
        let key = name.trim().to_ascii_lowercase();
        let key = if key.is_empty() { "dala".to_string() } else { key };
        self.voices.get(&key).ok_or_else(|| LkError::Other(format!("голос «{name}»: есть {}", self.voice_names().join(", "))))
    }

    fn run(&self, v: &Voice, ids: &[i64], speed: f32) -> Result<Vec<f32>> {
        let mut x = Vec::with_capacity(ids.len() + 2);
        x.push(0);
        x.extend_from_slice(ids);
        x.push(0);
        let mel = v.acoustic.forward(&x, speed)?;
        v.vocoder.forward(&mel)
    }

    /// Английский текст → PCM 24 кГц. `speed` — темп (1 — обычный).
    pub fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let v = self.voice(voice)?;
        let t0 = Instant::now();
        let chunks = self.g2p.phonemize_chunks(text);
        let t1 = Instant::now();
        let mut parts = Vec::new();
        for c in &chunks {
            parts.extend(self.split(&self.ids(c)));
        }
        if parts.is_empty() {
            return Err(LkError::Other("нечего говорить".into()));
        }
        let mut out = Vec::new();
        for p in &parts {
            out.extend(self.run(v, p, speed.clamp(0.25, 4.0))?);
        }
        if let Ok(mut t) = self.last.lock() {
            *t = Timings { chunks: parts.len(), audio_s: out.len() as f64 / SR as f64, g2p_s: (t1 - t0).as_secs_f64(), model_s: t1.elapsed().as_secs_f64() };
        }
        Ok(out)
    }
}
