//! Конвейер Silero TTS v5: текст → (омографы, ударения и «ё» для русских) → символы → сеть → звук.
//! Длинный текст (больше ~900 символов) — кусками по предложениям: позиционная таблица модели
//! ограничена 5000 кадрами (~62 с).

use std::path::Path;
use std::time::Instant;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::syn_bundle::SynBundleLoader;

use crate::accentor::{Accentor, AccentorKind};
use crate::config::{BundleConfig, PackageConfig};
use crate::frontend;
use crate::homosolver::HomoSolver;
use crate::model::{Extra, NetOut, RootKind, SileroNet, Weights};
use crate::{Result, SileroError, SAMPLE_RATES};

#[derive(Debug, Clone)]
pub struct Options {
    pub speaker: String,
    /// 48000 (родная), 24000 или 8000.
    pub sample_rate: u32,
    /// Темп: длительности делятся на него (как `prosody rate` SSML).
    pub rate: f32,
    /// Высота голоса: множитель нормированной высоты тона (как `prosody pitch`; 0 — «робот»).
    pub pitch: f32,
    /// Ударения, «ё» и омографы автоматически (русские версии).
    pub accents: bool,
    /// Язык для замены алфавита (пакеты СНГ); по умолчанию — префикс голоса.
    pub lang: Option<String>,
    /// Числа словами (русский голос): в алфавите модели цифр нет.
    pub numbers: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { speaker: String::new(), sample_rate: 48_000, rate: 1.0, pitch: 1.0, accents: true, lang: None, numbers: true }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub chunks: usize,
    pub audio_s: f64,
    pub frontend_s: f64,
    pub net_s: f64,
}

pub struct SileroPipeline {
    pub config: BundleConfig,
    net: SileroNet,
    accentor: Option<Accentor>,
    homo: Option<HomoSolver>,
    /// Типы предложений: нет, по символам текста (v5_4/v5_5), по токенам (v5_6).
    types: TypeMode,
    last: std::sync::Mutex<Timings>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TypeMode {
    None,
    Chars,
    Tokens,
}

impl SileroPipeline {
    pub fn from_syn(path: &Path, device: Device) -> Result<Self> {
        if !path.exists() {
            return Err(SileroError::Load(format!("нет файла {}", path.display())));
        }
        let b = Bundle::open(path).map_err(|e| SileroError::Load(e.to_string()))?;
        let read = |n: &str| b.read_file(n).map(|c| c.into_owned()).map_err(|e| SileroError::Load(format!("{n}: {e}")));
        let config: BundleConfig = serde_json::from_slice(&read("config.json")?).map_err(|e| SileroError::Load(format!("config.json: {e}")))?;
        let pkg = config.packages.first().ok_or_else(|| SileroError::Load("в config.json нет пакетов".into()))?;
        let mc = pkg.models.first().ok_or_else(|| SileroError::Load("в пакете нет моделей".into()))?;
        let open = |c: &str| -> Result<SynBundleLoader> { Ok(SynBundleLoader::open(path).map_err(|e| SileroError::Load(e.to_string()))?.with_component(c).with_device(device)) };
        // ~23M параметров: счёт в F32 (как оригинал на CPU)
        let tl = open(&mc.component)?;
        let net = SileroNet::load(&Weights { l: &tl, device, dtype: DType::F32 }, mc)?;
        let homo = match read("homosolver/vocab.txt") {
            Ok(vocab) => {
                let hl = open("homosolver")?;
                let hcfg: serde_json::Value = serde_json::from_slice(&read("homosolver/config.json")?).map_err(|e| SileroError::Load(format!("homosolver/config.json: {e}")))?;
                let phrases = read("homosolver/phrases.json").ok().filter(|p| p.len() > 2);
                Some(HomoSolver::new(&String::from_utf8_lossy(&vocab), &read("homosolver/homodict.json")?, phrases.as_deref(), &hcfg, &Weights { l: &hl, device, dtype: DType::F32 })?)
            }
            Err(_) => None,
        };
        let accentor = match read("accentor/ngrams.txt") {
            Ok(ng) => {
                let al = open("accentor")?;
                let get = |n: &str| -> Result<(Vec<f32>, Vec<usize>)> {
                    let t = synaptix_io::weights::WeightLoader::load_to(&al, n, Device::Cpu, DType::F32).map_err(|e| SileroError::Load(format!("accentor {n}: {e}")))?;
                    let dims = t.dims().to_vec();
                    Ok((t.flatten_all()?.to_vec1::<f32>()?, dims))
                };
                let kind = if homo.as_ref().is_some_and(|h| h.is_new()) { AccentorKind::New } else { AccentorKind::Old };
                Some(Accentor::new(&String::from_utf8_lossy(&ng), &String::from_utf8_lossy(&read("accentor/exceptions.tsv")?), &get, kind)?)
            }
            Err(_) => None,
        };
        let uses_types = mc.modules.get("pitch_predictor.pitch_pred").and_then(|m| m.utt_type_emb).unwrap_or(false);
        let types = match (uses_types, net.kind) {
            (false, _) => TypeMode::None,
            (true, RootKind::Emph) => TypeMode::Tokens,
            (true, RootKind::Classic) => TypeMode::Chars,
        };
        Ok(Self { config, net, accentor, homo, types, last: Default::default() })
    }

    fn pkg(&self) -> &PackageConfig {
        &self.config.packages[0]
    }

    pub fn speakers(&self) -> Vec<String> {
        self.pkg().speakers.clone()
    }

    pub fn has_accentor(&self) -> bool {
        self.accentor.is_some()
    }

    pub fn last_timings(&self) -> Timings {
        self.last.lock().map(|t| t.clone()).unwrap_or_default()
    }

    fn speaker_id(&self, name: &str) -> Result<usize> {
        let p = self.pkg();
        let m = p.speaker_to_model.get(name).ok_or_else(|| SileroError::Other(format!("голоса «{name}» нет; есть: {}", p.speakers.join(", "))))?;
        p.speaker_to_ids.get(*m).and_then(|d| d.get(name)).copied().ok_or_else(|| SileroError::Other(format!("голос «{name}»: нет индекса")))
    }

    /// Текст после омографов и ударений (то, что видит модель, без sos/eos).
    pub fn accentuate(&self, text: &str, lang: Option<&str>, accents: bool) -> Result<String> {
        let s = frontend::prepare(self.pkg(), text, lang)?;
        if !accents {
            return Ok(s);
        }
        let s = match &self.homo {
            Some(h) => h.apply(&s)?,
            None => s,
        };
        match &self.accentor {
            Some(a) => a.apply(&s),
            None => Ok(s),
        }
    }

    /// Омографы и ударения на готовой строке (как `accentor(text)` оригинала, без подготовки).
    pub fn accentuate_raw(&self, text: &str) -> Result<String> {
        let s = match &self.homo {
            Some(h) => h.apply(text)?,
            None => text.to_string(),
        };
        match &self.accentor {
            Some(a) => a.apply(&s),
            None => Ok(s),
        }
    }

    /// Входы сети для одного куска: последовательность и доп. входы.
    pub fn inputs(&self, text: &str, o: &Options) -> Result<(Vec<i64>, Extra)> {
        let lang = o.lang.clone().or_else(|| o.speaker.split('_').next().filter(|_| o.speaker.contains('_')).map(String::from));
        let expanded;
        let text = if o.numbers && text.chars().any(|c| c.is_ascii_digit()) && lang.as_deref().is_none_or(|l| l == "ru") {
            expanded = crate::numbers::expand(text);
            expanded.as_str()
        } else {
            text
        };
        let acc = self.accentuate(text, lang.as_deref(), o.accents)?;
        let seq = frontend::sequence(self.pkg(), &acc)?;
        let extra = match self.types {
            TypeMode::None => Extra::default(),
            TypeMode::Chars => {
                let types = frontend::classify_text(text);
                Extra { type_ids: Some(frontend::type_ids_by_chars(text, &types, seq.len())), focus: None, emphasis: None }
            }
            TypeMode::Tokens => {
                let types = frontend::classify_text(text);
                Extra { type_ids: Some(frontend::type_ids_by_tokens(self.pkg(), &seq, &types)), focus: Some(vec![0; seq.len()]), emphasis: None }
            }
        };
        Ok((seq, extra))
    }

    /// Один кусок → выход сети (звук, длительности, высота тона, мел).
    pub fn infer(&self, text: &str, o: &Options) -> Result<NetOut> {
        if !SAMPLE_RATES.contains(&o.sample_rate) {
            return Err(SileroError::Other(format!("частота {} Гц: есть 48000, 24000, 8000", o.sample_rate)));
        }
        let t0 = Instant::now();
        let spk = self.speaker_id(&o.speaker)?;
        let (seq, extra) = self.inputs(text, o)?;
        let t1 = Instant::now();
        let out = self.net.forward(&seq, spk, &extra, o.sample_rate, o.rate, o.pitch)?;
        if let Ok(mut t) = self.last.lock() {
            t.frontend_s += (t1 - t0).as_secs_f64();
            t.net_s += t1.elapsed().as_secs_f64();
        }
        Ok(out)
    }

    /// Вокодер отдельно (сверка): мел `[каналы × кадры]`.
    pub fn vocode(&self, mel: &[f32], channels: usize, sr: u32) -> Result<Vec<f32>> {
        self.net.vocode(mel, channels, sr)
    }

    /// Текст любой длины → звук `o.sample_rate`.
    pub fn synthesize(&self, text: &str, o: &Options) -> Result<Vec<f32>> {
        if let Ok(mut t) = self.last.lock() {
            *t = Timings::default();
        }
        let chunks = chunk(text, 900);
        if chunks.is_empty() {
            return Err(SileroError::Other("пустой текст".into()));
        }
        let mut out = Vec::new();
        for c in &chunks {
            out.extend(self.infer(c, o)?.audio);
        }
        if let Ok(mut t) = self.last.lock() {
            t.chunks = chunks.len();
            t.audio_s = out.len() as f64 / o.sample_rate as f64;
        }
        Ok(out)
    }
}

/// Куски не длиннее `max` символов по границам предложений (`.!?…` + пробел), иначе — по пробелам.
pub fn chunk(text: &str, max: usize) -> Vec<String> {
    let t = text.trim();
    if t.is_empty() {
        return Vec::new();
    }
    if t.chars().count() <= max {
        return vec![t.to_string()];
    }
    let c: Vec<char> = t.chars().collect();
    let mut sents = Vec::new();
    let mut cur = String::new();
    for i in 0..c.len() {
        cur.push(c[i]);
        if ".!?…\n".contains(c[i]) && c.get(i + 1).is_none_or(|n| n.is_whitespace()) {
            sents.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        sents.push(cur);
    }
    let mut out: Vec<String> = Vec::new();
    let mut acc = String::new();
    for s in sents {
        let s = s.trim().to_string();
        if s.is_empty() {
            continue;
        }
        if !acc.is_empty() && acc.chars().count() + 1 + s.chars().count() > max {
            out.push(std::mem::take(&mut acc));
        }
        if s.chars().count() > max {
            // очень длинное предложение — по словам
            for w in s.split_whitespace() {
                if !acc.is_empty() && acc.chars().count() + 1 + w.chars().count() > max {
                    out.push(std::mem::take(&mut acc));
                }
                if !acc.is_empty() {
                    acc.push(' ');
                }
                acc.push_str(w);
            }
            continue;
        }
        if !acc.is_empty() {
            acc.push(' ');
        }
        acc.push_str(&s);
    }
    if !acc.is_empty() {
        out.push(acc);
    }
    out
}
