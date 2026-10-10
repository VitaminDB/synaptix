//! Модель MFA: переходная модель Kaldi, GMM, дерево, LDA, признаки, словарь.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::features::FeatureConfig;
use crate::gmm::DiagGmms;
use crate::kaldi_text::{self, Tokens};
use crate::lexicon::Lexicon;
use crate::tree::ContextDependency;
use crate::{MfaError, Result};

/// Состояние HMM-топологии: класс pdf и переходы (`dst`, вероятность).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoState {
    /// `None` — финальное (неэмитирующее) состояние.
    pub pdf_class: Option<i32>,
    pub transitions: Vec<(usize, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    pub phones: Vec<i32>,
    pub states: Vec<TopoState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionModel {
    pub topologies: Vec<Topology>,
    /// Тройки (фонема, состояние HMM, pdf) — transition-state = индекс + 1.
    pub triples: Vec<(i32, usize, usize)>,
    /// log P по transition-id (индекс 0 не используется).
    pub log_probs: Vec<f32>,
    /// Первый transition-id каждого transition-state (индекс = state − 1).
    pub state_first_tid: Vec<usize>,
    #[serde(skip)]
    lookup: HashMap<(i32, usize, usize), usize>,
    #[serde(skip)]
    topo_of_phone: HashMap<i32, usize>,
}

impl TransitionModel {
    fn derive(&mut self) {
        self.topo_of_phone.clear();
        for (ti, t) in self.topologies.iter().enumerate() {
            for &p in &t.phones {
                self.topo_of_phone.insert(p, ti);
            }
        }
        self.lookup = self.triples.iter().enumerate().map(|(i, &tr)| (tr, i + 1)).collect();
        let mut first = Vec::with_capacity(self.triples.len());
        let mut cur = 1;
        for &(phone, hs, _) in &self.triples {
            first.push(cur);
            cur += self.topology(phone).map_or(0, |t| t.states[hs].transitions.len());
        }
        self.state_first_tid = first;
    }

    /// transition-id → (фонема, состояние HMM, pdf, номер перехода).
    pub fn decode_tid(&self, tid: usize) -> Option<(i32, usize, usize, usize)> {
        let ts = self.state_first_tid.partition_point(|&f| f <= tid);
        if ts == 0 {
            return None;
        }
        let (p, hs, pdf) = self.triples[ts - 1];
        Some((p, hs, pdf, tid - self.state_first_tid[ts - 1]))
    }

    pub fn topology(&self, phone: i32) -> Option<&Topology> {
        self.topo_of_phone.get(&phone).map(|&i| &self.topologies[i])
    }

    /// log P перехода `j` из состояния HMM `hs` фонемы с данным pdf.
    pub fn log_prob(&self, phone: i32, hs: usize, pdf: usize, j: usize) -> Option<f32> {
        let ts = *self.lookup.get(&(phone, hs, pdf))?;
        self.log_probs.get(self.state_first_tid[ts - 1] + j).copied()
    }

    fn parse(t: &mut Tokens) -> Result<Self> {
        t.expect("<TransitionModel>")?;
        t.expect("<Topology>")?;
        let mut topologies = Vec::new();
        while t.peek() == Some("<TopologyEntry>") {
            t.next()?;
            t.expect("<ForPhones>")?;
            let mut phones = Vec::new();
            while t.peek() != Some("</ForPhones>") {
                phones.push(t.int()? as i32);
            }
            t.next()?;
            let mut states = Vec::new();
            while t.peek() == Some("<State>") {
                t.next()?;
                let _id = t.int()?;
                let mut st = TopoState { pdf_class: None, transitions: Vec::new() };
                loop {
                    match t.next()? {
                        "<PdfClass>" => st.pdf_class = Some(t.int()? as i32),
                        "<ForwardPdfClass>" => st.pdf_class = Some(t.int()? as i32),
                        "<SelfLoopPdfClass>" => {
                            t.int()?;
                        }
                        "<Transition>" => {
                            let d = t.int()? as usize;
                            let p = t.float()?;
                            st.transitions.push((d, p));
                        }
                        "</State>" => break,
                        x => return Err(MfaError::Format(format!("State: `{x}`"))),
                    }
                }
                states.push(st);
            }
            t.expect("</TopologyEntry>")?;
            topologies.push(Topology { phones, states });
        }
        t.expect("</Topology>")?;
        t.expect("<Triples>")?;
        let n = t.int()? as usize;
        let mut triples = Vec::with_capacity(n);
        for _ in 0..n {
            let p = t.int()? as i32;
            let hs = t.int()? as usize;
            let pdf = t.int()? as usize;
            triples.push((p, hs, pdf));
        }
        t.expect("</Triples>")?;
        t.expect("<LogProbs>")?;
        let log_probs = t.bracketed()?;
        t.expect("</LogProbs>")?;
        t.expect("</TransitionModel>")?;
        let mut tm = Self { topologies, triples, log_probs, state_first_tid: Vec::new(), lookup: HashMap::new(), topo_of_phone: HashMap::new() };
        tm.derive();
        Ok(tm)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaModel {
    pub features: FeatureConfig,
    pub transitions: TransitionModel,
    #[serde(skip)]
    pub gmms: DiagGmms,
    pub tree: ContextDependency,
    /// LDA+MLLT `[rows × cols]`.
    pub lda_rows: usize,
    pub lda_cols: usize,
    #[serde(skip)]
    pub lda: Vec<f32>,
    /// Символ фонемы → id (как в phones.txt / meta.json).
    pub phones: HashMap<String, i32>,
    pub silence_phone: String,
    pub silence_probability: f32,
    pub initial_silence_probability: f32,
    /// Поправки финала: фраза кончается паузой / без неё.
    #[serde(default = "one")]
    pub final_silence_correction: f32,
    #[serde(default = "one")]
    pub final_non_silence_correction: f32,
    pub lexicon: Lexicon,
}

fn one() -> f32 {
    1.0
}

impl MfaModel {
    /// Из текстовых форм Kaldi и meta.json распакованной модели MFA.
    pub fn from_kaldi_text(mdl: &str, tree: &str, lda: &str, meta_json: &str, lexicon: &str) -> Result<Self> {
        let mut t = Tokens::new(mdl);
        let transitions = TransitionModel::parse(&mut t)?;
        t.expect("<DIMENSION>")?;
        let dim = t.int()? as usize;
        t.expect("<NUMPDFS>")?;
        let npdfs = t.int()? as usize;
        let mut gmms = DiagGmms { dim, offsets: vec![0], ..Default::default() };
        for _ in 0..npdfs {
            t.expect("<DiagGMM>")?;
            t.expect("<GCONSTS>")?;
            let gc = t.bracketed()?;
            t.expect("<WEIGHTS>")?;
            let _w = t.bracketed()?;
            t.expect("<MEANS_INVVARS>")?;
            let mi = t.bracketed()?;
            t.expect("<INV_VARS>")?;
            let iv = t.bracketed()?;
            t.expect("</DiagGMM>")?;
            if mi.len() != gc.len() * dim || iv.len() != gc.len() * dim {
                return Err(MfaError::Format("DiagGMM: размеры".into()));
            }
            gmms.gconsts.extend(gc);
            gmms.means_invvars.extend(mi);
            gmms.inv_vars.extend(iv);
            gmms.offsets.push(gmms.gconsts.len() as u32);
        }
        let tree = ContextDependency::from_text(tree)?;
        let (lda_rows, lda_cols, lda) = kaldi_text::matrix(lda)?;
        let meta: serde_json::Value = serde_json::from_str(meta_json).map_err(|e| MfaError::Format(format!("meta.json: {e}")))?;
        let f = &meta["features"];
        let num = |k: &str, d: f64| f[k].as_f64().unwrap_or(d) as f32;
        let features = FeatureConfig {
            sample_frequency: num("sample_frequency", 16000.0),
            frame_shift: num("frame_shift", 10.0),
            frame_length: num("frame_length", 25.0),
            low_frequency: num("low_frequency", 20.0),
            high_frequency: num("high_frequency", 7800.0),
            num_mel_bins: num("num_mel_bins", 23.0) as usize,
            num_coefficients: num("num_coefficients", 13.0) as usize,
            cepstral_lifter: num("cepstral_lifter", 22.0),
            preemphasis_coefficient: num("preemphasis_coefficient", 0.97),
            snip_edges: f["snip_edges"].as_bool().unwrap_or(false),
            use_energy: f["use_energy"].as_bool().unwrap_or(false),
            splice_left_context: num("splice_left_context", 3.0) as usize,
            splice_right_context: num("splice_right_context", 3.0) as usize,
        };
        let phones: HashMap<String, i32> = meta["phone_mapping"]
            .as_object()
            .ok_or_else(|| MfaError::Format("meta.json: нет phone_mapping".into()))?
            .iter()
            .filter_map(|(k, v)| v.as_i64().map(|i| (k.clone(), i as i32)))
            .collect();
        Ok(Self {
            features,
            transitions,
            gmms,
            tree,
            lda_rows,
            lda_cols,
            lda,
            phones,
            silence_phone: meta["optional_silence_phone"].as_str().unwrap_or("sil").to_string(),
            silence_probability: meta["silence_probability"].as_f64().unwrap_or(0.5) as f32,
            initial_silence_probability: meta["initial_silence_probability"].as_f64().unwrap_or(0.5) as f32,
            final_silence_correction: meta["final_silence_correction"].as_f64().unwrap_or(1.0) as f32,
            final_non_silence_correction: meta["final_non_silence_correction"].as_f64().unwrap_or(1.0) as f32,
            lexicon: Lexicon::from_text(lexicon),
        })
    }

    /// Компактный формат: `MFA1`, длина и JSON заголовка, затем f32-массивы.
    pub fn save(&self, path: &Path) -> Result<()> {
        let header = serde_json::to_vec(self).map_err(|e| MfaError::Format(e.to_string()))?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| MfaError::Io(path.display().to_string(), e))?);
        let io = |e| MfaError::Io(path.display().to_string(), e);
        f.write_all(b"MFA1").map_err(io)?;
        f.write_all(&(header.len() as u64).to_le_bytes()).map_err(io)?;
        f.write_all(&header).map_err(io)?;
        f.write_all(&(self.gmms.dim as u64).to_le_bytes()).map_err(io)?;
        let off: Vec<f32> = self.gmms.offsets.iter().map(|&o| f32::from_bits(o)).collect();
        for arr in [&off, &self.gmms.gconsts, &self.gmms.means_invvars, &self.gmms.inv_vars, &self.lda] {
            f.write_all(&(arr.len() as u64).to_le_bytes()).map_err(io)?;
            for v in arr.iter() {
                f.write_all(&v.to_le_bytes()).map_err(io)?;
            }
        }
        f.flush().map_err(io)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let io = |e| MfaError::Io(path.display().to_string(), e);
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).map_err(io)?;
        if &magic != b"MFA1" {
            return Err(MfaError::Format("не модель MFA1".into()));
        }
        let mut u = [0u8; 8];
        f.read_exact(&mut u).map_err(io)?;
        let mut header = vec![0u8; u64::from_le_bytes(u) as usize];
        f.read_exact(&mut header).map_err(io)?;
        let mut m: MfaModel = serde_json::from_slice(&header).map_err(|e| MfaError::Format(e.to_string()))?;
        m.transitions.derive();
        f.read_exact(&mut u).map_err(io)?;
        m.gmms.dim = u64::from_le_bytes(u) as usize;
        let mut read_arr = || -> Result<Vec<f32>> {
            f.read_exact(&mut u).map_err(io)?;
            let n = u64::from_le_bytes(u) as usize;
            let mut b = vec![0u8; n * 4];
            f.read_exact(&mut b).map_err(io)?;
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        m.gmms.offsets = read_arr()?.into_iter().map(f32::to_bits).collect();
        m.gmms.gconsts = read_arr()?;
        m.gmms.means_invvars = read_arr()?;
        m.gmms.inv_vars = read_arr()?;
        m.lda = read_arr()?;
        Ok(m)
    }

    pub fn phone_id(&self, sym: &str) -> Option<i32> {
        self.phones.get(sym).copied()
    }

    /// Признаки для выравнивания: MFCC → CMVN → splice → LDA (`[T × lda_rows]`).
    /// `pcm` — 16 кГц в [-1, 1].
    pub fn features(&self, pcm: &[f32]) -> Vec<f32> {
        let mfcc = crate::features::Mfcc::new(self.features.clone());
        let scaled: Vec<f32> = pcm.iter().map(|x| x * 32768.0).collect();
        let mut f = mfcc.compute(&scaled);
        let d = self.features.num_coefficients;
        crate::features::cmvn(&mut f, d);
        let s = crate::features::splice(&f, d, self.features.splice_left_context, self.features.splice_right_context);
        let ds = d * (self.features.splice_left_context + self.features.splice_right_context + 1);
        crate::features::transform(&s, ds, &self.lda, self.lda_rows, self.lda_cols)
    }
}
