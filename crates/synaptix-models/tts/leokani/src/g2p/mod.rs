//! Leokani G2P — нативный порт `g2p/leokani_g2p.py` (Milliwatt Labs): английский текст → фонемы
//! Kokoro/misaki с ударениями, без espeak-ng.
//!
//! Для каждой клаузы («пьесы» — слов без пунктуации между ними):
//! 1. база каждого слова — лексикон (`leokani-g2p.json`), иначе читатель чисел, иначе модель
//!    буквы→звуки (`oov-encoder` + `oov-decoder`, жадно);
//! 2. контекстная модель (`context`) смотрит на всю клаузу и выбирает для слова форму (смену ударения
//!    или слабую форму "to" -> tə), край (флэп, связующий ɹ) и слитность со следующим словом.
//!
//! Перед этим строку переписывает нормализатор (`norm.rs` + `rules*.rs`: числа, деньги, время,
//! телефоны, сокращения, буквы, омографы), а строковый фронтенд (`line.rs`) режет её на куски и клаузы.
//! Модули: `py` — семантика строк/`re` Python, `tables` — таблицы, `words` — числа словами и проверки,
//! `numbers` — чтение цифр как espeak, `nets` — три сети.

mod line;
mod nets;
mod norm;
mod numbers;
mod py;
mod rules1;
mod rules2;
mod rules3;
mod tables;
mod words;

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Deserialize;
use synaptix_core::tensor::Tensor;

use crate::{LkError, Result};
use words::{PH, PRIMARY, SECONDARY};

pub(crate) const VOWELS: &str = "aeiouæɑɐɒɔəɛɜɪʊʌAEIOUWYᵊᵻɚɝ";
const FORMAT: &str = "leokani-g2p/1";
/// Слоты контекстной модели: правки ударения базы, затем до N_VARIANTS вариантов слова из лексикона.
const STRESS_OPS: [&str; 7] = ["same", "demote", "strip", "promote1", "demote1", "demoteLast", "promote"];
const N_VARIANTS: usize = 6;
const EDGE_OPS: [&str; 4] = ["none", "flap", "linkr", "unflap"];
const SRC: [&str; 5] = ["lexicon", "oov", "number", "symbol", "empty"];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Doc {
    format: String,
    lexicon: HashMap<String, String>,
    cased: HashMap<String, String>,
    variants: HashMap<String, Vec<String>>,
    context_words: Vec<String>,
    symbols: Vec<String>,
    chars: Vec<String>,
    out_symbols: Vec<String>,
    number_atoms: HashMap<String, String>,
    symbol_words: HashMap<String, String>,
    oov: OovCfg,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OovCfg {
    max_out: usize,
}

/// Данные лексикона; на нём же — нормализатор и читатель чисел.
pub(crate) struct Lex {
    lexicon: HashMap<String, String>,
    cased: HashMap<String, String>,
    variants: HashMap<String, Vec<String>>,
    word_ids: HashMap<String, i64>,
    sym_ids: HashMap<String, i64>,
    chars: HashMap<String, usize>,
    out_syms: Vec<String>,
    atoms: HashMap<String, String>,
    symbol_words: HashMap<String, String>,
    /// Имена букв (letter:* из атомов чисел; "s" — имя буквы, а не суффикс).
    letters: HashMap<String, String>,
    max_out: usize,
}

impl Lex {
    /// `LeokaniG2P._word`: слово для читателя чисел.
    fn word(&self, w: &str) -> Option<String> {
        match self.cased.get(w) {
            Some(v) if !v.is_empty() => Some(v.clone()),
            _ => self.lexicon.get(&py::lower(w)).cloned(),
        }
    }
}

// ---------------------------------------------------------------- ударение и правки фонем

fn stress_op(b: &str, op: &str) -> String {
    let (p, s) = (PRIMARY.to_string(), SECONDARY.to_string());
    match op {
        "same" => b.to_string(),
        "demote" => b.replace(PRIMARY, &s),
        "strip" => b.replace([PRIMARY, SECONDARY], ""),
        "promote1" => b.replacen(SECONDARY, &p, 1),
        "demote1" => b.replacen(PRIMARY, &s, 1),
        "demoteLast" => match b.rfind(PRIMARY) {
            Some(i) => format!("{}{s}{}", &b[..i], &b[i + PRIMARY.len_utf8()..]),
            None => b.to_string(),
        },
        "promote" => b.replace(SECONDARY, &p),
        _ => unreachable!(),
    }
}

fn edge_op(s: &str, op: &str) -> String {
    if op == "none" || s.is_empty() {
        return s.to_string();
    }
    match op {
        "flap" => match s.strip_suffix('t') {
            Some(h) => format!("{h}ɾ"),
            None => s.to_string(),
        },
        "linkr" => format!("{s}ɹ"),
        "unflap" => match s.strip_suffix('ɾ') {
            Some(h) => format!("{h}t"),
            None => s.to_string(),
        },
        _ => unreachable!(),
    }
}

/// Слоты контекстной модели для слова: STRESS_OPS базы, затем до N_VARIANTS вариантов (None — нет).
fn candidates(base: &str, variants: &[String]) -> Vec<Option<String>> {
    let mut out: Vec<Option<String>> = STRESS_OPS.iter().map(|op| Some(stress_op(base, op))).collect();
    out.extend((0..N_VARIANTS).map(|i| variants.get(i).cloned()));
    out
}

fn shape(w: &str) -> usize {
    if w.chars().any(py::c_digit) {
        return 4;
    }
    if !w.chars().any(py::c_alpha) {
        return 5;
    }
    if w == py::lower(w) {
        return 0;
    }
    if py::is_upper(w) {
        return 2;
    }
    let rest = py::from(w, 1);
    if py::first_upper(w) && rest == py::lower(&rest) {
        return 1;
    }
    3
}

/// Признаки клаузы для контекстной модели.
fn features(lx: &Lex, words: &[String], bases: &[String], srcs: &[&str]) -> Vec<[i64; nets::N_FEATURES]> {
    let n = words.len();
    let mut out = Vec::with_capacity(n);
    for (i, ((w, b), s)) in words.iter().zip(bases).zip(srcs).enumerate() {
        let bb: Vec<char> = b.chars().filter(|&c| c != PRIMARY && c != SECONDARY && c != ' ').collect();
        let sym = |c: char| *lx.sym_ids.get(&c.to_string()).unwrap_or(&1);
        let k = bb.len();
        out.push([
            *lx.word_ids.get(&py::lower(w)).unwrap_or(&1),
            shape(w) as i64,
            if k > 0 { sym(bb[0]) } else { 0 },
            if k > 0 { sym(bb[k - 1]) } else { 0 },
            if k > 1 { sym(bb[k - 2]) } else { 0 },
            b.matches(PRIMARY).count().min(3) as i64,
            b.matches(SECONDARY).count().min(3) as i64,
            bb.iter().filter(|c| VOWELS.contains(**c)).count().min(7) as i64,
            if b.is_empty() { 0 } else { (b.matches(' ').count() + 1).min(3) as i64 },
            SRC.iter().position(|x| x == s).unwrap() as i64,
            (i == 0) as i64,
            (i == n - 1) as i64,
            i.min(7) as i64,
            (n - 1 - i).min(7) as i64,
        ]);
    }
    out
}

/// Лучшая пара (слот, край), которая у слова есть: логиты по слотам × краям, по убыванию (стабильно).
fn pick(logits: &[f32], cands: &[Option<String>]) -> Option<String> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| (-logits[a]).partial_cmp(&-logits[b]).unwrap_or(std::cmp::Ordering::Equal));
    for j in idx {
        let (k, e) = (j / EDGE_OPS.len(), j % EDGE_OPS.len());
        if let Some(Some(c)) = cands.get(k) {
            return Some(edge_op(c, EDGE_OPS[e]));
        }
    }
    None
}

fn assemble(forms: &[String], joins: &[bool]) -> String {
    let mut out = String::new();
    for (i, f) in forms.iter().enumerate() {
        if f.is_empty() {
            continue;
        }
        if !out.is_empty() && !out.ends_with(' ') && !(i > 0 && joins[i - 1] && !forms[i - 1].is_empty()) {
            out.push(' ');
        }
        out.push_str(f);
    }
    py::strip_ws(&out).to_string()
}

/// `re.findall(r"\d+|[^\W\d_]+", w)` с классами символов Python.
fn digit_letter_runs(w: &str) -> Vec<String> {
    let class = |c: char| -> u8 {
        if py::c_decimal(c) {
            1
        } else if py::c_word(c) && c != '_' {
            2
        } else {
            0
        }
    };
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cls = 0u8;
    for c in w.chars() {
        let k = class(c);
        if k != cls && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if k != 0 {
            cur.push(c);
        }
        cls = k;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Leokani G2P: нормализатор + лексикон + читатель чисел + три сети.
pub struct LeokaniG2P {
    lx: Lex,
    ctx: nets::Context,
    oov_net: nets::Oov,
    cache: Mutex<HashMap<String, String>>,
}

impl LeokaniG2P {
    /// Только G2P из бандла Leokani (`leokani-g2p.json` + сети `g2p_nets_*`), без голосов — английское
    /// произношение для других движков (Silero: латиница в русском тексте).
    pub fn from_syn(path: &std::path::Path) -> Result<Self> {
        use synaptix_core::{device::Device, dtype::DType};
        use synaptix_io::weights::{syn_bundle::SynBundleLoader, WeightLoader};
        let b = synaptix_bundle::Bundle::open(path).map_err(|e| LkError::Load(e.to_string()))?;
        let doc = b.read_file("leokani-g2p.json").map_err(|e| LkError::Load(format!("leokani-g2p.json: {e}")))?.into_owned();
        let nets: std::collections::BTreeMap<&str, SynBundleLoader> = [("context", "g2p_nets_context"), ("enc", "g2p_nets_enc"), ("dec", "g2p_nets_dec")]
            .into_iter()
            .map(|(p, c)| Ok((p, SynBundleLoader::open(path).map_err(|e| LkError::Load(e.to_string()))?.with_component(c).with_device(Device::Cpu))))
            .collect::<Result<_>>()?;
        let get = |n: &str| -> Result<Tensor> {
            let l = n.split('.').next().and_then(|p| nets.get(p)).ok_or_else(|| LkError::Load(format!("{n}: нет сети G2P")))?;
            l.load_to(n, Device::Cpu, DType::F32).map_err(|e| LkError::Load(format!("{n}: {e}")))
        };
        Self::new(&doc, &get)
    }

    /// doc_json — содержимое leokani-g2p.json; get — вес сети по полному имени (с префиксом), F32 на CPU.
    pub fn new(doc_json: &[u8], get: &dyn Fn(&str) -> Result<Tensor>) -> Result<Self> {
        let doc: Doc = serde_json::from_slice(doc_json).map_err(|e| LkError::Load(format!("leokani-g2p.json: {e}")))?;
        if doc.format != FORMAT {
            return Err(LkError::Load(format!("leokani-g2p.json: формат {:?}, нужен {FORMAT}", doc.format)));
        }
        let mut letters: HashMap<String, String> = doc
            .number_atoms
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("letter:").map(|l| (l.to_string(), v.clone())))
            .collect();
        letters.insert("s".into(), "ˈɛs".into()); // атом "s" — суффикс множественного ("80s")
        let lx = Lex {
            word_ids: doc.context_words.iter().enumerate().map(|(i, w)| (w.clone(), i as i64 + 2)).collect(),
            sym_ids: doc.symbols.iter().enumerate().map(|(i, s)| (s.clone(), i as i64 + 2)).collect(),
            chars: doc.chars.iter().enumerate().map(|(i, c)| (c.clone(), i + 4)).collect(),
            lexicon: doc.lexicon,
            cased: doc.cased,
            variants: doc.variants,
            out_syms: doc.out_symbols,
            atoms: doc.number_atoms,
            symbol_words: doc.symbol_words,
            letters,
            max_out: doc.oov.max_out,
        };
        Ok(Self { lx, ctx: nets::Context::new(get)?, oov_net: nets::Oov::new(get)?, cache: Mutex::new(HashMap::new()) })
    }

    /// Строка так, как её прочёл бы человек (`Normalizer.__call__`); pauses=false — без запятых между
    /// группами цифр телефона (внутри клаузы).
    pub fn normalize(&self, text: &str, pauses: bool) -> String {
        self.lx.normalize(text, pauses)
    }

    /// Одна клауза (слова без пунктуации) → фонемы (`LeokaniG2P.piece`).
    pub fn piece(&self, piece: &str) -> String {
        let words: Vec<String> = self.normalize(piece, false).split(' ').filter(|w| !w.is_empty()).map(str::to_string).collect();
        if words.is_empty() {
            return String::new();
        }
        let bs: Vec<(String, &str)> = words.iter().map(|w| self.base(w)).collect();
        let bases: Vec<String> = bs.iter().map(|b| b.0.clone()).collect();
        let srcs: Vec<&str> = bs.iter().map(|b| b.1).collect();
        let given: Vec<bool> = words.iter().map(|w| w.contains(PH)).collect();
        let words: Vec<String> = words.iter().map(|w| w.split(PH).next().unwrap().to_string()).collect();
        let feats = features(&self.lx, &words, &bases, &srcs);
        let (form, join) = self.ctx.run(&feats);
        let mut forms = Vec::with_capacity(words.len());
        for (i, (w, b)) in words.iter().zip(&bases).enumerate() {
            let f = if given[i] {
                let mut logits = form[i].clone();
                if !b.ends_with('ɹ') {
                    let li = EDGE_OPS.iter().position(|e| *e == "linkr").unwrap();
                    for v in logits.iter_mut().skip(li).step_by(EDGE_OPS.len()) {
                        *v = f32::NEG_INFINITY;
                    }
                }
                pick(&logits, &candidates(b, &[]))
            } else {
                let vars = self.lx.variants.get(&py::lower(w)).cloned().unwrap_or_default();
                pick(&form[i], &candidates(b, &vars))
            };
            forms.push(match f {
                Some(f) if !f.is_empty() => f,
                _ => b.clone(),
            });
        }
        let joins: Vec<bool> = join.iter().map(|j| j[1] > j[0]).collect();
        assemble(&forms, &joins)
    }

    /// Строка/абзац → фонемы по кускам-клаузам, ровно как `phonemize_chunks` в Python.
    pub fn phonemize_chunks(&self, text: &str) -> Vec<String> {
        line::chunks(&line::clean(&self.normalize(text, true)))
            .iter()
            .map(|c| line::line(c, |p| self.piece(p)))
            .collect()
    }

    /// Строка → одна строка фонем (куски через пробел).
    pub fn phonemize(&self, text: &str) -> String {
        self.phonemize_chunks(text).join(" ")
    }

    // ---- базы
    fn base(&self, w: &str) -> (String, &'static str) {
        let lx = &self.lx;
        if let Some((_, ph)) = w.split_once(PH) {
            return (ph.to_string(), "lexicon");
        }
        if let Some(v) = lx.cased.get(w) {
            return (v.clone(), "lexicon");
        }
        let lw = py::lower(w);
        if let Some(v) = lx.lexicon.get(&lw) {
            return (v.clone(), "lexicon");
        }
        if w.chars().any(py::c_digit) {
            if let Some(said) = lx.num_say(w) {
                return (said, "number");
            }
            // форма без правила ("(212)555-0100", "5'11"): её буквенные и цифровые куски, но не пусто
            let runs = digit_letter_runs(w);
            if !runs.is_empty() && !(runs.len() == 1 && runs[0] == w) {
                let parts: Vec<String> = runs.iter().map(|r| self.base(r).0).filter(|b| !b.is_empty()).collect();
                return (parts.join(" "), "number");
            }
        }
        if let Some(v) = lx.symbol_words.get(w) {
            return (v.clone(), "symbol");
        }
        if !w.chars().any(py::c_alpha) {
            return (String::new(), "empty");
        }
        let st = w.trim_end_matches('.');
        if w.ends_with('.') && !st.is_empty() && !st.contains('.') {
            return self.base(st); // "Street." -> Street (точка сокращения)
        }
        (self.oov(w), "oov")
    }

    /// Буквы → звуки для незнакомого слова (с кэшем; длинное — по частям).
    fn oov(&self, w: &str) -> String {
        if let Some(v) = self.cache.lock().unwrap().get(w) {
            return v.clone();
        }
        let said: String = words::split_long(w).iter().map(|p| self.decode(p)).collect();
        self.cache.lock().unwrap().insert(w.to_string(), said.clone());
        said
    }

    fn decode(&self, w: &str) -> String {
        let lx = &self.lx;
        let mut ids = vec![2usize];
        ids.extend(w.chars().map(|c| *lx.chars.get(&c.to_string()).unwrap_or(&3)));
        ids.push(1);
        let mut s = String::new();
        for t in self.oov_net.decode(&ids, lx.max_out) {
            if t <= 1 {
                break;
            }
            if t >= 4 {
                s.push_str(lx.out_syms.get(t - 4).map(|x| x.as_str()).unwrap_or(""));
            }
        }
        s
    }
}
