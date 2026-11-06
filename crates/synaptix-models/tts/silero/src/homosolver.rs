//! Омографы Silero (`models/homosolver.py`): слово из словаря омографов размечается `[HOMO] … [/HOMO]`
//! в контексте, BERT (3 слоя × 312) → [вектор маркера ‖ среднее по слову] → MLP → один из двух
//! вариантов (по алфавиту). v5_6: сначала фразы-шаблоны, контекст чистится и режется окном 300;
//! v5…v5_5: контекст — всё предложение как есть.
//!
//! Батч всех омографов предложения дополняется `[PAD]` без маски внимания — как в оригинале
//! (паддинг влияет на векторы, поэтому батч тот же).

use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::model::Weights;
use crate::{Result, SileroError};

const VOWELS: &str = "аоуыэиеяёю";

fn lower(s: &str) -> String {
    s.chars().flat_map(|c| c.to_lowercase()).collect()
}

fn is_cyr(c: char) -> bool {
    matches!(c, 'А'..='Я' | 'а'..='я' | 'ё' | 'Ё')
}

// ---------------------------------------------------------------- токенизатор (SimpleBertTokenizer)

struct WordPiece {
    vocab: HashMap<String, i64>,
    never_split: Vec<String>,
    unk: i64,
    cls: i64,
    sep: i64,
    pad: i64,
}

fn is_punct(c: char) -> bool {
    let cp = c as u32;
    if (33..=47).contains(&cp) || (58..=64).contains(&cp) || (91..=96).contains(&cp) || (123..=126).contains(&cp) {
        return true;
    }
    // unicodedata.category(c).startswith("P") — общая пунктуация Юникода
    matches!(c, '«' | '»' | '–' | '—' | '…' | '„' | '“' | '”' | '‘' | '’' | '‚' | '‹' | '›' | '¡' | '¿' | '·' | '§' | '¶' | '‐' | '‑' | '‒' | '―' | '′' | '″' | '‰' | '†' | '‡' | '•')
        || ('\u{2010}'..='\u{2027}').contains(&c)
        || ('\u{2030}'..='\u{205E}').contains(&c)
        || ('\u{3000}'..='\u{303F}').contains(&c) && !matches!(c, '\u{3000}')
}

impl WordPiece {
    fn new(vocab_txt: &str, never_split: Vec<String>) -> Self {
        let vocab: HashMap<String, i64> = vocab_txt.lines().enumerate().map(|(i, l)| (l.to_string(), i as i64)).collect();
        let g = |t: &str| vocab.get(t).copied().unwrap_or(0);
        Self { unk: g("[UNK]"), cls: g("[CLS]"), sep: g("[SEP]"), pad: g("[PAD]"), never_split, vocab }
    }

    fn basic(&self, text: &str) -> Vec<String> {
        let clean: String = text
            .chars()
            .filter(|&c| !(c == '\0' || c == '\u{fffd}' || (c.is_control() && !matches!(c, '\t' | '\n' | '\r'))))
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect();
        let mut out = Vec::new();
        for tok in clean.split_whitespace() {
            if self.never_split.iter().any(|n| n == tok) {
                out.push(tok.to_string());
                continue;
            }
            let mut cur = String::new();
            for c in tok.chars() {
                if is_punct(c) {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    out.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                out.push(cur);
            }
        }
        out
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = vec![self.cls];
        for tok in self.basic(text) {
            let chars: Vec<char> = tok.chars().collect();
            if chars.len() > 100 {
                ids.push(self.unk);
                continue;
            }
            let mut sub = Vec::new();
            let mut start = 0;
            let mut bad = false;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found = None;
                while start < end {
                    let s: String = chars[start..end].iter().collect();
                    let s = if start > 0 { format!("##{s}") } else { s };
                    if let Some(&id) = self.vocab.get(&s) {
                        found = Some(id);
                        break;
                    }
                    end -= 1;
                }
                match found {
                    Some(id) => {
                        sub.push(id);
                        start = end;
                    }
                    None => {
                        bad = true;
                        break;
                    }
                }
            }
            if bad {
                ids.push(self.unk);
            } else {
                ids.extend(sub);
            }
        }
        ids.push(self.sep);
        ids
    }
}

// ---------------------------------------------------------------- BERT

struct BertLayer {
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    ln1: (Tensor, Tensor),
    inter: QuantLinear,
    out: QuantLinear,
    ln2: (Tensor, Tensor),
}

struct Bert {
    word: Vec<f32>,
    pos: Vec<f32>,
    tok_type: Vec<f32>,
    ln: (Tensor, Tensor),
    layers: Vec<BertLayer>,
    heads: usize,
    eps: f32,
    hidden: usize,
    clf: Vec<QuantLinear>,
}

impl Bert {
    fn load(w: &Weights, heads: usize, eps: f32) -> Result<Self> {
        let lin = |p: &str| -> Result<QuantLinear> { Ok(QuantLinear::dense(w.get(&format!("{p}.weight"))?, Some(w.get(&format!("{p}.bias"))?))?) };
        let pair = |p: &str| -> Result<(Tensor, Tensor)> { Ok((w.get(&format!("{p}.weight"))?, w.get(&format!("{p}.bias"))?)) };
        let mut layers = Vec::new();
        while w.has(&format!("bert.encoder.layer.{}.attention.self.query.weight", layers.len())) {
            let p = format!("bert.encoder.layer.{}", layers.len());
            layers.push(BertLayer {
                q: lin(&format!("{p}.attention.self.query"))?,
                k: lin(&format!("{p}.attention.self.key"))?,
                v: lin(&format!("{p}.attention.self.value"))?,
                o: lin(&format!("{p}.attention.output.dense"))?,
                ln1: pair(&format!("{p}.attention.output.LayerNorm"))?,
                inter: lin(&format!("{p}.intermediate.dense"))?,
                out: lin(&format!("{p}.output.dense"))?,
                ln2: pair(&format!("{p}.output.LayerNorm"))?,
            });
        }
        let word = w.host("bert.embeddings.word_embeddings.weight")?;
        let tok_type = w.host("bert.embeddings.token_type_embeddings.weight")?;
        let hidden = tok_type.len() / 2;
        Ok(Self {
            word,
            pos: w.host("bert.embeddings.position_embeddings.weight")?,
            tok_type,
            ln: pair("bert.embeddings.LayerNorm")?,
            layers,
            heads,
            eps,
            hidden,
            clf: vec![lin("homo_clf.0")?, lin("homo_clf.3")?],
        })
    }

    /// Логиты для батча `[B][L]` (дополнен `pad`), маркеры `[HOMO]`/`[/HOMO]`.
    fn logits(&self, batch: &[Vec<i64>], starts: &[usize], ends: &[usize], dev: Device) -> Result<Vec<f32>> {
        let b = batch.len();
        let l = batch[0].len();
        let h = self.hidden;
        let mut x = vec![0f32; b * l * h];
        for (bi, ids) in batch.iter().enumerate() {
            for (t, &id) in ids.iter().enumerate() {
                let o = &mut x[(bi * l + t) * h..(bi * l + t + 1) * h];
                let wr = &self.word[id as usize * h..(id as usize + 1) * h];
                let pr = &self.pos[t * h..(t + 1) * h];
                for j in 0..h {
                    o[j] = wr[j] + pr[j] + self.tok_type[j];
                }
            }
        }
        let f = || -> std::result::Result<Vec<f32>, SynaptixError> {
            let x = Tensor::from_vec(x, vec![b, l, h], Device::Cpu)?.to_device(dev)?;
            let mut x = layer_norm(&x, Some(&self.ln.0), Some(&self.ln.1), self.eps)?;
            let hd = h / self.heads;
            for ly in &self.layers {
                let sp = |t: Tensor| -> std::result::Result<Tensor, SynaptixError> { t.reshape(vec![b, l, self.heads, hd])?.permute(&[0, 2, 1, 3])?.contiguous() };
                let a = scaled_dot_attention(&sp(ly.q.forward(&x)?)?, &sp(ly.k.forward(&x)?)?, &sp(ly.v.forward(&x)?)?, 1.0 / (hd as f32).sqrt(), None)?;
                let a = ly.o.forward(&a.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![b, l, h])?)?;
                x = layer_norm(&a.add(&x)?, Some(&ly.ln1.0), Some(&ly.ln1.1), self.eps)?;
                let f = ly.out.forward(&ly.inter.forward(&x)?.gelu_exact()?)?;
                x = layer_norm(&f.add(&x)?, Some(&ly.ln2.0), Some(&ly.ln2.1), self.eps)?;
            }
            x.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
        };
        let hid = f()?;
        let mut feats = Vec::with_capacity(b * 2 * h);
        for bi in 0..b {
            let row = |t: usize| &hid[(bi * l + t) * h..(bi * l + t + 1) * h];
            feats.extend_from_slice(row(starts[bi]));
            let (s, e) = (starts[bi] + 1, ends[bi]);
            let mut m = vec![0f32; h];
            for t in s..e {
                m.iter_mut().zip(row(t)).for_each(|(a, v)| *a += v);
            }
            let n = (e.saturating_sub(s)).max(1) as f32;
            m.iter_mut().for_each(|v| *v /= if e > s { n } else { f32::NAN });
            feats.extend(m);
        }
        let g = || -> std::result::Result<Vec<f32>, SynaptixError> {
            let t = Tensor::from_vec(feats, vec![b, 2 * h], Device::Cpu)?.to_device(dev)?;
            let y = self.clf[1].forward(&self.clf[0].forward(&t)?.relu()?)?;
            y.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
        };
        Ok(g()?)
    }
}

// ---------------------------------------------------------------- решатель

pub struct HomoSolver {
    tok: WordPiece,
    bert: Bert,
    homodict: HashMap<String, Vec<String>>,
    /// v5_6: {слово: [(вариант, [фразы])]}; пусто — старый вариант.
    phrases: HashMap<String, Vec<(String, Vec<Vec<char>>)>>,
    window: usize,
    homo_start: i64,
    homo_end: i64,
    device: Device,
}

/// `_clean_text` (v5_6): чистка контекста по краям омографа.
fn clean_text(text: &str, is_start: bool) -> String {
    if text.is_empty() {
        return String::new();
    }
    let keep = |c: char| c.is_ascii_alphanumeric() || is_cyr(c) || c.is_whitespace() || ".!?,-".contains(c);
    let t: String = text.chars().filter(|&c| keep(c)).collect();
    let spaces = |s: &str| -> String {
        let mut o = String::new();
        let mut ws = false;
        for c in s.chars() {
            if c.is_whitespace() {
                if !ws {
                    o.push(' ');
                }
                ws = true;
            } else {
                o.push(c);
                ws = false;
            }
        }
        o
    };
    let double_dash = |s: &str| -> String {
        let c: Vec<char> = s.chars().collect();
        let mut o = String::new();
        let mut i = 0;
        while i < c.len() {
            if c[i] == '-' && i + 1 < c.len() && c[i + 1] == '-' {
                while i < c.len() && c[i] == '-' {
                    i += 1;
                }
                o.push_str(" - ");
            } else {
                o.push(c[i]);
                i += 1;
            }
        }
        o
    };
    // ([.!?])\1+ → \1 ; ,{2,} → ,
    let repeat = |s: &str| -> String {
        let mut o = String::new();
        let mut prev: Option<char> = None;
        for c in s.chars() {
            if prev == Some(c) && ".!?,".contains(c) {
                continue;
            }
            o.push(c);
            prev = Some(c);
        }
        o
    };
    // \s+([.,!?]) → \1
    let space_before = |s: &str| -> String {
        let c: Vec<char> = s.chars().collect();
        let mut o = String::new();
        let mut i = 0;
        while i < c.len() {
            if c[i].is_whitespace() {
                let mut j = i;
                while j < c.len() && c[j].is_whitespace() {
                    j += 1;
                }
                if j < c.len() && ".,!?".contains(c[j]) {
                    i = j;
                    continue;
                }
                o.extend(&c[i..j]);
                i = j;
            } else {
                o.push(c[i]);
                i += 1;
            }
        }
        o
    };
    // ([.,!?])(?=\S) → '\1 '
    let add_space = |s: &str| -> String {
        let c: Vec<char> = s.chars().collect();
        let mut o = String::new();
        for i in 0..c.len() {
            o.push(c[i]);
            if ".,!?".contains(c[i]) && i + 1 < c.len() && !c[i + 1].is_whitespace() {
                o.push(' ');
            }
        }
        o
    };
    let t = spaces(&t);
    let t = double_dash(&t);
    let t = repeat(&t);
    let t = space_before(&t);
    let t = repeat(&t);
    let t = add_space(&t);
    let mut t = spaces(&t).trim().to_string();
    if is_start {
        t = t.trim_start_matches([' ', '.', ',', '!', '?', '-']).to_string();
        let mut c = t.chars();
        if let Some(f) = c.next() {
            t = f.to_uppercase().collect::<String>() + &lower(c.as_str());
        }
    } else {
        let mut c = t.chars();
        if let Some(f) = c.next() {
            t = f.to_string() + &lower(c.as_str());
        }
        if t.chars().last().is_some_and(|c| !".!?".contains(c)) {
            t.push('.');
        }
    }
    t
}

fn tail_chars(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    c[c.len().saturating_sub(n)..].iter().collect()
}

fn head_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

impl HomoSolver {
    pub fn new(vocab: &str, homodict_json: &[u8], phrases_json: Option<&[u8]>, cfg: &serde_json::Value, w: &Weights) -> Result<Self> {
        let never: Vec<String> = cfg.get("never_split").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let tok = WordPiece::new(vocab, never);
        let heads = cfg.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
        let eps = cfg.get("layer_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-12) as f32;
        let bert = Bert::load(w, heads, eps)?;
        let homodict: HashMap<String, Vec<String>> = serde_json::from_slice(homodict_json).map_err(|e| SileroError::Load(format!("homodict: {e}")))?;
        let mut phrases = HashMap::new();
        if let Some(p) = phrases_json {
            let raw: HashMap<String, Vec<(String, Vec<String>)>> = serde_json::from_slice(p).map_err(|e| SileroError::Load(format!("phrases: {e}")))?;
            for (k, v) in raw {
                phrases.insert(k, v.into_iter().map(|(var, ps)| (var, ps.into_iter().map(|s| lower(&s).chars().collect()).collect())).collect());
            }
        }
        let homo_start = cfg.get("homo_start_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let homo_end = cfg.get("homo_end_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let window = cfg.get("window_size").and_then(|v| v.as_u64()).unwrap_or(300) as usize;
        Ok(Self { tok, bert, homodict, phrases, window, homo_start, homo_end, device: w.device })
    }

    pub fn is_new(&self) -> bool {
        !self.phrases.is_empty()
    }

    /// Поиск фразы-шаблона в размеченном контексте (как regex.search: слева направо, варианты по порядку).
    fn by_phrases(&self, word: &str, mark: &str) -> Option<String> {
        let vars = self.phrases.get(word)?;
        let m: Vec<char> = lower(mark).chars().collect();
        let boundary = |c: Option<&char>| c.is_none_or(|c| !(is_cyr(*c) || *c == '-'));
        for i in 0..m.len() {
            if !boundary(if i == 0 { None } else { m.get(i - 1) }) {
                continue;
            }
            for (var, ps) in vars {
                for p in ps {
                    if m.len() - i >= p.len() && m[i..i + p.len()] == p[..] && boundary(m.get(i + p.len())) {
                        return Some(var.clone());
                    }
                }
            }
        }
        None
    }

    /// Ударения омографов в предложении (put_stress, put_yo, stress_single_vowel = True).
    pub fn apply(&self, sentence: &str) -> Result<String> {
        let chars: Vec<char> = sentence.chars().collect();
        // (?=.*[а-яё])[а-яё+]+ c IGNORECASE: серии кириллицы и «+», если дальше в строке есть буква
        let mut found: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if is_cyr(chars[i]) || chars[i] == '+' {
                let mut j = i;
                while j < chars.len() && (is_cyr(chars[j]) || chars[j] == '+') {
                    j += 1;
                }
                let has_letter = chars[i..].iter().take_while(|&&c| c != '\n').any(|&c| is_cyr(c));
                if has_letter {
                    found.push((i, j));
                }
                i = j;
            } else {
                i += 1;
            }
        }
        let new = self.is_new();
        let mut items: Vec<(usize, usize, String, Option<String>)> = Vec::new(); // (start, end, word, вариант по фразам)
        let mut batch: Vec<Vec<i64>> = Vec::new();
        let (mut starts, mut ends) = (Vec::new(), Vec::new());
        for (s, e) in found {
            let word: String = chars[s..e].iter().collect();
            let wl = lower(&word);
            let in_dict = self.homodict.contains_key(&wl);
            let is_h = in_dict || self.phrases.contains_key(&wl);
            if !is_h {
                continue;
            }
            let mark = if new {
                let before: String = chars[..s].iter().collect();
                let after: String = chars[e..].iter().collect();
                let a = tail_chars(&clean_text(&before, true), self.window / 2);
                let b = head_chars(&clean_text(&after, false), self.window / 2);
                format!("{a} [HOMO] {wl} [/HOMO] {b}").trim().to_string()
            } else {
                let before: String = chars[..s].iter().collect();
                let after: String = chars[e..].iter().collect();
                format!("{before} [HOMO] {word} [/HOMO] {after}")
            };
            let by_phrase = if new { self.by_phrases(&wl, &mark) } else { None };
            if by_phrase.is_none() {
                if !in_dict {
                    continue;
                }
                let ids = self.tok.encode(&mark);
                let st = ids.iter().position(|&x| x == self.homo_start).ok_or_else(|| SileroError::Other("нет [HOMO] в токенах".into()))?;
                let en = ids.iter().position(|&x| x == self.homo_end).ok_or_else(|| SileroError::Other("нет [/HOMO] в токенах".into()))?;
                batch.push(ids);
                starts.push(st);
                ends.push(en);
            }
            items.push((s, e, word, by_phrase));
        }
        if items.is_empty() {
            return Ok(sentence.to_string());
        }
        let preds: Vec<bool> = if batch.is_empty() {
            Vec::new()
        } else {
            let l = batch.iter().map(Vec::len).max().unwrap_or(0);
            for b in batch.iter_mut() {
                b.resize(l, self.tok.pad);
            }
            // round(sigmoid(x)): 0.5 → 0 (округление к чётному)
            self.bert.logits(&batch, &starts, &ends, self.device)?.into_iter().map(|x| x > 0.0).collect()
        };
        let mut out: Vec<char> = chars.clone();
        let mut offset = 0usize;
        let mut ni = 0;
        for (s, e, word, by_phrase) in items {
            let pred = match by_phrase {
                Some(v) => v,
                None => {
                    let mut vars = self.homodict[&lower(&word)].clone();
                    vars.sort();
                    let p = preds[ni] as usize;
                    ni += 1;
                    vars.get(p).cloned().unwrap_or_else(|| vars[0].clone())
                }
            };
            let pc: Vec<char> = pred.chars().collect();
            let n_vowels = pc.iter().filter(|c| VOWELS.contains(c.to_lowercase().next().unwrap_or(**c))).count();
            let stress_idx = pc.iter().position(|&c| c == '+').unwrap_or(0);
            let plain: Vec<char> = pc.iter().copied().filter(|&c| c != '+').collect();
            let wc: Vec<char> = word.chars().collect();
            let mut nw: Vec<char> = wc
                .iter()
                .zip(&plain)
                .flat_map(|(c1, c2)| if c1.is_lowercase() { c2.to_lowercase().collect::<Vec<_>>() } else { c2.to_uppercase().collect::<Vec<_>>() })
                .collect();
            let _ = n_vowels;
            nw.insert(stress_idx.min(nw.len()), '+');
            let (s, e) = (s + offset, e + offset);
            out.splice(s..e, nw);
            offset += 1;
        }
        Ok(out.into_iter().collect())
    }
}
