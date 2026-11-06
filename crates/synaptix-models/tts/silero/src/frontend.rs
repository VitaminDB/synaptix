//! Подготовка текста как в `multi_acc_v3_package.py` (без SSML): переводы строк, нижний регистр,
//! тире, замена алфавита для языков СНГ (`ext_alph`), фильтр по символам модели, `^`-чистка;
//! типы предложений (вопросы, восклицания) и их раскладка по токенам — v5_4/v5_5 по символам
//! исходного текста, v5_6 — по токенам последовательности.

use std::collections::{HashMap, HashSet};

use crate::config::PackageConfig;
use crate::{Result, SileroError};

pub const TYPE_ST: i64 = 0;

fn type_id(t: &str) -> i64 {
    match t {
        "wh_q" => 1,
        "general_q" => 2,
        "alternative_q" => 3,
        "tag_q" => 4,
        "exclam" => 5,
        _ => 0,
    }
}

fn lower(s: &str) -> String {
    s.chars().flat_map(|c| c.to_lowercase()).collect()
}

/// Переводы строк: после знака препинания — пробел, иначе «. ».
pub fn newlines(text: &str) -> String {
    let t = text.trim();
    let c: Vec<char> = t.chars().collect();
    let mut o = String::new();
    for (i, &ch) in c.iter().enumerate() {
        if ch == '\n' {
            if i > 0 && ",;:.!?".contains(c[i - 1]) {
                o.push(' ');
            } else {
                o.push_str(". ");
            }
        } else {
            o.push(ch);
        }
    }
    o
}

fn clean_star_text(text: &str) -> String {
    let t = text.replace('^', " ^");
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.replace("^ ", "^");
    let mut o = String::new();
    for ch in t.chars() {
        if ch == '^' && o.ends_with('^') {
            continue;
        }
        o.push(ch);
    }
    let mut o = o.trim().to_string();
    if o.ends_with('^') {
        o.pop();
        o = o.trim().to_string();
    }
    o
}

/// `prepare_text_input`: текст для модели (нижний регистр, только её символы).
pub fn prepare(cfg: &PackageConfig, text: &str, lang: Option<&str>) -> Result<String> {
    let t = newlines(text);
    let t = lower(&t);
    let t = t.replace('—', "–").replace('‑', "-");
    let t = match (&cfg.ext_alph, lang) {
        (Some(ext), Some(l)) => {
            let alph: HashSet<&str> = cfg.alphabet.iter().map(String::as_str).collect();
            let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
            let all_known = letters.iter().all(|c| alph.contains(c.to_string().as_str()));
            match ext.get(l) {
                Some(map) if !letters.is_empty() && !all_known => t.chars().map(|c| map.get(&c.to_string()).cloned().unwrap_or_else(|| c.to_string())).collect(),
                _ => t,
            }
        }
        _ => t,
    };
    let allowed: HashSet<char> = cfg.symbols.chars().skip(3).chain(['^']).collect();
    let t: String = t.chars().filter(|c| allowed.contains(c)).collect();
    let s = clean_star_text(&t);
    let has_text = s.chars().any(|c| matches!(c, 'а'..='я' | '-'));
    if !has_text && !s.chars().any(|c| c.is_alphabetic()) {
        return Err(SileroError::Other("в тексте нет букв, которые знает модель".into()));
    }
    Ok(s)
}

/// Последовательность: sos + текст + eos → индексы символов.
pub fn sequence(cfg: &PackageConfig, text: &str) -> Result<Vec<i64>> {
    let full = format!("{}{}{}", cfg.sos_token, text, cfg.eos_token);
    full.chars()
        .map(|c| cfg.symbol_to_id.get(&c.to_string()).copied().ok_or_else(|| SileroError::Other(format!("символ «{c}» не знаком модели"))))
        .collect()
}

// ---------------------------------------------------------------- типы предложений

const WH_FORMS: &[&str] = &[
    "кто", "кого", "кому", "кем", "ком", "что", "чего", "чему", "чем", "чём", "чё", "чо", "где", "куда", "откуда", "когда", "почему", "зачем", "как", "почём", "отчего", "насколько", "сколько", "скольких", "скольким", "сколькими", "какой", "какая", "какое", "какие", "какого", "каких", "какому", "каким", "какими", "каком", "какую", "чей", "чья", "чьё", "чье", "чьи", "чьего", "чьей", "чьих", "чьему", "чьим", "чьими", "чьём", "чьем", "чью", "который", "которая", "которое", "которые", "которого", "которой", "которых", "которому", "которым", "которыми", "котором", "которую", "каков", "какова", "каково", "каковы",
];
const FILLERS: &[&str] = &["а", "ну", "и", "так", "вот", "слышь", "слушай", "скажите", "скажи", "пожалуйста", "вобще", "вообще", "типа", "короче"];
const TAGS: &[&[&str]] = &[&["правда"], &["верно"], &["да"], &["не", "так", "ли"], &["не", "правда", "ли"], &["разве", "не", "так"], &["ведь", "так"], &["ведь"], &["а"]];

fn strip_outer_quotes(s: &str) -> String {
    let open = ['"', '«', '\u{201C}', '\u{201E}'];
    let close = ['"', '»', '\u{201D}', '\u{2019}'];
    let mut s = s.trim().to_string();
    while s.starts_with(open) {
        s = s.chars().skip(1).collect::<String>().trim().to_string();
    }
    while s.ends_with(close) {
        s.pop();
        s = s.trim().to_string();
    }
    s
}

fn normalize_for_clf(s: &str) -> String {
    let s = strip_outer_quotes(s).replace('+', "").replace('ё', "е").replace('Ё', "Е");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// TAG_RE: `[,\s]+(хвост)\s*\?$` без учёта регистра.
fn has_tag(q: &str) -> bool {
    let l = lower(q);
    let Some(body) = l.strip_suffix('?') else { return false };
    let body = body.trim_end();
    for tag in TAGS {
        // слова хвоста с конца, между ними — пробелы
        let mut rest = body;
        let mut ok = true;
        for (k, w) in tag.iter().rev().enumerate() {
            if k > 0 {
                let t = rest.trim_end();
                if t.len() == rest.len() {
                    ok = false;
                    break;
                }
                rest = t;
            }
            match rest.strip_suffix(w) {
                Some(r) => rest = r,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && rest.chars().last().is_some_and(|c| c == ',' || c.is_whitespace()) {
            return true;
        }
    }
    false
}

fn words_ru_en(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower(s).chars() {
        if matches!(c, 'а'..='я' | 'ё' | 'a'..='z') {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn has_wh_prefix(q: &str) -> bool {
    let mut content = Vec::new();
    for w in words_ru_en(q).into_iter().take(8) {
        if FILLERS.contains(&w.as_str()) {
            continue;
        }
        content.push(w);
        if content.len() >= 4 {
            break;
        }
    }
    content.iter().any(|w| WH_FORMS.contains(&w.as_str()))
}

/// `\bили\b` без учёта регистра.
fn has_ili(q: &str) -> bool {
    let c: Vec<char> = lower(q).chars().collect();
    let word = |x: Option<&char>| x.is_some_and(|c| c.is_alphanumeric() || *c == '_');
    let p = ['и', 'л', 'и'];
    (0..c.len().saturating_sub(2)).any(|i| c[i..i + 3] == p && !word(if i == 0 { None } else { c.get(i - 1) }) && !word(c.get(i + 3)))
}

pub fn classify_sentence(s: &str) -> &'static str {
    let s = s.trim();
    if s.is_empty() {
        return "st";
    }
    let sc = strip_outer_quotes(s);
    if sc.is_empty() {
        return "st";
    }
    let mut tail = sc.clone();
    if tail.ends_with("?!") || tail.ends_with("?..") || tail.ends_with("?...") {
        tail = tail.trim_end_matches(['.', '!']).to_string();
    }
    if tail.ends_with('?') {
        let q = normalize_for_clf(&tail);
        if has_tag(&q) {
            return "tag_q";
        }
        if has_wh_prefix(&q) {
            return "wh_q";
        }
        if has_ili(&q) {
            return "alternative_q";
        }
        return "general_q";
    }
    if sc.ends_with('!') {
        return "exclam";
    }
    "st"
}

/// SENT_SPLIT_RE `(?<=[.!?])\s+`: куски и номер предложения для каждого символа (разделитель —
/// к предыдущему предложению).
fn sent_split(text: &str) -> (Vec<String>, Vec<usize>) {
    let c: Vec<char> = text.chars().collect();
    let mut parts = Vec::new();
    let mut ids = Vec::with_capacity(c.len());
    let mut cur = String::new();
    let mut k = 0;
    let mut i = 0;
    while i < c.len() {
        if c[i].is_whitespace() && i > 0 && ".!?".contains(c[i - 1]) {
            let mut j = i;
            while j < c.len() && c[j].is_whitespace() {
                j += 1;
            }
            ids.extend(std::iter::repeat_n(k, j - i));
            parts.push(std::mem::take(&mut cur));
            k += 1;
            i = j;
        } else {
            cur.push(c[i]);
            ids.push(k);
            i += 1;
        }
    }
    parts.push(cur);
    (parts, ids)
}

pub fn classify_text(text: &str) -> Vec<&'static str> {
    let t = text.trim();
    if t.is_empty() {
        return vec!["st"];
    }
    let (parts, _) = sent_split(t);
    let v: Vec<&str> = parts.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).map(classify_sentence).collect();
    if v.is_empty() { vec!["st"] } else { v }
}

/// v5_4/v5_5: типы по символам исходного текста (`build_type_ids_inference`).
pub fn type_ids_by_chars(text: &str, types: &[&str], seq_len: usize) -> Vec<i64> {
    let t = text.trim();
    let sents: Vec<String> = if t.is_empty() { vec![String::new()] } else { sent_split(t).0 };
    let mut per_char = Vec::new();
    for (i, s) in sents.iter().enumerate() {
        let ty = types.get(i).or(types.last()).copied().unwrap_or("st");
        let tid = type_id(ty);
        per_char.extend(std::iter::repeat_n(tid, s.chars().count()));
        if i + 1 < sents.len() {
            per_char.push(tid);
        }
    }
    let mut out = vec![0i64; seq_len];
    if per_char.is_empty() {
        return out;
    }
    let def = type_id(types.first().copied().unwrap_or("st"));
    out[0] = def;
    let mut idx = 1;
    for tid in per_char {
        if idx < seq_len {
            out[idx] = tid;
            idx += 1;
        }
    }
    while idx < seq_len {
        out[idx] = def;
        idx += 1;
    }
    out
}

/// v5_6: типы по токенам последовательности (`build_type_ids_tokens` без эмфазы).
pub fn type_ids_by_tokens(cfg: &PackageConfig, seq: &[i64], types: &[&str]) -> Vec<i64> {
    let id2s: HashMap<i64, &str> = cfg.symbol_to_id.iter().map(|(k, &v)| (v, k.as_str())).collect();
    let toks: Vec<&str> = seq.iter().map(|i| id2s.get(i).copied().unwrap_or("")).collect();
    let n = toks.len();
    let inner: String = toks[1..n - 1].concat();
    let (_, sent_ids) = sent_split(&inner);
    let nsent = sent_ids.last().map(|&k| k + 1).unwrap_or(0);
    let mut spans = Vec::new();
    for k in 0..nsent {
        let idx: Vec<usize> = sent_ids.iter().enumerate().filter(|(_, &s)| s == k).map(|(i, _)| i + 1).collect();
        if idx.is_empty() {
            continue;
        }
        let a = idx[0];
        let mut b = idx[idx.len() - 1] + 1;
        while b > a && toks[b - 1] == " " {
            b -= 1;
        }
        spans.push((a, b));
    }
    let owned: Vec<&'static str>;
    let types: &[&str] = if types.len() != spans.len() {
        owned = spans.iter().map(|&(a, b)| classify_sentence(&toks[a..b].concat().replace('+', ""))).collect();
        &owned
    } else {
        types
    };
    let tids: Vec<i64> = types.iter().map(|t| type_id(t)).collect();
    let values: Vec<i64> = sent_ids.iter().map(|&s| tids.get(s.min(tids.len().saturating_sub(1))).copied().unwrap_or(0)).collect();
    let mut out = vec![0i64; n];
    if values.is_empty() {
        return out;
    }
    for (i, &v) in values.iter().enumerate() {
        let j = i + 1;
        if j < n - 1 {
            out[j] = v;
        }
    }
    out[0] = values[0];
    out[n - 1] = *values.last().unwrap_or(&0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types() {
        assert_eq!(classify_sentence("Как дела?"), "wh_q");
        assert_eq!(classify_sentence("Ты придёшь?"), "general_q");
        assert_eq!(classify_sentence("Чай или кофе?"), "alternative_q");
        assert_eq!(classify_sentence("Хорошо, правда?"), "tag_q");
        assert_eq!(classify_sentence("Ну, не так ли?"), "tag_q");
        assert_eq!(classify_sentence("Неправда?"), "general_q");
        assert_eq!(classify_sentence("Ура!"), "exclam");
        assert_eq!(classify_sentence("Ну а кто там?"), "wh_q");
        assert_eq!(classify_text("Привет! Как у тебя дела сегодня? Замок был старый."), vec!["exclam", "wh_q", "st"]);
    }

    #[test]
    fn stars() {
        assert_eq!(clean_star_text("а ^^ б^"), "а ^б");
        assert_eq!(newlines("раз\nдва,\nтри"), "раз. два, три");
    }
}
