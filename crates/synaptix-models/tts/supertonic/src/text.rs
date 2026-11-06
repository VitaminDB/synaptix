//! Текст как у `UnicodeProcessor` / `chunk_text` upstream (`py/helper.py`).

use unicode_normalization::UnicodeNormalization;

pub const LANGS: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi", "na",
];

fn is_emoji(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x1F600..=0x1F64F | 0x1F300..=0x1F5FF | 0x1F680..=0x1F6FF | 0x1F700..=0x1F77F | 0x1F780..=0x1F7FF |
        0x1F800..=0x1F8FF | 0x1F900..=0x1F9FF | 0x1FA00..=0x1FA6F | 0x1FA70..=0x1FAFF | 0x2600..=0x26FF |
        0x2700..=0x27BF | 0x1F1E6..=0x1F1FF)
}

/// `_preprocess_text`: NFKD, без эмодзи, замены знаков, пробелы у пунктуации,
/// точка в конце, теги языка `<xx>…</xx>`.
pub fn preprocess(text: &str, lang: &str) -> Result<String, String> {
    if !LANGS.contains(&lang) {
        return Err(format!("язык «{lang}» не поддержан; есть: {}", LANGS.join(", ")));
    }
    let mut t: String = text.nfkd().filter(|c| !is_emoji(*c)).collect();
    for (k, v) in [
        ("–", "-"), ("‑", "-"), ("—", "-"), ("_", " "), ("\u{201c}", "\""), ("\u{201d}", "\""), ("\u{2018}", "'"), ("\u{2019}", "'"),
        ("´", "'"), ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "), ("→", " "), ("←", " "),
    ] {
        t = t.replace(k, v);
    }
    t.retain(|c| !matches!(c, '♥' | '☆' | '♡' | '©' | '\\'));
    for (k, v) in [("@", " at "), ("e.g.,", "for example, "), ("i.e.,", "that is, ")] {
        t = t.replace(k, v);
    }
    for p in [",", ".", "!", "?", ";", ":", "'"] {
        t = t.replace(&format!(" {p}"), p);
    }
    while t.contains("\"\"") {
        t = t.replace("\"\"", "\"");
    }
    while t.contains("''") {
        t = t.replace("''", "'");
    }
    while t.contains("``") {
        t = t.replace("``", "`");
    }
    let mut t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let ends = ['.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', '…', '。', '」', '』', '】', '〉', '》', '›', '»'];
    if !t.chars().last().is_some_and(|c| ends.contains(&c)) {
        t.push('.');
    }
    Ok(format!("<{lang}>{t}</{lang}>"))
}

/// Индексы символов по `unicode_indexer.json` (код символа → номер; −1 —
/// неизвестный: как Gather ONNX с индексом −1 — последняя строка таблицы).
pub fn ids(text: &str, indexer: &[i64], vocab: usize) -> Vec<i64> {
    text.chars()
        .map(|c| {
            // upstream: np.uint16(ord(c)) — коды выше 0xFFFF обрезаются.
            let u = (c as u32 & 0xFFFF) as usize;
            let v = indexer.get(u).copied().unwrap_or(-1);
            if v < 0 { vocab as i64 + v } else { v }
        })
        .collect()
}

const ABBR: &[&str] = &["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ph.D.", "etc.", "e.g.", "i.e.", "vs.", "Inc.", "Ltd.", "Co.", "Corp.", "St.", "Ave.", "Blvd."];

/// `chunk_text`: абзацы по пустым строкам, предложения по `.!?` + пробел
/// (кроме сокращений и одиночной заглавной буквы), куски до `max_len` символов.
pub fn chunks(text: &str, max_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let norm = text.trim().replace("\r\n", "\n");
    let mut paragraphs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut blank = false;
    for line in norm.split('\n') {
        if line.trim().is_empty() {
            blank = true;
            continue;
        }
        if blank && !cur.trim().is_empty() {
            paragraphs.push(std::mem::take(&mut cur));
        }
        blank = false;
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }
    if !cur.trim().is_empty() {
        paragraphs.push(cur);
    }
    for p in paragraphs {
        let p = p.trim();
        let chars: Vec<char> = p.chars().collect();
        let mut sentences: Vec<String> = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            if matches!(chars[i], '.' | '!' | '?') && i + 1 < chars.len() && chars[i + 1].is_whitespace() {
                let before: String = chars[start..=i].iter().collect();
                let abbr = ABBR.iter().any(|a| before.ends_with(a));
                let single = i >= 1 && chars[i - 1].is_ascii_uppercase() && (i < 2 || !chars[i - 2].is_alphanumeric());
                if !abbr && !single {
                    sentences.push(before);
                    let mut j = i + 1;
                    while j < chars.len() && chars[j].is_whitespace() {
                        j += 1;
                    }
                    start = j;
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        if start < chars.len() {
            sentences.push(chars[start..].iter().collect());
        }
        let mut cur = String::new();
        for s in sentences {
            if cur.chars().count() + s.chars().count() + 1 <= max_len {
                if !cur.is_empty() {
                    cur.push(' ');
                }
                cur.push_str(&s);
            } else {
                if !cur.is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur = s;
            }
        }
        if !cur.is_empty() {
            out.push(cur.trim().to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_like_upstream() {
        assert_eq!(preprocess("Hello  world", "en").unwrap(), "<en>Hello world.</en>");
        assert_eq!(preprocess("Hi , there!", "en").unwrap(), "<en>Hi, there!</en>");
        assert!(preprocess("x", "zz").is_err());
    }

    #[test]
    fn chunking() {
        let c = chunks("Mr. Smith came. He left! Ok?", 300);
        assert_eq!(c, vec!["Mr. Smith came. He left! Ok?"]);
        let c = chunks("One two. Three four.", 10);
        assert_eq!(c, vec!["One two.", "Three four."]);
    }
}
