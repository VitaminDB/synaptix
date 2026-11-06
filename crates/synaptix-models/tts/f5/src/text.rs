//! Текст как у F5-TTS: `convert_char_to_pinyin` (без китайского — посимвольно, с заменами `;“”‘’`
//! и пробелом перед ASCII-сегментом длиннее символа, как после сегментации rjieba), словарь
//! `vocab.txt` (строка = индекс, неизвестное — 0), деление на куски `chunk_text`.

use std::collections::HashMap;

pub struct Vocab {
    map: HashMap<String, i64>,
}

impl Vocab {
    /// `vocab.txt`: символ на строку, индекс — номер строки (как `char[:-1]` upstream). Файл
    /// открывается в Python текстовым режимом — `\r\n` и `\r` там тоже конец строки (у русских
    /// версий словарь с CRLF).
    pub fn parse(s: &str) -> Self {
        let s = s.replace("\r\n", "\n").replace('\r', "\n");
        let mut map = HashMap::new();
        for (i, line) in s.split_inclusive('\n').enumerate() {
            let tok = line.strip_suffix('\n').unwrap_or_else(|| {
                // последняя строка без перевода — upstream всё равно отрезает символ
                let mut c = line.chars();
                c.next_back();
                c.as_str()
            });
            map.insert(tok.to_string(), i as i64);
        }
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn ids(&self, tokens: &[String]) -> Vec<i64> {
        tokens.iter().map(|t| self.map.get(t).copied().unwrap_or(0)).collect()
    }
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

/// Сегменты как у rjieba на тексте без китайского: ASCII-слова (`GPT-4`, `A.B`), серии одинаковой
/// ASCII-пунктуации из `.+#&%_-` (`...`), остальное — по символу.
fn segments(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if is_word_char(c[i]) {
            let mut j = i;
            while j < c.len() && is_word_char(c[j]) {
                j += 1;
            }
            // связки: слово [.-+#&%_] слово
            while j + 1 < c.len() && ".-+#&%_".contains(c[j]) && is_word_char(c[j + 1]) {
                j += 1;
                while j < c.len() && is_word_char(c[j]) {
                    j += 1;
                }
            }
            out.push(c[i..j].iter().collect());
            i = j;
        } else if ".+#&%_-".contains(c[i]) {
            let mut j = i;
            while j < c.len() && c[j] == c[i] {
                j += 1;
            }
            out.push(c[i..j].iter().collect());
            i = j;
        } else {
            out.push(c[i].to_string());
            i += 1;
        }
    }
    out
}

/// `convert_char_to_pinyin` для одного текста (без китайского): список токенов-символов.
pub fn to_tokens(text: &str) -> Vec<String> {
    let t: String = text
        .chars()
        .map(|c| match c {
            ';' => ',',
            '“' | '”' => '"',
            '‘' | '’' => '\'',
            c => c,
        })
        .collect();
    let mut out: Vec<String> = Vec::new();
    for seg in segments(&t) {
        let ascii = seg.is_ascii();
        if ascii {
            if out.last().is_some_and(|l| !" :'\"".contains(l.as_str())) && seg.len() > 1 {
                out.push(" ".into());
            }
            out.extend(seg.chars().map(|c| c.to_string()));
        } else {
            out.extend(seg.chars().map(|c| c.to_string()));
        }
    }
    out
}

/// `chunk_text`: предложения по `;:,.!?` + пробел, куски до `max_bytes` байт UTF-8.
pub fn chunk_text(text: &str, max_bytes: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if ";:,.!?".contains(c) && i + 1 < chars.len() && chars[i + 1].is_whitespace() {
            sentences.push(std::mem::take(&mut cur));
            while i + 1 < chars.len() && chars[i + 1].is_whitespace() {
                i += 1;
            }
        } else if "；：，。！？".contains(c) {
            sentences.push(std::mem::take(&mut cur));
        }
        i += 1;
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    let one_byte_end = |s: &str| s.chars().last().is_some_and(|c| c.len_utf8() == 1);
    let mut chunks = Vec::new();
    let mut chunk = String::new();
    for s in sentences.into_iter().filter(|s| !s.is_empty()) {
        if chunk.len() + s.len() <= max_bytes {
            chunk.push_str(&s);
            if one_byte_end(&s) {
                chunk.push(' ');
            }
        } else {
            if !chunk.is_empty() {
                chunks.push(chunk.trim().to_string());
            }
            chunk = s.clone();
            if one_byte_end(&s) {
                chunk.push(' ');
            }
        }
    }
    if !chunk.is_empty() {
        chunks.push(chunk.trim().to_string());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(s: &str) -> String {
        to_tokens(s).concat()
    }

    #[test]
    fn like_upstream() {
        assert_eq!(conv("Привет, мир!"), "Привет, мир!");
        assert_eq!(conv("Привет..."), "Привет ...");
        assert_eq!(conv("Это GPT-4 и iPhone 15 Pro."), "Это GPT-4 и iPhone 15 Pro.");
        assert_eq!(conv("молок+о; «ёлка» “q” ‘a’"), "молок+о, «ёлка» \"q\" 'a'");
        assert_eq!(conv("тестHello мир"), "тест Hello мир");
        assert_eq!(conv("x...y"), "x ...y");
        assert_eq!(conv("Он сказал:\"да\""), "Он сказал:\"да\"");
    }

    #[test]
    fn vocab_crlf() {
        let v = Vocab::parse(" \r\n!\r\nа\r\n");
        assert_eq!(v.ids(&["!".into(), "а".into(), " ".into(), "?".into()]), vec![1, 2, 0, 0]);
    }

    #[test]
    fn chunks() {
        assert_eq!(chunk_text("Раз, два. Три!", 1000), vec!["Раз, два. Три!"]);
        assert_eq!(chunk_text("Раз. Два.", 6).len(), 2);
    }
}
