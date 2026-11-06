//! Строковый фронтенд (как в `leokani.py`): чистка разметки, деление на куски-клаузы, разбиение на
//! безпунктуационные «пьесы», переписывание местоимения I и омографов, сборка строки фонем.

use unicode_properties::GeneralCategory as Gc;

use super::py::{self, re};
use super::tables::t;

const PUNCT: [&str; 8] = [",", ".", "!", "?", ";", ":", "—", "…"];

fn is_punct(s: &str) -> bool {
    PUNCT.contains(&s)
}

/// Символы с последующими комбинирующими знаками (и ZWJ) — одним кластером.
pub fn clusters(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in s.chars() {
        let mark = matches!(py::cat(c), Gc::NonspacingMark | Gc::SpacingMark | Gc::EnclosingMark) || c == '\u{200D}';
        match out.last_mut() {
            Some(l) if mark => l.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

/// Убрать знаки разметки и свернуть пробелы.
pub fn clean(text: &str) -> String {
    let s = py::re_sub(re!(r"[*_`#>|]+"), text, |_| " ".to_string());
    let s = py::re_sub(re!(r"\s+"), &s, |_| " ".to_string());
    py::strip_ws(&s).to_string()
}

/// Куски-клаузы: разрыв после . ! ? ; : (не после титула вроде 'Dr.') и после запятой, когда в куске
/// 12 слов. Первому куску нужно 3 слова.
pub fn chunks(text: &str) -> Vec<String> {
    let tt = t();
    let mut out: Vec<String> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for w in text.split(' ') {
        if w.is_empty() {
            continue;
        }
        cur.push(w);
        let end = py::last(w).unwrap();
        let brk = "!?;:".contains(end)
            || (end == '.' && !tt.titles.contains(py::rstrip_set(py::strip_set(&py::lower(w), "(\"'"), ".")))
            || (end == ',' && cur.len() >= 12);
        if brk && cur.len() >= if out.is_empty() { 3 } else { 1 } {
            out.push(cur.join(" "));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.join(" "));
    }
    out.into_iter().filter(|c| c.chars().any(py::c_alnum)).collect()
}

/// Местоимение I, омографы и несколько имён, переписанные по контексту.
fn respell(word: &str, prev: Option<&str>, nxt: &[String]) -> Option<String> {
    let tt = t();
    let w = py::lower(word);
    let p = prev.map(py::lower);
    let n: Vec<String> = nxt.iter().map(|x| py::lower(x)).collect();
    let wq = w.replace('’', "'");
    if word.starts_with('I') && tt.pronoun.contains_key(&wq) {
        if let Some(p) = &p {
            if tt.pronoun_after.contains(&p.replace('’', "'")) {
                return None;
            }
        }
        return Some(tt.pronoun[&wq].clone());
    }
    if let Some(x) = tt.names.get(&w) {
        return Some(x.clone());
    }
    let p_in = |set: &std::collections::HashSet<String>| p.as_ref().is_some_and(|p| set.contains(p));
    let n0_in = |set: &std::collections::HashSet<String>| n.first().is_some_and(|x| set.contains(x));
    if w == "read" {
        if p_in(&tt.read_past_before) || n.iter().any(|x| tt.read_past_after.contains(x)) {
            return Some("red".into());
        }
    } else if w == "lead" {
        if n0_in(&tt.lead_metal) || p.as_deref() == Some("of") {
            return Some("led".into());
        }
    } else if w == "wind" || w == "winds" {
        let s = if w == "winds" { "s" } else { "" };
        if n.first().is_some_and(|x| x == "up" || x == "down") {
            return Some(format!("wined{s}"));
        }
        if n0_in(&tt.wind_object) && (p.is_none() || p_in(&tt.wind_verb_before)) {
            return Some(format!("wined{s}"));
        }
    } else if w == "lives" {
        if p_in(&tt.lives_subject) {
            return Some("livs".into());
        }
        if let (Some(pp), Some(praw)) = (&p, prev) {
            if py::first(praw).is_some_and(py::c_upper) && !tt.lives_owner.contains(pp) {
                return Some("livs".into());
            }
        }
    }
    if p_in(&tt.year_before) && py::clen(&w) == 4 && w.is_ascii() && py::is_digit(&w) {
        let y: i64 = w.parse().ok()?;
        if (1100..=1999).contains(&y) || (2010..=2099).contains(&y) {
            let (hi, lo) = (y / 100, y % 100);
            return Some(if lo == 0 {
                format!("{hi} hundred")
            } else if lo < 10 {
                format!("{hi} oh {lo}")
            } else {
                format!("{hi} {lo}")
            });
        }
    }
    None
}

/// Слова, сгруппированные в клаузы без пунктуации, со знаками между ними отдельными кусками.
pub fn pieces(text: &str) -> Vec<String> {
    let tt = t();
    let mut words: Vec<(String, String)> = Vec::new();
    for tk in py::split_ws(text) {
        let mut core = clusters(tk);
        let mut a = 0;
        while a < core.len() && tt.dropped.contains(&core[a]) {
            a += 1;
        }
        core.drain(..a);
        let mut trail = String::new();
        while let Some(last) = core.last() {
            if !(tt.dropped.contains(last) || is_punct(last)) {
                break;
            }
            if is_punct(last) {
                if last == "." && tt.abbrev.contains(&py::lower(&core.concat())) {
                    break;
                }
                trail = format!("{last}{trail}");
            }
            core.pop();
        }
        if !core.is_empty() && core.iter().all(|c| c == "-") {
            words.push((String::new(), "—".into()));
            continue;
        }
        words.push((core.concat(), trail));
    }
    let mut out: Vec<String> = Vec::new();
    let mut clause: Vec<String> = Vec::new();
    for (i, (word, punct)) in words.iter().enumerate() {
        if !word.is_empty() {
            let prev = if i > 0 && words[i - 1].1.is_empty() { Some(words[i - 1].0.as_str()) } else { None };
            let mut nxt: Vec<String> = Vec::new();
            if punct.is_empty() {
                for w in words.iter().take(words.len().min(i + 5)).skip(i + 1) {
                    nxt.push(w.0.clone());
                    if !w.1.is_empty() {
                        break;
                    }
                }
            }
            clause.push(respell(word, prev, &nxt).unwrap_or_else(|| word.clone()));
        }
        if !punct.is_empty() {
            if !clause.is_empty() {
                out.push(clause.join(" "));
            }
            clause.clear();
            out.push(if punct.contains("..") { "…".to_string() } else { clusters(punct).into_iter().take(2).collect() });
        }
    }
    if !clause.is_empty() {
        out.push(clause.join(" "));
    }
    out
}

/// Символы, которые платформы читают по-разному (римские цифры, буквы в кружках), — пробелы.
pub fn desilence(piece: &str) -> String {
    piece
        .chars()
        .map(|c| {
            let u = c as u32;
            if (0x2160..=0x2182).contains(&u) || (0x2185..=0x2188).contains(&u) || (0x24B6..=0x24E9).contains(&u) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// Кусок текста → фонемы: клаузы через `piece`, пунктуация как есть.
pub fn line(text: &str, piece: impl Fn(&str) -> String) -> String {
    let mut o = String::new();
    for p in pieces(text) {
        if clusters(&p).iter().all(|c| is_punct(c)) {
            o.push_str(&p);
            continue;
        }
        let ph = piece(&desilence(&p));
        if ph.is_empty() {
            continue;
        }
        if !o.is_empty() && !o.ends_with(' ') {
            o.push(' ');
        }
        o.push_str(&ph);
    }
    py::strip_ws(&o).to_string()
}
