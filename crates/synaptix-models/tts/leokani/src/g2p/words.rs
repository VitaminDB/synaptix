//! Числа словами, римские цифры, проверки слов (произносимость, эмодзи, ʻokina), тесты омографов
//! и деление длинного незнакомого слова — свободные функции `leokani_g2p.py`.

use unicode_normalization::UnicodeNormalization;

use super::py::{self, re, Bail, R};
use super::tables::t;

pub const PRIMARY: char = 'ˈ';
pub const SECONDARY: char = 'ˌ';
/// "word" + PH + phonemes: слово с заданной базой (контекстная модель всё равно видит "word").
pub const PH: char = '\u{E000}';
pub const PH_S: &str = "\u{E000}";

/// `ONES[n]` с отрицательными индексами Python.
fn ones_py(n: i128) -> R<String> {
    let k = if n < 0 { n + 20 } else { n };
    t().one(k)
}

pub fn cardinal(n: i128) -> R<String> {
    if n < 20 {
        return ones_py(n);
    }
    if n < 100 {
        let tens = &t().tens[(n / 10) as usize];
        return Ok(if n % 10 == 0 { tens.clone() } else { format!("{tens}-{}", t().one(n % 10)?) });
    }
    for (scale, name) in [(1_000_000_000i128, "billion"), (1_000_000, "million"), (1000, "thousand"), (100, "hundred")] {
        if n >= scale {
            let (q, r) = (n / scale, n % scale);
            let mut s = format!("{} {name}", cardinal(q)?);
            if r != 0 {
                s.push(' ');
                s.push_str(&cardinal(r)?);
            }
            return Ok(s);
        }
    }
    Err(Bail)
}

pub fn ordinal(n: i128) -> R<String> {
    let c = cardinal(n)?;
    let mut words: Vec<String> = c.split(' ').map(str::to_string).collect();
    let lastw = words.last().unwrap().clone();
    let (head, last) = match lastw.rfind('-') {
        Some(k) => (lastw[..k].to_string(), lastw[k + 1..].to_string()),
        None => (String::new(), lastw.clone()),
    };
    let last = if let Some(o) = t().ordinal_words.get(&last) {
        o.clone()
    } else if let Some(stem) = last.strip_suffix('y') {
        format!("{stem}ieth")
    } else {
        format!("{last}th")
    };
    *words.last_mut().unwrap() = if head.is_empty() { last } else { format!("{head}-{last}") };
    Ok(words.join(" "))
}

pub fn plural_number(n: i128) -> R<String> {
    if (1000..=9999).contains(&n) && n % 100 == 0 && n % 1000 != 0 {
        return Ok(format!("{} hundreds", cardinal(n / 100)?));
    }
    if (1100..=1999).contains(&n) || (2010..=2099).contains(&n) {
        return Ok(format!("{} {}", cardinal(n / 100)?, plural_number(n % 100)?));
    }
    let w = cardinal(n)?;
    Ok(match w.strip_suffix('y') {
        Some(s) => format!("{s}ies"),
        None => format!("{w}s"),
    })
}

pub fn split_words(s: &str) -> Vec<String> {
    py::split_sp(s)
}

pub fn year_words(y: i128) -> R<Vec<String>> {
    if (2000..=2009).contains(&y) || !(1000..=2099).contains(&y) {
        return Ok(split_words(&cardinal(y)?));
    }
    let (hi, lo) = (y / 100, y % 100);
    if lo == 0 {
        return Ok(vec![cardinal(hi)?, "hundred".into()]);
    }
    let mut out = vec![cardinal(hi)?];
    if lo < 10 {
        out.push("oh".into());
        out.push(t().one(lo)?);
    } else {
        out.push(cardinal(lo)?);
    }
    Ok(out)
}

pub fn pair_words(n: i128) -> R<Vec<String>> {
    if n.rem_euclid(1000) == 0 {
        return Ok(split_words(&cardinal(n)?));
    }
    if (1000..=9999).contains(&n) {
        year_words(n)
    } else {
        Ok(split_words(&cardinal(n)?))
    }
}

pub fn digit_words(s: &str) -> R<Vec<String>> {
    let mut out = Vec::new();
    for c in s.chars() {
        if !py::c_digit(c) {
            continue;
        }
        if c == '0' {
            out.push("zero".to_string());
        } else {
            out.push(t().one(py::int(&c.to_string())?)?);
        }
    }
    Ok(out)
}

pub fn roman_value(s: &str) -> R<i128> {
    let val = |c: char| -> R<i128> {
        Ok(match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return Err(Bail),
        })
    };
    let cs: Vec<char> = s.chars().collect();
    let mut total = 0;
    for (k, &c) in cs.iter().enumerate() {
        let v = val(c)?;
        if k + 1 < cs.len() && val(cs[k + 1])? > v {
            total -= v;
        } else {
            total += v;
        }
    }
    Ok(total)
}

pub fn roman_match(s: &str) -> bool {
    py::is_m(re!(r"^(?=[IVXLCDM])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"), s)
}

/// Гавайское написание для поиска: без ʻokina и макронов.
pub fn strip_okina(w: &str) -> String {
    let w = py::re_sub(re!(r"(?<=[\p{L}\p{N}_])[ʻ‘'’](?=[\p{L}\p{N}_])"), w, |_| String::new());
    w.nfd().filter(|&c| py::cat(c) != unicode_properties::GeneralCategory::NonspacingMark).collect()
}

pub fn is_emoji(w: &str) -> bool {
    use unicode_properties::GeneralCategory as Gc;
    !w.is_empty()
        && w.chars().all(|c| {
            matches!(py::cat(c), Gc::OtherSymbol | Gc::ModifierSymbol | Gc::Format | Gc::NonspacingMark)
                || (0x1F000..=0x1FAFF).contains(&(c as u32))
                || c == '\u{FE0F}'
                || c == '\u{200D}'
        })
        && w.chars().any(|c| (0x1F000..=0x1FAFF).contains(&(c as u32)) || (0x2600..=0x27BF).contains(&(c as u32)))
}

/// Мог бы англоязычный читатель сказать этот ряд букв словом ("BLORB" да, "AGC"/"QRT" нет)?
pub fn pronounceable(w: &str) -> bool {
    let w = py::lower(w);
    if !w.is_ascii() || !py::is_alpha(&w) {
        return false;
    }
    if py::clen(&w) <= 3 {
        return py::is_m(re!(r"^[bcdfghjklmnpqrstvwxz]?[aeiou][bcdfgklmnprstvxz]?$"), &w);
    }
    let groups = py::findall(re!(r"[aeiou]+|(?<=[^aeiou])y(?![aeiou])|[^aeiou]"), &w);
    let vowel = |g: &str| g.starts_with(|c| "aeiou".contains(c)) || g == "y";
    let mut merged: Vec<String> = Vec::new();
    for g in groups {
        if let Some(l) = merged.last_mut() {
            if !vowel(&g) && !vowel(l) {
                l.push_str(&g);
                continue;
            }
        }
        merged.push(g);
    }
    let vows: Vec<&String> = merged.iter().filter(|g| vowel(g)).collect();
    if vows.is_empty() || vows.iter().any(|v| v.len() > 2) {
        return false;
    }
    let tt = t();
    for (k, g) in merged.iter().enumerate() {
        if vowel(g) {
            continue;
        }
        if k == 0 {
            if !tt.onsets.contains(g.as_str()) {
                return false;
            }
        } else if k == merged.len() - 1 {
            if !tt.codas.contains(g.as_str()) {
                return false;
            }
        } else if !(0..=g.len()).any(|j| tt.codas.contains(&g[..j]) && tt.onsets.contains(&g[j..])) || g.len() > 4 {
            return false;
        }
    }
    true
}

/// `w in hg_set(vals)`.
fn in_hg(w: &str, vals: &str) -> bool {
    vals.split('|').any(|x| match x.strip_prefix('@') {
        Some(cls) => t().hg_sets.get(cls).is_some_and(|s| s.contains(w)),
        None => x == w,
    })
}

pub fn hg_test(tst: &str, prev: &[String], nxt: &[String], cap: bool) -> bool {
    if tst == "cap" {
        return cap;
    }
    if tst == "!cap" {
        return !cap;
    }
    let mut op = "~";
    let mut k: isize = -1;
    for o in ["!=", "!~", "=", "~"] {
        op = o;
        k = tst.find(o).map(|x| x as isize).unwrap_or(-1);
        if k > 0 {
            break;
        }
    }
    if k < 0 {
        return false;
    }
    let k = k as usize;
    let field = &tst[..k];
    let vals = &tst[k + op.len()..];
    let ws: Vec<&str> = if field == "P" || field == "N" {
        let seq = if field == "P" { prev } else { nxt };
        let v: Vec<&str> = seq.iter().map(|s| s.as_str()).filter(|w| *w != "^" && *w != "$").collect();
        if v.is_empty() {
            vec![seq.last().map(|s| s.as_str()).unwrap_or("")]
        } else {
            v
        }
    } else {
        let seq = if field.starts_with('p') { prev } else { nxt };
        let n = field[1..2].parse::<usize>().unwrap_or(1) - 1;
        vec![if n < seq.len() { seq[n].as_str() } else { seq.last().map(|s| s.as_str()).unwrap_or("") }]
    };
    if op == "~" || op == "!~" {
        let ends: Vec<&str> = vals.split('|').collect();
        let hit = ws.iter().any(|w| *w != "^" && *w != "$" && ends.iter().any(|e| w.ends_with(e)));
        return if op == "~" { hit } else { !hit };
    }
    let hit = ws.iter().any(|w| in_hg(w, vals));
    if op == "!=" {
        !hit
    } else {
        hit
    }
}

const LONG_WORD: usize = 20;
const PART: usize = 12;

/// Незнакомое слово длиннее LONG_WORD говорится частями (модель зацикливается на длинных входах).
pub fn split_long(w: &str) -> Vec<String> {
    let cs: Vec<char> = w.chars().collect();
    let len = cs.len();
    if len <= LONG_WORD {
        return vec![w.to_string()];
    }
    let n = len.div_ceil(PART);
    let size = len as f64 / n as f64;
    let vowel = |c: char| -> bool {
        let l: String = c.to_lowercase().collect();
        "aeiouy".contains(l.as_str())
    };
    let mut cuts: Vec<usize> = Vec::new();
    let mut last: i64 = 0;
    for k in 1..n {
        let aim = py::round_half_even(k as f64 * size);
        let mut best = aim;
        for d in [0i64, 1, -1, 2, -2, 3, -3] {
            let j = aim + d;
            if last < j && j < len as i64 && vowel(cs[(j - 1) as usize]) && !vowel(cs[j as usize]) {
                best = j;
                break;
            }
        }
        cuts.push(best as usize);
        last = best;
    }
    let mut starts = vec![0usize];
    starts.extend(cuts.iter().copied());
    let mut ends = cuts.clone();
    ends.push(len);
    starts
        .iter()
        .zip(ends.iter())
        .filter(|(a, b)| a < b)
        .map(|(&a, &b)| cs[a..b].iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_words() {
        assert_eq!(cardinal(1234).unwrap(), "one thousand two hundred thirty-four");
        assert_eq!(ordinal(21).unwrap(), "twenty-first");
        assert_eq!(ordinal(40).unwrap(), "fortieth");
        assert_eq!(plural_number(1990).unwrap(), "nineteen nineties");
        assert_eq!(year_words(1905).unwrap(), vec!["nineteen", "oh", "five"]);
        assert_eq!(roman_value("MCMXCIV").unwrap(), 1994);
        assert!(roman_match("XIV") && !roman_match("") && !roman_match("IIII"));
        assert!(pronounceable("BLORB") && !pronounceable("AGC"));
        assert_eq!(strip_okina("Hawaiʻi"), "Hawaii");
    }
}
