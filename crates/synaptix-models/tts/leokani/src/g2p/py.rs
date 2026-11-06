//! Семантика строк и `re` из Python, нужная для точного порта: `str.isupper()/isalpha()/isdigit()…`
//! по символам Unicode, срезы по кодовым точкам, `int()`, `round()` (банковское), разбиение `split()`,
//! кэш регулярных выражений (`fancy_regex`: lookbehind/lookahead как в `re`).

use fancy_regex::{Captures, Regex};
use unicode_properties::{GeneralCategory as Gc, UnicodeGeneralCategory};

/// Ошибка «как исключение в Python» (ValueError/KeyError/IndexError…): правило нормализатора
/// при ней оставляет токен как есть, читатель чисел возвращает None.
#[derive(Debug, Clone, Copy)]
pub struct Bail;
pub type R<T> = std::result::Result<T, Bail>;

/// Регулярное выражение, скомпилированное один раз на место вызова.
macro_rules! re {
    ($p:expr) => {{
        static RE: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| fancy_regex::Regex::new($p).expect("leokani g2p: регулярное выражение"))
    }};
}
pub(crate) use re;

// ---------------------------------------------------------------- символы

pub fn cat(c: char) -> Gc {
    c.general_category()
}

/// `str.isalpha()` для символа: категории L*.
pub fn c_alpha(c: char) -> bool {
    matches!(cat(c), Gc::UppercaseLetter | Gc::LowercaseLetter | Gc::TitlecaseLetter | Gc::ModifierLetter | Gc::OtherLetter)
}

/// `str.isdecimal()`: Nd (то же, что `\d` в `re`).
pub fn c_decimal(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    cat(c) == Gc::DecimalNumber
}

/// `str.isdigit()`: Nd и символы с Numeric_Type=Digit (верхние/нижние индексы, цифры в кружках…).
pub fn c_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    if cat(c) == Gc::DecimalNumber {
        return true;
    }
    let u = c as u32;
    matches!(u,
        0xB2 | 0xB3 | 0xB9 | 0x1369..=0x1371 | 0x19DA | 0x2070 | 0x2074..=0x2079 | 0x2080..=0x2089
        | 0x2460..=0x2468 | 0x2474..=0x247C | 0x2488..=0x2490 | 0x24EA | 0x24F5..=0x24FD | 0x24FF
        | 0x2776..=0x277E | 0x2780..=0x2788 | 0x278A..=0x2792 | 0x10A40..=0x10A43 | 0x10E60..=0x10E68
        | 0x11052..=0x1105A | 0x1F100..=0x1F10A)
}

/// `str.isnumeric()` (приближённо: N*).
pub fn c_numeric(c: char) -> bool {
    c_digit(c) || matches!(cat(c), Gc::DecimalNumber | Gc::LetterNumber | Gc::OtherNumber)
}

/// `str.isalnum()`.
pub fn c_alnum(c: char) -> bool {
    c_alpha(c) || c_numeric(c)
}

/// Символ `\w` в `re` для str: isalnum или `_`.
pub fn c_word(c: char) -> bool {
    c == '_' || c_alnum(c)
}

/// `str.isspace()`.
pub fn c_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn c_title(c: char) -> bool {
    cat(c) == Gc::TitlecaseLetter
}

/// `c.isupper()` для одного символа.
pub fn c_upper(c: char) -> bool {
    c.is_uppercase() && !c.is_lowercase() && !c_title(c)
}

/// `c.islower()` для одного символа.
pub fn c_lower(c: char) -> bool {
    c.is_lowercase() && !c.is_uppercase() && !c_title(c)
}

// ---------------------------------------------------------------- строки

pub fn is_upper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() || c_title(c) {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}

pub fn is_lower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_uppercase() || c_title(c) {
            return false;
        }
        if c.is_lowercase() {
            cased = true;
        }
    }
    cased
}

pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(c_alpha)
}

pub fn is_digit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(c_digit)
}

pub fn lower(s: &str) -> String {
    s.to_lowercase()
}

pub fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// `str.capitalize()`.
pub fn capitalize(s: &str) -> String {
    let mut it = s.chars();
    match it.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(it.as_str().to_lowercase().chars()).collect(),
    }
}

/// Первый символ (`s[:1]`).
pub fn first(s: &str) -> Option<char> {
    s.chars().next()
}

pub fn last(s: &str) -> Option<char> {
    s.chars().next_back()
}

/// `s[:1].isupper()`.
pub fn first_upper(s: &str) -> bool {
    first(s).is_some_and(c_upper)
}

/// `s[:1].islower()`.
pub fn first_lower(s: &str) -> bool {
    first(s).is_some_and(c_lower)
}

/// `s[:1].isdigit()`.
pub fn first_digit(s: &str) -> bool {
    first(s).is_some_and(c_digit)
}

/// `s[-1:] in set` (set — строка символов).
pub fn last_in(s: &str, set: &str) -> bool {
    last(s).is_some_and(|c| set.contains(c))
}

/// `s[:1] in set` (непустой `s`).
pub fn first_in(s: &str, set: &str) -> bool {
    first(s).is_some_and(|c| set.contains(c))
}

/// Длина в кодовых точках.
pub fn clen(s: &str) -> usize {
    s.chars().count()
}

/// Срез `s[a:b]` по кодовым точкам (индексы Python, отрицательные — с конца, `None` — край).
pub fn sl(s: &str, a: Option<isize>, b: Option<isize>) -> String {
    let cs: Vec<char> = s.chars().collect();
    let n = cs.len() as isize;
    let norm = |x: isize| -> usize {
        let y = if x < 0 { x + n } else { x };
        y.clamp(0, n) as usize
    };
    let a = a.map(norm).unwrap_or(0);
    let b = b.map(norm).unwrap_or(n as usize);
    if a >= b {
        return String::new();
    }
    cs[a..b].iter().collect()
}

/// `s[k:]`.
pub fn from(s: &str, k: isize) -> String {
    sl(s, Some(k), None)
}

/// `s[:k]`.
pub fn to(s: &str, k: isize) -> String {
    sl(s, None, Some(k))
}

/// `s.strip(chars)`.
pub fn strip_set<'a>(s: &'a str, set: &str) -> &'a str {
    s.trim_matches(|c| set.contains(c))
}

pub fn lstrip_set<'a>(s: &'a str, set: &str) -> &'a str {
    s.trim_start_matches(|c| set.contains(c))
}

pub fn rstrip_set<'a>(s: &'a str, set: &str) -> &'a str {
    s.trim_end_matches(|c| set.contains(c))
}

/// `s.strip()` (пробельные символы Python).
pub fn strip_ws(s: &str) -> &str {
    s.trim_matches(c_space)
}

/// `s.split()` без аргументов.
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(c_space).filter(|x| !x.is_empty()).collect()
}

/// Значение десятичной цифры Unicode (Nd).
fn digit_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    if cat(c) != Gc::DecimalNumber {
        return None;
    }
    let u = c as u32;
    // Блоки Nd идут по 10 подряд от нуля; математические цифры (1D7CE…1D7FF) — пять блоков подряд.
    if (0x1D7CE..=0x1D7FF).contains(&u) {
        return Some((u - 0x1D7CE) % 10);
    }
    // Ищем ноль блока: ближайший снизу символ Nd, перед которым не Nd.
    let mut z = u;
    while z > 0 && u - z < 9 {
        match char::from_u32(z - 1) {
            Some(p) if cat(p) == Gc::DecimalNumber => z -= 1,
            _ => break,
        }
    }
    Some(u - z)
}

/// `int(s)` для строки цифр (знак и пробелы по краям, `_` между цифрами — как в Python).
pub fn int(s: &str) -> R<i128> {
    let t = strip_ws(s);
    let (neg, body) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return Err(Bail);
    }
    let mut v: i128 = 0;
    for c in body.chars() {
        if c == '_' {
            continue;
        }
        let d = digit_value(c).ok_or(Bail)?;
        v = v.checked_mul(10).and_then(|x| x.checked_add(d as i128)).ok_or(Bail)?;
    }
    Ok(if neg { -v } else { v })
}

/// `round(x)` в Python: к ближайшему, половина — к чётному.
pub fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    let r = if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    };
    r as i64
}

// ---------------------------------------------------------------- re

/// `re.match(...)`/`re.search(...)` как bool (шаблон сам несёт `^`, если нужен якорь).
pub fn is_m(r: &Regex, s: &str) -> bool {
    r.is_match(s).unwrap_or(false)
}

pub fn caps<'t>(r: &Regex, s: &'t str) -> Option<Captures<'t>> {
    r.captures(s).ok().flatten()
}

/// Группа или None (`m.group(i)`).
pub fn g<'t>(c: &Captures<'t>, i: usize) -> Option<&'t str> {
    c.get(i).map(|m| m.as_str())
}

/// Группа или "" (для групп, которые точно участвуют).
pub fn gs<'t>(c: &Captures<'t>, i: usize) -> &'t str {
    c.get(i).map(|m| m.as_str()).unwrap_or("")
}

/// `re.findall` для шаблона без групп.
pub fn findall(r: &Regex, s: &str) -> Vec<String> {
    r.find_iter(s).filter_map(|m| m.ok()).map(|m| m.as_str().to_string()).collect()
}

/// `re.split` (с захватывающими группами — они тоже попадают в результат, как в Python).
pub fn re_split(r: &Regex, s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for c in r.captures_iter(s).filter_map(|c| c.ok()) {
        let m = c.get(0).unwrap();
        out.push(s[last..m.start()].to_string());
        for i in 1..c.len() {
            out.push(c.get(i).map(|x| x.as_str().to_string()).unwrap_or_default());
        }
        last = m.end();
    }
    out.push(s[last..].to_string());
    out
}

/// `re.sub` с функцией замены.
pub fn re_sub(r: &Regex, s: &str, f: impl Fn(&Captures) -> String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    for c in r.captures_iter(s).filter_map(|c| c.ok()) {
        let m = c.get(0).unwrap();
        out.push_str(&s[last..m.start()]);
        out.push_str(&f(&c));
        last = m.end();
    }
    out.push_str(&s[last..]);
    out
}

/// Список строк из `&str`.
pub fn sv(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

/// `" ".split(" ")`-подобное разбиение по одному пробелу (как `str.split(" ")`).
pub fn split_sp(s: &str) -> Vec<String> {
    s.split(' ').map(|x| x.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_str_semantics() {
        assert!(is_upper("ABC1"));
        assert!(!is_upper("123"));
        assert!(is_lower("abc'"));
        assert!(is_digit("²"));
        assert!(int("²").is_err());
        assert_eq!(int("١٢").unwrap(), 12);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(sl("abcdef", Some(-2), Some(-1)), "e");
        assert_eq!(re_split(re!(r"([.-])"), "a.b-c"), vec!["a", ".", "b", "-", "c"]);
        assert_eq!(capitalize("hELLO"), "Hello");
    }
}
