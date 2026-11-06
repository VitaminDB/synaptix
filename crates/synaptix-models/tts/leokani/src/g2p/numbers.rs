//! Цифры так, как их читает фронтенд espeak (`Numbers` в `leokani_g2p.py`): "740" -> sˈɛvənhˈʌndɹɪd fˈɔɹɾi,
//! "$3.50" -> dˈɑləɹ θɹˈi pYnt fˈIv zˈiəɹO. Атомы (0–99, порядки, знаки, буквы, порядковые) — из лексикона.

use super::py::{self, re, Bail, R};
use super::words::{PRIMARY, SECONDARY};
use super::{Lex, VOWELS};

const ORD: [&str; 4] = ["st", "nd", "rd", "th"];

fn suffix_sound(s: &str) -> R<&'static str> {
    Ok(match s {
        "st" => "st",
        "nd" => "ənd",
        "rd" => "d",
        "th" => "θ",
        _ => return Err(Bail),
    })
}

fn right_suffix(n: i128) -> &'static str {
    let r = n.rem_euclid(100);
    if (10..=20).contains(&r) {
        "th"
    } else {
        match n.rem_euclid(10) {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    }
}

impl Lex {
    fn a(&self, k: &str) -> R<String> {
        self.atoms.get(k).cloned().ok_or(Bail)
    }

    /// 0 < n < 1000 -> токены; pre: форма перед словом порядка.
    fn n_group(&self, n: i128, pre: bool) -> R<Vec<String>> {
        let (h, r) = (n / 100, n % 100);
        let mut out = Vec::new();
        if h != 0 {
            out.push(self.a(&h.to_string())? + &self.a("hundred")?);
        }
        if r != 0 {
            let p = format!("pre:{r}");
            out.push(if pre && self.atoms.contains_key(&p) { self.a(&p)? } else { self.a(&r.to_string())? });
        }
        Ok(out)
    }

    fn n_integer(&self, digits: &str) -> R<Option<Vec<String>>> {
        let n = match py::int(digits) {
            Ok(n) => n,
            // больше i128 — заведомо >= 10**12
            Err(_) if !digits.is_empty() && digits.chars().all(py::c_decimal) => return Ok(None),
            Err(e) => return Err(e),
        };
        if n >= 1_000_000_000_000 {
            return Ok(None);
        }
        if n == 0 {
            return Ok(Some(vec![self.a("0")?]));
        }
        if (1900..=1999).contains(&n) && py::clen(digits) == 4 {
            let mut out = vec![self.a("19")? + &self.a("hundred")?];
            if n % 100 != 0 {
                out.push(self.a(&(n % 100).to_string())?);
            }
            return Ok(Some(out));
        }
        let mut out = Vec::new();
        let mut n = n;
        for (scale, name) in [(1_000_000_000i128, "billion"), (1_000_000, "million"), (1000, "thousand")] {
            let q = n.div_euclid(scale);
            n = n.rem_euclid(scale);
            if q != 0 {
                out.extend(self.n_group(q, true)?);
                out.push(self.a(name)?);
            }
        }
        if n != 0 {
            out.extend(self.n_group(n, false)?);
        }
        Ok(Some(out))
    }

    fn join_atoms(&self, d: &str) -> R<String> {
        let mut s = String::new();
        for c in d.chars() {
            s += &self.a(&c.to_string())?;
        }
        Ok(s)
    }

    /// Ряд с ведущими нулями.
    fn n_digits(&self, d: &str) -> R<Vec<String>> {
        let cs: Vec<char> = d.chars().collect();
        if cs.len() <= 3 {
            let z = cs.len() - d.trim_start_matches('0').chars().count();
            let z = z.min(cs.len() - 1);
            let mut out = vec![(0..z).map(|_| self.a("0")).collect::<R<Vec<_>>>()?.concat()];
            let rest: String = cs[z..].iter().collect();
            out.extend(self.n_integer(&rest)?.ok_or(Bail)?);
            return Ok(out);
        }
        if cs.len() == 4 {
            return Ok(vec![self.join_atoms(d)?]);
        }
        let mut groups: Vec<String> = Vec::new();
        let mut rest = &cs[..];
        while !rest.is_empty() {
            let k = rest.len().saturating_sub(3);
            groups.insert(0, rest[k..].iter().collect());
            rest = &rest[..k];
        }
        groups.iter().map(|g| self.join_atoms(g)).collect()
    }

    fn n_number(&self, s: &str) -> R<Option<Vec<String>>> {
        let (whole, frac) = match s.find('.') {
            Some(k) => (&s[..k], &s[k + 1..]),
            None => (s, ""),
        };
        let digits = whole.replace(',', "");
        let out = if py::clen(&digits) > 1 && digits.starts_with('0') && !whole.contains(',') {
            Some(self.n_digits(&digits)?)
        } else {
            self.n_integer(&digits)?
        };
        let Some(mut out) = out else { return Ok(None) };
        if !frac.is_empty() {
            out.push(self.a(".")?);
            for c in frac.chars() {
                out.push(self.a(&c.to_string())?);
            }
        }
        Ok(Some(out))
    }

    /// espeak говорит основу порядкового и затем суффикс как написан: 22nd sˈɛkənd, 22st sˈɛkst.
    fn n_ordinal(&self, n: i128, suffix: &str) -> R<Vec<String>> {
        let r = n.rem_euclid(100);
        if r == 0 {
            let mut out = self.n_integer(&n.to_string())?.unwrap_or_default();
            out.push(self.a("ord:th")?);
            return Ok(out);
        }
        let mut head = if n - r != 0 { self.n_integer(&(n - r).to_string())?.unwrap_or_default() } else { vec![] };
        let whole = self.a(&format!("ord:{r}"))?;
        let right = suffix_sound(right_suffix(r))?;
        if !whole.ends_with(right) {
            head.push(whole);
            return Ok(head);
        }
        head.push(format!("{}{}", &whole[..whole.len() - right.len()], suffix_sound(suffix)?));
        Ok(head)
    }

    /// По буквам: имя каждой буквы со вторичным ударением, кроме последней.
    fn n_letters(&self, run: &str, first: bool) -> Option<Vec<String>> {
        let names: Vec<Option<&String>> =
            run.chars().map(|c| self.atoms.get(&format!("letter:{}", py::lower(&c.to_string())))).collect();
        if names.iter().any(|n| n.is_none()) {
            return None;
        }
        let names: Vec<&String> = names.into_iter().map(|n| n.unwrap()).collect();
        let k = names.len();
        let mut spelled: String =
            names[..k - 1].iter().map(|n| n.replace(PRIMARY, &SECONDARY.to_string())).collect();
        let tail = if first && py::lower(run) == "a" {
            names[k - 1].replace(PRIMARY, &SECONDARY.to_string())
        } else {
            names[k - 1].clone()
        };
        spelled.push_str(&tail);
        Some(vec![spelled])
    }

    /// Чтение токена с цифрами; None — правила нет.
    pub fn num_say(&self, w: &str) -> Option<String> {
        self.n_say(w).ok().flatten()
    }

    fn n_say(&self, w: &str) -> R<Option<String>> {
        let segs = py::findall(re!(r"\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z]+|."), &w.replace('−', "-"));
        let mut toks: Vec<String> = Vec::new();
        let mut glue = false;
        let mut i = 0;
        let empty = String::new();
        while i < segs.len() {
            let g = &segs[i];
            let nxt = segs.get(i + 1).unwrap_or(&empty);
            let prev = if i > 0 { &segs[i - 1] } else { &empty };
            if py::first_digit(g) {
                let part: Option<Vec<String>> = if prev == ":" && py::clen(g) == 2 && g.starts_with('0') {
                    Some(g.chars().map(|c| self.a(&c.to_string())).collect::<R<Vec<_>>>()?)
                } else if nxt == ":" && py::is_digit(g) {
                    self.n_integer(g)?
                } else if ORD.contains(&nxt.as_str()) && !g.contains('.') {
                    let p = self.n_ordinal(py::int(&g.replace(',', ""))?, nxt)?;
                    i += 1;
                    Some(p)
                } else if nxt == "s" && !g.contains('.') {
                    let Some(mut p) = self.n_number(g)? else { return Ok(None) };
                    p.push(self.a("suffix:s")?);
                    i += 1;
                    Some(p)
                } else {
                    self.n_number(g)?
                };
                let Some(mut part) = part else { return Ok(None) };
                if glue && !toks.is_empty() && !part.is_empty() {
                    let f = part.remove(0);
                    toks.last_mut().unwrap().push_str(&f);
                } else if glue && !toks.is_empty() {
                    return Err(Bail); // part[0] на пустом списке — IndexError
                }
                toks.extend(part);
                glue = false;
            } else if py::is_alpha(g) {
                let n = py::clen(g);
                let part: Option<Vec<String>> = if prev == "-" && n > 1 {
                    self.word(g).map(|y| vec![y])
                } else if n > 1 && py::is_lower(g) && self.word(g).is_some() && (nxt == "-" || prev.is_empty()) {
                    Some(vec![self.word(g).unwrap()])
                } else if n > 3 && self.word(g).is_some() {
                    Some(vec![self.word(g).unwrap()])
                } else if n <= 3 || py::is_upper(g) {
                    self.n_letters(g, toks.is_empty())
                } else {
                    None
                };
                let Some(mut part) = part else { return Ok(None) };
                if glue && !toks.is_empty() {
                    let f = part.remove(0);
                    toks.last_mut().unwrap().push_str(&f);
                }
                toks.extend(part);
                glue = false;
            } else if (g == "'" || g == "’") && nxt == "s" && !toks.is_empty() {
                toks.push(self.a("suffix:s")?);
                i += 1;
            } else if (g == "$" || g == "£" || g == "€") && toks.is_empty() {
                toks.push(self.a(g)?);
            } else if g == "-" {
                if toks.is_empty() && py::first_digit(nxt) {
                    toks.push(self.a("minus")?);
                } else if !toks.is_empty() && py::first_digit(nxt) && py::is_alpha(prev) {
                    toks.push(String::new()); // espeak: слово, пустой элемент, затем число
                } else if !toks.is_empty() && py::first_digit(nxt) {
                    toks.push(self.a("-")?);
                } else if !toks.is_empty() && py::is_alpha(nxt) {
                    glue = true;
                } else if nxt.is_empty() {
                } else {
                    return Ok(None);
                }
            } else if g == "%" || g == "°" || g == "/" {
                toks.push(self.a(g)?);
            } else if g == ":" {
                if !(py::clen(nxt) == 2 && py::is_digit(nxt)) {
                    toks.push(self.a(":")?);
                }
            } else if g == "–" {
            } else {
                return Ok(None);
            }
            i += 1;
        }
        Ok(if toks.is_empty() { None } else { Some(Self::link(&toks)) })
    }

    /// espeak связывает слово на ɚ (здесь əɹ) с гласным началом следующего лишним ɹ.
    fn link(toks: &[String]) -> String {
        let mut out = Vec::with_capacity(toks.len());
        for (i, tk) in toks.iter().enumerate() {
            let nxt = toks.get(i + 1).map(|s| s.as_str()).unwrap_or("");
            let nb: String = nxt.chars().filter(|&c| c != PRIMARY && c != SECONDARY).collect();
            let mut tk = tk.clone();
            if tk.ends_with("əɹ") && nb.chars().next().is_some_and(|c| VOWELS.contains(c)) {
                tk.push('ɹ');
            }
            out.push(tk);
        }
        out.join(" ")
    }
}
