//! Нормализатор (`Normalizer` в `leokani_g2p.py`): переписывает строку так, как её прочёл бы человек —
//! деньги, телефоны, время и даты, почта, адреса сайтов, версии, единицы, дроби, римские цифры,
//! сокращения, омографы. Здесь — разбор строки на токены, общие помощники и порядок правил;
//! сами правила — в `rules_*.rs`.

use super::py::{self, re, Bail, R};
use super::tables::t;
use super::words::{self, cardinal, digit_words, ordinal, PH, PH_S, PRIMARY, SECONDARY};
use super::Lex;

pub const OPEN: &str = "\"'“‘([{«";
pub const CLOSE: &str = "\"'”’)]}»,.!?;:…";
const HG_STOP: &str = ".,!?;:…—–";
const LETTER_PLURAL_ES: &str = "sxhz";
const LETTER_PLURAL_S: &str = "f";
pub const SIBILANT_END: &str = "szʃʒʧʤ";
pub const VOICELESS_END: &str = "ptkfθ";

/// Результат правила: (слова, сколько токенов съедено, хвостовая пунктуация).
pub type Out = Option<(Vec<String>, usize, String)>;

pub fn out(words: Vec<String>, used: usize, trail: impl Into<String>) -> R<Out> {
    Ok(Some((words, used, trail.into())))
}

/// Контекст токена для правил.
pub struct Ctx<'a> {
    pub core: &'a str,
    pub trail: &'a str,
    pub prev: &'a str,
    pub prev2: &'a str,
    pub prev_end: bool,
    pub nxt: &'a str,
    pub nxt_raw: &'a str,
    pub toks: &'a [String],
    pub i: usize,
    pub pauses: bool,
    pub low: String,
    pub pl: String,
    pub nl: String,
}

/// `re.match(r"^-?\d[\d,]*(\.\d+)?$", x)`.
pub fn is_num(x: &str) -> bool {
    py::is_m(re!(r"^-?\d[\d,]*(\.\d+)?$"), x)
}

fn bare(x: &str) -> String {
    x.chars().filter(|&c| c != PRIMARY && c != SECONDARY && c != ' ').collect()
}

/// `Normalizer.split`: (открывающая пунктуация, ядро, закрывающая).
pub fn nsplit(tk: &str) -> (String, String, String) {
    let cs: Vec<char> = tk.chars().collect();
    let mut a = 0;
    while a < cs.len() && OPEN.contains(cs[a]) {
        let rest: String = cs[a..].iter().collect();
        if py::is_m(re!(r"^['’]\d0s"), &rest) {
            break;
        }
        a += 1;
    }
    let mut b = cs.len();
    while b > a && CLOSE.contains(cs[b - 1]) {
        b -= 1;
    }
    (cs[..a].iter().collect(), cs[a..b].iter().collect(), cs[b..].iter().collect())
}

pub fn core_of(tk: &str) -> String {
    nsplit(tk).1
}

impl Lex {
    // ---- лексикон глазами нормализатора
    pub fn known(&self, w: &str) -> bool {
        t().say_as.contains_key(w) || self.lexicon.contains_key(w)
    }

    pub fn is_cased(&self, w: &str) -> bool {
        self.cased.contains_key(w)
    }

    pub fn common(&self, w: &str) -> bool {
        self.word_ids.contains_key(w)
    }

    pub fn reading(&self, w: &str) -> Vec<String> {
        let mut v: Vec<String> = self.lexicon.get(w).cloned().into_iter().collect();
        if let Some(x) = self.variants.get(w) {
            v.extend(x.iter().cloned());
        }
        v
    }

    // ---- помощники
    /// Лексикон уже читает w по буквам (базой или одной из форм) — или не знает его.
    pub fn spelled(&self, w: &str) -> bool {
        let forms = self.reading(w);
        let letters: String =
            w.chars().map(|c| bare(self.letters.get(&c.to_string()).map(|s| s.as_str()).unwrap_or("?"))).collect();
        forms.is_empty() || forms.iter().any(|f| bare(f) == letters)
    }

    pub fn spelled_form(&self, f: &str, w: &str) -> bool {
        let letters: String =
            w.chars().map(|c| bare(self.letters.get(&c.to_string()).map(|s| s.as_str()).unwrap_or("?"))).collect();
        bare(f) == letters
    }

    pub fn letter(&self, c: &str) -> String {
        match self.letters.get(&py::lower(c)) {
            Some(ph) if !ph.is_empty() => format!("{}{PH}{ph}", py::upper(c)),
            _ => py::upper(c),
        }
    }

    pub fn spell(&self, s: &str) -> R<Vec<String>> {
        let mut out = Vec::new();
        for c in s.chars() {
            if !(py::c_alnum(c) && c.is_ascii()) {
                continue;
            }
            if py::c_alpha(c) {
                out.push(self.letter(&c.to_string()));
            } else {
                out.push(t().one(py::int(&c.to_string())?)?);
            }
        }
        Ok(out)
    }

    /// Инициализм одним словом с ударением на последней букве ("NBA"); end — "s"/"'s".
    pub fn spell_word(&self, s: &str, end: &str) -> R<Vec<String>> {
        let names: Vec<Option<&String>> = s.chars().map(|c| self.letters.get(&py::lower(&c.to_string()))).collect();
        if !py::is_alpha(s) || !s.is_ascii() || names.iter().any(|n| n.is_none()) {
            return self.spell(s);
        }
        let names: Vec<&String> = names.into_iter().map(|n| n.unwrap()).collect();
        let k = names.len();
        let last = if !end.is_empty() {
            let lp = self.letter_plural(&py::last(s).unwrap().to_string(), if end == "s" { "" } else { end });
            lp.split_once(PH).ok_or(Bail)?.1.to_string()
        } else {
            names[k - 1].clone()
        };
        let head: String = names[..k - 1].iter().map(|n| n.replace(PRIMARY, "ˌ")).collect();
        Ok(vec![format!("{s}{end}{PH}{head}{last}")])
    }

    pub fn letter_plural(&self, c: &str, possessive: &str) -> String {
        let lc = py::lower(c);
        let ph = self.letters.get(&lc).map(|s| s.as_str()).unwrap_or("");
        let end = if LETTER_PLURAL_ES.contains(lc.as_str()) && !lc.is_empty() {
            "ᵻz"
        } else if LETTER_PLURAL_S.contains(lc.as_str()) && !lc.is_empty() {
            "s"
        } else {
            "z"
        };
        let pos = if possessive.is_empty() { "s" } else { possessive };
        format!("{}{pos}{PH}{ph}{end}", py::upper(c))
    }

    /// Ряд букв без пробелов ('milliwattlabs') — слова лексикона, наименьшим числом кусков.
    pub fn segment(&self, run: &str) -> Option<Vec<String>> {
        let low: Vec<char> = py::lower(run).chars().collect();
        let n = low.len();
        let mut best: Vec<Option<Vec<String>>> = vec![None; n + 1];
        best[0] = Some(vec![]);
        for i in 1..=n {
            for j in i.saturating_sub(20)..i {
                let wl = i - j;
                if best[j].is_none() || !(wl >= 3 || (j == 0 && wl == 1 && i < n)) {
                    continue;
                }
                let w: String = low[j..i].iter().collect();
                if wl > 1 && !self.known(&w) {
                    continue;
                }
                let mut cand = best[j].clone().unwrap();
                cand.push(w);
                if best[i].as_ref().is_none_or(|b| cand.len() < b.len()) {
                    best[i] = Some(cand);
                }
            }
        }
        match best[n].take() {
            Some(o) if o.len() > 1 => Some(o),
            _ => None,
        }
    }

    /// Слово с базой из SAY_AS.
    pub fn sword(w: &str) -> String {
        match t().say_as.get(w) {
            Some(p) => format!("{w}{PH}{p}"),
            None => w.to_string(),
        }
    }

    /// Буквенно-цифровой кусок почты, адреса сайта или ника.
    pub fn say_run(&self, run: &str) -> R<Vec<String>> {
        let mut out = Vec::new();
        for g in py::findall(re!(r"[A-Za-z]+|\d+"), run) {
            if py::is_digit(&g) {
                out.extend(digit_words(&g)?);
            } else if self.known(&py::lower(&g)) {
                out.push(Self::sword(&py::lower(&g)));
            } else if let Some(seg) = self.segment(&g) {
                for w in seg {
                    out.push(if py::clen(&w) == 1 { self.letter(&w) } else { Self::sword(&w) });
                }
            } else if py::clen(&g) > 3 && py::is_m(re!(r"[aeiouy]"), &py::lower(&g)) {
                out.push(py::lower(&g));
            } else {
                out.extend(self.spell(&g)?);
            }
        }
        Ok(out)
    }

    pub fn say_parts(&self, s: &str) -> R<Vec<String>> {
        let mut out = Vec::new();
        for part in py::re_split(re!(r"([._+\-/~=?&#:%])"), s) {
            if part.is_empty() {
                continue;
            }
            if let Some(w) = t().sep_words.get(&part) {
                out.push(w.clone());
            } else {
                out.extend(self.say_run(&part)?);
            }
        }
        Ok(out)
    }

    pub fn domain(&self, d: &str) -> R<Vec<String>> {
        let labels: Vec<&str> = d.split('.').collect();
        let mut out = Vec::new();
        for (i, lab) in labels.iter().enumerate() {
            if i > 0 {
                out.push("dot".to_string());
            }
            let low = py::lower(lab);
            if i == labels.len() - 1 && t().tlds.contains(&low) {
                match t().tld_say.get(&low) {
                    Some(s) => out.push(s.clone()),
                    None => out.extend(self.spell(&low)?),
                }
            } else if low == "www" {
                out.extend(self.spell("www")?);
            } else {
                out.extend(self.say_parts(lab)?);
            }
        }
        Ok(out)
    }

    pub fn money(
        &self,
        sign: &str,
        cur: &str,
        whole: Option<&str>,
        frac: Option<&str>,
        scale: Option<&str>,
        next_scale: Option<&str>,
    ) -> R<Vec<String>> {
        let (one, many, cent, cents) = t().currency.get(cur).ok_or(Bail)?.clone();
        let mut o: Vec<String> = if sign.is_empty() { vec![] } else { vec!["minus".into()] };
        let whole = whole.unwrap_or("0");
        let wd = whole.replace(',', "");
        let n = if wd.is_empty() { 0 } else { py::int(&wd)? };
        let sc = match t().scales.get(scale.unwrap_or("")) {
            Some(s) if !s.is_empty() => Some(s.as_str()),
            _ => next_scale,
        };
        if let Some(sc) = sc.filter(|s| !s.is_empty()) {
            o.extend([format!("{whole}{}", frac.unwrap_or("")), sc.to_string(), many]);
            return Ok(o);
        }
        if let Some(fr) = frac.filter(|f| !f.is_empty()) {
            if py::clen(fr) == 3 && !cents.is_empty() {
                let c = py::int(&py::from(fr, 1))?;
                if n == 0 && c != 0 {
                    o.extend([c.to_string(), if c == 1 { cent } else { cents }]);
                    return Ok(o);
                }
                let unit = if n == 1 { one } else { many };
                if c == 0 {
                    o.extend([whole.to_string(), unit]);
                    return Ok(o);
                }
                if cur == "£" {
                    o.extend([whole.to_string(), unit, c.to_string()]);
                    return Ok(o);
                }
                o.extend([whole.to_string(), unit, "and".into(), c.to_string(), if c == 1 { cent } else { cents }]);
                return Ok(o);
            }
            o.extend([format!("{whole}{fr}"), many]);
            return Ok(o);
        }
        o.extend([whole.to_string(), if n == 1 && !"¥".contains(cur) { one } else { many }]);
        Ok(o)
    }

    pub fn phone(&self, groups: &[String], pauses: bool, plus: bool) -> R<Vec<String>> {
        let mut o: Vec<String> = if plus { vec!["plus".into()] } else { vec![] };
        for (i, g) in groups.iter().enumerate() {
            let mut w = if py::is_digit(g) { digit_words(g)? } else { vec![py::lower(g)] };
            if g == "800" && i == 1 && groups[0] == "1" {
                w = py::sv(&["eight", "hundred"]);
            }
            if pauses && i + 1 < groups.len() {
                w.last_mut().ok_or(Bail)?.push(',');
            }
            o.extend(w);
        }
        Ok(o)
    }

    pub fn time(&self, h: &str, m: &str, s: Option<&str>, ampm: Option<&str>) -> R<Option<Vec<String>>> {
        let (hh, mm) = (py::int(h)?, py::int(m)?);
        if hh > 24 || mm > 59 {
            return Ok(None);
        }
        if let Some(s) = s {
            let ss = py::int(s)?;
            if ss > 59 {
                return Ok(None);
            }
            let mut o = vec![hh.to_string()];
            if 0 < mm && mm < 10 {
                o.extend(["oh".into(), mm.to_string()]);
            } else if mm != 0 {
                o.push(m.to_string());
            } else {
                o.push("hundred".into());
            }
            if 0 < ss && ss < 10 {
                o.extend(["oh".into(), ss.to_string()]);
            } else if ss != 0 {
                o.push(s.to_string());
            } else {
                o.push("hundred".into());
            }
            return Ok(Some(o));
        }
        let mut o = vec![hh.to_string()];
        let ampm = ampm.filter(|a| !a.is_empty());
        if mm == 0 {
            if hh >= 13 || (h.starts_with('0') && ampm.is_none()) {
                o.push("hundred".into());
            }
        } else if mm < 10 {
            o.extend(["oh".into(), mm.to_string()]);
        } else {
            o.push(m.to_string());
        }
        if let Some(a) = ampm {
            o.extend([self.letter(a), "M".into()]);
        }
        Ok(Some(o))
    }

    pub fn fraction(n: i128, d: i128) -> R<Option<Vec<String>>> {
        let (one, many) = match d {
            2 => ("half".to_string(), "halves".to_string()),
            4 => ("quarter".to_string(), "quarters".to_string()),
            3 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 16 | 32 | 64 | 100 => {
                let o = ordinal(d)?;
                let m = format!("{o}s");
                (o, m)
            }
            _ => return Ok(None),
        };
        if !(0 < n && n < d) {
            return Ok(None);
        }
        Ok(Some(vec![cardinal(n)?, if n == 1 { one } else { many }]))
    }

    /// Ряд заглавных, читаемый по буквам: лексикон читает его по буквам или не знает, и по-английски
    /// его словом не сказать.
    pub fn initialism(&self, w: &str) -> bool {
        let tt = t();
        if tt.acronym_words.contains(w) || tt.say_as_cased.contains_key(w) || words::roman_match(w) || !(w.is_ascii() && py::is_alpha(w)) {
            return false;
        }
        if self.is_cased(w) {
            return false;
        }
        let lw = py::lower(w);
        if self.known(&lw) {
            return self.spelled(&lw) && !self.reading(&lw).is_empty();
        }
        !words::pronounceable(w)
    }

    pub fn single(tk: &str) -> bool {
        let c = core_of(tk);
        py::clen(&c) == 1 && py::is_alpha(&c) && py::is_upper(&c) && c.is_ascii()
    }

    /// Слова вокруг токена i написаны заглавными как обычные слова ("PLEASE READ THE MANUAL").
    pub fn shouted(&self, toks: &[String], i: usize) -> bool {
        let mut n = 0;
        for j in [i as isize - 2, i as isize - 1, i as isize + 1, i as isize + 2] {
            if 0 <= j && (j as usize) < toks.len() {
                let c = core_of(&toks[j as usize]);
                if py::clen(&c) >= 3
                    && py::is_upper(&c)
                    && py::is_alpha(&c.replace(['\'', '’'], ""))
                    && self.known(&py::lower(&c))
                    && !self.initialism(&c)
                {
                    n += 1;
                }
            }
        }
        n >= 2
    }

    /// 192.168.1.1 -> one ninety-two dot one sixty-eight dot one dot one.
    pub fn ip(&self, host: &str) -> R<Vec<String>> {
        let mut o = Vec::new();
        for (k, x) in host.split('.').enumerate() {
            let n = py::int(x)?;
            if k > 0 {
                o.push("dot".to_string());
            }
            if n < 100 || n % 100 == 0 {
                o.extend(words::split_words(&cardinal(n)?));
            } else {
                o.push(t().one(n / 100)?);
                if n % 100 < 10 {
                    o.extend(["oh".to_string(), t().one(n % 10)?]);
                } else {
                    o.push(cardinal(n % 100)?);
                }
            }
        }
        Ok(o)
    }

    /// Четыре цифры размера парами ('1920' -> 19 20).
    pub fn pairs(d: &str) -> Vec<String> {
        if py::clen(d) == 4 && py::from(d, 2) != "00" && !d.starts_with('0') {
            let c2 = py::sl(d, Some(2), Some(3));
            let second = if c2 != "0" { py::from(d, 2) } else { format!("oh {}", py::sl(d, Some(3), Some(4))) };
            return vec![py::to(d, 2), second];
        }
        vec![d.to_string()]
    }

    // ---- омографы
    fn ctx_word(tk: &str) -> String {
        let cs: Vec<char> = tk.chars().collect();
        let (mut a, mut b) = (0, cs.len());
        while a < b && OPEN.contains(cs[a]) {
            a += 1;
        }
        while b > a && CLOSE.contains(cs[b - 1]) {
            b -= 1;
        }
        py::lower(&cs[a..b].iter().collect::<String>()).replace('’', "'")
    }

    /// Фонемы значения омографа по контексту или None (решает контекстная модель).
    pub fn homograph(&self, core: &str, toks: &[String], i: usize) -> Option<String> {
        let rules = t().homographs.get(&py::lower(core))?;
        if !py::is_alpha(core) {
            return None;
        }
        let mut prev: Vec<String> = Vec::new();
        let mut j = i as isize - 1;
        while j > i as isize - 4 {
            if j < 0 || py::last_in(&toks[j as usize], HG_STOP) {
                prev.push("^".into());
                break;
            }
            prev.push(Self::ctx_word(&toks[j as usize]));
            j -= 1;
        }
        let mut nxt: Vec<String> = Vec::new();
        if !py::last_in(&toks[i], HG_STOP) {
            for j in i + 1..i + 4 {
                if j >= toks.len() {
                    nxt.push("$".into());
                    break;
                }
                nxt.push(Self::ctx_word(&toks[j]));
                if py::last_in(&toks[j], HG_STOP) {
                    break;
                }
            }
        }
        if nxt.is_empty() || (nxt.len() < 3 && nxt.last().unwrap() != "$") {
            nxt.push("$".into());
        }
        if prev.last().map(|s| s.as_str()) != Some("^") && prev.len() < 3 {
            prev.push("^".into());
        }
        let cap = py::first_upper(core) && prev[0] != "^";
        for (said, tests) in rules {
            if tests.iter().all(|ts| words::hg_test(ts, &prev, &nxt, cap)) {
                return if said.is_empty() { None } else { Some(said.clone()) };
            }
        }
        None
    }

    /// Последнее: заглавные, которых нет в лексиконе, — по буквам, если словом их не сказать.
    fn capitals(&self, c: &Ctx) -> R<Out> {
        let core = c.core;
        if py::clen(core) >= 2
            && py::is_upper(core)
            && py::is_alpha(core)
            && core.is_ascii()
            && !self.known(&py::lower(core))
            && !self.is_cased(core)
            && self.initialism(core)
        {
            return out(self.spell_word(core, "")?, 1, c.trail);
        }
        Ok(None)
    }

    // ---- строка
    /// `Normalizer.__call__`.
    pub fn normalize(&self, text: &str, pauses: bool) -> String {
        let tt = t();
        let text = text.replace(['\u{2011}', '\u{2010}'], "-");
        let text = py::re_sub(re!(r"(\d)\s*([½¼¾⅓⅔⅛⅜⅝⅞])"), &text, |m| {
            format!("{} {} ", py::gs(m, 1), tt.vulgar.get(py::gs(m, 2)).map(|s| s.as_str()).unwrap_or(""))
        });
        let text = py::re_sub(re!(r"½|¼|¾|⅓|⅔|⅛|⅜|⅝|⅞"), &text, |m| {
            format!(" {} ", tt.vulgar_words.get(py::gs(m, 0)).map(|s| s.as_str()).unwrap_or(""))
        });
        let text = py::re_sub(
            re!(r"(?<=[\p{L}\p{N}_])([–—])(?=(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?\b)"),
            &text,
            |m| format!(" {} ", py::gs(m, 1)),
        );
        let toks: Vec<String> = py::split_ws(&text).into_iter().map(str::to_string).collect();
        let mut o: Vec<String> = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            let (w, used) = self.token(&toks, i, pauses);
            o.extend(w);
            i += used;
        }
        o.join(" ")
    }

    /// `self.sub`: нормализовать кусок и разбить по пробелу.
    pub fn sub(&self, text: &str, pauses: bool) -> Vec<String> {
        py::split_sp(&self.normalize(text, pauses))
    }

    fn token(&self, toks: &[String], i: usize, pauses: bool) -> (Vec<String>, usize) {
        let tk = &toks[i];
        if tk.contains(PH) {
            return (vec![tk.clone()], 1);
        }
        let (lead, core, trail) = nsplit(tk);
        let prev = if i > 0 { core_of(&toks[i - 1]) } else { String::new() };
        let prev_end = i == 0 || py::last_in(&toks[i - 1], ".!?:;");
        let prev2 = if i > 1 { py::lower(&core_of(&toks[i - 2])) } else { String::new() };
        let nxt_raw = toks.get(i + 1).cloned().unwrap_or_default();
        let nxt = core_of(&nxt_raw);
        let c = Ctx {
            core: &core,
            trail: &trail,
            prev: &prev,
            prev2: &prev2,
            prev_end,
            nxt: &nxt,
            nxt_raw: &nxt_raw,
            toks,
            i,
            pauses,
            low: py::lower(&core),
            pl: py::lower(&prev),
            nl: py::lower(&nxt),
        };
        match self.rewrite(&c) {
            Ok(Some((w, used, tr))) => Self::finish(w, used, &lead, &core, &tr),
            _ => (vec![tk.clone()], 1),
        }
    }

    fn finish(words: Vec<String>, used: usize, lead: &str, core: &str, trail: &str) -> (Vec<String>, usize) {
        let mut lead = lead.to_string();
        if lead.contains('(') && !trail.contains(')') && (used > 1 || core.contains(')')) {
            lead = lead.replace('(', ""); // "(555) 867-5309": скобка закрылась внутри
        }
        if words.is_empty() {
            let keep = !py::strip_set(trail, "\"'”’)]}»").is_empty();
            return (if keep { vec![format!("{}{trail}", py::strip_set(&lead, "\"'“‘([{«"))] } else { vec![] }, used);
        }
        let mut ws: Vec<String> = Vec::new();
        for w in words {
            if w.contains(PH) && w.contains(' ') {
                let (head, ph) = w.split_once(PH).unwrap();
                for p in ph.split(' ') {
                    ws.push(format!("{head}{PH_S}{p}"));
                }
            } else {
                ws.push(w);
            }
        }
        ws[0] = format!("{lead}{}", ws[0]);
        let l = ws.len() - 1;
        ws[l].push_str(trail);
        (ws, used)
    }

    fn rewrite(&self, c: &Ctx) -> R<Out> {
        if c.core.is_empty() {
            return Ok(None);
        }
        if let Some(r) = self.letters_first(c)? {
            return Ok(Some(r));
        }
        if let Some(r) = self.more2(c)? {
            return Ok(Some(r));
        }
        if let Some(r) = self.more(c)? {
            return Ok(Some(r));
        }
        if let Some(r) = self.more3(c)? {
            return Ok(Some(r));
        }
        if let Some(r) = self.basic(c)? {
            return Ok(Some(r));
        }
        self.capitals(c)
    }
}
