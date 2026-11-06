//! Правила нормализатора: `more` (составные через тире, деньги за единицу, годы, адреса, телефоны,
//! версии, футы и дюймы, римские цифры) и `more3` (CASED_SAY, диапазоны годов и процентов, порты,
//! телефоны группами, коды после слова-подсказки, единицы со степенями и скоростями, веб).

use super::norm::{core_of, is_num, nsplit, out, Ctx, Out};
use super::py::{self, caps, g, gs, is_m, re, sv, Bail, R};
use super::rules3::{DATE_ISO, IPV4, MONEY, PHONE, TIME, TIME_AMPM, TOLLWORD, VERSION_DASH};
use super::tables::t;
use super::words::{self, cardinal, digit_words, ordinal, pair_words, roman_value, year_words, PH};
use super::Lex;

/// Чтение номера порта.
fn port_words(p: &str) -> R<Vec<String>> {
    let pc: Vec<char> = p.chars().collect();
    let n = py::int(p)?;
    Ok(if pc.len() == 4 && py::from(p, 1) != "000" && n < 2100 {
        pair_words(n)?
    } else if pc.len() == 4 && pc[2] != '0' && py::from(p, 1) != "000" {
        vec![py::to(p, 2), py::from(p, 2)]
    } else {
        vec![p.to_string()]
    })
}

impl Lex {
    pub(super) fn more(&self, c: &Ctx) -> R<Out> {
        let tt = t();
        let (core, trail, prev, nxt, nxt_raw, toks, i, pauses) = (c.core, c.trail, c.prev, c.nxt, c.nxt_raw, c.toks, c.i, c.pauses);
        let (low, pl, nl) = (c.low.as_str(), c.pl.as_str(), c.nl.as_str());
        // диапазон или составное через тире: "3pm-5pm", "$5M-$7M", "10am–noon", "3-for-$10"
        if is_m(re!(r"[–—-]"), core)
            && is_m(re!(r"\d"), core)
            && !is_m(PHONE(), core)
            && !is_m(TOLLWORD(), core)
            && !is_m(re!(r"^[\d,.]+([–-][\d,.]+)+$"), core)
            && !is_m(DATE_ISO(), core)
            && !is_m(VERSION_DASH(), core)
            && !is_m(re!(r"^\d{1,2}:\d{2}[-–]\d{1,2}:\d{2}$"), core)
        {
            let parts = py::re_split(re!(r"[–—-]"), core);
            let timey = |p: &str| is_m(TIME(), p) || is_m(TIME_AMPM(), p) || matches!(py::lower(p).as_str(), "noon" | "midnight");
            let money = |p: &str| is_m(MONEY(), p);
            let all_nonempty = parts.iter().all(|p| !p.is_empty());
            if parts.len() == 2
                && all_nonempty
                && (parts.iter().all(|p| timey(p)) || parts.iter().all(|p| money(p)))
            {
                let mut a = self.sub(&parts[0], pauses);
                let b = self.sub(&parts[1], pauses);
                if money(&parts[0]) && a.last() == b.last() {
                    a.pop(); // "5 million to 7 million dollars"
                }
                a.push("to".into());
                a.extend(b);
                return out(a, 1, trail);
            }
            if all_nonempty && parts.iter().any(|p| money(p) || is_m(TIME_AMPM(), p)) {
                let mut o = Vec::new();
                for p in &parts {
                    o.extend(self.sub(p, pauses));
                }
                return out(o, 1, trail);
            }
        }
        // деньги за единицу: "£2,450/month", "$12/hr"
        if let Some(m) = caps(re!(r"^([$€£¥₹][\d,.]+[kKmMbB]?)/([A-Za-z]+)$"), core) {
            if let Some(per) = tt.per_unit.get(&py::lower(gs(&m, 2))) {
                let mut w = self.sub(gs(&m, 1), pauses);
                w.extend(py::split_sp(per));
                return out(w, 1, trail);
            }
        }
        // деньги как определение: "a $100 gift card", "£5 notes"
        if let Some(m) = caps(MONEY(), core) {
            if g(&m, 3).is_some() && g(&m, 4).is_none() && g(&m, 5).is_none() && tt.attributive.contains(nl) {
                let mut w: Vec<String> = if gs(&m, 1).is_empty() { vec![] } else { vec!["minus".into()] };
                w.push(gs(&m, 3).into());
                w.push(tt.currency.get(gs(&m, 2)).ok_or(Bail)?.0.clone());
                return out(w, 1, trail);
            }
        }
        // куски телефонов: "(212)555-0100", добавочные, звёздочка, решётка
        if let Some(m) = caps(re!(r"^\(?(\d{3})\)(\d{3})-(\d{4})$"), core) {
            let gr = vec![gs(&m, 1).to_string(), gs(&m, 2).to_string(), gs(&m, 3).to_string()];
            return out(self.phone(&gr, pauses, false)?, 1, trail);
        }
        if let Some(m) = caps(re!(r"(?i)^(?:x|ext\.?)(\d{1,5})$"), core) {
            let d = gs(&m, 1);
            let mut w = vec!["extension".to_string()];
            if py::clen(d) > 2 {
                w.extend(digit_words(d)?);
            } else {
                w.push(d.into());
            }
            return out(w, 1, trail);
        }
        if low == "ext" && trail.starts_with('.') && py::first_digit(nxt) {
            return out(sv(&["extension"]), 1, py::from(trail, 1));
        }
        if let Some(m) = caps(re!(r"^\*(\d{1,4})#?$"), core) {
            let mut w = vec!["star".to_string()];
            w.extend(digit_words(gs(&m, 1))?);
            if core.ends_with('#') {
                w.push("pound".into());
            }
            return out(w, 1, trail);
        }
        if core == "#" && (matches!(pl, "press" | "then" | "hit" | "dial" | "enter" | "and" | "or") || nxt.is_empty()) {
            return out(sv(&["pound"]), 1, trail);
        }
        if core == "*" && !(is_num(prev) && is_num(nxt)) {
            return out(sv(&["star"]), 1, trail);
        }
        // военное время: "0800 hours"
        if is_m(re!(r"^[0-2]\d[0-5]\d$"), core) && matches!(nl, "hours" | "hrs" | "hr" | "h") {
            let (h, mm) = (py::int(&py::to(core, 2))?, py::int(&py::from(core, 2))?);
            let mut w: Vec<String> = if core.starts_with('0') { vec!["oh".into()] } else { vec![] };
            w.push(cardinal(h)?);
            if mm == 0 {
                w.push("hundred".into());
            } else if mm < 10 {
                w.extend(["oh".into(), tt.one(mm)?]);
            } else {
                w.push(cardinal(mm)?);
            }
            return out(w, 1, trail);
        }
        // веб: "localhost:8080/v1/chat"
        if let Some(m) = caps(
            re!(r"^(?:https?://)?(localhost|(?:[\p{L}\p{N}_-]+\.)+[a-z]{2,}|\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})(/\S*)?$"),
            core,
        ) {
            let (host, port) = (gs(&m, 1), gs(&m, 2));
            let mut w = if is_m(IPV4(), host) {
                self.ip(host)?
            } else if host == "localhost" {
                sv(&["local", "host"])
            } else {
                self.domain(host)?
            };
            w.push("colon".into());
            w.extend(port_words(port)?);
            for seg in g(&m, 3).unwrap_or("").split('/').skip(1) {
                w.push("slash".into());
                if !seg.is_empty() {
                    w.extend(self.say_parts(seg)?);
                }
            }
            return out(w, 1, trail);
        }
        if low.starts_with("ftp://") && py::clen(core) > 6 {
            let mut w = self.spell("FTP")?;
            w.extend(sv(&["colon", "slash", "slash"]));
            w.extend(self.sub(&py::from(core, 6), pauses));
            return out(w, 1, trail);
        }
        // версии: "Python 3.12", "1.10", "v20.x", "6.8.0-45-generic", сборка "23A344"
        if let Some(m) = caps(re!(r"^([vV])?(\d+)\.(\d{2,3}|x)$"), core) {
            let minor = gs(&m, 3);
            if tt.version_cue.contains(pl)
                || g(&m, 1).is_some()
                || minor == "x"
                || (minor.ends_with('0')
                    && !minor.starts_with('0')
                    && !tt.units.contains_key(nl)
                    && !is_num(prev)
                    && !matches!(pl, "is" | "of" | "at" | "was" | "by" | "about" | "gpa" | "score"))
            {
                let mut w: Vec<String> = if g(&m, 1).is_some() { vec!["version".into()] } else { vec![] };
                w.extend([gs(&m, 2).to_string(), "point".into()]);
                w.push(if minor == "x" { self.letter("x") } else { minor.to_string() });
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(VERSION_DASH(), core) {
            let mut w = Vec::new();
            for (k, x) in gs(&m, 1).split('.').enumerate() {
                if k > 0 {
                    w.push("point".to_string());
                }
                w.push(x.to_string());
            }
            for part in gs(&m, 2).split('-').skip(1) {
                w.push("dash".into());
                if !part.is_empty() {
                    w.extend(self.sub(part, pauses));
                }
            }
            return out(w, 1, trail);
        }
        if is_m(re!(r"^\d+[A-Z]+\d+[A-Z0-9]*$"), core) && py::clen(core) >= 4 {
            let mut w = Vec::new();
            for gr in py::findall(re!(r"\d+|[A-Z]+"), core) {
                if py::is_digit(&gr) {
                    if py::clen(&gr) <= 2 {
                        w.push(gr);
                    } else {
                        w.extend(digit_words(&gr)?);
                    }
                } else {
                    w.extend(self.spell(&gr)?);
                }
            }
            return out(w, 1, trail);
        }
        // шестнадцатеричные строки — по символу
        if py::clen(core) >= 8 && is_m(re!(r"^[0-9a-fA-F]+$"), core) && is_m(re!(r"\d"), core) && is_m(re!(r"[a-fA-F]"), core) {
            return out(self.spell(core)?, 1, trail);
        }
        // группы цифр подряд (карты, счета, IBAN) — по цифре
        if is_m(re!(r"^\d{2,4}$"), core) {
            let near: Vec<String> = toks[i.saturating_sub(2)..toks.len().min(i + 3)].iter().map(|x| core_of(x)).collect();
            if near.iter().filter(|x| is_m(re!(r"^(\d{4}|[A-Z]{2}\d{2})$"), x)).count() >= 3 {
                let mut w = digit_words(core)?;
                if c.pauses && trail.is_empty() && i + 1 < toks.len() {
                    w.last_mut().ok_or(Bail)?.push(',');
                }
                return out(w, 1, trail);
            }
        }
        if is_m(re!(r"^[A-Z]{2}\d{2}$"), core) && is_m(re!(r"^\d{4}$"), nxt) {
            let mut w = self.spell(&py::to(core, 2))?;
            w.extend(digit_words(&py::from(core, 2))?);
            if pauses && trail.is_empty() {
                w.last_mut().ok_or(Bail)?.push(',');
            }
            return out(w, 1, trail);
        }
        // годы после месяца или дня: "Dec. 1999", "March 14, 1988"
        if is_m(re!(r"^1[1-9]\d\d$"), core)
            && !core.ends_with("00")
            && i > 0
            && (tt.months.iter().any(|x| x == prev)
                || (tt.month_abbr.contains_key(pl) && py::first_upper(prev))
                || is_m(re!(r"^\d{1,2}(st|nd|rd|th)?,$"), &toks[i - 1])
                || is_m(re!(r"^([\p{L}\p{N}_]+ )?[\p{L}\p{N}_]+(st|nd|rd|th),$"), &toks[i - 1]))
        {
            return out(year_words(py::int(core)?)?, 1, trail);
        }
        // день после месяца — порядковое ("July 5" -> July fifth); день перед месяцем — "the fifth of July"
        let after_month = tt.months.iter().any(|x| x == prev) || (tt.month_abbr.contains_key(pl) && py::first_upper(prev));
        if is_m(re!(r"^\d{1,2}$"), core)
            && (1..=31).contains(&py::int(core)?)
            && after_month
            && !tt.units.contains_key(nl)
            && !matches!(nl, "percent" | "degrees" | "hours" | "minutes" | "days" | "people" | "times")
        {
            return out(vec![ordinal(py::int(core)?)?], 1, trail);
        }
        if is_m(re!(r"^\d{1,2}$"), core)
            && (1..=31).contains(&py::int(core)?)
            && tt.months.iter().any(|x| x == nxt)
            && !matches!(pl, "the" | "of")
            && !is_num(prev)
        {
            return out(vec!["the".into(), ordinal(py::int(core)?)?, "of".into()], 1, trail);
        }
        // адреса
        if let Some(dir) = tt.directions.get(low) {
            if py::is_upper(core) {
                if py::clen(core) == 2 && (tt.street_words.contains(py::rstrip_set(pl, ".")) || tt.street_names.contains(prev)) {
                    return out(vec![dir.clone()], 1, trail);
                }
                if py::clen(core) == 1
                    && trail.starts_with('.')
                    && (py::first_upper(nxt) || is_m(re!(r"^\d+(st|nd|rd|th)$"), nxt))
                {
                    return out(vec![dir.clone()], 1, py::from(trail, 1));
                }
            }
        }
        if matches!(low, "ste" | "hwy" | "rte" | "ave" | "blvd")
            && py::first_upper(core)
            && (py::first_digit(nxt) || (matches!(low, "ave" | "blvd") && py::first_upper(prev)))
        {
            let w = match low {
                "ste" => "Suite",
                "hwy" => "Highway",
                "rte" => "Route",
                "ave" => "Avenue",
                _ => "Boulevard",
            };
            let mut tr = trail.to_string();
            if trail.starts_with('.') && (py::clen(trail) > 1 || (!nxt.is_empty() && !py::first_upper(nxt))) {
                tr = py::from(trail, 1);
            }
            return out(sv(&[w]), 1, tr);
        }
        if let Some(st) = tt.states.get(core) {
            if is_m(re!(r"^\d{5}(-\d{4})?$"), nxt) && i > 0 && toks[i - 1].ends_with(',') {
                let z = nsplit(nxt_raw);
                let mut w = py::split_sp(st);
                w.extend(digit_words(&z.1)?);
                return out(w, 2, z.2);
            }
        }
        if let Some(m) = caps(re!(r"^I-(\d{1,3})$"), core) {
            let d = gs(&m, 1);
            let dc: Vec<char> = d.chars().collect();
            let mut w = vec![self.letter("I")];
            if dc.len() <= 2 {
                w.push(d.into());
            } else {
                w.push(tt.one(py::int(&dc[0].to_string())?)?);
                if dc[1] == '0' {
                    w.extend(["oh".into(), tt.one(py::int(&dc[2].to_string())?)?]);
                } else {
                    w.push(py::from(d, 1));
                }
            }
            return out(w, 1, trail);
        }
        if is_m(re!(r"^[1-9]\d{3}$"), core)
            && (matches!(pl, "box" | "suite" | "ste" | "room" | "unit" | "apt" | "apartment")
                || (py::first_upper(nxt)
                    && toks[(i + 1).min(toks.len())..toks.len().min(i + 4)]
                        .iter()
                        .any(|x| tt.street_words.contains(py::rstrip_set(&py::lower(&core_of(x)), ".")))))
        {
            return out(pair_words(py::int(core)?)?, 1, trail);
        }
        // футы и дюймы, экспоненциальная запись, единица после дефиса
        let inch = py::first_in(trail, "\"”″");
        if let Some(m) = caps(re!(r"^(\d{1,2})[′'’](\d{1,2})$"), core) {
            if inch {
                return out(vec![gs(&m, 1).into(), "foot".into(), gs(&m, 2).into()], 1, py::from(trail, 1));
            }
        }
        if is_m(re!(r"^\d+(\.\d+)?$"), core)
            && inch
            && (matches!(prev, "a" | "an" | "the")
                || matches!(
                    nl,
                    "bolt" | "screw" | "pipe" | "screen" | "display" | "tv" | "monitor" | "wide" | "long" | "tall" | "deep" | "thick" | "of" | "nail"
                ))
        {
            let unit = if core == "1" || matches!(prev, "a" | "an" | "the") { "inch" } else { "inches" };
            return out(vec![core.into(), unit.into()], 1, py::from(trail, 1));
        }
        if let Some(m) = caps(re!(r"^(\d{1,3})/(\d{1,3})$"), core) {
            if inch {
                if let Some(mut fr) = Self::fraction(py::int(gs(&m, 1))?, py::int(gs(&m, 2))?)? {
                    fr.push("inch".into());
                    return out(fr, 1, py::from(trail, 1));
                }
            }
        }
        if let Some(m) = caps(re!(r"^(-?\d+(?:\.\d+)?)[eE]([-+]?)(\d{1,3})$"), core) {
            let mut w = vec![gs(&m, 1).to_string()];
            w.extend(sv(&["times", "ten", "to", "the"]));
            if gs(&m, 2) == "-" {
                w.push("minus".into());
            }
            w.push(gs(&m, 3).into());
            return out(w, 1, trail);
        }
        if let Some(m) = caps(re!(r"^(\d+(?:\.\d+)?)-([A-Za-zµμ]+)$"), core) {
            if let Some(u) = tt.units.get(gs(&m, 2)) {
                let mut w = vec![gs(&m, 1).to_string()];
                w.extend(py::split_sp(&u.0));
                return out(w, 1, trail);
            }
        }
        // римские цифры: "Elizabeth I ruled", "Act I", "section ii", год "MCMXCIV"
        if core == "I"
            && py::first_upper(prev)
            && tt.regnal.contains(pl)
            && (nl.ends_with("ed")
                || matches!(nl, "was" | "had" | "is" | "of" | "and" | "became" | "died" | "who" | "'s" | "reigned" | "ruled"))
        {
            return out(sv(&["the", "first"]), 1, trail);
        }
        if core == "I" && tt.roman_cue.contains(pl) && py::first_upper(prev) {
            return out(sv(&["one"]), 1, trail);
        }
        if py::is_lower(core) && py::clen(core) >= 2 && words::roman_match(&py::upper(core)) && tt.roman_cue.contains(pl) {
            return out(vec![cardinal(roman_value(&py::upper(core))?)?], 1, trail);
        }
        if core.starts_with('M') && py::clen(core) >= 4 && words::roman_match(core) {
            let v = roman_value(core)?;
            if (1000..=2099).contains(&v) {
                return out(year_words(v)?, 1, trail);
            }
        }
        if py::clen(core) == 2
            && py::is_upper(core)
            && py::is_alpha(core)
            && core.is_ascii()
            && !self.is_cased(core)
            && !self.common(low)
            && !words::roman_match(core)
            && !self.spelled(low)
            && !(i > 0 && py::is_upper(&toks[i - 1]) && py::is_alpha(&toks[i - 1]) && py::clen(&toks[i - 1]) > 3)
        {
            return out(self.spell(core)?, 1, trail); // "UA": лексикон читает словом, а не буквами
        }
        if is_m(re!(r"^(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}$"), core)
            && trail.starts_with('.')
            && !matches!(low, "e.g" | "i.e" | "a.m" | "p.m")
            && core.chars().any(py::c_lower)
        {
            let tr = if py::clen(trail) > 1 { py::from(trail, 1) } else { trail.to_string() };
            return out(self.spell(&core.replace('.', ""))?, 1, tr);
        }
        // аббревиатуры, совпадающие со словами: "the WHO", "the IT team"
        if matches!(core, "WHO" | "IT" | "AM" | "OR" | "IS" | "AS") && pl == "the" {
            return out(self.spell(core)?, 1, trail);
        }
        // омографы-прилагательные
        if low == "minute"
            && (matches!(pl, "is" | "are" | "was" | "were" | "very" | "so" | "quite" | "extremely" | "too" | "seemed" | "looked")
                || matches!(
                    nl,
                    "details" | "detail" | "amounts" | "amount" | "traces" | "trace" | "particles" | "quantities" | "changes" | "differences"
                        | "fraction" | "portion" | "particle"
                ))
        {
            return out(vec![format!("{core}{PH}{}", tt.adj_forms["minute"])], 1, trail);
        }
        if low == "content"
            && matches!(nl, "with" | "to")
            && matches!(
                pl,
                "seemed" | "seem" | "seems" | "feel" | "felt" | "feels" | "be" | "been" | "is" | "was" | "are" | "were" | "am" | "remain"
                    | "remained" | "quite" | "very" | "perfectly" | "more" | "less" | "not" | "looked" | "looks"
            )
        {
            return out(vec![format!("{core}{PH}{}", tt.adj_forms["content"])], 1, trail);
        }
        if low == "fl" && trail.starts_with('.') && is_m(re!(r"^\d+(st|nd|rd|th)$"), prev) {
            return out(sv(&["floor"]), 1, py::from(trail, 1));
        }
        Ok(None)
    }

    pub(super) fn more3(&self, c: &Ctx) -> R<Out> {
        let tt = t();
        let (core, trail, prev, prev2, nxt, nxt_raw, toks, i, pauses) =
            (c.core, c.trail, c.prev, c.prev2, c.nxt, c.nxt_raw, c.toks, c.i, c.pauses);
        let (low, pl, nl) = (c.low.as_str(), c.pl.as_str(), c.nl.as_str());
        let tail1 = |tr: &str| if tr.starts_with('.') { py::from(tr, 1) } else { tr.to_string() };
        let one_n = |n: &str| n == "1" || n == "-1";
        // заглавные, переписанные словами: "IEEE", "WWII", "PhD"
        if let Some(v) = tt.cased_say.get(core) {
            return out(self.sub(v, pauses), 1, trail);
        }
        if core == "ID" {
            return out(vec![format!("{core}{PH}ˌIdˈi")], 1, trail);
        }
        // диапазоны годов: "1990–95", "2019-2020 season"
        let m = caps(re!(r"^(1[1-9]\d\d|20\d\d)[–-](\d\d|1[1-9]\d\d|20\d\d)$"), core);
        if let Some(m) = m {
            let cond = !core.ends_with("00")
                || (is_m(re!(r"^(1[1-9]\d\d|20\d\d)[–-](\d\d|\d{4})$"), core) && py::int(gs(&m, 1))? % 100 != 0);
            if cond {
                let (a, b) = (py::int(gs(&m, 1))?, gs(&m, 2));
                let bc: Vec<char> = b.chars().collect();
                let bw = if bc.len() == 4 {
                    year_words(py::int(b)?)?
                } else if bc[0] == '0' {
                    vec!["oh".into(), tt.one(py::int(&bc[1].to_string())?)?]
                } else {
                    words::split_words(&cardinal(py::int(b)?)?)
                };
                let not_range = (bc.len() == 4 && py::int(b)? <= a) || (bc.len() == 2 && py::int(b)? <= a % 100);
                if !not_range {
                    let mut w = year_words(a)?;
                    if !matches!(nl, "season" | "seasons" | "school" | "academic" | "fiscal" | "tax" | "budget" | "financial") {
                        w.push("to".into());
                    }
                    w.extend(bw);
                    return out(w, 1, trail);
                }
            }
        }
        // диапазон в процентах: "20–30%"
        if let Some(m) = caps(re!(r"^(\d[\d,]*(?:\.\d+)?)[–-](\d[\d,]*(?:\.\d+)?)%$"), core) {
            return out(vec![gs(&m, 1).into(), "to".into(), gs(&m, 2).into(), "percent".into()], 1, trail);
        }
        // диапазоны дней после месяца: "Jan 3–7"
        if let Some(m) = caps(re!(r"^(\d{1,2})[–-](\d{1,2})$"), core) {
            let (a, b) = (py::int(gs(&m, 1))?, py::int(gs(&m, 2))?);
            if (tt.months.iter().any(|x| x == prev) || (tt.month_abbr.contains_key(py::rstrip_set(pl, ".")) && py::first_upper(prev)))
                && 1 <= a
                && a < b
                && b <= 31
            {
                return out(vec![ordinal(a)?, "to".into(), ordinal(b)?], 1, trail);
            }
        }
        // диапазоны дробей: "1/2-1 cup", "1–1/2 tsp"
        if let Some(m) = caps(re!(r"^(\d{1,2}(?:/\d{1,2})?)[–-](\d{1,2}(?:/\d{1,2})?)$"), core) {
            if core.contains('/') {
                let mut o: Option<Vec<String>> = Some(Vec::new());
                for (k, part) in [gs(&m, 1), gs(&m, 2)].iter().enumerate() {
                    if part.contains('/') {
                        let (x, y) = part.split_once('/').unwrap();
                        match Self::fraction(py::int(x)?, py::int(y)?)? {
                            None => {
                                o = None;
                                break;
                            }
                            Some(fr) => {
                                let v = o.as_mut().unwrap();
                                if k > 0 {
                                    v.push("to".into());
                                }
                                v.extend(fr);
                            }
                        }
                    } else {
                        let v = o.as_mut().unwrap();
                        if k > 0 {
                            v.push("to".into());
                        }
                        v.push(part.to_string());
                    }
                }
                if let Some(o) = o.filter(|v| !v.is_empty()) {
                    return out(o, 1, trail);
                }
            }
        }
        // самолёты: "Boeing 737-800"
        if let Some(m) = caps(re!(r"^([1-9])(\d\d)(?:-(\d{1,3}))?$"), core) {
            if matches!(pl, "boeing" | "airbus" | "embraer" | "bombardier" | "douglas" | "md" | "dc") {
                let mut w = vec![tt.one(py::int(gs(&m, 1))?)?];
                w.extend(words::split_words(&cardinal(py::int(gs(&m, 2))?)?));
                if let Some(x) = g(&m, 3) {
                    w.extend(words::split_words(&cardinal(py::int(x)?)?));
                }
                return out(w, 1, trail);
            }
        }
        // годы с эрой: "A.D. 1250", "500 BC"
        let era = |x: &str| matches!(py::rstrip_set(x, "."), "A.D" | "AD" | "B.C" | "BC" | "CE" | "BCE" | "B.C.E");
        if is_m(re!(r"^\d{3,4}$"), core) && (era(prev) || era(nxt)) && !core.ends_with("00") {
            return out(year_words(py::int(core)?)?, 1, trail);
        }
        // месяц перед диапазоном дней: "Jan 3–7"
        if let Some(&mo) = tt.month_abbr.get(py::rstrip_set(low, ".")) {
            if py::first_upper(core) && py::is_alpha(core) && is_m(re!(r"^\d{1,2}[–-]\d{1,2}$"), nxt) {
                return out(vec![tt.months[mo - 1].clone()], 1, tail1(trail));
            }
        }
        // порт: "port 443", "port 8080"
        if is_m(re!(r"^\d{2,5}$"), core) && matches!(pl, "port" | "ports") && !matches!(prev2, "the" | "a") {
            let cc: Vec<char> = core.chars().collect();
            let n = py::int(core)?;
            let w = if cc.len() == 4 && py::from(core, 1) != "000" && n < 2100 {
                pair_words(n)?
            } else if cc.len() == 4 && cc[2] != '0' && py::from(core, 1) != "000" {
                vec![py::to(core, 2), py::from(core, 2)]
            } else if cc.len() == 3 && cc[1] != '0' {
                vec![py::to(core, 1), py::from(core, 1)]
            } else {
                vec![core.to_string()]
            };
            return out(w, 1, trail);
        }
        // день недели после тире: "Tue–Thu", "Mon - Fri"
        if let Some(day) = tt.day_abbr.get(low) {
            if py::first_upper(core)
                && py::is_alpha(core)
                && (matches!(prev, "–" | "-" | "—" | "through" | "to") || matches!(pl, "thru" | "until" | "till"))
                && !(matches!(prev, "through" | "to")
                    && py::first_upper(nxt)
                    && py::is_alpha(nxt)
                    && !tt.day_abbr.contains_key(nl)
                    && !tt.day_names.contains(nxt)
                    && !tt.month_abbr.contains_key(nl))
            {
                return out(vec![day.clone()], 1, tail1(trail));
            }
        }
        // телефоны группами: "312 555 0199", добавочные, "988", ZIP+4
        if is_m(re!(r"^\d{3}$"), core)
            && is_m(re!(r"^\d{3}$"), nxt)
            && i + 2 < toks.len()
            && !toks[i].starts_with('(')
            && !(i > 0 && is_m(re!(r"^\+\d{1,3}$"), &toks[i - 1]))
            && is_m(re!(r"^\d{4}$"), &core_of(&toks[i + 2]))
            && nsplit(nxt_raw).2.is_empty()
        {
            let z = nsplit(&toks[i + 2]);
            let gr = vec![core.to_string(), nxt.to_string(), z.1.clone()];
            return out(self.phone(&gr, pauses, false)?, 3, z.2);
        }
        if matches!(low, "ext" | "extension" | "x")
            && trail.is_empty()
            && is_m(re!(r"^\d{1,5}$"), nxt)
            && (low != "x" || is_m(re!(r"\d"), prev))
            && i > 0
        {
            let z = nsplit(nxt_raw);
            let mut w = vec!["extension".to_string()];
            if py::clen(&z.1) > 2 {
                w.extend(digit_words(&z.1)?);
            } else {
                w.push(z.1.clone());
            }
            return out(w, 2, z.2);
        }
        if core == "988" && (matches!(pl, "call" | "text" | "dial" | "or") || nxt.is_empty()) {
            return out(digit_words(core)?, 1, trail);
        }
        if is_m(re!(r"^\d{5}-\d{4}$"), core) {
            return out(digit_words(&core.replace('-', ""))?, 1, trail);
        }
        if is_m(re!(r"^\d{4}$"), core) && pl == "in" && matches!(prev2, "ends" | "ending") {
            let n = py::int(core)?;
            let cue = toks[i.saturating_sub(6)..i].iter().any(|x| {
                let w = py::lower(&core_of(x));
                tt.code_cue.contains(&w) || w == "card" || w == "phone"
            });
            if !(1100..=2099).contains(&n) || cue {
                return out(digit_words(core)?, 1, trail);
            }
        }
        // коды после слова-подсказки: "tracking number 9400111899", "code 482913", "SN 555123"
        let cue: Vec<String> =
            toks[i.saturating_sub(3)..i].iter().map(|x| py::rstrip_set(&py::lower(&core_of(x)), ":#").to_string()).collect();
        let cue_last2 = &cue[cue.len().saturating_sub(2)..];
        let any_cue = |xs: &[String]| xs.iter().any(|x| tt.code_cue.contains(x));
        if (is_m(re!(r"^#?\d{4,}([-]\d{2,})*$"), core) && !is_num(&core.replace('-', "x")) && core.contains('-'))
            || is_m(re!(r"^\d{5,}$"), core)
        {
            let last3: Vec<&str> = cue[cue.len().saturating_sub(3)..].iter().map(|s| s.as_str()).collect();
            if last3 == ["the", "code", "is"] {
            } else if any_cue(cue_last2) || (cue.last().map(|s| s.as_str()) == Some("is") && any_cue(&cue)) {
                let digits: String = core.chars().filter(|&ch| py::c_decimal(ch)).collect();
                return out(digit_words(&digits)?, 1, trail);
            }
        }
        if is_m(re!(r"^\d{4}-\d{3,}$"), core) && any_cue(cue_last2) {
            return out(digit_words(&core.replace('-', ""))?, 1, trail);
        }
        // единицы: "100 m²", "500 MB/s", "10 sq ft", "2 in long", "180 °C", "250 g of flour"
        if let Some(m) = caps(re!(r"^(\d[\d,]*(?:\.\d+)?)?(km|m|cm|mm|ft|in|mi|yd)([²³])$"), core) {
            if g(&m, 1).is_some() || is_num(prev) {
                let n = g(&m, 1).unwrap_or(prev);
                let name = tt.units.get(gs(&m, 2)).cloned().unwrap_or(("inch".into(), "inches".into()));
                let wds = format!("{}{}", if gs(&m, 3) == "²" { "square " } else { "cubic " }, if one_n(n) { name.0 } else { name.1 });
                let mut w: Vec<String> = if g(&m, 1).is_some() { vec![n.to_string()] } else { vec![] };
                w.extend(py::split_sp(&wds));
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^(\d[\d,]*(?:\.\d+)?)?([KMGT]B|[kMG]b|km|m|ft|mi)/(s|sec|h|hr|min)$"), core) {
            if g(&m, 1).is_some() || is_num(prev) {
                let n = g(&m, 1).unwrap_or(prev);
                let u = gs(&m, 2);
                let name = match tt.units.get(u) {
                    Some(x) => x.clone(),
                    None => match u {
                        "Mb" => ("megabit".into(), "megabits".into()),
                        "Gb" => ("gigabit".into(), "gigabits".into()),
                        "kb" => ("kilobit".into(), "kilobits".into()),
                        _ => return Err(Bail),
                    },
                };
                let per = match gs(&m, 3) {
                    "s" | "sec" => "second",
                    "h" | "hr" => "hour",
                    _ => "minute",
                };
                let mut w: Vec<String> = if g(&m, 1).is_some() { vec![n.to_string()] } else { vec![] };
                w.extend(py::split_sp(if one_n(n) { &name.0 } else { &name.1 }));
                w.extend(["per".into(), per.into()]);
                return out(w, 1, trail);
            }
        }
        if matches!(low, "sq" | "sq." | "cu") && is_num(prev) && (tt.units.contains_key(nxt) || matches!(nl, "in" | "ft")) {
            let z = nsplit(nxt_raw);
            let u = if tt.units.contains_key(&z.1) { z.1.clone() } else { py::lower(&z.1) };
            let name = tt.units.get(&u).cloned().unwrap_or(("inch".into(), "inches".into()));
            let mut w = vec![if low.starts_with("sq") { "square" } else { "cubic" }.to_string()];
            w.extend(py::split_sp(if one_n(prev) { &name.0 } else { &name.1 }));
            let tr = if z.2 != "." { py::lstrip_set(&z.2, ".").to_string() } else { z.2.clone() };
            return out(w, 2, tr);
        }
        if core == "in"
            && is_num(prev)
            && trail.is_empty()
            && matches!(nl, "long" | "wide" | "tall" | "deep" | "thick" | "high" | "across" | "diameter" | "square")
        {
            return out(sv(&[if prev == "1" { "inch" } else { "inches" }]), 1, trail);
        }
        if matches!(core, "°C" | "°F" | "°") && is_num(prev) {
            let mut w = sv(&[if one_n(prev) { "degree" } else { "degrees" }]);
            match core {
                "°C" => w.push("Celsius".into()),
                "°F" => w.push("Fahrenheit".into()),
                _ => {}
            }
            return out(w, 1, trail);
        }
        if core == "g" && is_num(prev) && (matches!(nl, "of" | "") || !trail.is_empty()) {
            let u = tt.units.get("g").ok_or(Bail)?;
            return out(py::split_sp(if one_n(prev) { &u.0 } else { &u.1 }), 1, trail);
        }
        // римские цифры: пункт списка "i." в начале строки, "Grand Theft Auto V"
        if i == 0
            && trail == "."
            && words::roman_match(&py::upper(core))
            && (py::is_lower(core) || (py::clen(core) > 1 && py::is_upper(core)))
            && py::first_upper(nxt)
        {
            return out(vec![py::capitalize(&cardinal(roman_value(&py::upper(core))?)?)], 1, trail);
        }
        if core == "V"
            && py::first_upper(prev)
            && py::is_alpha(prev)
            && !py::is_upper(prev)
            && !tt.regnal.contains(pl)
            && !tt.letter_cue.contains(pl)
            && i > 1
            && !py::last_in(&toks[i - 1], ".!?:;")
            && (nxt.is_empty() || py::first_lower(nxt))
        {
            return out(sv(&["five"]), 1, trail);
        }
        // веб: "https" по буквам, "@bakes.by.bella", "@NYTimes"
        if matches!(low, "http" | "https") && py::is_lower(core) {
            return out(self.spell(low)?, 1, trail);
        }
        if is_m(re!(r"^@[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+$"), core) {
            let mut w = vec!["at".to_string()];
            w.extend(self.say_parts(&py::from(core, 1))?);
            return out(w, 1, trail);
        }
        if is_m(re!(r"^@[A-Za-z0-9_]+$"), core) && is_m(re!(r"[a-z]"), core) && is_m(re!(r"[A-Z]{2}[a-z]"), core) {
            let mut w = vec!["at".to_string()];
            for part in py::findall(re!(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|_"), &py::from(core, 1)) {
                if part == "_" {
                    w.push("underscore".into());
                } else if !py::is_upper(&part) {
                    w.extend(self.say_run(&part)?);
                } else if self.initialism(&part) || py::clen(&part) <= 3 {
                    w.extend(self.spell(&part)?);
                } else {
                    w.push(part);
                }
            }
            return out(w, 1, trail);
        }
        Ok(None)
    }
}
