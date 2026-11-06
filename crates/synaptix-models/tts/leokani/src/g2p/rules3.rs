//! Правила нормализатора первого круга (`basic`): почта, хэштеги, адреса сайтов, деньги, градусы,
//! телефоны, IP, версии, десятилетия, даты, дроби, время, диапазоны, единицы, сокращения с точкой,
//! римские цифры, имена и омографы. Здесь же общие регулярные выражения `Normalizer`.

#![allow(non_snake_case)]

use fancy_regex::Regex;

use super::norm::{is_num, nsplit, out, Ctx, Out};
use super::py::{self, caps, g, gs, is_m, re, sv, Bail, R};
use super::tables::t;
use super::words::{self, cardinal, digit_words, ordinal, plural_number, roman_value, year_words, PH};
use super::Lex;

pub fn TIME() -> &'static Regex {
    re!(r"(?i)^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:([ap])\.?m\.?)?$")
}
pub fn TIME_AMPM() -> &'static Regex {
    re!(r"(?i)^(\d{1,2})([ap])\.?m\.?$")
}
pub fn MONEY() -> &'static Regex {
    re!(r"^(-?)([$€£¥₹])(\d[\d,]*)?(\.\d+)?(k|K|m|M|mm|b|B|bn|t|T)?$")
}
pub fn PHONE() -> &'static Regex {
    re!(r"^(?:(\+?1)[-. ])?\(?(\d{3})\)?[-.](\d{3})[-.](\d{4})$")
}
pub fn TOLLWORD() -> &'static Regex {
    re!(r"^1-(\d{3})-([A-Za-z]{3,})$")
}
pub fn IPV4() -> &'static Regex {
    re!(r"^\d{1,3}(?:\.\d{1,3}){3}$")
}
pub fn VERSION_DASH() -> &'static Regex {
    re!(r"^(\d+(?:\.\d+){2,})((?:-[A-Za-z0-9.]+){2,})$")
}
pub fn DATE_ISO() -> &'static Regex {
    re!(r"^(\d{4})-(\d{2})-(\d{2})$")
}
pub fn FRACTION() -> &'static Regex {
    re!(r"^(\d{1,3})/(\d{1,3})$")
}

impl Lex {
    pub(super) fn basic(&self, c: &Ctx) -> R<Out> {
        let tt = t();
        let (core, trail, prev, prev2, prev_end, nxt, nxt_raw, toks, i, pauses) =
            (c.core, c.trail, c.prev, c.prev2, c.prev_end, c.nxt, c.nxt_raw, c.toks, c.i, c.pauses);
        let low = c.low.as_str();
        let dot = trail.starts_with('.');
        // конец предложения после сокращения: точка остаётся только в конце строки или перед заглавной
        let keep_dot = if i + 1 >= toks.len() || py::first_upper(nxt) { trail.to_string() } else { py::from(trail, 1) };
        let nxt_cap = py::first_upper(nxt);
        let prev_cap = (py::first_upper(prev) && py::is_alpha(prev)) || is_m(re!(r"^\d+(st|nd|rd|th)?$"), prev);
        let nxt_num = py::first_digit(nxt);
        let prev_num = is_num(prev);
        let pl = c.pl.as_str();
        let nl = c.nl.as_str();

        // -- символы и цифры
        if let Some(m) = caps(re!(r"^([A-Za-z0-9][A-Za-z0-9._%+-]*)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)$"), core) {
            let mut w = self.say_parts(gs(&m, 1))?;
            w.push("at".into());
            w.extend(self.domain(gs(&m, 2))?);
            return out(w, 1, trail);
        }
        if core.starts_with('@') && is_m(re!(r"^@[A-Za-z0-9_]+$"), core) {
            let mut w = sv(&["at"]);
            w.extend(self.say_parts(&py::from(core, 1))?);
            return out(w, 1, trail);
        }
        if core.starts_with('#') && py::clen(core) > 1 {
            let rest = py::from(core, 1);
            if py::is_digit(&rest) {
                let mut w = sv(&["number"]);
                if py::clen(core) > 4 {
                    w.extend(digit_words(&rest)?);
                } else {
                    w.push(rest);
                }
                return out(w, 1, trail);
            }
            if is_m(re!(r"^#[A-Za-z][\p{L}\p{N}_]*$"), core) {
                let mut w = sv(&["hashtag"]);
                for p in py::findall(re!(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+"), &rest) {
                    let lp = py::lower(&p);
                    w.push(if self.known(&lp) { lp } else { p });
                }
                return out(w, 1, trail);
            }
        }
        if is_m(re!(r"^[A-G]#$"), core) {
            return out(vec![py::to(core, 1), "sharp".into()], 1, trail);
        }
        if let Some(m) = caps(re!(r"(?i)^(?:(https?|ftp)://)?((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(/\S*)?$"), core) {
            let host0 = gs(&m, 2);
            let tld = host0.rsplit_once('.').map(|x| x.1).unwrap_or(host0);
            let first = host0.split('.').next().unwrap_or("");
            if (g(&m, 1).is_some() || tt.tlds.contains(&py::lower(tld)))
                && is_m(re!(r"[A-Za-z]{2}"), first)
                && (g(&m, 1).is_some() || !dot || low.matches('.').count() >= 1)
            {
                if g(&m, 1).is_none() && !tt.tlds.contains(tld) {
                    return Ok(None);
                }
                if tt.word_abbr.contains_key(low) || matches!(format!("{low}.").as_str(), "e.g." | "i.e.") {
                    return Ok(None);
                }
                let mut w = Vec::new();
                let mut host = host0.to_string();
                if py::lower(&host).starts_with("www.") {
                    w.extend(self.spell("www")?);
                    w.push("dot".into());
                    host = py::from(&host, 4);
                }
                w.extend(self.domain(&host)?);
                for seg in g(&m, 3).unwrap_or("").split('/').skip(1) {
                    w.push("slash".into());
                    if is_m(re!(r"(?i)^[a-z]{2}[-_][a-z]{2}$"), seg) {
                        w.extend(self.spell(&py::to(seg, 2))?);
                        w.push(tt.sep_words.get(&py::sl(seg, Some(2), Some(3))).ok_or(Bail)?.clone());
                        w.extend(self.spell(&py::from(seg, 3))?);
                    } else if is_m(re!(r"^[1-9]\d{0,3}$"), seg) {
                        w.push(seg.into());
                    } else if !seg.is_empty() {
                        w.extend(self.say_parts(seg)?);
                    }
                }
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(MONEY(), core) {
            if g(&m, 3).is_some() || g(&m, 4).is_some() {
                let nscale = if tt.scale_words.contains(nl) && g(&m, 5).is_none() { Some(nl) } else { None };
                let w = self.money(gs(&m, 1), gs(&m, 2), g(&m, 3), g(&m, 4), g(&m, 5), nscale)?;
                if nscale.is_some() {
                    return out(w, 2, nsplit(nxt_raw).2);
                }
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^(\d+)¢$"), core) {
            let n = gs(&m, 1);
            return out(vec![n.into(), if n == "1" { "cent" } else { "cents" }.into()], 1, trail);
        }
        if let Some(m) = caps(re!(r"^(-?\d[\d,]*(?:\.\d+)?)°([CF])?$"), core) {
            if g(&m, 2).is_some() || core.ends_with('°') {
                let n = gs(&m, 1);
                let mut w = if let Some(r) = n.strip_prefix('-') { vec!["minus".into(), r.to_string()] } else { vec![n.to_string()] };
                w.push(if n == "1" || n == "-1" { "degree" } else { "degrees" }.into());
                match g(&m, 2) {
                    Some("C") => w.push("Celsius".into()),
                    Some("F") => w.push("Fahrenheit".into()),
                    _ => {}
                }
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(TOLLWORD(), core) {
            let gr = vec!["1".to_string(), gs(&m, 1).to_string(), gs(&m, 2).to_string()];
            return out(self.phone(&gr, pauses, false)?, 1, trail);
        }
        if let Some(m) = caps(PHONE(), core) {
            let gr: Vec<&str> = (1..=4).filter_map(|k| g(&m, k)).filter(|x| !x.is_empty()).collect();
            let plus = gr[0].starts_with('+');
            let gr: Vec<String> = gr.iter().map(|x| py::lstrip_set(x, "+").to_string()).collect();
            return out(self.phone(&gr, pauses, plus)?, 1, trail);
        }
        if is_m(re!(r"^\(\d{3}\)$"), &toks[i]) || (is_m(re!(r"^\d{3}$"), core) && toks[i].starts_with('(')) {
            if let Some(m2) = caps(re!(r"^(\d{3})-(\d{4})$"), nxt) {
                let gr = vec![core.to_string(), gs(&m2, 1).to_string(), gs(&m2, 2).to_string()];
                return out(self.phone(&gr, pauses, false)?, 2, nsplit(nxt_raw).2);
            }
        }
        if is_m(re!(r"^\+\d{1,3}$"), core) {
            let mut groups = vec![py::from(core, 1)];
            let mut j = i + 1;
            while j < toks.len() && is_m(re!(r"^\d{1,4}(-\d{1,4})*$"), &nsplit(&toks[j]).1) {
                groups.extend(nsplit(&toks[j]).1.split('-').map(str::to_string));
                j += 1;
                if !nsplit(&toks[j - 1]).2.is_empty() {
                    break;
                }
            }
            if groups.len() >= 3 {
                return out(self.phone(&groups, pauses, true)?, j - i, nsplit(&toks[j - 1]).2);
            }
        }
        if let Some(m) = caps(re!(r"^(\d{3})-(\d{4})$"), core) {
            if !prev_num {
                let gr = vec![gs(&m, 1).to_string(), gs(&m, 2).to_string()];
                return out(self.phone(&gr, pauses, false)?, 1, trail);
            }
        }
        if is_m(re!(r"^[2-9]11$"), core) && (core == "911" || matches!(pl, "dial" | "call" | "called" | "calling")) {
            return out(digit_words(core)?, 1, trail);
        }
        if is_m(IPV4(), core) {
            let mut all = true;
            for x in core.split('.') {
                if py::int(x)? >= 256 {
                    all = false;
                    break;
                }
            }
            if all {
                return out(self.ip(core)?, 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^([vV])?(\d+(?:\.\d+)+)([-+][A-Za-z0-9.]+)?$"), core) {
            if gs(&m, 2).matches('.').count() >= 2 || g(&m, 1).is_some() {
                let mut w: Vec<String> =
                    if g(&m, 1).is_some() && !matches!(pl, "version" | "release" | "v") { sv(&["version"]) } else { vec![] };
                for (k, x) in gs(&m, 2).split('.').enumerate() {
                    if k > 0 {
                        w.push("point".into());
                    }
                    if !(py::clen(x) > 1 && x.starts_with('0')) {
                        w.push(x.into());
                    } else {
                        w.extend(digit_words(x)?);
                    }
                }
                if let Some(x) = g(&m, 3) {
                    w.extend(py::findall(re!(r"[A-Za-z]+|\d+"), x));
                }
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^\.(\d+)$"), core) {
            let mut w = sv(&["point"]);
            w.extend(digit_words(gs(&m, 1))?);
            return out(w, 1, trail);
        }
        if let Some(m) = caps(re!(r"^['’]?(\d{2}|\d{4})s$"), core) {
            let d = gs(&m, 1);
            if d.ends_with('0') && (py::clen(d) == 2 || (1000..=2090).contains(&py::int(d)?)) {
                return out(py::split_sp(&plural_number(py::int(d)?)?), 1, trail);
            }
        }
        if let Some(m) = caps(DATE_ISO(), core) {
            let (mo, dd) = (py::int(gs(&m, 2))?, py::int(gs(&m, 3))?);
            if (1..=12).contains(&mo) && (1..=31).contains(&dd) {
                let sep = if pauses { "," } else { "" };
                let mut w = vec![tt.months[(mo - 1) as usize].clone(), format!("{}{sep}", ordinal(dd)?)];
                w.extend(year_words(py::int(gs(&m, 1))?)?);
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$"), core) {
            let (mo, dd) = (py::int(gs(&m, 1))?, py::int(gs(&m, 2))?);
            if (1..=12).contains(&mo) && (1..=31).contains(&dd) {
                let g3 = gs(&m, 3);
                let y = if py::clen(g3) == 4 {
                    g3.to_string()
                } else {
                    format!("{}{g3}", if py::int(g3)? < 50 { "20" } else { "19" })
                };
                let sep = if pauses { "," } else { "" };
                let mut w = vec![tt.months[(mo - 1) as usize].clone(), format!("{}{sep}", ordinal(dd)?)];
                w.extend(year_words(py::int(&y)?)?);
                return out(w, 1, trail);
            }
        }
        if let Some(m) = caps(FRACTION(), core) {
            let (n, d) = (py::int(gs(&m, 1))?, py::int(gs(&m, 2))?);
            let fr = Self::fraction(n, d)?;
            let unit_next = matches!(
                nl,
                "of" | "inch" | "inches" | "cup" | "cups" | "mile" | "miles" | "teaspoon" | "teaspoons" | "tablespoon" | "tablespoons"
                    | "pound" | "pounds" | "ounce" | "ounces" | "hour" | "hours" | "an" | "a" | "in" | "tsp" | "tbsp" | "lb" | "oz"
            );
            if let Some(fr) = fr {
                if matches!(d, 2 | 3 | 4 | 8) || unit_next || (matches!(d, 16 | 32 | 64) && n % 2 != 0) {
                    return out(fr, 1, trail);
                }
            }
            if (1..=12).contains(&n) && (13..=31).contains(&d) {
                return out(vec![tt.months[(n - 1) as usize].clone(), ordinal(d)?], 1, trail);
            }
            if matches!((n, d), (24, 7) | (9, 11) | (7, 11) | (50, 50) | (24, 365)) {
                return out(vec![n.to_string(), d.to_string()], 1, trail);
            }
            return Ok(None);
        }
        if is_m(re!(r"^\d+$"), core) && is_m(FRACTION(), nxt) {
            let (x, y) = nxt.split_once('/').ok_or(Bail)?;
            let (n, d) = (py::int(x)?, py::int(y)?);
            if let Some(fr) = Self::fraction(n, d)? {
                if py::int(core)? < 1000 {
                    let mut w = vec![cardinal(py::int(core)?)?, "and".into()];
                    if n == 1 {
                        w.extend(["a".into(), fr[1].clone()]);
                    } else {
                        w.extend(fr);
                    }
                    return out(w, 2, nsplit(nxt_raw).2);
                }
            }
        }
        if let Some(m) = caps(TIME(), core) {
            if g(&m, 3).is_some() || g(&m, 4).is_some() || gs(&m, 2).starts_with('0') {
                if let Some(w) = self.time(gs(&m, 1), gs(&m, 2), g(&m, 3), g(&m, 4))? {
                    if !w.is_empty() {
                        return out(w, 1, trail);
                    }
                }
            }
        }
        if let Some(m) = caps(TIME_AMPM(), core) {
            if py::int(gs(&m, 1))? <= 12 {
                return out(vec![gs(&m, 1).into(), self.letter(gs(&m, 2)), "M".into()], 1, trail);
            }
        }
        if matches!(core, "am" | "pm") && (prev_num || is_m(TIME(), prev)) {
            return out(vec![self.letter(&py::to(core, 1)), "M".into()], 1, trail);
        }
        if let Some(m) = caps(re!(r"^(\d{1,2}):(\d)$"), core) {
            if !py::first_upper(prev) && !prev_num {
                return out(vec![gs(&m, 1).into(), "to".into(), gs(&m, 2).into()], 1, trail);
            }
        }
        if is_m(re!(r"^\d{1,2}:\d{2}[-–]\d{1,2}:\d{2}$"), core) {
            let ab = py::re_split(re!(r"[-–]"), core);
            if ab.len() != 2 {
                return Err(Bail);
            }
            let tm = |s: &str| -> R<Vec<String>> {
                let (h, m) = s.split_once(':').ok_or(Bail)?;
                Ok(self.time(h, m, None, None)?.filter(|v| !v.is_empty()).unwrap_or_else(|| vec![s.to_string()]))
            };
            let mut w = tm(&ab[0])?;
            w.push("to".into());
            w.extend(tm(&ab[1])?);
            return out(w, 1, trail);
        }
        if let Some(m) = caps(re!(r"^(\d[\d,]*(?:\.\d+)?)[-–](\d[\d,]*(?:\.\d+)?)$"), core) {
            let (a, b) = (gs(&m, 1), gs(&m, 2));
            if a == b {
                return out(vec![a.into(), b.into()], 1, trail);
            }
            return out(vec![a.into(), "to".into(), b.into()], 1, trail);
        }
        if let Some(m) = caps(re!(r"^(\d+)[x×](\d+)$"), core) {
            let mut w = Self::pairs(gs(&m, 1));
            w.push("by".into());
            w.extend(Self::pairs(gs(&m, 2)));
            return out(w, 1, trail);
        }
        if let Some(m) = caps(re!(r"^(\d+(?:\.\d+)?)[x×]$"), core) {
            return out(vec![gs(&m, 1).into(), "times".into()], 1, trail);
        }
        if let Some(m) = caps(
            re!(r"^(\d[\d,]*(?:\.\d+)?)(km|cm|mm|nm|µm|μm|kg|mg|µg|mcg|lbs?|oz|ft|mi|mph|kph|ml|mL|tsp|tbsp|kWh|Wh|kW|MW|mW|W|mV|kV|V|mAh|mA|µA|Hz|kHz|MHz|GHz|KB|kB|MB|GB|TB|Mbps|Gbps|ms|µs|ns|hrs?|mins?|secs?|dB|rpm|psi)$"),
            core,
        ) {
            let (sing, plur) = tt.units.get(gs(&m, 2)).ok_or(Bail)?;
            let mut w = vec![gs(&m, 1).to_string()];
            w.extend(py::split_sp(if gs(&m, 1) == "1" { sing } else { plur }));
            return out(w, 1, trail);
        }
        if is_m(re!(r"^\d{13,}$"), core) {
            return out(digit_words(core)?, 1, trail);
        }
        if matches!(core, "~" | "≈") && nxt_num {
            return out(sv(&["about"]), 1, trail);
        }
        if core.starts_with('~') && is_m(re!(r"^~\d"), core) {
            return out(vec!["about".into(), py::from(core, 1)], 1, trail);
        }
        if matches!(core, ">" | "<" | "*" | "=" | "+" | "&" | "±" | ">=" | "<=") && (prev_num || nxt_num || core == "&") {
            let mut say: Vec<&str> = match core {
                ">" => vec!["is", "greater", "than"],
                "<" => vec!["is", "less", "than"],
                "*" => vec!["times"],
                "=" => vec!["equals"],
                "+" => vec!["plus"],
                "&" => vec!["and"],
                "±" => vec!["plus", "or", "minus"],
                ">=" => vec!["is", "at", "least"],
                _ => vec!["is", "at", "most"],
            };
            if matches!(core, ">" | "<") && matches!(pl, "is" | "are" | "was" | "were") {
                say.remove(0);
            }
            return out(sv(&say), 1, trail);
        }
        if is_m(re!(r"^[A-Z]{1,3}&[A-Z]{1,3}$"), core) {
            let (a, b) = core.split_once('&').unwrap();
            let mut w = self.spell(a)?;
            w.push("and".into());
            w.extend(self.spell(b)?);
            return out(w, 1, trail);
        }
        if is_m(re!(r"^[A-Za-z]+&[A-Za-z]+$"), core) {
            let (a, b) = core.split_once('&').unwrap();
            return out(vec![a.into(), "and".into(), b.into()], 1, trail);
        }
        if is_m(re!(r"[A-Za-z](—|--|–)[A-Za-z]"), core) {
            let w: Vec<String> = py::re_split(re!(r"—|--|–"), core).into_iter().filter(|p| !p.is_empty()).collect();
            return out(w, 1, trail);
        }
        if is_m(re!(r"^(?i:rock)['’]n['’](?i:roll)$"), core) {
            let head = core.split('\'').next().unwrap().split('’').next().unwrap().to_string();
            return out(vec![head, "and".into(), "roll".into()], 1, trail);
        }

        // -- числа с единицей после
        if prev_num && tt.units.contains_key(core) && !(matches!(core, "in" | "m" | "g") && !dot && core != "m") {
            if core == "in" {
                return Ok(None);
            }
            let (sing, plur) = &tt.units[core];
            return out(py::split_sp(if matches!(prev, "1" | "-1") { sing } else { plur }), 1, trail);
        }
        if prev_num && low == "in" && dot {
            let tr = if keep_dot != trail { py::from(&keep_dot, 1) } else { trail.to_string() };
            return out(sv(&[if prev == "1" { "inch" } else { "inches" }]), 1, tr);
        }
        if prev_num
            && matches!(
                low,
                "ft" | "lbs" | "lb" | "oz" | "hrs" | "hr" | "mins" | "min" | "secs" | "sec" | "yrs" | "yr" | "tsp" | "tbsp" | "mi" | "yds" | "yd"
                    | "gal" | "qt"
            )
            && dot
        {
            let mut w = tt.word_abbr.get(low).ok_or(Bail)?.clone();
            if !matches!(prev, "1" | "-1") {
                w = match w.as_str() {
                    "foot" => "feet",
                    "inch" => "inches",
                    "pound" => "pounds",
                    "ounce" => "ounces",
                    "hour" => "hours",
                    "minute" => "minutes",
                    "second" => "seconds",
                    "year" => "years",
                    "teaspoon" => "teaspoons",
                    "tablespoon" => "tablespoons",
                    "mile" => "miles",
                    "yard" => "yards",
                    "gallon" => "gallons",
                    "quart" => "quarts",
                    x => x,
                }
                .to_string();
            }
            let tr = if !(i + 1 >= toks.len() || nxt_cap) { py::from(trail, 1) } else { trail.to_string() };
            return out(vec![w], 1, tr);
        }

        // -- сокращения с точкой
        let first_cap = py::first_upper(core);
        if dot || low == "e.g" {
            let key = low;
            if key == "e.g" {
                return out(py::split_sp(&tt.word_abbr[key]), 1, if dot { py::from(trail, 1) } else { trail.to_string() });
            }
            if key == "no" && matches!(core, "No" | "no") && (nxt_num || nxt.starts_with('#')) {
                return out(sv(&["number"]), 1, py::from(trail, 1));
            }
            if matches!(key, "st" | "dr") && first_cap {
                // "Pine St. near", "Mulberry Dr." -> улица; "St. James", "Dr. Smith" остаются
                if (prev_cap || prev_num) && !prev_end && !(nxt_cap && i + 1 < toks.len() && trail == ".") {
                    let tr = if py::clen(trail) == 1 { keep_dot.clone() } else { py::from(trail, 1) };
                    return out(vec![tt.street_abbr[key].clone()], 1, tr);
                }
                if key == "st" && nxt_cap && trail == "." {
                    return out(sv(&["Saint"]), 1, "");
                }
                return Ok(None);
            }
            if matches!(key, "jr" | "sr") && first_cap && prev_cap {
                return out(vec![tt.street_abbr[key].clone()], 1, keep_dot);
            }
            if let Some(ti) = tt.title_abbr.get(key) {
                if first_cap && (nxt_cap || nxt_num) {
                    if key == "ft" && !nxt_cap {
                        return Ok(None);
                    }
                    return out(vec![ti.clone()], 1, py::from(trail, 1));
                }
            }
            if let Some(sa) = tt.street_abbr.get(key) {
                if first_cap && !matches!(key, "st" | "dr" | "jr" | "sr") && (prev_cap || prev_num) {
                    return out(vec![sa.clone()], 1, keep_dot);
                }
            }
            if let Some(&mo) = tt.month_abbr.get(key) {
                if first_cap && (nxt_num || prev_num) {
                    return out(vec![tt.months[mo - 1].clone()], 1, keep_dot);
                }
            }
            if let Some(day) = tt.day_abbr.get(key) {
                if first_cap {
                    return out(vec![day.clone()], 1, keep_dot);
                }
            }
            if matches!(
                key,
                "vs" | "approx" | "apt" | "dept" | "est" | "fig" | "vol" | "ch" | "pp" | "ca" | "cf" | "corp" | "govt" | "intl" | "misc"
            ) && (key != "ch" || nxt_num)
                && (key != "est" || nxt_num)
                && (!matches!(key, "fig" | "vol" | "pp" | "apt") || nxt_num || py::first_upper(nxt))
            {
                let tr = if !(i + 1 >= toks.len() || nxt_cap) { py::from(trail, 1) } else { trail.to_string() };
                return out(py::split_sp(&tt.word_abbr[key]), 1, tr);
            }
        }
        let day_n = is_m(re!(r"^\d{1,2}(st|nd|rd|th)?$"), nxt);
        if let Some(&mo) = tt.month_abbr.get(low) {
            if (first_cap && py::is_alpha(core) && nxt_num && day_n && !matches!(low, "mar" | "jun"))
                || (matches!(low, "mar" | "jun") && first_cap && day_n)
            {
                return out(vec![tt.months[mo - 1].clone()], 1, trail);
            }
        }
        if low == "vs" {
            return out(sv(&["versus"]), 1, trail);
        }

        // -- римские цифры
        if words::roman_match(core) && py::is_upper(core) {
            let v = roman_value(core)?;
            let cl = py::clen(core);
            if v != 0 && tt.regnal.contains(pl) && py::first_upper(prev) && (cl > 1 || !trail.is_empty() || nxt.is_empty() || nl == "of") {
                return out(vec!["the".into(), ordinal(v)?], 1, trail);
            }
            if (v != 0 && tt.roman_cue.contains(pl) && py::first_upper(prev) && (cl > 1 || matches!(core, "V" | "X")))
                || (v != 0 && tt.roman_cue.contains(pl) && core == "I" && (!trail.is_empty() || nxt.is_empty()))
            {
                return out(vec![cardinal(v)?], 1, trail);
            }
            if v != 0
                && cl > 1
                && !tt.roman_ambiguous.contains(core)
                && py::first_upper(prev)
                && py::is_alpha(prev)
                && !py::is_upper(prev)
            {
                return out(vec![cardinal(v)?], 1, trail);
            }
        }

        // -- имена, марки, омографы
        let plain = words::strip_okina(core);
        if let Some(p) = tt.say_as_cased.get(core) {
            return out(vec![format!("{core}{PH}{p}")], 1, trail);
        }
        let plow = py::lower(&plain);
        if let Some(p) = tt.say_as.get(&plow) {
            if py::is_alpha(&plain.replace('-', "")) && (!matches!(plow.as_str(), "kea" | "lanai") || pl == "mauna" || py::first_upper(&plain)) {
                return out(vec![format!("{plain}{PH}{p}")], 1, trail);
            }
        }
        if let Some(v) = tt.verb_forms.get(low) {
            if tt.verb_before.contains(pl) && (low != "live" || matches!(pl, "i" | "eye")) && (low != "present" || pl == "please") {
                return out(vec![format!("{core}{PH}{v}")], 1, trail);
            }
        }
        if low == "close" && (tt.verb_before.contains(pl) || prev_end) && tt.objects.contains(nl) {
            return out(vec![format!("{core}{PH}klˈOz")], 1, trail);
        }
        if low == "bow"
            && ((matches!(pl, "a" | "his" | "her" | "their" | "final") && matches!(prev2, "take" | "took" | "takes" | "taking" | "a"))
                || matches!(nl, "down" | "before" | "deeply" | "low"))
        {
            return out(vec![format!("{core}{PH}bˈW")], 1, trail);
        }
        if low == "wound"
            && (matches!(nl, "up" | "down" | "around" | "round" | "through" | "back") || tt.objects.contains(nl))
            && !matches!(
                pl,
                "the" | "a" | "his" | "her" | "my" | "your" | "our" | "their" | "its" | "this" | "that" | "open" | "deep" | "fresh" | "stab"
                    | "gunshot" | "head" | "old"
            )
        {
            return out(vec![format!("{core}{PH}wˈWnd")], 1, trail);
        }
        if low == "dove"
            && (tt.subjects.contains(pl)
                || matches!(nl, "into" | "in" | "under" | "off" | "for" | "down" | "through" | "onto" | "deep" | "headfirst" | "back"))
        {
            return out(vec![format!("{core}{PH}dˈOv")], 1, trail);
        }
        if low == "bass"
            && (matches!(pl, "largemouth" | "smallmouth" | "striped" | "sea" | "rock" | "peacock")
                || (pl == "a" && matches!(prev2, "caught" | "catch" | "catching" | "landed" | "hooked"))
                || matches!(nl, "fishing" | "fisherman" | "boat" | "lure" | "lures" | "pond"))
        {
            return out(vec![format!("{core}{PH}bˈæs")], 1, trail);
        }
        if words::is_emoji(core) {
            return out(vec![], 1, trail);
        }
        if let Some(said) = self.homograph(core, toks, i) {
            return out(vec![format!("{core}{PH}{said}")], 1, trail);
        }
        Ok(None)
    }
}
