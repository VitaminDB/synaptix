//! Правила нормализатора: `letters_first` (буквы и инициализмы v1.2, раньше всех) и `more2`
//! (сокращения из переписки, диапазоны дней и дат, коды, часовые пояса, единицы после дробей).

use super::norm::{core_of, is_num, out, Ctx, Out, SIBILANT_END, VOICELESS_END};
use super::py::{self, caps, g, gs, is_m, re, sv, R};
use super::tables::t;
use super::words::{self, cardinal, digit_words, ordinal, roman_value, PH};
use super::Lex;

impl Lex {
    pub(super) fn letters_first(&self, c: &Ctx) -> R<Out> {
        let tt = t();
        let (core, trail, prev, nxt, nxt_raw, toks, i) = (c.core, c.trail, c.prev, c.nxt, c.nxt_raw, c.toks, c.i);
        let (low, pl, nl) = (c.low.as_str(), c.pl.as_str(), c.nl.as_str());
        // одиночные буквы через пробел: "A G C"; одинокое "A"/"I" перед словом — артикль/местоимение
        if core == "A" || core == "I" {
            let after_letter = i > 0
                && Self::single(&toks[i - 1])
                && !matches!(core_of(&toks[i - 1]).as_str(), "A" | "I")
                && !py::last_in(&toks[i - 1], ".!?;:");
            let before_letter = trail.is_empty() && i + 1 < toks.len() && Self::single(nxt_raw);
            let tr0 = trail.is_empty() || py::first_in(trail, ",;");
            if after_letter
                || before_letter
                || (core == "A"
                    && (matches!(pl, "letter" | "letters" | "an")
                        || (matches!(nl, "and" | "or") && trail.is_empty())
                        || (tr0 && Self::single(nxt_raw))))
                || (matches!(pl, "and" | "or") && i > 1 && Self::single(&toks[i - 2]))
            {
                return out(vec![self.letter(core)], 1, trail);
            }
        }
        // множественное и притяжательное инициализмов: "CEOs", "PDFs", "AGC's"
        if let Some(m) = caps(re!(r"^([A-Z]{2,6})(s|'s|’s)$"), core) {
            let st = gs(&m, 1);
            if !tt.shorthand_cased.contains_key(st) && self.initialism(st) && !tt.shorthand_cased.contains_key(core) && core != "IDs" {
                return out(self.spell_word(st, gs(&m, 2))?, 1, trail);
            }
        }
        // заглавные с точками: "A.G.C.", "N.Y.C."
        if is_m(re!(r"^(?:[A-Z]\.){1,}[A-Z]$"), core) && trail.starts_with('.') && py::clen(core) >= 3 {
            let keep = if i + 1 >= toks.len() || (py::first_upper(nxt) && nxt != "I") { trail.to_string() } else { py::from(trail, 1) };
            let plain = core.replace('.', "");
            if tt.acronym_words.contains(&plain) {
                return out(vec![plain], 1, keep);
            }
            return out(self.spell_word(&plain, "")?, 1, keep);
        }
        // инициал в середине имени: "George W. Bush"
        let initial = |x: &str| is_m(re!(r"^[A-Z]\.$"), x);
        if py::clen(core) == 1
            && py::is_upper(core)
            && trail == "."
            && py::first_upper(nxt)
            && !tt.sentence_start.contains(nl)
            && (initial(nxt_raw)
                || (i > 0 && initial(&toks[i - 1]) && py::is_alpha(nxt))
                || (py::first_upper(prev)
                    && py::is_alpha(prev)
                    && !py::is_upper(prev)
                    && py::clen(nxt) > 1
                    && py::is_alpha(nxt)
                    && !py::last_in(&toks[i - 1], ".!?:;,")))
            && !toks[i + 1..toks.len().min(i + 4)]
                .iter()
                .any(|x| tt.street_words.contains(py::rstrip_set(&py::lower(&core_of(x)), ".")))
        {
            return out(vec![self.letter(core)], 1, py::from(trail, 1));
        }
        // буква перед словом: "p-value", "t-test"; слово перед буквой: "Type-C"
        if let Some(m) = caps(re!(r"^([a-zA-Z])-(value|values|test|tests|axis|score|scores|statistic|coordinate|intercept)$"), core) {
            return out(vec![self.letter(gs(&m, 1)), gs(&m, 2).to_string()], 1, trail);
        }
        if let Some(m) = caps(re!(r"^([A-Za-z]{2,})-([A-Z])$"), core) {
            if py::is_alpha(gs(&m, 1)) {
                return out(vec![gs(&m, 1).to_string(), self.letter(gs(&m, 2))], 1, trail);
            }
        }
        // буква и номер модели: "F-150", "B-52"
        if let Some(m) = caps(re!(r"^([A-Z]{1,2})-(\d{2,3})$"), core) {
            if py::to(core, 2) != "I-" {
                let d = gs(&m, 2);
                let dc: Vec<char> = d.chars().collect();
                let num = if dc.len() == 2 || (dc[1] == '0' && dc[2] == '0') {
                    words::split_words(&cardinal(py::int(d)?)?)
                } else {
                    let mut v = vec![tt.one(py::int(&dc[0].to_string())?)?];
                    if dc[1] == '0' {
                        v.extend(["oh".into(), tt.one(py::int(&dc[2].to_string())?)?]);
                    } else {
                        v.extend(words::split_words(&cardinal(py::int(&py::from(d, 1))?)?));
                    }
                    v
                };
                let mut w = self.spell(gs(&m, 1))?;
                w.extend(num);
                return out(w, 1, trail);
            }
        }
        // инициализмы смешанного регистра: "DoD", "MoU"
        if is_m(re!(r"^[A-Z]o[A-Z]s?$"), core) && !tt.say_as_cased.contains_key(core) {
            return out(self.spell_word(&py::to(core, 3), &py::from(core, 3))?, 1, trail);
        }
        // разрешение: "1080p", "720p"
        if let Some(m) = caps(re!(r"^(\d{3,4})([pi])$"), core) {
            let d = gs(&m, 1);
            let mut w = if py::clen(d) == 4 {
                words::pair_words(py::int(d)?)?
            } else {
                let mut v = vec![tt.one(py::int(&py::to(d, 1))?)?];
                v.extend(words::split_words(&cardinal(py::int(&py::from(d, 1))?)?));
                v
            };
            w.push(self.letter(gs(&m, 2)));
            return out(w, 1, trail);
        }
        // притяжательное слова заглавными: "NATO's"
        if let Some(m) = caps(re!(r"^([A-Z]{4,})(['’]s)$"), core) {
            let st = gs(&m, 1);
            let rd = self.reading(&py::lower(st));
            if !self.initialism(st) && !rd.is_empty() {
                let ph = &rd[0];
                let lc = py::last(ph).map(|x| x.to_string()).unwrap_or_default();
                let end = if !lc.is_empty() && SIBILANT_END.contains(lc.as_str()) {
                    "ᵻz"
                } else if !lc.is_empty() && VOICELESS_END.contains(lc.as_str()) {
                    "s"
                } else {
                    "z"
                };
                return out(vec![format!("{core}{PH}{ph}{end}")], 1, trail);
            }
        }
        // обычное слово, выкрикнутое заглавными, читается словом ("I CAN'T BELIEVE IT")
        if py::clen(core) >= 2
            && py::is_upper(core)
            && py::is_alpha(&core.replace(['\'', '’'], ""))
            && self.known(low)
            && !tt.acronym_words.contains(core)
            && !tt.say_as_cased.contains_key(core)
            && !tt.shorthand_cased.contains_key(core)
            && !tt.zones.contains_key(core)
            && !words::roman_match(core)
            && (self.common(low) || !self.is_cased(core))
            && self.shouted(toks, i)
            && self.reading(low).iter().any(|f| !self.spelled_form(f, low))
        {
            let w = if i > 0 && !py::last_in(&toks[i - 1], ".!?") { low.to_string() } else { py::capitalize(low) };
            return out(vec![w], 1, trail);
        }
        Ok(None)
    }

    pub(super) fn more2(&self, c: &Ctx) -> R<Out> {
        let tt = t();
        let (core, trail, prev, prev2, nxt, toks, i, pauses) = (c.core, c.trail, c.prev, c.prev2, c.nxt, c.toks, c.i, c.pauses);
        let (low, pl, nl) = (c.low.as_str(), c.pl.as_str(), c.nl.as_str());
        let tail1 = |tr: &str| if tr.starts_with('.') { py::from(tr, 1) } else { tr.to_string() };
        // сокращения переписки
        if let Some(v) = tt.shorthand_cased.get(core) {
            if pl != "the" {
                return out(py::split_sp(v), 1, trail);
            }
        }
        if let Some(v) = tt.shorthand.get(low) {
            if !py::is_upper(core) && pl != "the" && (low != "re" || trail.starts_with(':')) {
                return out(py::split_sp(v), 1, trail);
            }
        }
        match low {
            "w/" => return out(sv(&["with"]), 1, trail),
            "w/o" => return out(sv(&["without"]), 1, trail),
            "b/c" => return out(sv(&["because"]), 1, trail),
            _ => {}
        }
        if matches!(core, "→" | "->" | "=>") {
            return out(sv(&["to"]), 1, trail);
        }
        if core == "@" && !prev.is_empty() && !nxt.is_empty() {
            return out(sv(&["at"]), 1, trail);
        }
        if core == "St"
            && !trail.starts_with('.')
            && py::first_upper(prev)
            && py::is_alpha(prev)
            && (!trail.is_empty() || nxt.is_empty() || py::first_lower(nxt))
        {
            return out(sv(&["Street"]), 1, trail);
        }
        if is_m(re!(r"^[1-9]\d{2}[A-Z]?$"), core)
            && matches!(py::rstrip_set(pl, "."), "room" | "rm" | "suite" | "ste" | "gate" | "apt" | "apartment" | "unit" | "flight")
        {
            let d: Vec<char> = py::to(core, 3).chars().collect();
            let tail = self.spell(&py::from(core, 3))?;
            let mut w = vec![tt.one(py::int(&d[0].to_string())?)?];
            if d[1] == '0' && d[2] == '0' {
                w.push("hundred".into());
            } else if d[1] == '0' {
                w.extend(["oh".into(), tt.one(py::int(&d[2].to_string())?)?]);
            } else {
                w.push(d[1..].iter().collect());
            }
            w.extend(tail);
            return out(w, 1, trail);
        }
        if let Some(m) = caps(re!(r"^([A-Z]{2,4})(?:→|->)([A-Z]{2,4})$"), core) {
            let mut w = self.spell(gs(&m, 1))?;
            w.push("to".into());
            w.extend(self.spell(gs(&m, 2))?);
            return out(w, 1, trail);
        }
        // одинокое тире между днями, датами или числами
        if matches!(core, "–" | "-" | "—") && !prev.is_empty() && !nxt.is_empty() {
            if (tt.day_abbr.contains_key(py::rstrip_set(pl, ".")) || tt.day_names.contains(prev))
                && (tt.day_abbr.contains_key(py::rstrip_set(nl, ".")) || tt.day_names.contains(nxt))
            {
                return out(sv(&["through"]), 1, trail);
            }
            if (py::first_digit(prev) || tt.month_abbr.contains_key(py::rstrip_set(pl, ".")))
                && (py::first_digit(nxt) || tt.month_abbr.contains_key(py::rstrip_set(nl, ".")))
            {
                return out(sv(&["to"]), 1, trail);
            }
        }
        // дни недели коротко без точки перед датой, временем или диапазоном
        if let Some(day) = tt.day_abbr.get(low) {
            if py::first_upper(core)
                && py::is_alpha(core)
                && (py::first_digit(nxt)
                    || tt.day_abbr.contains_key(nl)
                    || matches!(nxt, "–" | "-")
                    || (trail == "," && i + 1 < toks.len()))
            {
                return out(vec![day.clone()], 1, trail);
            }
        }
        if let Some(m) = caps(re!(r"^([A-Z][a-z]{2})[–-]([A-Z][a-z]{2})$"), core) {
            let (a, b) = (py::lower(gs(&m, 1)), py::lower(gs(&m, 2)));
            if let (Some(da), Some(db)) = (tt.day_abbr.get(&a), tt.day_abbr.get(&b)) {
                return out(vec![da.clone(), "through".into(), db.clone()], 1, trail);
            }
        }
        // дата после дня недели или "on": "Tue 11/4" -> November fourth
        if let Some(m) = caps(re!(r"^(\d{1,2})/(\d{1,2})$"), core) {
            let (a, b) = (py::int(gs(&m, 1))?, py::int(gs(&m, 2))?);
            if (1..=12).contains(&a)
                && (1..=31).contains(&b)
                && (tt.day_names.contains(prev)
                    || tt.day_abbr.contains_key(py::rstrip_set(pl, ".,"))
                    || matches!(pl, "on" | "by" | "due" | "until" | "till" | "from" | "before" | "after" | "dated" | "thru"))
            {
                return out(vec![tt.months[(a - 1) as usize].clone(), ordinal(b)?], 1, trail);
            }
        }
        // месяц после порядкового: "1st Feb" -> first of February
        if let Some(&mo) = tt.month_abbr.get(py::rstrip_set(low, ".")) {
            if py::first_upper(core) && is_m(re!(r"^\d{1,2}(st|nd|rd|th)$"), prev) {
                return out(vec!["of".into(), tt.months[mo - 1].clone()], 1, tail1(trail));
            }
        }
        // часовые пояса после времени
        if let Some(z) = tt.zones.get(core) {
            if is_m(super::rules3::TIME(), prev) || is_m(re!(r"^(?i:[ap]\.?m\.?)$"), prev) || matches!(prev, "AM" | "PM") {
                return out(py::split_sp(z), 1, trail);
            }
        }
        // "8:40a", "11:25p"
        if let Some(m) = caps(re!(r"^(\d{1,2}):(\d{2})([ap])$"), core) {
            return out(self.time(gs(&m, 1), gs(&m, 2), None, g(&m, 3))?.unwrap_or_default(), 1, trail);
        }
        // диапазоны часов: "10–11:30", "9–5"
        if let Some(m) = caps(re!(r"^(\d{1,2})(?::(\d{2}))?[–-](\d{1,2})(?::(\d{2}))?$"), core) {
            if (g(&m, 2).is_some() || g(&m, 4).is_some())
                && py::int(gs(&m, 1))? <= 24
                && py::int(gs(&m, 3))? <= 24
                && !py::first_upper(prev)
            {
                let a = self.time(gs(&m, 1), g(&m, 2).unwrap_or("00"), None, None)?.unwrap_or_else(|| vec![gs(&m, 1).to_string()]);
                let b = self.time(gs(&m, 3), g(&m, 4).unwrap_or("00"), None, None)?.unwrap_or_else(|| vec![gs(&m, 3).to_string()]);
                let mut w: Vec<String> = a.into_iter().filter(|x| x != "hundred").collect();
                w.push("to".into());
                w.extend(b.into_iter().filter(|x| x != "hundred"));
                return out(w, 1, trail);
            }
        }
        // слитно длительности и расстояния: "2h", "15m" (после часов), "400m"
        if let Some(m) = caps(re!(r"^(\d+(?:\.\d+)?)(h|hr|m|min|s)$"), core) {
            let (n, u) = (gs(&m, 1), gs(&m, 2));
            let one = n == "1";
            if u == "h" || u == "hr" {
                return out(vec![n.into(), if one { "hour" } else { "hours" }.into()], 1, trail);
            }
            if u == "min" || (u == "m" && is_m(re!(r"^\d+(h|hr)$"), prev)) {
                return out(vec![n.into(), if one { "minute" } else { "minutes" }.into()], 1, trail);
            }
            if u == "s" && is_m(re!(r"^\d+(m|min)$"), prev) {
                return out(vec![n.into(), if one { "second" } else { "seconds" }.into()], 1, trail);
            }
            if u == "m"
                && ((matches!(
                    pl,
                    "the" | "a" | "ran" | "run" | "runs" | "swam" | "swim" | "won" | "in" | "over" | "under" | "about" | "climbed" | "dove"
                        | "fell" | "jumped"
                ) && !matches!(nl, "for" | "in" | "dollars" | "people" | "users"))
                    || matches!(
                        nl,
                        "race" | "dash" | "sprint" | "freestyle" | "hurdles" | "relay" | "run" | "swim" | "high" | "deep" | "long" | "tall"
                            | "wide" | "away" | "above" | "below"
                    ))
            {
                return out(vec![n.into(), if one { "meter" } else { "meters" }.into()], 1, trail);
            }
        }
        // "$75+", "1,000+ users"
        if let Some(m) = caps(re!(r"^([$€£¥]?)(\d[\d,]*(?:\.\d+)?)\+$"), core) {
            if !gs(&m, 1).is_empty() {
                let mut w = self.sub(&format!("{}{}", gs(&m, 1), gs(&m, 2)), pauses);
                w.extend(sv(&["or", "more"]));
                return out(w, 1, trail);
            }
            return out(vec!["more".into(), "than".into(), gs(&m, 2).into()], 1, trail);
        }
        // цена после "a"/"an" — определение: "a $5 late fee"
        if let Some(m) = caps(super::rules3::MONEY(), core) {
            if g(&m, 3).is_some()
                && g(&m, 4).is_none()
                && g(&m, 5).is_none()
                && matches!(pl, "a" | "an")
                && py::first_lower(nxt)
                && !matches!(nl, "a" | "an" | "the" | "and" | "or" | "for" | "per" | "each" | "to")
            {
                let cur = tt.currency.get(gs(&m, 2)).ok_or(py::Bail)?;
                return out(vec![gs(&m, 3).into(), cur.0.clone()], 1, trail);
            }
        }
        // последние четыре цифры карты, короткий номер для SMS
        if is_m(re!(r"^\d{4}$"), core) && (pl == "ending" || (pl == "in" && prev2 == "ending")) {
            return out(digit_words(core)?, 1, trail);
        }
        if is_m(re!(r"^\d{5,6}$"), core) && pl == "to" && toks[i.saturating_sub(4)..i].iter().any(|x| py::lower(&core_of(x)) == "text") {
            return out(digit_words(core)?, 1, trail);
        }
        // почтовый индекс перед городом
        if is_m(re!(r"^\d{5}$"), core) && py::first_upper(nxt) && i > 0 && toks[i - 1].ends_with(',') {
            return out(digit_words(core)?, 1, trail);
        }
        // коды и пароли: "abc123", "B12", "Win10"
        if let Some(m) = caps(re!(r"^(?i:win)(\d{1,2})$"), core) {
            return out(vec!["Windows".into(), gs(&m, 1).into()], 1, trail);
        }
        if let Some(m) = caps(re!(r"^([a-z]{1,4})(\d{1,6})$"), core) {
            let a = gs(&m, 1);
            if a != "x" && a != "ext" && !self.known(low) && (py::clen(a) <= 3 || !self.known(a)) {
                let d = gs(&m, 2);
                let mut w = self.spell(a)?;
                if py::clen(d) <= 2 {
                    w.push(d.into());
                } else {
                    w.extend(digit_words(d)?);
                }
                return out(w, 1, trail);
            }
        }
        // стороны света после номера шоссе или перед номерной улицей
        if matches!(core, "N" | "S" | "E" | "W")
            && is_num(prev)
            && (matches!(prev2, "hwy" | "highway" | "route" | "rte" | "interstate" | "i" | "us" | "sr")
                || is_m(re!(r"^\d+(st|nd|rd|th)$"), nxt)
                || (py::first_upper(nxt) && py::clen(nxt) > 2))
        {
            return out(vec![tt.directions.get(low).ok_or(py::Bail)?.clone()], 1, trail);
        }
        if matches!(low, "rm" | "apt" | "bldg" | "fl") && py::first_upper(core) && (py::first_digit(nxt) || nxt.starts_with('#')) {
            let w = match low {
                "rm" => "Room",
                "apt" => "Apartment",
                "bldg" => "Building",
                _ => "Floor",
            };
            return out(sv(&[w]), 1, tail1(trail));
        }
        if low == "approx" && (py::first_digit(nxt) || nxt.is_empty() || py::first_in(nxt, "$~")) {
            return out(sv(&["approximately"]), 1, tail1(trail));
        }
        // единицы после дроби или диапазона: "1/2 tsp", "40–45 min"
        let frac_word = matches!(pl, "half" | "quarter" | "third" | "eighth" | "quarters" | "thirds" | "eighths");
        if let Some((sing, plur)) = tt.units.get(core) {
            if !matches!(core, "m" | "g" | "L" | "W" | "V" | "s")
                && (is_m(super::rules3::FRACTION(), prev) || is_m(re!(r"^\d+[–-]\d+$"), prev) || frac_word)
            {
                let one = is_m(super::rules3::FRACTION(), prev) || matches!(pl, "half" | "quarter" | "third" | "eighth");
                return out(py::split_sp(if one { sing } else { plur }), 1, trail);
            }
        }
        // одна буква после слова-класса: "Series A", "Plan B"
        if core == "A" && tt.letter_cue.contains(pl) && py::first_upper(prev) {
            return out(vec![self.letter("A")], 1, trail);
        }
        // римские цифры после имени правителя ("Henry V tonight"), "Rocky IV"
        if matches!(core, "V" | "X") && tt.regnal.contains(pl) && py::first_upper(prev) {
            return out(vec!["the".into(), ordinal(roman_value(core)?)?], 1, trail);
        }
        if matches!(core, "IV" | "VI" | "IX" | "XI") && py::first_upper(prev) && py::is_alpha(prev) && !py::is_upper(prev) {
            let v = roman_value(core)?;
            return out(if tt.regnal.contains(pl) { vec!["the".into(), ordinal(v)?] } else { vec![cardinal(v)?] }, 1, trail);
        }
        // повелительное "Read" в начале предложения
        if core == "Read" && (i == 0 || py::last_in(&toks[i - 1], ".!?:;")) && (tt.objects.contains(nl) || py::first_upper(nxt)) {
            return out(vec![format!("{core}{PH}ɹˈid")], 1, trail);
        }
        // диакритика, которой нет в лексиконе: "Kīlauea", "Hawaiʻi", "Ó"
        let plain = words::strip_okina(core);
        if plain != core
            && plain.is_ascii()
            && py::is_alpha(&plain)
            && !self.known(low)
            && !tt.say_as.contains_key(&py::lower(&plain))
            && (self.known(&py::lower(&plain)) || py::clen(&plain) == 1)
        {
            return out(vec![plain], 1, trail);
        }
        // три заглавные, которые лексикон читает словом, а никто бы не стал: "HOA"
        if py::clen(core) == 3
            && py::is_upper(core)
            && py::is_alpha(core)
            && core.is_ascii()
            && !self.is_cased(core)
            && !self.common(low)
            && !tt.acronym_words.contains(core)
            && !words::roman_match(core)
            && !self.spelled(low)
            && !(i > 0 && py::is_upper(&toks[i - 1]) && py::clen(&toks[i - 1]) > 3)
        {
            return out(self.spell(core)?, 1, trail);
        }
        // омографы
        if low == "lead"
            && matches!(pl, "for" | "of" | "contains" | "containing" | "with" | "no" | "free" | "test" | "testing" | "tested")
            && !matches!(nl, "the" | "a" | "an" | "us" | "them" | "you" | "him" | "her" | "me" | "to" | "our" | "your" | "their" | "his" | "my")
        {
            return out(vec![format!("{core}{PH}lˈɛd")], 1, trail);
        }
        if low == "does"
            && (tt.number_words.contains(pl)
                || py::is_digit(prev)
                || matches!(pl, "the" | "some" | "many" | "several" | "two" | "three" | "few" | "and"))
            && !matches!(nl, "not" | "it" | "he" | "she" | "this" | "that" | "anyone" | "anybody" | "everyone")
        {
            return out(vec![format!("{core}{PH}dˈOz")], 1, trail);
        }
        if low == "row"
            && matches!(nl, "over" | "about" | "with" | "between")
            && matches!(pl, "a" | "the" | "big" | "huge" | "blazing" | "public" | "family" | "another" | "furious")
        {
            return out(vec![format!("{core}{PH}ɹˈW")], 1, trail);
        }
        if low == "minute" && matches!(pl, "a" | "the") && tt.minute_nouns.contains(nl) {
            return out(vec![format!("{core}{PH}{}", tt.adj_forms["minute"])], 1, trail);
        }
        Ok(None)
    }
}
