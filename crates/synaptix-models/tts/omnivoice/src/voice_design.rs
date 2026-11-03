//! Описание голоса (`instruct`) и язык — как в upstream `_resolve_instruct`,
//! `_resolve_language` (`omnivoice/models/omnivoice.py`) и
//! `omnivoice/utils/voice_design.py`, `utils/lang_map.py` (k2-fsa, Apache-2.0).
//!
//! Модель обучена на описаниях строго из набора признаков в нижнем регистре
//! через «, » (китайские — через «，»): «Male, Low Pitch» или произвольный
//! текст для неё — вне распределения обучения, голос выходит случайным, а
//! описание порой читается вслух. Upstream приводит регистр и разделители и
//! отвергает неизвестные признаки с подсказкой; здесь — то же.

use crate::{OmniVoiceError, Result};

/// Категории признаков: (английский, китайский). Внутри категории — не
/// больше одного признака.
const GENDER: &[(&str, &str)] = &[("male", "男"), ("female", "女")];
const AGE: &[(&str, &str)] = &[
    ("child", "儿童"),
    ("teenager", "少年"),
    ("young adult", "青年"),
    ("middle-aged", "中年"),
    ("elderly", "老年"),
];
const PITCH: &[(&str, &str)] = &[
    ("very low pitch", "极低音调"),
    ("low pitch", "低音调"),
    ("moderate pitch", "中音调"),
    ("high pitch", "高音调"),
    ("very high pitch", "极高音调"),
];
const STYLE: &[(&str, &str)] = &[("whisper", "耳语")];
/// Акценты — только для английской речи.
const ACCENTS: &[&str] = &[
    "american accent",
    "british accent",
    "australian accent",
    "chinese accent",
    "canadian accent",
    "indian accent",
    "korean accent",
    "portuguese accent",
    "russian accent",
    "japanese accent",
];
/// Диалекты — только для китайской речи.
const DIALECTS: &[&str] = &[
    "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
];

const PAIRED: &[&[(&str, &str)]] = &[GENDER, AGE, PITCH, STYLE];

fn has_cjk(s: &str) -> bool {
    s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

fn all_valid() -> impl Iterator<Item = &'static str> {
    PAIRED.iter().flat_map(|c| c.iter().flat_map(|(e, z)| [*e, *z])).chain(ACCENTS.iter().copied()).chain(DIALECTS.iter().copied())
}

/// Номер категории признака (для проверки «одно на категорию»).
fn category(item: &str) -> usize {
    for (i, c) in PAIRED.iter().enumerate() {
        if c.iter().any(|(e, z)| *e == item || *z == item) {
            return i;
        }
    }
    if ACCENTS.contains(&item) {
        PAIRED.len()
    } else {
        PAIRED.len() + 1
    }
}

/// Похожесть строк как `difflib.SequenceMatcher.ratio` (приближённо, через
/// наибольшую общую подпоследовательность): 2·M / (|a| + |b|).
fn ratio(a: &str, b: &str) -> f32 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            dp[i + 1][j + 1] = if a[i] == b[j] { dp[i][j] + 1 } else { dp[i][j + 1].max(dp[i + 1][j]) };
        }
    }
    2.0 * dp[a.len()][b.len()] as f32 / (a.len() + b.len()).max(1) as f32
}

/// Привести описание голоса к виду, на котором обучена модель (порт
/// `_resolve_instruct`): регистр, разделители «, » / «，», язык признаков
/// (`use_zh` — в тексте есть китайские иероглифы; диалект делает описание
/// китайским, акцент — английским). Неизвестный признак, акцент вместе с
/// диалектом или два признака одной категории — ошибка с подсказкой.
/// Пустое описание — `None`.
pub fn resolve_instruct(instruct: Option<&str>, use_zh: bool) -> Result<Option<String>> {
    let Some(s) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let mut items = Vec::new();
    let mut unknown = Vec::new();
    for raw in s.split([',', '，']).map(str::trim).filter(|x| !x.is_empty()) {
        let n = raw.to_lowercase();
        if let Some(v) = all_valid().find(|v| *v == n) {
            items.push(v);
        } else {
            let best = all_valid().map(|v| (v, ratio(&n, v))).max_by(|a, b| a.1.total_cmp(&b.1)).filter(|(_, r)| *r >= 0.6);
            unknown.push(match best {
                Some((v, _)) => format!("«{raw}» (может быть, «{v}»?)"),
                None => format!("«{raw}»"),
            });
        }
    }
    if !unknown.is_empty() {
        return Err(OmniVoiceError::Other(format!(
            "описание голоса OmniVoice: неизвестные признаки {}. Допустимы (через запятую, по одному из группы): \
             male/female; child, teenager, young adult, middle-aged, elderly; very low pitch, low pitch, \
             moderate pitch, high pitch, very high pitch; whisper; {}; китайские диалекты {}",
            unknown.join(", "),
            ACCENTS.join(", "),
            DIALECTS.join("，")
        )));
    }
    let has_dialect = items.iter().any(|n| n.ends_with('话'));
    let has_accent = items.iter().any(|n| n.contains(" accent"));
    if has_dialect && has_accent {
        return Err(OmniVoiceError::Other("описание голоса OmniVoice: акцент (английская речь) и диалект (китайская) вместе нельзя".into()));
    }
    let use_zh = if has_dialect { true } else if has_accent { false } else { use_zh };
    let items: Vec<&str> = items
        .into_iter()
        .map(|n| {
            PAIRED
                .iter()
                .flat_map(|c| c.iter())
                .find(|(e, z)| *e == n || *z == n)
                .map(|(e, z)| if use_zh { *z } else { *e })
                .unwrap_or(n)
        })
        .collect();
    for i in 0..items.len() {
        if let Some(j) = (i + 1..items.len()).find(|&j| category(items[j]) == category(items[i])) {
            return Err(OmniVoiceError::Other(format!("описание голоса OmniVoice: «{}» и «{}» — из одной группы, нужен один", items[i], items[j])));
        }
    }
    let sep = if items.iter().any(|n| has_cjk(n)) { "，" } else { ", " };
    Ok(Some(items.join(sep)))
}

/// Описание для текста `text`: признаки — по-китайски, если в тексте есть
/// иероглифы (как `generate` upstream).
pub fn resolve_instruct_for(instruct: Option<&str>, text: &str) -> Result<Option<String>> {
    resolve_instruct(instruct, has_cjk(text))
}

/// Таблица языков upstream (`docs/lang_id_name_map.tsv`): код → имя в нижнем регистре.
const LANG_MAP: &str = include_str!("lang_map.tsv");

/// Язык как `_resolve_language`: `None`/«none» — без языка; код из таблицы
/// (`en`, `ru`, `kk`, …) — как есть; имя языка («English», «russian») — его
/// код; неизвестное — без языка (upstream так же, с предупреждением).
/// Дополнительно: «auto» и пустая строка — без языка, код без учёта регистра.
pub fn resolve_language(lang: Option<&str>) -> Option<&'static str> {
    let l = lang.map(str::trim).filter(|l| !l.is_empty())?;
    let low = l.to_lowercase();
    if low == "none" || low == "auto" {
        return None;
    }
    let rows = || LANG_MAP.lines().filter_map(|r| r.split_once('\t'));
    rows().find(|(id, _)| *id == l).or_else(|| rows().find(|(_, name)| *name == low)).or_else(|| rows().find(|(id, _)| *id == low)).map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruct_like_upstream() {
        assert_eq!(resolve_instruct(Some("Male, Middle-aged,Low Pitch , British Accent"), false).unwrap().as_deref(), Some("male, middle-aged, low pitch, british accent"));
        assert_eq!(resolve_instruct(Some("female，young adult"), false).unwrap().as_deref(), Some("female, young adult"));
        // текст китайский и нет акцента — признаки по-китайски
        assert_eq!(resolve_instruct(Some("female, young adult"), true).unwrap().as_deref(), Some("女，青年"));
        // диалект делает описание китайским
        assert_eq!(resolve_instruct(Some("female, young adult, 四川话"), false).unwrap().as_deref(), Some("女，青年，四川话"));
        // акцент — английским даже для китайского текста
        assert_eq!(resolve_instruct(Some("男, british accent"), true).unwrap().as_deref(), Some("male, british accent"));
        assert_eq!(resolve_instruct(Some("  "), false).unwrap(), None);
        assert!(resolve_instruct(Some("male, female"), false).is_err());
        assert!(resolve_instruct(Some("deep male voice"), false).is_err());
        let e = resolve_instruct(Some("mal"), false).unwrap_err().to_string();
        assert!(e.contains("«male»"), "{e}");
    }

    #[test]
    fn language_like_upstream() {
        assert_eq!(resolve_language(Some("en")), Some("en"));
        assert_eq!(resolve_language(Some("English")), Some("en"));
        assert_eq!(resolve_language(Some("russian")), Some("ru"));
        assert_eq!(resolve_language(Some("kk")), Some("kk"));
        assert_eq!(resolve_language(Some("RU")), Some("ru"));
        assert_eq!(resolve_language(Some("None")), None);
        assert_eq!(resolve_language(Some("auto")), None);
        assert_eq!(resolve_language(Some("klingon")), None);
        assert_eq!(resolve_language(None), None);
    }
}
