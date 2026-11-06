//! Таблицы нормализатора и строкового фронтенда (`tables.json` — выгрузка констант `leokani_g2p.py`:
//! сокращения, единицы, штаты, омографы и т. п.; порядок правил омографов сохранён).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;

type Set = HashSet<String>;
type Map = HashMap<String, String>;

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub struct Tables {
    pub ones: Vec<String>,
    pub tens: Vec<String>,
    pub ordinal_words: Map,
    pub months: Vec<String>,
    pub month_abbr: HashMap<String, usize>,
    pub day_abbr: Map,
    pub title_abbr: Map,
    pub street_abbr: Map,
    pub word_abbr: Map,
    pub units: HashMap<String, (String, String)>,
    pub currency: HashMap<String, (String, String, String, String)>,
    pub scales: Map,
    pub scale_words: Set,
    pub attributive: Set,
    pub per_unit: Map,
    pub version_cue: Set,
    pub acronym_words: Set,
    pub shorthand: Map,
    pub shorthand_cased: Map,
    pub zones: Map,
    pub letter_cue: Set,
    pub number_words: Set,
    pub minute_nouns: Set,
    pub directions: Map,
    pub street_words: Set,
    pub states: Map,
    pub tlds: Set,
    pub tld_say: Map,
    pub sep_words: Map,
    pub regnal: Set,
    pub roman_cue: Set,
    pub roman_ambiguous: Set,
    pub say_as: Map,
    pub say_as_cased: Map,
    pub verb_before: Set,
    pub adj_forms: Map,
    pub verb_forms: Map,
    pub subjects: Set,
    pub objects: Set,
    pub cased_say: Map,
    pub code_cue: Set,
    pub onsets: Set,
    pub codas: Set,
    pub sentence_start: Set,
    pub hg_classes: Map,
    pub homographs: HashMap<String, Vec<(String, Vec<String>)>>,
    pub titles: Set,
    pub abbrev: Set,
    pub read_past_before: Set,
    pub read_past_after: Set,
    pub lead_metal: Set,
    pub wind_verb_before: Set,
    pub wind_object: Set,
    pub lives_subject: Set,
    pub lives_owner: Set,
    pub year_before: Set,
    pub names: Map,
    pub pronoun: Map,
    pub pronoun_after: Set,
    pub dropped: Set,
    pub vulgar: Map,
    pub vulgar_words: Map,
    /// HG_CLASSES, разобранные в множества.
    #[serde(skip)]
    pub hg_sets: HashMap<String, Set>,
    /// DAY_ABBR.values() / STREET_ABBR.values().
    #[serde(skip)]
    pub day_names: Set,
    #[serde(skip)]
    pub street_names: Set,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

/// Таблицы (разбираются один раз).
pub fn t() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut t: Tables =
            serde_json::from_str(include_str!("tables.json")).expect("leokani g2p: tables.json");
        t.hg_sets = t
            .hg_classes
            .iter()
            .map(|(k, v)| (k.clone(), v.split_whitespace().map(str::to_string).collect()))
            .collect();
        t.day_names = t.day_abbr.values().cloned().collect();
        t.street_names = t.street_abbr.values().cloned().collect();
        t
    })
}

impl Tables {
    /// `ONES[n]` (IndexError вне 0…19).
    pub fn one(&self, n: i128) -> super::py::R<String> {
        if (0..20).contains(&n) {
            Ok(self.ones[n as usize].clone())
        } else {
            Err(super::py::Bail)
        }
    }
}
