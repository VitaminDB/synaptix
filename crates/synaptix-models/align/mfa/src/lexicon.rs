//! Словарь MFA: `слово  [p_произн  p_паузы_после  поправка  поправка]  фонемы`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pron {
    pub phones: Vec<String>,
    pub prob: f32,
    /// Вероятность паузы после слова (`None` — общая из модели).
    pub silence_after: Option<f32>,
    /// Поправки вероятности входа в слово после паузы / без неё.
    #[serde(default)]
    pub silence_before_correction: Option<f32>,
    #[serde(default)]
    pub non_silence_before_correction: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lexicon {
    pub words: HashMap<String, Vec<Pron>>,
}

impl Lexicon {
    pub fn from_text(src: &str) -> Self {
        let mut words: HashMap<String, Vec<Pron>> = HashMap::new();
        for line in src.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 2 {
                continue;
            }
            let word = cols[0].to_lowercase();
            let nums: Vec<f32> = cols[1..cols.len() - 1].iter().filter_map(|c| c.trim().parse().ok()).collect();
            let phones: Vec<String> = cols[cols.len() - 1].split_whitespace().map(String::from).collect();
            if phones.is_empty() {
                continue;
            }
            words.entry(word).or_default().push(Pron {
                phones,
                prob: nums.first().copied().unwrap_or(1.0),
                silence_after: nums.get(1).copied(),
                silence_before_correction: nums.get(2).copied(),
                non_silence_before_correction: nums.get(3).copied(),
            });
        }
        Self { words }
    }

    pub fn get(&self, word: &str) -> Option<&[Pron]> {
        self.words.get(&word.to_lowercase()).map(|v| v.as_slice())
    }
}
