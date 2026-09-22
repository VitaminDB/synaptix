//! Грамматика событий при генерации: какие токены допустимы на следующем шаге.
//!
//! Повторяет `PromptGrammarState` релиза: после сдвига по сетке поля события
//! идут строго по порядку схемы (время → ритм → секция → тональность → аккорд
//! → ноты), за размером обязана следовать позиция восьмой, за высотой ноты —
//! длительность или следующая высота; подряд не больше четырёх сдвигов, а конец
//! последовательности возможен только после непустого события.

use crate::vocab::{TokenType, Vocab, EOS};
use crate::SheetError;

const TIMESTAMP: i32 = 0;
const RHYTHM: i32 = 1;
const STRUCTURE: i32 = 2;
const KEY: i32 = 3;
const CHORD: i32 = 4;
const MELODY: i32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Incomplete {
    RhythmAfterMeter,
    MelodyAfterPitch,
}

#[derive(Debug, Clone)]
pub struct GrammarState {
    in_shift: bool,
    shift_run: u32,
    payload_count: u32,
    last_field: i32,
    incomplete: Option<Incomplete>,
}

impl Default for GrammarState {
    fn default() -> Self {
        Self { in_shift: true, shift_run: 0, payload_count: 0, last_field: -1, incomplete: None }
    }
}

impl GrammarState {
    fn allow_range(mask: &mut [bool], start: u32, end: u32) {
        for m in &mut mask[start as usize..end as usize] {
            *m = true;
        }
    }

    /// Маска разрешённых токенов (длина — размер словаря).
    pub fn allowed(&self, vocab: &Vocab, mask: &mut Vec<bool>) {
        mask.clear();
        mask.resize(vocab.n_tokens as usize, false);
        if self.payload_count > 0 {
            mask[EOS as usize] = true;
        }
        if (self.payload_count > 0 || self.in_shift) && self.shift_run < 4 {
            Self::allow_range(mask, vocab.subbeat_shift_start, vocab.subbeat_shift_end);
        }
        match self.incomplete {
            Some(Incomplete::RhythmAfterMeter) => {
                Self::allow_range(mask, vocab.eighth_start, vocab.eighth_end);
                return;
            }
            Some(Incomplete::MelodyAfterPitch) => {
                Self::allow_range(mask, vocab.duration_start, vocab.duration_end);
                Self::allow_range(mask, vocab.pitch_start, vocab.pitch_end);
                return;
            }
            None => {}
        }
        if self.last_field < TIMESTAMP {
            Self::allow_range(mask, vocab.time_start, vocab.time_end);
        }
        if self.last_field < RHYTHM {
            Self::allow_range(mask, vocab.meter_start, vocab.meter_end);
            Self::allow_range(mask, vocab.eighth_start, vocab.eighth_end);
        }
        if self.last_field < STRUCTURE {
            Self::allow_range(mask, vocab.structure_start, vocab.structure_end);
        }
        if self.last_field < KEY {
            Self::allow_range(mask, vocab.key_start, vocab.key_end);
        }
        if self.last_field < CHORD {
            Self::allow_range(mask, vocab.full_chord_start, vocab.full_chord_end);
        }
        if self.last_field <= MELODY {
            Self::allow_range(mask, vocab.pitch_start, vocab.pitch_end);
        }
    }

    /// Учесть выданный токен. `true` — последовательность закончена.
    pub fn update(&mut self, vocab: &Vocab, token: u32) -> Result<bool, SheetError> {
        if token == EOS {
            return Ok(true);
        }
        let kind = vocab.token_type(token)?;
        if kind == TokenType::SubbeatShift {
            if !self.in_shift && self.payload_count > 0 {
                self.payload_count = 0;
                self.last_field = -1;
                self.incomplete = None;
            }
            self.in_shift = true;
            self.shift_run += 1;
            return Ok(false);
        }
        self.in_shift = false;
        self.shift_run = 0;
        self.payload_count += 1;
        let (field, incomplete) = match kind {
            TokenType::Time => (TIMESTAMP, None),
            TokenType::Meter => (RHYTHM, Some(Incomplete::RhythmAfterMeter)),
            TokenType::EighthPosition => (RHYTHM, None),
            TokenType::Structure => (STRUCTURE, None),
            TokenType::Key => (KEY, None),
            TokenType::ChordFull => (CHORD, None),
            TokenType::Pitch => (MELODY, Some(Incomplete::MelodyAfterPitch)),
            TokenType::Duration => (MELODY, None),
            other => {
                return Err(SheetError::Sequence(format!(
                    "неожиданный тип токена в событии: {}",
                    other.name()
                )))
            }
        };
        self.last_field = field;
        self.incomplete = incomplete;
        Ok(false)
    }
}
