use crate::structured::outlines::OutlinesConstraint;

/// Грамматика на основе regex (байтовый словарь, токен == байт). `start` —
/// regex-паттерн; если добавлены `rules`, эффективный паттерн = их альтернация
/// `(r0)|(r1)|...`. Допустимые токены вычисляются через [`OutlinesConstraint`]
/// (Thompson-NFA). Для более выразительных CFG нужен отдельный движок.
pub struct Grammar {
    pub rules: Vec<String>,
    pub start: String,
}

impl Grammar {
    /// `start` трактуется как regex-паттерн.
    pub fn new(start: impl Into<String>) -> Self {
        Self { rules: Vec::new(), start: start.into() }
    }

    /// Явный конструктор regex-грамматики (синоним `new`).
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self::new(pattern)
    }

    pub fn add_rule(&mut self, rule: impl Into<String>) -> &mut Self {
        self.rules.push(rule.into());
        self
    }

    fn effective_pattern(&self) -> String {
        if self.rules.is_empty() {
            self.start.clone()
        } else {
            self.rules.iter().map(|r| format!("({r})")).collect::<Vec<_>>().join("|")
        }
    }

    /// Допустимые токены (байты 0..vocab_size) после уже принятой байт-строки
    /// `state`. Компилирует regex и шагает NFA по `state`.
    pub fn allowed_tokens(&self, state: &[u32], vocab_size: usize) -> Vec<u32> {
        let con = OutlinesConstraint::new(self.effective_pattern());
        let consumed: Vec<u8> = state.iter().filter_map(|&t| u8::try_from(t).ok()).collect();
        con.allowed_bytes(&consumed)
            .into_iter()
            .map(|b| b as u32)
            .filter(|&t| (t as usize) < vocab_size)
            .collect()
    }
}

pub struct LinearGrammar {
    pub steps: Vec<Vec<u32>>,
    pub state: usize,
}

impl LinearGrammar {
    pub fn new(steps: Vec<Vec<u32>>) -> Self {
        Self { steps, state: 0 }
    }

    pub fn is_finished(&self) -> bool {
        self.state >= self.steps.len()
    }

    pub fn allowed_tokens(&self) -> Vec<u32> {
        if self.is_finished() {
            return Vec::new();
        }
        self.steps[self.state].clone()
    }

    pub fn advance(&mut self, token: u32) -> bool {
        if self.is_finished() {
            return false;
        }
        if !self.steps[self.state].contains(&token) {
            return false;
        }
        self.state += 1;
        true
    }

    pub fn reset(&mut self) {
        self.state = 0;
    }
}
