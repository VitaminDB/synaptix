//! Разбор текстовых форм Kaldi (`--binary=false`): поток токенов.

use crate::{MfaError, Result};

pub struct Tokens<'a> {
    toks: Vec<&'a str>,
    pos: usize,
}

impl<'a> Tokens<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { toks: src.split_whitespace().collect(), pos: 0 }
    }

    pub fn peek(&self) -> Option<&'a str> {
        self.toks.get(self.pos).copied()
    }

    pub fn next(&mut self) -> Result<&'a str> {
        let t = self.toks.get(self.pos).copied().ok_or_else(|| MfaError::Format("неожиданный конец".into()))?;
        self.pos += 1;
        Ok(t)
    }

    pub fn expect(&mut self, tok: &str) -> Result<()> {
        let t = self.next()?;
        if t != tok {
            return Err(MfaError::Format(format!("ожидалось `{tok}`, получено `{t}`")));
        }
        Ok(())
    }

    pub fn int(&mut self) -> Result<i64> {
        let t = self.next()?;
        t.parse().map_err(|_| MfaError::Format(format!("не число: `{t}`")))
    }

    pub fn float(&mut self) -> Result<f32> {
        let t = self.next()?;
        t.parse().map_err(|_| MfaError::Format(format!("не число: `{t}`")))
    }

    /// `[ a b c ]` (вектор или матрица построчно — все числа подряд).
    pub fn bracketed(&mut self) -> Result<Vec<f32>> {
        self.expect("[")?;
        let mut v = Vec::new();
        loop {
            let t = self.next()?;
            if t == "]" {
                return Ok(v);
            }
            v.push(t.parse().map_err(|_| MfaError::Format(format!("не число: `{t}`")))?);
        }
    }
}

/// Матрица `copy-matrix --binary=false`: строки по переводам строк.
pub fn matrix(src: &str) -> Result<(usize, usize, Vec<f32>)> {
    let mut rows = 0;
    let mut data = Vec::new();
    for line in src.lines() {
        let nums: Vec<f32> = line
            .split_whitespace()
            .filter(|t| *t != "[" && *t != "]")
            .map(|t| t.parse::<f32>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| MfaError::Format(format!("матрица: {e}")))?;
        if !nums.is_empty() {
            rows += 1;
            data.extend(nums);
        }
    }
    let cols = if rows > 0 { data.len() / rows } else { 0 };
    if rows * cols != data.len() {
        return Err(MfaError::Format("матрица: неровные строки".into()));
    }
    Ok((rows, cols, data))
}
