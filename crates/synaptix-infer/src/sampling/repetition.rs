use std::collections::HashMap;

use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

/// Окно по хвосту контекста: `last_n == 0` — весь контекст.
fn window(ids: &[u32], last_n: usize) -> &[u32] {
    if last_n == 0 || last_n >= ids.len() {
        ids
    } else {
        &ids[ids.len() - last_n..]
    }
}

/// Штраф за повтор в llama.cpp-семантике: логит уже встреченного токена
/// делится (положительный) или умножается (отрицательный) на `penalty`.
///
/// `last_n` — размер окна по хвосту контекста, 0 = весь контекст. Окно
/// принципиально для агент-режима: на промпте в десятки тысяч токенов штраф
/// «по всему контексту» задевает половину словаря и работает как шум, а от
/// вырождения в цикл при этом не спасает.
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
    pub last_n: usize,
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, ctx: &ProcessorContext) -> Result<()> {
        if self.penalty == 1.0 {
            return Ok(());
        }
        for &token_id in window(&ctx.input_ids, self.last_n) {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
        Ok(())
    }
}

/// Presence/frequency-штрафы в OpenAI-семантике: логит сдвигается вниз на
/// `presence` за сам факт появления токена и ещё на `frequency × count` за
/// каждое повторение. В отличие от `repeat_penalty` это аддитивный сдвиг —
/// он не зависит от знака логита и предсказуемо усиливается с числом
/// повторов, поэтому именно он гасит зацикливание агента.
///
/// `last_n` — то же окно, что и у [`RepetitionPenaltyProcessor`].
pub struct PresenceFrequencyProcessor {
    pub presence: f32,
    pub frequency: f32,
    pub last_n: usize,
}

impl LogitProcessor for PresenceFrequencyProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, ctx: &ProcessorContext) -> Result<()> {
        if self.presence == 0.0 && self.frequency == 0.0 {
            return Ok(());
        }
        let ids = window(&ctx.input_ids, self.last_n);
        let mut counts: HashMap<u32, f32> = HashMap::with_capacity(ids.len().min(1024));
        for &t in ids {
            *counts.entry(t).or_insert(0.0) += 1.0;
        }
        for (t, c) in counts {
            let idx = t as usize;
            if idx < logits.len() {
                logits[idx] -= self.presence + self.frequency * c;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(ids: &[u32]) -> ProcessorContext {
        ProcessorContext { input_ids: ids.to_vec(), step: 0, batch_idx: 0 }
    }

    #[test]
    fn repetition_window_ignores_head() {
        // Токен 0 встречался только в голове контекста — при окне 2 он
        // штрафоваться не должен, токен 3 (в окне) — должен.
        let mut p = RepetitionPenaltyProcessor { penalty: 2.0, last_n: 2 };
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        p.process(&mut logits, &ctx(&[0, 1, 2, 3])).unwrap();
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[3], 0.5);
        assert_eq!(logits[2], 0.5);
    }

    #[test]
    fn repetition_zero_window_is_whole_context() {
        let mut p = RepetitionPenaltyProcessor { penalty: 2.0, last_n: 0 };
        let mut logits = vec![1.0, 1.0];
        p.process(&mut logits, &ctx(&[0])).unwrap();
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[1], 1.0);
    }

    #[test]
    fn repetition_negative_logit_scales_down() {
        let mut p = RepetitionPenaltyProcessor { penalty: 2.0, last_n: 0 };
        let mut logits = vec![-1.0];
        p.process(&mut logits, &ctx(&[0])).unwrap();
        assert_eq!(logits[0], -2.0);
    }

    #[test]
    fn presence_frequency_scales_with_count() {
        let mut p = PresenceFrequencyProcessor { presence: 0.5, frequency: 0.25, last_n: 0 };
        let mut logits = vec![0.0, 0.0];
        // Токен 0 встречается трижды, токен 1 — один раз.
        p.process(&mut logits, &ctx(&[0, 0, 0, 1])).unwrap();
        assert!((logits[0] - -(0.5 + 0.75)).abs() < 1e-6);
        assert!((logits[1] - -(0.5 + 0.25)).abs() < 1e-6);
    }

    #[test]
    fn presence_frequency_noop_when_zero() {
        let mut p = PresenceFrequencyProcessor { presence: 0.0, frequency: 0.0, last_n: 0 };
        let mut logits = vec![1.0];
        p.process(&mut logits, &ctx(&[0, 0])).unwrap();
        assert_eq!(logits[0], 1.0);
    }
}
