use synaptix_ops::rng::Philox4x32;
use crate::error::Result;
use super::Sampler;

pub struct MultinomialSampler;

impl Sampler for MultinomialSampler {
    fn sample(&mut self, logits: &[f32], rng: &mut Philox4x32) -> Result<u32> {
        // После top-k почти весь словарь — −inf: exp и накопление — только по конечным.
        let finite: Vec<(u32, f32)> = logits.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, &v)| (i as u32, v)).collect();
        if finite.is_empty() {
            return Err(crate::error::InferError::Sampling("все вероятности нулевые или NaN".into()));
        }
        let max = finite.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = finite.iter().map(|p| ((p.1 - max) as f64).exp()).collect();
        // Накопление в f64 и `u` из [0, 1): у словаря в 150–250k сумма в f32
        // не добирала до 1, `u` мог быть ровно 1.0 — и срабатывал запасной
        // выход «последний токен словаря» (замаскированный/служебный).
        let u = next_unit(rng) * exps.iter().sum::<f64>();
        let mut cumsum = 0.0f64;
        for (p, &e) in finite.iter().zip(&exps) {
            cumsum += e;
            if u < cumsum {
                return Ok(p.0);
            }
        }
        // Погрешность на последнем шаге — последний токен с p > 0.
        finite
            .iter()
            .zip(&exps)
            .rev()
            .find(|(_, &e)| e > 0.0)
            .map(|(p, _)| p.0)
            .ok_or_else(|| crate::error::InferError::Sampling("все вероятности нулевые или NaN".into()))
    }
}

/// Равномерное в [0, 1): делитель 2^32, а не `u32::MAX` (тот давал ровно 1.0).
fn next_unit(rng: &mut Philox4x32) -> f64 {
    rng.next_u32() as f64 / 4_294_967_296.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Маскированный хвост (-inf) не выпадает никогда, даже на краю `u`.
    #[test]
    fn never_picks_masked_tail() {
        let mut rng = Philox4x32::new(7);
        let mut s = MultinomialSampler;
        let mut logits = vec![0.0f32; 1000];
        logits[999] = f32::NEG_INFINITY;
        for _ in 0..20_000 {
            assert_ne!(s.sample(&logits, &mut rng).unwrap(), 999);
        }
    }
}
