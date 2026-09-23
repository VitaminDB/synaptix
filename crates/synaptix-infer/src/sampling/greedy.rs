use synaptix_ops::rng::Philox4x32;
use crate::error::{InferError, Result};
use super::Sampler;

pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32], _rng: &mut Philox4x32) -> Result<u32> {
        // NaN (inf→NaN после MXFP8→F16) не участвует: `partial_cmp` с
        // `Equal` для NaN выбирал случайный токен без всякой ошибки.
        logits.iter().enumerate()
            .filter(|(_, x)| !x.is_nan())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| InferError::Sampling("все логиты NaN или пусты".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_skips_nan() {
        let mut rng = Philox4x32::new(0);
        let mut s = GreedySampler;
        assert_eq!(s.sample(&[f32::NAN, 1.0, 3.0, 2.0], &mut rng).unwrap(), 2);
        assert!(s.sample(&[f32::NAN, f32::NAN], &mut rng).is_err());
    }
}
