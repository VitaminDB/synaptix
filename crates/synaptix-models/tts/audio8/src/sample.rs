//! Сэмплинг как в upstream (`ArkttsModel._processed_scores` / `_sample` / RAS).
//!
//! Порядок: фильтр (семантика + EOS) → top-k/top-p «legacy» по сырым логитам
//! (сортировка, cumsum softmax, убрать cum > p или ранг ≥ k, первый всегда остаётся)
//! → деление на температуру → softmax → argmax(p / −ln U). U — из счётчикового
//! ГПСЧ: u_i зависит только от зерна и номера, каждый вызов съедает ширину
//! словаря шага (как `torch.rand(V)` upstream), поэтому шум можно брать только
//! для допустимых индексов и всё равно совпасть с эталоном `reference/audio8_ref.py`.

const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct CounterRng {
    seed: u64,
    n: u64,
}

impl CounterRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, n: 0 }
    }

    /// u ∈ (0, 1) для номера `n + i` (без сдвига счётчика).
    #[inline]
    pub fn at(&self, i: u64) -> f32 {
        let z = mix(self.seed.wrapping_add((self.n + i + 1).wrapping_mul(GOLD)));
        // 23 бита: (k + 0.5)/2^23 точно представимо в f32 и строго внутри (0, 1)
        // (с 24 битами верхнее значение округлялось до 1.0 → шум 0 → NaN).
        (((z >> 41) as f64 + 0.5) / (1u64 << 23) as f64) as f32
    }

    pub fn advance(&mut self, n: u64) {
        self.n += n;
    }

    pub fn drawn(&self) -> u64 {
        self.n
    }
}

/// Фильтр top-k/top-p upstream: на месте выставляет −∞ отброшенным.
/// `scores` — допустимые кандидаты (−∞ уже отброшены фильтром выше).
pub fn legacy_top_k_top_p(scores: &mut [f32], top_k: usize, top_p: f32) {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // torch.sort(descending) — стабильная по убыванию для равных? Равенство на
    // практике только у −∞, их порядок не влияет.
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let max = scores[order[0]];
    if !max.is_finite() {
        return;
    }
    let mut denom = 0f64;
    for &i in &order {
        denom += ((scores[i] - max) as f64).exp();
    }
    let mut cum = 0f64;
    let mut remove = vec![false; scores.len()];
    for (rank, &i) in order.iter().enumerate() {
        cum += ((scores[i] - max) as f64).exp() / denom;
        // Как torch: cumsum в f32.
        let c = cum as f32;
        if rank > 0 && (c > top_p || rank >= top_k) {
            remove[i] = true;
        }
    }
    for (s, r) in scores.iter_mut().zip(remove) {
        if r {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Обработка логитов шага: top-k/top-p, затем деление на температуру.
pub fn processed(logits: &[f32], top_k: usize, top_p: f32, temperature: f32) -> Vec<f32> {
    let mut s = logits.to_vec();
    legacy_top_k_top_p(&mut s, top_k, top_p);
    let t = temperature.max(1e-5);
    for v in &mut s {
        *v /= t;
    }
    s
}

pub fn argmax(scores: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in scores.iter().enumerate() {
        if v > scores[best] {
            best = i;
        }
    }
    best
}

/// `argmax(softmax(s) / −ln U)`; `index[i]` — номер i-го кандидата в полном
/// словаре шага шириной `vocab` (для шума). Съедает `vocab` чисел ГПСЧ.
pub fn sample(scores: &[f32], index: &dyn Fn(usize) -> u64, vocab: u64, rng: &mut CounterRng) -> usize {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        let noise = -rng.at(index(i)).ln();
        let v = p / noise;
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    rng.advance(vocab);
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_keeps_first_and_k() {
        let mut s = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        legacy_top_k_top_p(&mut s, 2, 1.0);
        assert!(s[1].is_finite() && s[3].is_finite());
        assert!(s[0].is_infinite() && s[2].is_infinite() && s[4].is_infinite());
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        let mut s = vec![10.0, 0.0, 0.0];
        legacy_top_k_top_p(&mut s, 50, 0.1);
        assert!(s[0].is_finite() && s[1].is_infinite());
    }

    #[test]
    fn rng_is_counter_based() {
        let mut a = CounterRng::new(7);
        let x = a.at(5);
        a.advance(3);
        assert_eq!(a.at(2), x);
        assert!(x > 0.0 && x < 1.0);
    }
}
