//! Сэмплинг как в эталоне (`sampling.py`): fp32, деление на температуру,
//! сортировка по убыванию, кумулятивная вероятность по полному распределению,
//! отбрасывается всё с `cum > top_p` (у головного токена токен, пересёкший
//! порог, остаётся; у акустических голов — нет), ранг 0 остаётся всегда.
//! RAS: если первый розыгрыш — семантический токен из последних 16 головных,
//! берётся второй (T = 1, top-k 30, top-p 0,95); оба розыгрыша делаются всегда.

pub const RAS_WINDOW: usize = 16;
const RAS_TEMPERATURE: f32 = 1.0;
const RAS_TOP_P: f32 = 0.95;
const RAS_TOP_K: usize = 30;
/// Потолок top-k акустических голов.
pub const CODEBOOK_TOP_K: usize = 30;

/// splitmix64 — воспроизводимость по сиду.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Индекс выбранного логита. `top_k = 0` — без ограничения.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, keep_crossing: bool, rng: &mut Rng) -> usize {
    let t = temperature.max(1e-5);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let max = logits[order[0]] / t;
    let p: Vec<f64> = order.iter().map(|&i| ((logits[i] / t - max) as f64).exp()).collect();
    let z: f64 = p.iter().sum();
    let k = if top_k == 0 { order.len() } else { top_k.min(order.len()) };
    let mut keep = 1usize;
    let mut cum = p[0] / z;
    let mut prev_dropped = cum > top_p as f64;
    for r in 1..k {
        cum += p[r] / z;
        let dropped = cum > top_p as f64;
        // Со сдвигом (keep_crossing) ранг r отбрасывается, если отброшен ранг r−1.
        let d = if keep_crossing { prev_dropped } else { dropped };
        prev_dropped = dropped;
        if d {
            break;
        }
        keep = r + 1;
    }
    let total: f64 = p[..keep].iter().sum();
    let mut u = rng.next() * total;
    for r in 0..keep {
        u -= p[r];
        if u <= 0.0 {
            return order[r];
        }
    }
    order[keep - 1]
}

/// Головной токен кадра с RAS. `logits` — по кандидатам `ids`; `window` —
/// последние головные токены.
pub fn sample_main(logits: &[f32], ids: &[u32], window: &[u32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    let first = ids[sample(logits, temperature, top_k, top_p, true, rng)];
    let ras = ids[sample(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, true, rng)];
    let tail = &window[window.len().saturating_sub(RAS_WINDOW)..];
    if crate::is_semantic(first) && tail.contains(&first) {
        ras
    } else {
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_at_zero_temperature() {
        let mut r = Rng::new(1);
        for _ in 0..20 {
            assert_eq!(sample(&[0.1, 2.0, 1.9, -1.0], 0.0, 0, 0.9, false, &mut r), 1);
        }
    }

    #[test]
    fn crossing_token_kept_only_for_main() {
        // p ≈ [0.6, 0.3, 0.1]; top_p 0.7: без сдвига остаётся ранг 0, со сдвигом — 0 и 1.
        let l = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let mut r = Rng::new(3);
        let mut seen = [0usize; 3];
        for _ in 0..200 {
            assert_eq!(sample(&l, 1.0, 0, 0.7, false, &mut r), 0);
            seen[sample(&l, 1.0, 0, 0.7, true, &mut r)] += 1;
        }
        assert!(seen[1] > 0 && seen[2] == 0);
    }
}
