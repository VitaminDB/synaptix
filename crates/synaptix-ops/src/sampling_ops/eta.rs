/// Eta sampling (Hewitt et al., 2022): порог `ε = min(eta, sqrt(eta)·exp(−H))`,
/// где `H` — энтропия softmax-распределения (натуральный логарифм). Маскирует
/// (в `-inf`) токены с вероятностью < ε. Argmax всегда сохраняется.
pub fn apply_eta(logits: &mut Vec<f32>, eta: f32) {
    if eta <= 0.0 || logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let exp_sum: f32 = exps.iter().sum();
    if exp_sum <= 0.0 {
        return;
    }
    // энтропия H = -Σ p ln p
    let mut h = 0.0f32;
    for &e in &exps {
        let p = e / exp_sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    let threshold = eta.min(eta.sqrt() * (-h).exp());

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();

    for i in 0..logits.len() {
        if i == argmax {
            continue;
        }
        let p = exps[i] / exp_sum;
        if p < threshold {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}
