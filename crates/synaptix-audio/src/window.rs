use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    Hann,
    Hamming,
    Rectangular,
}

pub fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (n - 1) as f32).cos())
        .collect()
}

pub fn hann_periodic(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

pub fn hamming(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let alpha = 0.54f32;
    let beta = 0.46f32;
    (0..n)
        .map(|i| alpha - beta * (2.0 * PI * i as f32 / (n - 1) as f32).cos())
        .collect()
}

pub fn rectangular(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

pub fn build(kind: WindowKind, n: usize) -> Vec<f32> {
    match kind {
        WindowKind::Hann => hann_periodic(n),
        WindowKind::Hamming => hamming(n),
        WindowKind::Rectangular => rectangular(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_n4_known_values() {
        let w = hann(4);
        assert_eq!(w.len(), 4);
        assert!((w[0] - 0.0).abs() < 1e-6);
        assert!((w[3] - 0.0).abs() < 1e-6);
        assert!((w[1] - 0.75).abs() < 1e-6);
        assert!((w[2] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn hann_periodic_no_zero_at_end() {
        let w = hann_periodic(4);
        assert_eq!(w.len(), 4);
        assert!(w[0].abs() < 1e-6);
        assert!(w[3] > 0.4);
    }

    #[test]
    fn hamming_basic() {
        let w = hamming(3);
        assert!((w[0] - 0.08).abs() < 1e-5);
        assert!((w[1] - 1.0).abs() < 1e-5);
        assert!((w[2] - 0.08).abs() < 1e-5);
    }
}
