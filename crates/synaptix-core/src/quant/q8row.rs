use rayon::prelude::*;

pub const QMAX: f32 = 127.0;
pub const CLIP_STEPS: usize = 40;
pub const CLIP_MIN: f32 = 0.6;

fn row_error(w: &[f32], h: Option<&[f32]>, s: f32) -> f64 {
    let inv = 1.0 / s;
    let mut e = 0.0f64;
    for (i, &v) in w.iter().enumerate() {
        let q = (v * inv).round().clamp(-QMAX, QMAX);
        let d = (v - q * s) as f64;
        e += d * d * h.map_or(1.0, |h| h[i] as f64);
    }
    e
}

pub fn row_scale(w: &[f32], h: Option<&[f32]>) -> f32 {
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        return 1.0;
    }
    let Some(h) = h else { return amax / QMAX };
    let mut best = (row_error(w, Some(h), amax / QMAX), amax / QMAX);
    for i in 1..=CLIP_STEPS {
        let a = 1.0 - (1.0 - CLIP_MIN) * i as f32 / CLIP_STEPS as f32;
        let s = a * amax / QMAX;
        let e = row_error(w, Some(h), s);
        if e < best.0 {
            best = (e, s);
        }
    }
    best.1
}

pub fn quantize_row(w: &[f32], h: Option<&[f32]>, q: &mut [u8]) -> f32 {
    let s = row_scale(w, h);
    let inv = 1.0 / s;
    for (d, &v) in q.iter_mut().zip(w) {
        *d = (v * inv).round().clamp(-QMAX, QMAX) as i8 as u8;
    }
    s
}

pub fn quantize(w: &[f32], n: usize, k: usize, h: Option<&[f32]>) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(w.len(), n * k);
    let h = h.filter(|h| h.len() == k);
    let mut packed = vec![0u8; n * k];
    let scales: Vec<f32> = packed
        .par_chunks_mut(k.max(1))
        .zip(w.par_chunks(k.max(1)))
        .map(|(q, row)| quantize_row(row, h, q))
        .collect();
    (packed, scales)
}

pub fn dequant_row(src: &[u8], scale: f32, dst: &mut [f32]) {
    for (d, &b) in dst.iter_mut().zip(src) {
        *d = b as i8 as f32 * scale;
    }
}

pub fn scales_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn scales_to_bytes(scales: &[f32]) -> Vec<u8> {
    scales.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_scale_never_worse_than_absmax() {
        let mut w: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.37).sin()) * 0.02).collect();
        w[5] = 0.9;
        let h: Vec<f32> = (0..1024).map(|i| if i == 5 { 0.001 } else { 1.0 }).collect();
        let s = row_scale(&w, Some(&h));
        assert!(row_error(&w, Some(&h), s) <= row_error(&w, Some(&h), 0.9 / QMAX));
        assert_eq!(row_scale(&w, None), 0.9 / QMAX);
        let mut q = vec![0u8; w.len()];
        let s = quantize_row(&w, Some(&h), &mut q);
        let mut back = vec![0f32; w.len()];
        dequant_row(&q, s, &mut back);
        assert!((back[100] - w[100]).abs() <= s * 0.5 + 1e-6);
    }

    #[test]
    fn zero_row() {
        let w = vec![0.0f32; 64];
        let (p, s) = quantize(&w, 1, 64, None);
        assert!(p.iter().all(|&b| b == 0));
        assert_eq!(s, vec![1.0]);
    }
}
