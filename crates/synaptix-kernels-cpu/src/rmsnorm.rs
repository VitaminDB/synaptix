pub fn rms_norm_f32(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * scale * wi;
    }
}

pub fn rms_norm_f32_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(w) {
        *xi = *xi * scale * wi;
    }
}

pub fn rms_norm_gated_f32(x: &[f32], gate: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for (((o, &xi), &gi), &wi) in out.iter_mut().zip(x).zip(gate).zip(w) {
        let silu_g = gi / (1.0 + (-gi).exp());
        *o = xi * scale * wi * silu_g;
    }
}
