pub fn softmax_f32_inplace(buf: &mut [f32]) {
    let max = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in buf.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    for x in buf.iter_mut() { *x *= inv; }
}

pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    output.copy_from_slice(input);
    softmax_f32_inplace(output);
}

pub fn log_softmax_f32_inplace(buf: &mut [f32]) {
    let max = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = buf.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
    for x in buf.iter_mut() { *x -= log_sum; }
}

pub fn online_softmax_f32(seq: &[f32], output: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    let mut d = 0.0f32;
    for &x in seq {
        let m_new = m.max(x);
        d = d * (m - m_new).exp() + (x - m_new).exp();
        m = m_new;
    }
    let inv_d = 1.0 / d.max(f32::MIN_POSITIVE);
    for (out, &x) in output.iter_mut().zip(seq) {
        *out = (x - m).exp() * inv_d;
    }
}
