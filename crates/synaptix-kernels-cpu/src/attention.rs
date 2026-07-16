use crate::softmax::softmax_f32_inplace;

pub fn scaled_dot_attention_f32(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    for qi in 0..seq_len {
        for ki in 0..seq_len {
            let dot: f32 = (0..head_dim)
                .map(|d| q[qi * head_dim + d] * k[ki * head_dim + d])
                .sum();
            scores[ki] = dot * scale;
        }
        softmax_f32_inplace(&mut scores);
        for d in 0..head_dim {
            out[qi * head_dim + d] = scores.iter().enumerate()
                .map(|(ki, &p)| p * v[ki * head_dim + d])
                .sum();
        }
    }
}

pub fn causal_attention_f32(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    for qi in 0..seq_len {
        for ki in 0..=qi {
            let dot: f32 = (0..head_dim)
                .map(|d| q[qi * head_dim + d] * k[ki * head_dim + d])
                .sum();
            scores[ki] = dot * scale;
        }
        for ki in (qi + 1)..seq_len { scores[ki] = f32::NEG_INFINITY; }
        softmax_f32_inplace(&mut scores[..=qi]);
        for ki in (qi + 1)..seq_len { scores[ki] = 0.0; }
        for d in 0..head_dim {
            out[qi * head_dim + d] = scores.iter().enumerate()
                .map(|(ki, &p)| p * v[ki * head_dim + d])
                .sum();
        }
    }
}
