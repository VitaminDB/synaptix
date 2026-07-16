use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(num_heads);
    let next_pow2 = next_power_of_two(num_heads);
    let base_slopes = pow2_slopes(next_pow2);
    if num_heads <= next_pow2 {
        slopes.extend_from_slice(&base_slopes[..num_heads]);
    } else {
        slopes.extend_from_slice(&base_slopes);
        let half = pow2_slopes(next_pow2 * 2);
        let mut idx = 0usize;
        while slopes.len() < num_heads {
            slopes.push(half[idx * 2 + 1]);
            idx += 1;
        }
    }
    slopes
}

fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn pow2_slopes(n: usize) -> Vec<f32> {
    let start = 2.0_f32.powf(-8.0 / n as f32);
    let mut slopes = Vec::with_capacity(n);
    let mut s = 1.0_f32;
    for _ in 0..n {
        s *= start;
        slopes.push(s);
    }
    slopes
}

pub fn alibi_bias(num_heads: usize, seq_len: usize, device: Device) -> Result<Tensor> {
    if num_heads == 0 || seq_len == 0 {
        return Err(SynaptixError::Unsupported("alibi_bias: zero size"));
    }
    let slopes = alibi_slopes(num_heads);
    let mut data = vec![0.0_f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        let m = slopes[h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = ((j as f32) - (i as f32)).abs();
                data[(h * seq_len + i) * seq_len + j] = -m * dist;
            }
        }
    }
    Tensor::from_vec(data, (1, num_heads, seq_len, seq_len), device)
}
