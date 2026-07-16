use crate::error::{AudioError, Result};

pub fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(AudioError::invalid_arg("sample rates must be > 0"));
    }
    if src_rate == dst_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let frac = (pos - lo as f64) as f32;
        let v = input[lo] * (1.0 - frac) + input[hi] * frac;
        out.push(v);
    }
    Ok(out)
}
