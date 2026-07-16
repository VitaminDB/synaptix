use synaptix_core::error::{Result, SynaptixError};

use super::fp8_e4m3::{decode_e4m3, encode_e4m3, FP8_E4M3_MAX};

pub const FP8_BLOCK_SIZE: usize = 32;

pub fn quantize_f32_to_fp8(input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
    let total = input.len();
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let block_size = FP8_BLOCK_SIZE;
    let num_blocks = total.div_ceil(block_size);

    let mut data = vec![0u8; total];
    let mut scales = vec![0f32; num_blocks];

    for bi in 0..num_blocks {
        let start = bi * block_size;
        let end = (start + block_size).min(total);
        let block = &input[start..end];

        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if amax > 0.0 { amax / FP8_E4M3_MAX } else { 1.0 };
        let inv_scale = if amax > 0.0 { FP8_E4M3_MAX / amax } else { 0.0 };
        scales[bi] = scale;

        for (i, &v) in block.iter().enumerate() {
            let scaled = v * inv_scale;
            data[start + i] = encode_e4m3(scaled);
        }
    }

    Ok((data, scales))
}

pub fn dequantize_fp8_to_f32(data: &[u8], scales: &[f32], total: usize) -> Result<Vec<f32>> {
    let block_size = FP8_BLOCK_SIZE;
    let num_blocks = total.div_ceil(block_size);
    if scales.len() != num_blocks {
        return Err(SynaptixError::Other(format!(
            "dequantize_fp8: scales {} != num_blocks {}",
            scales.len(),
            num_blocks
        )));
    }
    if data.len() != total {
        return Err(SynaptixError::Other(format!(
            "dequantize_fp8: data {} != total {}",
            data.len(),
            total
        )));
    }
    let mut out = vec![0f32; total];
    for bi in 0..num_blocks {
        let start = bi * block_size;
        let end = (start + block_size).min(total);
        let scale = scales[bi];
        for i in start..end {
            out[i] = decode_e4m3(data[i]) * scale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x - *y) as f64;
            s += d * d;
        }
        (s / a.len() as f64).sqrt() as f32
    }

    #[test]
    fn quantize_dequantize_zero() {
        let input = vec![0f32; 64];
        let (data, scales) = quantize_f32_to_fp8(&input).unwrap();
        let back = dequantize_fp8_to_f32(&data, &scales, 64).unwrap();
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn quantize_dequantize_random_small_amax() {
        let input: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.05).collect();
        let (data, scales) = quantize_f32_to_fp8(&input).unwrap();
        assert_eq!(data.len(), 96);
        assert_eq!(scales.len(), 3);
        let back = dequantize_fp8_to_f32(&data, &scales, 96).unwrap();
        let err = rms(&input, &back);
        let amax = input.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(err < amax * 0.05, "FP8 RMS {err} > 5% of amax {amax}");
    }

    #[test]
    fn quantize_dequantize_random_large_amax() {
        let input: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 3.0).collect();
        let (data, scales) = quantize_f32_to_fp8(&input).unwrap();
        let back = dequantize_fp8_to_f32(&data, &scales, 128).unwrap();
        let err = rms(&input, &back);
        let amax = input.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(err < amax * 0.05, "FP8 RMS {err} > 5% of amax {amax}");
    }

    #[test]
    fn quantize_tail_block_shorter() {
        let input: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
        let (data, scales) = quantize_f32_to_fp8(&input).unwrap();
        assert_eq!(data.len(), 40);
        assert_eq!(scales.len(), 2);
        let back = dequantize_fp8_to_f32(&data, &scales, 40).unwrap();
        assert_eq!(back.len(), 40);
    }
}
