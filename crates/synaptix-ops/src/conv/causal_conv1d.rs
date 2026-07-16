use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

pub fn causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    // x: [B, C, L]  weight: [C, 1, K]  bias: [C]  →  out: [B, C, out_len]
    if x.rank() != 3 || weight.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "causal_conv1d: x must be [B,C,L], weight [C,1,K]",
        ));
    }
    let (b, c, l) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let k = weight.dims()[2];
    let stride = stride.max(1);
    let out_len = (l + stride - 1) / stride;

    let pad = Tensor::zeros(vec![b, c, k - 1], x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&pad, x], 2)?; // [B, C, L+K-1]

    let mut out = Tensor::zeros(vec![b, c, out_len], x.dtype(), x.device())?;
    for ki in 0..k {
        // weight[:, 0, ki] → [C] → [1, C, 1]
        let w_ki = weight
            .narrow(2, ki, 1)?
            .squeeze(2)?
            .squeeze(1)?
            .unsqueeze(0)?
            .unsqueeze(2)?; // [1, C, 1]

        if stride == 1 {
            let slice = x_padded.narrow(2, ki, l)?; // [B, C, L]
            out = out.add(&slice.broadcast_mul(&w_ki)?)?;
        } else {
            // Collect strided positions
            let mut parts: Vec<Tensor> = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let pos = ki + i * stride;
                if pos < x_padded.dims()[2] {
                    parts.push(x_padded.narrow(2, pos, 1)?);
                }
            }
            if !parts.is_empty() {
                let refs: Vec<&Tensor> = parts.iter().collect();
                let sliced = Tensor::cat(&refs, 2)?; // [B, C, out_len]
                out = out.add(&sliced.broadcast_mul(&w_ki)?)?;
            }
        }
    }

    if let Some(b_t) = bias {
        let b_shaped = b_t.unsqueeze(0)?.unsqueeze(2)?; // [1, C, 1]
        out = out.broadcast_add(&b_shaped)?;
    }
    Ok(out)
}

pub fn causal_conv1d_stateful(
    state: &mut [f32],
    x: &[f32],
    w: &[f32],
    s: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    let km1 = k - 1;
    let mut ext = vec![0.0f32; (km1 + s) * channels];
    ext[..km1 * channels].copy_from_slice(state);
    ext[km1 * channels..].copy_from_slice(&x[..s * channels]);
    let mut out = vec![0.0f32; s * channels];
    for i in 0..s {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w[c * k + j] * ext[(i + j) * channels + c];
            }
            out[i * channels + c] = acc;
        }
    }
    let start = s * channels;
    state.copy_from_slice(&ext[start..start + km1 * channels]);
    out
}

#[cfg(test)]
mod stateful_tests {
    use super::*;
    use synaptix_core::device::Device;

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn stateful_matches_full_seq() {
        synaptix_kernels_cpu::ensure_registered();
        let (s, channels, k) = (5usize, 7usize, 4usize);
        let mut seed = 999u64;
        let x = rand_vec(s * channels, &mut seed);
        let w = rand_vec(channels * k, &mut seed);

        let mut state = vec![0.0f32; (k - 1) * channels];
        let mine = causal_conv1d_stateful(&mut state, &x, &w, s, channels, k);

        let mut x_cl = vec![0.0f32; channels * s];
        for t in 0..s {
            for c in 0..channels {
                x_cl[c * s + t] = x[t * channels + c];
            }
        }
        let xt = Tensor::from_vec(x_cl, vec![1, channels, s], Device::Cpu).unwrap();
        let wt = Tensor::from_vec(w.clone(), vec![channels, 1, k], Device::Cpu).unwrap();
        let conv = causal_conv1d(&xt, &wt, None, 1).unwrap();
        let conv_v: Vec<f32> = conv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut maxerr = 0.0f32;
        for t in 0..s {
            for c in 0..channels {
                maxerr = maxerr.max((conv_v[c * s + t] - mine[t * channels + c]).abs());
            }
        }
        assert!(maxerr < 1e-4, "stateful conv vs full-seq max abs err {maxerr}");
    }
}
