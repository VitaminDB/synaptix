use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    dilation: usize,
) -> Result<Tensor> {
    if input.rank() != 3 || weight.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "conv_transpose1d: input [B,C_in,L], weight [C_in,C_out/groups,K]",
        ));
    }
    let (b, c_in, l) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let (c_in_w, c_out_g, k) = (weight.dims()[0], weight.dims()[1], weight.dims()[2]);
    if c_in_w != c_in {
        return Err(SynaptixError::shape_mismatch(input.dims(), weight.dims()));
    }
    let groups = groups.max(1);
    let stride = stride.max(1);
    let dilation = dilation.max(1);
    let c_out = c_out_g * groups;
    let out_len = (l - 1) * stride + dilation * (k - 1) + output_padding + 1;
    if out_len < 2 * padding + 1 {
        return Err(SynaptixError::Unsupported("conv_transpose1d: out_len < padding"));
    }
    let out_len_cropped = out_len - 2 * padding;
    let device = input.device();
    let dtype = input.dtype();

    let depthwise = groups == c_in && c_out_g == 1;

    if depthwise && dilation == 1 && output_padding == 0 {
        match input.dwconv1d(weight, bias, stride, 0, true) {
            Ok(full) => {
                let out = if padding > 0 {
                    full.narrow(2, padding, out_len_cropped)?.contiguous()?
                } else {
                    full
                };
                return Ok(out);
            }
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }

    // Fast path (VAE upsampler): groups==1, dilation==1, output_padding==0.
    // ConvTranspose1d(stride S, kernel K) ≡ Conv1d(kernel ⌈K/S⌉, Cout·S out-ch)
    // + pixel-shuffle. One tensor-core matmul + reshape instead of the per-tap
    // `place_strided` copy-storm (O(K) full-output materializations).
    if groups == 1 && !depthwise && dilation == 1 && output_padding == 0 {
        let out = convt1d_pixelshuffle(input, weight, stride, padding)?;
        let out = if let Some(bt) = bias {
            out.broadcast_add(&bt.reshape(vec![1, c_out, 1])?)?
        } else {
            out
        };
        return Ok(out);
    }

    let x_t_g1 = if groups == 1 && !depthwise {
        Some(input.permute(vec![0, 2, 1])?.contiguous()?)
    } else {
        None
    };
    let mut out = Tensor::zeros(vec![b, c_out, out_len], dtype, device)?;
    for kk in 0..k {
        let p_k = if depthwise {
            let w_k = weight.narrow(2, kk, 1)?.contiguous()?.reshape(vec![1, c_in, 1])?;
            input.broadcast_mul(&w_k)?
        } else if groups == 1 {
            let w_k = weight.narrow(2, kk, 1)?.contiguous()?.squeeze(2)?;
            x_t_g1.as_ref().unwrap().matmul(&w_k)?.permute(vec![0, 2, 1])?.contiguous()?
        } else {
            let cin_g = c_in / groups;
            let mut parts = Vec::with_capacity(groups);
            for g in 0..groups {
                let xg = input.narrow(1, g * cin_g, cin_g)?;
                let wg = weight.narrow(0, g * cin_g, cin_g)?.narrow(2, kk, 1)?.contiguous()?.squeeze(2)?;
                let xt = xg.permute(vec![0, 2, 1])?.contiguous()?;
                parts.push(xt.matmul(&wg)?.permute(vec![0, 2, 1])?.contiguous()?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 1)?
        };
        let placed = place_strided(&p_k, stride, kk * dilation, out_len)?;
        out = out.add(&placed)?;
    }

    let out = if padding > 0 {
        out.narrow(2, padding, out_len_cropped)?.contiguous()?
    } else {
        out
    };
    let out = if let Some(bt) = bias {
        out.broadcast_add(&bt.reshape(vec![1, c_out, 1])?)?
    } else {
        out
    };
    Ok(out)
}

/// ConvTranspose1d via the Conv1d + pixel-shuffle identity (groups==1,
/// dilation==1, output_padding==0). Avoids the per-tap `place_strided`
/// materialization that dominates VAE decode wall-time (copy-storm).
///
/// For residue `sr` in `[0,S)`, `out[b,co,t*S+sr] = Σ_pp x[b,:,t-pp]·w[:,co,sr+pp*S]`,
/// a Conv1d of size `P=⌈K/S⌉`. Stacking the S residues over `Cout·S` channels and
/// interleaving (pixel-shuffle) reconstructs the full transposed output.
type WcKey = (usize, Vec<usize>, usize);
type WcEntry = (Weak<synaptix_core::tensor::storage::Storage>, Tensor);
static WC_CACHE: OnceLock<Mutex<HashMap<WcKey, WcEntry>>> = OnceLock::new();

fn cached_pixelshuffle_weight(weight: &Tensor, stride: usize) -> Result<Option<Tensor>> {
    let key = (
        Arc::as_ptr(&weight.storage_arc()) as usize,
        weight.dims().to_vec(),
        stride,
    );
    let cache = WC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(g) = cache.lock() else {
        return Ok(None);
    };
    let Some((wk, t)) = g.get(&key) else {
        return Ok(None);
    };
    if wk.upgrade().is_some_and(|s| Arc::as_ptr(&s) as usize == key.0) {
        return Ok(Some(t.clone()));
    }
    Ok(None)
}

fn store_pixelshuffle_weight(weight: &Tensor, stride: usize, wc: &Tensor) {
    let storage = weight.storage_arc();
    let key = (Arc::as_ptr(&storage) as usize, weight.dims().to_vec(), stride);
    let cache = WC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut g) = cache.lock() {
        g.retain(|_, (wk, _)| wk.upgrade().is_some());
        g.insert(key, (Arc::downgrade(&storage), wc.clone()));
    }
}

fn convt1d_pixelshuffle(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let (b, c_in, l) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let c_out = weight.dims()[1];
    let k = weight.dims()[2];
    let s = stride.max(1);
    let p = (k + s - 1) / s; // ⌈K/S⌉
    let device = input.device();
    let dtype = input.dtype();

    // Rearranged Conv1d weight Wc: [Cout*S, Cin, P], cheap (weight-sized).
    //   Wc[co*S + sr, ci, pp] = w[ci, co, sr + (P-1-pp)*S]   (0 if index >= K)
    if let Some(wc) = cached_pixelshuffle_weight(weight, s)? {
        return convt1d_pixelshuffle_apply(input, &wc, b, c_out, k, s, p, padding);
    }
    let mut p_slices: Vec<Tensor> = Vec::with_capacity(p);
    for pp in 0..p {
        let mut s_mats: Vec<Tensor> = Vec::with_capacity(s);
        for sr in 0..s {
            let kidx = sr as isize + ((p - 1 - pp) as isize) * (s as isize);
            let mat = if kidx >= 0 && (kidx as usize) < k {
                weight.narrow(2, kidx as usize, 1)?.squeeze(2)?.transpose(0, 1)?.contiguous()?
            } else {
                Tensor::zeros(vec![c_out, c_in], dtype, device)?
            };
            s_mats.push(mat.reshape(vec![c_out, 1, c_in])?);
        }
        let refs: Vec<&Tensor> = s_mats.iter().collect();
        let inter = Tensor::cat(&refs, 1)?.reshape(vec![c_out * s, c_in])?;
        p_slices.push(inter.reshape(vec![c_out * s, c_in, 1])?);
    }
    let refs: Vec<&Tensor> = p_slices.iter().collect();
    let wc = Tensor::cat(&refs, 2)?.contiguous()?; // [Cout*S, Cin, P]
    store_pixelshuffle_weight(weight, s, &wc);
    convt1d_pixelshuffle_apply(input, &wc, b, c_out, k, s, p, padding)
}

#[allow(clippy::too_many_arguments)]
fn convt1d_pixelshuffle_apply(
    input: &Tensor,
    wc: &Tensor,
    b: usize,
    c_out: usize,
    k: usize,
    s: usize,
    p: usize,
    padding: usize,
) -> Result<Tensor> {
    // Conv1d(stride 1, padding P-1) -> [B, Cout*S, L+P-1] (single tensor-core GEMM).
    let conv = super::conv1d::conv1d_dilated(input, wc, None, 1, p - 1, 1)?;
    let lc = conv.dims()[2];

    // Pixel-shuffle: out[b,co,t*S+sr] = conv[b, co*S+sr, t].
    let out_full = conv
        .reshape(vec![b, c_out, s, lc])?
        .permute(vec![0, 1, 3, 2])?
        .contiguous()?
        .reshape(vec![b, c_out, lc * s])?;

    let out_len_full = (input.dims()[2] - 1) * s + k;
    let out_full = if lc * s > out_len_full {
        out_full.narrow(2, 0, out_len_full)?
    } else {
        out_full
    };
    if padding > 0 {
        let out_len_cropped = out_len_full - 2 * padding;
        out_full.narrow(2, padding, out_len_cropped)?.contiguous()
    } else {
        out_full.contiguous()
    }
}

fn place_strided(p: &Tensor, stride: usize, offset: usize, out_len: usize) -> Result<Tensor> {
    let p = p.contiguous()?;
    let (b, c, l) = (p.dims()[0], p.dims()[1], p.dims()[2]);
    let base = if stride == 1 {
        p.clone()
    } else {
        let z = Tensor::zeros(vec![b, c, l, stride - 1], p.dtype(), p.device())?;
        let p4 = p.reshape(vec![b, c, l, 1])?;
        Tensor::cat(&[&p4, &z], 3)?.contiguous()?.reshape(vec![b, c, l * stride])?
    };
    let base_len = (l - 1) * stride + 1;
    if offset >= out_len {
        return Tensor::zeros(vec![b, c, out_len], p.dtype(), p.device());
    }
    let take = base_len.min(out_len - offset);
    let win = base.narrow(2, 0, take)?;
    let right = out_len - offset - take;
    let mut parts: Vec<Tensor> = Vec::new();
    if offset > 0 {
        parts.push(Tensor::zeros(vec![b, c, offset], p.dtype(), p.device())?);
    }
    parts.push(win.contiguous()?);
    if right > 0 {
        parts.push(Tensor::zeros(vec![b, c, right], p.dtype(), p.device())?);
    }
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap().contiguous();
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;

    fn assert_matches_naive(c_in: usize, c_out: usize, k: usize, s: usize, l: usize, pad: usize) {
        synaptix_kernels_cpu::ensure_registered();
        let dev = Device::Cpu;
        let x: Vec<f32> = (0..c_in * l).map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6).collect();
        let w: Vec<f32> = (0..c_in * c_out * k).map(|i| ((i * 5 % 11) as f32) * 0.1 - 0.5).collect();
        let xt = Tensor::from_vec(x, vec![1, c_in, l], dev).unwrap();
        let wt = Tensor::from_vec(w, vec![c_in, c_out, k], dev).unwrap();
        // Fast path (pixel-shuffle) vs naive CPU scatter reference.
        let fast = conv_transpose1d(&xt, &wt, None, s, pad, 0, 1, 1).unwrap();
        let refr = crate::conv::transposed::transposed_conv(&xt, &wt, None, s, pad).unwrap();
        let a: Vec<f32> = fast.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = refr.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(fast.dims(), refr.dims(), "shape c{c_in}/{c_out} k{k} s{s} l{l} p{pad}");
        for (i, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((xa - xb).abs() < 1e-3, "idx{i}: {xa} vs {xb} (c{c_in}/{c_out} k{k} s{s} l{l} p{pad})");
        }
    }

    #[test]
    fn cached_weight_survives_repeat_calls() {
        synaptix_kernels_cpu::ensure_registered();
        let dev = Device::Cpu;
        let (c_in, c_out, k, s, pad) = (4usize, 3usize, 6usize, 3usize, 2usize);
        let w: Vec<f32> = (0..c_in * c_out * k).map(|i| ((i * 5 % 11) as f32) * 0.1 - 0.5).collect();
        let wt = Tensor::from_vec(w, vec![c_in, c_out, k], dev).unwrap();
        for l in [4usize, 9, 4, 17] {
            let x: Vec<f32> = (0..c_in * l).map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6).collect();
            let xt = Tensor::from_vec(x, vec![1, c_in, l], dev).unwrap();
            let fast = conv_transpose1d(&xt, &wt, None, s, pad, 0, 1, 1).unwrap();
            let refr = crate::conv::transposed::transposed_conv(&xt, &wt, None, s, pad).unwrap();
            let a: Vec<f32> = fast.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = refr.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(fast.dims(), refr.dims(), "shape при L={l}");
            for (i, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
                assert!((xa - xb).abs() < 1e-3, "L={l} idx{i}: {xa} vs {xb}");
            }
        }
    }

    #[test]
    fn pixelshuffle_matches_naive() {
        assert_matches_naive(3, 2, 4, 2, 5, 1); // K=2S
        assert_matches_naive(4, 3, 6, 3, 4, 2); // K=2S, S=3
        assert_matches_naive(2, 2, 5, 2, 6, 1); // K odd, P=3
        assert_matches_naive(1, 1, 8, 4, 3, 2); // K=2S, S=4
        assert_matches_naive(5, 4, 4, 2, 7, 0); // no padding
        assert_matches_naive(6, 6, 7, 2, 5, 3); // K=2S+1
    }
}
