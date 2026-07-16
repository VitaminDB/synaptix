use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

fn strided_mask(
    s_q: usize,
    s_kv: usize,
    stride: usize,
    causal: bool,
    device: Device,
) -> Result<Tensor> {
    if stride == 0 {
        return Err(SynaptixError::Unsupported("strided: stride > 0"));
    }
    let mut data = vec![f32::NEG_INFINITY; s_q * s_kv];
    for i in 0..s_q {
        let qi = i + (s_kv - s_q);
        for j in 0..s_kv {
            if causal && j > qi {
                continue;
            }
            let local = j + stride > qi && j <= qi;
            let strided = qi >= j && (qi - j) % stride == 0;
            if local || strided {
                data[i * s_kv + j] = 0.0;
            }
        }
    }
    Tensor::from_vec::<_, f32>(data, vec![s_q, s_kv], device)
}

pub fn strided_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    stride: usize,
    causal: bool,
) -> Result<Tensor> {
    let s_q = q.dims()[q.rank() - 2];
    let s_kv = k.dims()[k.rank() - 2];
    let mask = strided_mask(s_q, s_kv, stride, causal, q.device())?;
    scaled_dot_attention(q, k, v, scale, Some(&mask))
}
