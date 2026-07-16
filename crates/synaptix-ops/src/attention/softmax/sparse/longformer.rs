use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

fn longformer_mask(
    s_q: usize,
    s_kv: usize,
    window: usize,
    global_positions: &[usize],
    causal: bool,
    device: Device,
) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; s_q * s_kv];
    for i in 0..s_q {
        let qi = i + (s_kv - s_q);
        let is_global_q = global_positions.iter().any(|&g| g == qi);
        for j in 0..s_kv {
            if causal && j > qi {
                continue;
            }
            let local = (qi as isize - j as isize).abs() as usize <= window;
            let is_global_k = global_positions.iter().any(|&g| g == j);
            if local || is_global_q || is_global_k {
                data[i * s_kv + j] = 0.0;
            }
        }
    }
    Tensor::from_vec::<_, f32>(data, vec![s_q, s_kv], device)
}

pub fn longformer_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window: usize,
    global_positions: &[usize],
    causal: bool,
) -> Result<Tensor> {
    let s_q = q.dims()[q.rank() - 2];
    let s_kv = k.dims()[k.rank() - 2];
    let mask = longformer_mask(s_q, s_kv, window, global_positions, causal, q.device())?;
    scaled_dot_attention(q, k, v, scale, Some(&mask))
}
