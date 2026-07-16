use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

pub struct SinkConfig {
    pub num_sink_tokens: usize,
    pub window_size: usize,
}

impl Default for SinkConfig {
    fn default() -> Self {
        Self { num_sink_tokens: 4, window_size: 512 }
    }
}

fn streaming_sink_mask(
    s_q: usize,
    s_kv: usize,
    sinks: usize,
    window: usize,
    device: Device,
) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; s_q * s_kv];
    for i in 0..s_q {
        let q_pos = i + (s_kv - s_q);
        for j in 0..s_kv {
            if j > q_pos {
                continue;
            }
            let in_sink = j < sinks;
            let in_window = window == 0 || j + window > q_pos;
            if in_sink || in_window {
                data[i * s_kv + j] = 0.0;
            }
        }
    }
    Tensor::from_vec::<_, f32>(data, vec![s_q, s_kv], device)
}

pub fn streaming_sink_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    config: &SinkConfig,
) -> Result<Tensor> {
    let s_q = q.dims()[q.rank() - 2];
    let s_kv = k.dims()[k.rank() - 2];
    let mask = streaming_sink_mask(s_q, s_kv, config.num_sink_tokens, config.window_size, q.device())?;
    scaled_dot_attention(q, k, v, scale, Some(&mask))
}
