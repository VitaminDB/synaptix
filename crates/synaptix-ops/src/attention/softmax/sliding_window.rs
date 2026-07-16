use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;
use crate::mask::sliding_window_mask;

pub fn sliding_window_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window: usize,
    extra_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let s = q.dims()[q.rank() - 2];
    let device: Device = q.device();
    let sw = sliding_window_mask(s, window, device)?;
    let combined = match extra_mask {
        Some(m) => sw.broadcast_add(&m.to_dtype(synaptix_core::dtype::DType::F32)?)?,
        None => sw,
    };
    scaled_dot_attention(q, k, v, scale, Some(&combined))
}
