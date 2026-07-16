use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn causal_mask(seq_len: usize, device: Device) -> Result<Tensor> {
    synaptix_ops::mask::causal_mask(seq_len, device)
}

pub fn sliding_window_mask(seq_len: usize, window: usize, device: Device) -> Result<Tensor> {
    synaptix_ops::mask::sliding_window_mask(seq_len, window, device)
}
