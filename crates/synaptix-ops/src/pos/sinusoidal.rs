use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn sinusoidal_positional_embedding(
    seq_len: usize,
    dim: usize,
    device: Device,
) -> Result<Tensor> {
    sinusoidal_positional_embedding_with_period(seq_len, dim, 10000.0, device)
}

pub fn sinusoidal_positional_embedding_with_period(
    seq_len: usize,
    dim: usize,
    max_period: f32,
    device: Device,
) -> Result<Tensor> {
    if dim % 2 != 0 {
        return Err(SynaptixError::Unsupported(
            "sinusoidal_positional_embedding: dim must be even",
        ));
    }
    let half = dim / 2;
    let mut data = vec![0.0_f32; seq_len * dim];
    for pos in 0..seq_len {
        for i in 0..half {
            let exponent = (2.0 * i as f32) / (dim as f32);
            let denom = max_period.powf(exponent);
            let angle = (pos as f32) / denom;
            data[pos * dim + 2 * i] = angle.sin();
            data[pos * dim + 2 * i + 1] = angle.cos();
        }
    }
    Tensor::from_vec(data, (seq_len, dim), device)
}
