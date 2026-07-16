use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    max_period: f32,
) -> Result<Tensor> {
    if timesteps.rank() != 1 {
        return Err(SynaptixError::Unsupported("timestep_embedding: timesteps must be 1D"));
    }
    if dim % 2 != 0 {
        return Err(SynaptixError::Unsupported("timestep_embedding: dim must be even"));
    }
    let device: Device = timesteps.device();
    let n = timesteps.dims()[0];
    let half = dim / 2;
    let t_vec: Vec<f64> = match timesteps.dtype() {
        synaptix_core::dtype::DType::F32 => timesteps.to_vec1::<f32>()?.into_iter().map(|v| v as f64).collect(),
        synaptix_core::dtype::DType::F64 => timesteps.to_vec1::<f64>()?,
        synaptix_core::dtype::DType::U32 => timesteps.to_vec1::<u32>()?.into_iter().map(|v| v as f64).collect(),
        synaptix_core::dtype::DType::I64 => timesteps.to_vec1::<i64>()?.into_iter().map(|v| v as f64).collect(),
        _ => return Err(SynaptixError::Unsupported("timestep_embedding: dtype")),
    };
    let mut data = vec![0.0_f32; n * dim];
    for (i, t) in t_vec.iter().enumerate() {
        for k in 0..half {
            let exponent = -(k as f64) * (max_period.ln() as f64) / (half as f64);
            let freq = exponent.exp();
            let angle = (*t * freq) as f32;
            data[i * dim + k] = angle.cos();
            data[i * dim + half + k] = angle.sin();
        }
    }
    Tensor::from_vec(data, (n, dim), device)
}
