use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub struct PerChannelStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl PerChannelStats {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    fn make_mean_std(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        if x.rank() < 2 {
            return Err(SynaptixError::Unsupported("PerChannelStats: rank >= 2"));
        }
        let c = x.dims()[1];
        if self.mean.len() != c || self.std.len() != c {
            return Err(SynaptixError::shape_mismatch(&[c], &[self.mean.len()]));
        }
        let mut shape = vec![1usize; x.rank()];
        shape[1] = c;
        let m = Tensor::from_vec::<_, f32>(self.mean.clone(), shape.clone(), x.device())?;
        let s = Tensor::from_vec::<_, f32>(self.std.clone(), shape, x.device())?;
        Ok((m.to_dtype(x.dtype())?, s.to_dtype(x.dtype())?))
    }

    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let (m, s) = self.make_mean_std(x)?;
        x.broadcast_sub(&m)?.broadcast_div(&s)
    }

    pub fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let (m, s) = self.make_mean_std(x)?;
        x.broadcast_mul(&s)?.broadcast_add(&m)
    }
}
