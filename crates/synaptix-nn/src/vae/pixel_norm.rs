use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub struct PixelNorm {
    pub eps: f64,
}

impl PixelNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype_in = x.dtype();
        let xf = x.to_dtype(synaptix_core::dtype::DType::F32)?;
        let last = xf.rank() - 1;
        let sq = xf.mul(&xf)?;
        let mean = sq.mean_keepdim(last)?;
        let denom = mean.affine(1.0, self.eps as f32)?.sqrt()?;
        let normed = xf.broadcast_div(&denom)?;
        normed.to_dtype(dtype_in)
    }
}
