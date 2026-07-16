use std::sync::Arc;

use synaptix_core::error::Result;
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

pub struct SumGradFn {
    parents: [Tensor; 1],
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
}

impl SumGradFn {
    pub fn new(input: &Tensor, dims: Vec<usize>, keepdim: bool) -> Arc<dyn GradFn> {
        Arc::new(Self {
            parents: [input.clone()],
            input_shape: input.dims().to_vec(),
            dims,
            keepdim,
        })
    }
}

impl GradFn for SumGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = unsqueeze_reduced(output_grad, &self.dims, self.keepdim, &self.input_shape)?;
        let ones = Tensor::ones(
            self.input_shape.clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let out = g.broadcast_mul(&ones)?;
        Ok(vec![Some(out)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "SumGradFn"
    }
}

pub struct MeanGradFn {
    parents: [Tensor; 1],
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
    inv_n: f32,
}

impl MeanGradFn {
    pub fn new(input: &Tensor, dims: Vec<usize>, keepdim: bool) -> Arc<dyn GradFn> {
        let input_shape = input.dims().to_vec();
        let n: usize = if dims.is_empty() {
            input.dims().iter().product()
        } else {
            dims.iter().map(|&d| input_shape[d]).product()
        };
        let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f32 };
        Arc::new(Self { parents: [input.clone()], input_shape, dims, keepdim, inv_n })
    }
}

impl GradFn for MeanGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g_scaled = output_grad.affine(self.inv_n, 0.0)?;
        let g = unsqueeze_reduced(&g_scaled, &self.dims, self.keepdim, &self.input_shape)?;
        let ones = Tensor::ones(
            self.input_shape.clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let out = g.broadcast_mul(&ones)?;
        Ok(vec![Some(out)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "MeanGradFn"
    }
}

fn unsqueeze_reduced(
    t: &Tensor,
    dims: &[usize],
    keepdim: bool,
    input_shape: &[usize],
) -> Result<Tensor> {
    let mut g = t.clone();
    let all_dims: Vec<usize> = if dims.is_empty() {
        (0..input_shape.len()).collect()
    } else {
        let mut v = dims.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    };
    if !keepdim {
        for &d in &all_dims {
            g = g.unsqueeze(d)?;
        }
    }
    Ok(g)
}
