use synaptix_core::{error::SynaptixError, tensor::Tensor};

type R<T> = Result<T, SynaptixError>;

#[derive(Debug, Clone, Copy)]
pub struct GuiderParams {
    pub cfg_scale: f32,
    pub rescale: f32,
    pub skip_steps: usize,
}

impl GuiderParams {
    pub fn positive_only() -> Self {
        Self { cfg_scale: 1.0, rescale: 0.0, skip_steps: 0 }
    }

    pub fn cfg(scale: f32) -> Self {
        Self { cfg_scale: scale, rescale: 0.0, skip_steps: 0 }
    }

    pub fn needs_uncond(&self, step: usize) -> bool {
        self.cfg_scale > 1.0 && step >= self.skip_steps
    }
}

pub fn apply_cfg(cond: &Tensor, uncond: &Tensor, scale: f32) -> R<Tensor> {
    if scale == 1.0 {
        return Ok(cond.clone());
    }
    let diff = cond.sub(uncond)?;
    uncond.add(&diff.mul_scalar(scale)?)
}

pub fn rescale_to(pred: &Tensor, reference: &Tensor, factor: f32) -> R<Tensor> {
    if factor <= 0.0 {
        return Ok(pred.clone());
    }
    let std_p = tensor_std(pred)?;
    let std_r = tensor_std(reference)?;
    if std_p <= 0.0 {
        return Ok(pred.clone());
    }
    let scaled = pred.mul_scalar(std_r / std_p)?;
    let a = scaled.mul_scalar(factor)?;
    let b = pred.mul_scalar(1.0 - factor)?;
    a.add(&b)
}

fn tensor_std(t: &Tensor) -> R<f32> {
    let host = t
        .to_device(synaptix_core::device::Device::Cpu)?
        .to_dtype(synaptix_core::dtype::DType::F32)?;
    let v = host.to_vec1::<f32>()?;
    if v.is_empty() {
        return Ok(0.0);
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    Ok(var.sqrt())
}
