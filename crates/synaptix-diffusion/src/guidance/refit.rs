use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};

pub fn cfg_rescale(guided: &Tensor, x0_cfg: &Tensor, x0_cond: &Tensor, phi: f32) -> Result<Tensor> {
    if phi == 0.0 {
        return Ok(guided.clone());
    }
    let guided_flat: Vec<f32> = guided.flatten_all()?.to_vec1()?;
    let x0_cfg_flat: Vec<f32> = x0_cfg.flatten_all()?.to_vec1()?;
    let x0_cond_flat: Vec<f32> = x0_cond.flatten_all()?.to_vec1()?;
    let std_cfg = std_dev(&x0_cfg_flat);
    let std_cond = std_dev(&x0_cond_flat);
    let rescale = (std_cond / std_cfg.max(1e-12)).min(10.0);
    let rescaled: Vec<f32> = guided_flat.iter().map(|v| v * rescale).collect();
    let target: Vec<f32> = guided_flat.iter().zip(&rescaled)
        .map(|(g, r)| g + phi * (r - g))
        .collect();
    Tensor::from_vec(target, guided.dims().to_vec(), guided.device()).map_err(DiffusionError::from)
}

fn std_dev(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    var.sqrt()
}
