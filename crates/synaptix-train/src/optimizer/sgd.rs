use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self { lr: 0.01, momentum: 0.9, weight_decay: 0.0, nesterov: false }
    }
}

pub struct Sgd {
    pub config: SgdConfig,
    velocities: Vec<Tensor>,
}

impl Sgd {
    pub fn new(config: SgdConfig) -> Self {
        Self { config, velocities: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if self.velocities.is_empty() {
            for p in params.iter() {
                self.velocities.push(p.zeros_like()?);
            }
        }
        let lr = self.config.lr as f32;
        let mu = self.config.momentum as f32;
        let wd = self.config.weight_decay as f32;
        for i in 0..params.len() {
            let g = if wd > 0.0 {
                grads[i].add(&params[i].mul_scalar(wd)?)?
            } else {
                grads[i].clone()
            };
            if mu > 0.0 {
                self.velocities[i] = self.velocities[i].affine(mu, 0.0)?.add(&g)?;
                params[i] = if self.config.nesterov {
                    params[i].sub(&g.add(&self.velocities[i].mul_scalar(mu)?)?.mul_scalar(lr)?)?
                } else {
                    params[i].sub(&self.velocities[i].mul_scalar(lr)?)?
                };
            } else {
                params[i] = params[i].sub(&g.mul_scalar(lr)?)?;
            }
        }
        Ok(())
    }
}
