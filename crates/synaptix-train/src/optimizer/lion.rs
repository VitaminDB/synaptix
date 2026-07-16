use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct LionConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self { lr: 1e-4, betas: (0.9, 0.99), weight_decay: 0.0 }
    }
}

pub struct Lion {
    pub config: LionConfig,
    m: Vec<Tensor>,
}

impl Lion {
    pub fn new(config: LionConfig) -> Self {
        Self { config, m: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(p.zeros_like()?);
            }
        }
        let (beta1, beta2) = self.config.betas;
        let lr = self.config.lr as f32;
        let wd = self.config.weight_decay as f32;
        for i in 0..params.len() {
            let g = &grads[i];
            // update = sign(beta1 * m + (1 - beta1) * g)
            let update = self.m[i].affine(beta1 as f32, 0.0)?
                .add(&g.affine((1.0 - beta1) as f32, 0.0)?)?
                .sign()?;
            // m = beta2 * m + (1 - beta2) * g
            self.m[i] = self.m[i].affine(beta2 as f32, 0.0)?
                .add(&g.affine((1.0 - beta2) as f32, 0.0)?)?;
            // w = w * (1 - lr * wd) - lr * update
            params[i] = params[i].affine(1.0 - lr * wd, 0.0)?
                .sub(&update.mul_scalar(lr)?)?;
        }
        Ok(())
    }
}
