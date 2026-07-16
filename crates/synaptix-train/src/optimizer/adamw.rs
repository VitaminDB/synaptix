use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct AdamWConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self { lr: 1e-4, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.01 }
    }
}

pub struct AdamW {
    pub config: AdamWConfig,
    pub step: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(config: AdamWConfig) -> Self {
        Self { config, step: 0, m: Vec::new(), v: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "AdamW: {} params but {} grads",
                params.len(),
                grads.len()
            )));
        }
        self.step += 1;
        let t = self.step;

        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(p.zeros_like()?);
                self.v.push(p.zeros_like()?);
            }
        }

        let (beta1, beta2) = self.config.betas;
        let lr = self.config.lr as f32;
        let eps = self.config.eps as f32;
        let wd = self.config.weight_decay as f32;
        let bc1 = 1.0 - beta1.powi(t as i32) as f32;
        let bc2 = 1.0 - beta2.powi(t as i32) as f32;

        for i in 0..params.len() {
            let g = &grads[i];
            self.m[i] = self.m[i].affine(beta1 as f32, 0.0)?
                .add(&g.affine((1.0 - beta1) as f32, 0.0)?)?;
            self.v[i] = self.v[i].affine(beta2 as f32, 0.0)?
                .add(&g.sqr()?.affine((1.0 - beta2) as f32, 0.0)?)?;
            let m_hat = self.m[i].mul_scalar(1.0 / bc1)?;
            let v_hat = self.v[i].mul_scalar(1.0 / bc2)?;
            let update = m_hat.div(&v_hat.sqrt()?.add_scalar(eps)?)?;
            params[i] = params[i].affine(1.0 - lr * wd, 0.0)?
                .sub(&update.mul_scalar(lr)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {}
}
