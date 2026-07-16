use synaptix_core::tensor::Tensor;
use crate::error::Result;

/// AdEMAMix: две EMA градиента — быстрая (β1) и медленная (β3, ≈0.9999) — плюс
/// второй момент (β2). Обновление:
///   `m1 = β1·m1 + (1−β1)·g`, `m2 = β3·m2 + (1−β3)·g`, `v = β2·v + (1−β2)·g²`;
///   `upd = (m1/(1−β1ᵗ) + α·m2) / (√(v/(1−β2ᵗ)) + ε)`;
///   `θ = θ·(1 − lr·wd) − lr·upd`.
pub struct AdemAmixConfig {
    pub lr: f64,
    pub betas: (f64, f64, f64),
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
impl Default for AdemAmixConfig {
    fn default() -> Self {
        Self { lr: 1e-3, betas: (0.9, 0.999, 0.9999), alpha: 5.0, eps: 1e-8, weight_decay: 0.0 }
    }
}

pub struct AdemAmix {
    pub config: AdemAmixConfig,
    pub step: usize,
    m1: Vec<Tensor>,
    m2: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdemAmix {
    pub fn new(config: AdemAmixConfig) -> Self {
        Self { config, step: 0, m1: Vec::new(), m2: Vec::new(), v: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "AdEMAMix: {} params but {} grads",
                params.len(),
                grads.len()
            )));
        }
        self.step += 1;
        let t = self.step as i32;
        if self.m1.is_empty() {
            for p in params.iter() {
                self.m1.push(p.zeros_like()?);
                self.m2.push(p.zeros_like()?);
                self.v.push(p.zeros_like()?);
            }
        }
        let (b1, b2, b3) = self.config.betas;
        let (b1, b2, b3) = (b1 as f32, b2 as f32, b3 as f32);
        let lr = self.config.lr as f32;
        let alpha = self.config.alpha as f32;
        let eps = self.config.eps as f32;
        let wd = self.config.weight_decay as f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        for i in 0..params.len() {
            let g = &grads[i];
            self.m1[i] = self.m1[i].affine(b1, 0.0)?.add(&g.affine(1.0 - b1, 0.0)?)?;
            self.m2[i] = self.m2[i].affine(b3, 0.0)?.add(&g.affine(1.0 - b3, 0.0)?)?;
            self.v[i] = self.v[i].affine(b2, 0.0)?.add(&g.sqr()?.affine(1.0 - b2, 0.0)?)?;

            let m_hat = self.m1[i].mul_scalar(1.0 / bc1)?.add(&self.m2[i].mul_scalar(alpha)?)?;
            let v_hat = self.v[i].mul_scalar(1.0 / bc2)?;
            let upd = m_hat.div(&v_hat.sqrt()?.add_scalar(eps)?)?;
            params[i] = params[i].affine(1.0 - lr * wd, 0.0)?.sub(&upd.mul_scalar(lr)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {}
}
