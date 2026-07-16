use synaptix_core::tensor::Tensor;
use crate::error::Result;

/// Sophia (упрощённый: диагональная кривизна через эмпирический Fisher `g²`
/// вместо отдельной оценки Hessian). Обновление с per-координатным клиппингом:
///   `m = β1·m + (1−β1)·g`, `h = β2·h + (1−β2)·g²`;
///   `upd = clamp(m / (ρ·h + ε), −1, 1)`;  `θ = θ·(1 − lr·wd) − lr·upd`.
pub struct SophiaConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
impl Default for SophiaConfig {
    fn default() -> Self {
        Self { lr: 1e-4, betas: (0.96, 0.99), rho: 0.04, eps: 1e-12, weight_decay: 0.0 }
    }
}

pub struct Sophia {
    pub config: SophiaConfig,
    pub step: usize,
    m: Vec<Tensor>,
    h: Vec<Tensor>,
}

impl Sophia {
    pub fn new(config: SophiaConfig) -> Self {
        Self { config, step: 0, m: Vec::new(), h: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "Sophia: {} params but {} grads",
                params.len(),
                grads.len()
            )));
        }
        self.step += 1;
        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(p.zeros_like()?);
                self.h.push(p.zeros_like()?);
            }
        }
        let (b1, b2) = self.config.betas;
        let (b1, b2) = (b1 as f32, b2 as f32);
        let lr = self.config.lr as f32;
        let rho = self.config.rho as f32;
        let eps = self.config.eps as f32;
        let wd = self.config.weight_decay as f32;

        for i in 0..params.len() {
            let g = &grads[i];
            self.m[i] = self.m[i].affine(b1, 0.0)?.add(&g.affine(1.0 - b1, 0.0)?)?;
            self.h[i] = self.h[i].affine(b2, 0.0)?.add(&g.sqr()?.affine(1.0 - b2, 0.0)?)?;
            let denom = self.h[i].mul_scalar(rho)?.add_scalar(eps)?;
            let upd = self.m[i].div(&denom)?.clamp(-1.0, 1.0)?;
            params[i] = params[i].affine(1.0 - lr * wd, 0.0)?.sub(&upd.mul_scalar(lr)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {}
}
