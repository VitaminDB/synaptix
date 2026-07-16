use synaptix_core::tensor::Tensor;
use crate::error::Result;

/// Adam8bit (bitsandbytes-стиль). Числовая семантика обновления — обычный Adam
/// (decoupled weight decay): `m = β1·m + (1−β1)·g`, `v = β2·v + (1−β2)·g²`,
/// `θ = θ·(1 − lr·wd) − lr·m̂/(√v̂ + ε)`. 8-битное блочное квантование состояний
/// (m, v) — это оптимизация памяти на GPU; на CPU-пути состояния держатся в f32,
/// поэтому результат совпадает с эталонным Adam.
pub struct Adam8bitConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
}
impl Default for Adam8bitConfig {
    fn default() -> Self {
        Self { lr: 1e-3, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0 }
    }
}

pub struct Adam8bit {
    pub config: Adam8bitConfig,
    pub step: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Adam8bit {
    pub fn new(config: Adam8bitConfig) -> Self {
        Self { config, step: 0, m: Vec::new(), v: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "Adam8bit: {} params but {} grads",
                params.len(),
                grads.len()
            )));
        }
        self.step += 1;
        let t = self.step as i32;
        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(p.zeros_like()?);
                self.v.push(p.zeros_like()?);
            }
        }
        let (b1, b2) = self.config.betas;
        let (b1, b2) = (b1 as f32, b2 as f32);
        let lr = self.config.lr as f32;
        let eps = self.config.eps as f32;
        let wd = self.config.weight_decay as f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        for i in 0..params.len() {
            let g = &grads[i];
            self.m[i] = self.m[i].affine(b1, 0.0)?.add(&g.affine(1.0 - b1, 0.0)?)?;
            self.v[i] = self.v[i].affine(b2, 0.0)?.add(&g.sqr()?.affine(1.0 - b2, 0.0)?)?;
            let m_hat = self.m[i].mul_scalar(1.0 / bc1)?;
            let v_hat = self.v[i].mul_scalar(1.0 / bc2)?;
            let upd = m_hat.div(&v_hat.sqrt()?.add_scalar(eps)?)?;
            params[i] = params[i].affine(1.0 - lr * wd, 0.0)?.sub(&upd.mul_scalar(lr)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {}
}
