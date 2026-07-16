use synaptix_core::tensor::Tensor;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct AdafactorConfig {
    pub lr: f64,
    pub eps1: f64,
    pub clip_threshold: f64,
    pub decay_rate: f64,
    pub beta1: Option<f64>,
    pub weight_decay: f64,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            eps1: 1e-30,
            clip_threshold: 1.0,
            decay_rate: -0.8,
            beta1: None,
            weight_decay: 0.0,
        }
    }
}

pub struct Adafactor {
    pub config: AdafactorConfig,
    pub step: usize,
    rs: Vec<Option<Tensor>>,
    cs: Vec<Option<Tensor>>,
    vs: Vec<Option<Tensor>>,
    ms: Vec<Option<Tensor>>,
}

impl Adafactor {
    pub fn new(config: AdafactorConfig) -> Self {
        Self { config, step: 0, rs: Vec::new(), cs: Vec::new(), vs: Vec::new(), ms: Vec::new() }
    }

    fn init_state(&mut self, params: &[Tensor]) -> Result<()> {
        if !self.rs.is_empty() {
            return Ok(());
        }
        for p in params {
            if p.rank() >= 2 {
                let dims = p.dims();
                let nrows = dims[0];
                let ncols: usize = dims[1..].iter().product();
                self.rs.push(Some(Tensor::zeros(vec![nrows], p.dtype(), p.device())?));
                self.cs.push(Some(Tensor::zeros(vec![ncols], p.dtype(), p.device())?));
                self.vs.push(None);
            } else {
                self.rs.push(None);
                self.cs.push(None);
                self.vs.push(Some(p.zeros_like()?));
            }
            self.ms.push(if self.config.beta1.is_some() { Some(p.zeros_like()?) } else { None });
        }
        Ok(())
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "Adafactor: {} params but {} grads", params.len(), grads.len()
            )));
        }
        self.init_state(params)?;
        self.step += 1;
        let t = self.step;

        let beta2_t = 1.0 - (t as f64).powf(self.config.decay_rate);
        let beta2_t_f32 = beta2_t as f32;
        let one_minus_beta2 = (1.0 - beta2_t) as f32;
        let lr = self.config.lr as f32;
        let eps1 = self.config.eps1 as f32;
        let wd = self.config.weight_decay as f32;

        for i in 0..params.len() {
            let g = &grads[i];
            let g_sq = g.sqr()?.add_scalar(eps1)?;

            let update = if g.rank() >= 2 {
                let g_sq_2d = if g.rank() == 2 {
                    g_sq.clone()
                } else {
                    let dims = g.dims();
                    let nrows = dims[0];
                    let ncols: usize = dims[1..].iter().product();
                    g_sq.reshape(vec![nrows, ncols])?
                };
                let r_old = self.rs[i].as_ref().unwrap();
                let c_old = self.cs[i].as_ref().unwrap();
                let r_new = r_old.affine(beta2_t_f32, 0.0)?
                    .add(&g_sq_2d.mean_keepdim(1)?.squeeze(1)?.affine(one_minus_beta2, 0.0)?)?;
                let c_new = c_old.affine(beta2_t_f32, 0.0)?
                    .add(&g_sq_2d.mean_keepdim(0)?.squeeze(0)?.affine(one_minus_beta2, 0.0)?)?;
                self.rs[i] = Some(r_new.clone());
                self.cs[i] = Some(c_new.clone());

                let r_sum = r_new.flatten_all()?.sum_all()?.to_scalar::<f32>()?.max(eps1);
                let r_norm = r_new.affine(1.0 / r_sum, 0.0)?;
                let r_col = r_norm.unsqueeze(1)?;
                let c_row = c_new.unsqueeze(0)?;
                let v_2d = r_col.broadcast_mul(&c_row)?;
                let denom = v_2d.sqrt()?;
                let denom_full = if g.rank() == 2 {
                    denom
                } else {
                    denom.reshape(g.dims().to_vec())?
                };
                g.div(&denom_full)?
            } else {
                let v = self.vs[i].as_ref().unwrap();
                let v_new = v.affine(beta2_t_f32, 0.0)?
                    .add(&g_sq.affine(one_minus_beta2, 0.0)?)?;
                self.vs[i] = Some(v_new.clone());
                g.div(&v_new.sqrt()?)?
            };

            let n = update.shape().numel().max(1) as f32;
            let upd_rms = (update.sqr()?.flatten_all()?.sum_all()?.to_scalar::<f32>()? / n).sqrt();
            let clip_scale = 1.0_f32 / (upd_rms / self.config.clip_threshold as f32).max(1.0);
            let clipped = update.affine(clip_scale, 0.0)?;

            let final_update = if let Some(beta1) = self.config.beta1 {
                let m = self.ms[i].as_ref().unwrap();
                let m_new = m.affine(beta1 as f32, 0.0)?.add(&clipped.affine((1.0 - beta1) as f32, 0.0)?)?;
                self.ms[i] = Some(m_new.clone());
                m_new
            } else {
                clipped
            };

            let mut p_new = params[i].sub(&final_update.affine(lr, 0.0)?)?;
            if wd > 0.0 {
                p_new = p_new.affine(1.0 - lr * wd, 0.0)?;
            }
            params[i] = p_new;
        }
        Ok(())
    }
}
