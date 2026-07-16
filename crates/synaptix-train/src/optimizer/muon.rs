use synaptix_core::tensor::Tensor;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,
    pub nesterov: bool,
    pub ns_steps: usize,
    pub weight_decay: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
            weight_decay: 0.0,
        }
    }
}

pub struct Muon {
    pub config: MuonConfig,
    momentum_bufs: Vec<Tensor>,
}

impl Muon {
    pub fn new(config: MuonConfig) -> Self {
        Self { config, momentum_bufs: Vec::new() }
    }

    pub fn step_params(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::TrainError::Other(format!(
                "Muon: {} params but {} grads", params.len(), grads.len()
            )));
        }
        if self.momentum_bufs.is_empty() {
            for p in params.iter() {
                self.momentum_bufs.push(p.zeros_like()?);
            }
        }
        let momentum = self.config.momentum as f32;
        let lr = self.config.lr as f32;
        let wd = self.config.weight_decay as f32;

        for i in 0..params.len() {
            let g = &grads[i];
            let buf = self.momentum_bufs[i].affine(momentum, 0.0)?.add(g)?;
            self.momentum_bufs[i] = buf.clone();
            let update_input = if self.config.nesterov {
                g.add(&buf.affine(momentum, 0.0)?)?
            } else {
                buf
            };

            let update = if update_input.rank() == 2 {
                newton_schulz_zeropower(&update_input, self.config.ns_steps)?
            } else {
                update_input
            };

            let mut p_new = params[i].sub(&update.affine(lr, 0.0)?)?;
            if wd > 0.0 {
                p_new = p_new.affine(1.0 - lr * wd, 0.0)?;
            }
            params[i] = p_new;
        }
        Ok(())
    }
}

pub fn newton_schulz_zeropower(g: &Tensor, steps: usize) -> Result<Tensor> {
    let dims = g.dims();
    if dims.len() != 2 {
        return Err(crate::error::TrainError::Other("NS: rank-2 required".into()));
    }
    let transposed = dims[0] > dims[1];
    let x_t = if transposed { g.transpose(0, 1)?.contiguous()? } else { g.clone() };

    let sq = x_t.sqr()?;
    let frob = sq.flatten_all()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-7);
    let mut x = x_t.affine(1.0 / frob, 0.0)?;

    let a = 3.4445_f32;
    let b = -4.7750_f32;
    let c = 2.0315_f32;

    for _ in 0..steps {
        let xt = x.transpose(0, 1)?.contiguous()?;
        let a_mat = x.matmul(&xt)?;
        let b_mat = a_mat.matmul(&a_mat)?;
        let part = a_mat.affine(b, 0.0)?.add(&b_mat.affine(c, 0.0)?)?;
        let inner = part.matmul(&x)?;
        x = x.affine(a, 0.0)?.add(&inner)?;
    }

    let out = if transposed {
        x.transpose(0, 1)?.contiguous()?
    } else {
        x
    };
    Ok(out)
}
