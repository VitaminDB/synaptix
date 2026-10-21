//! AdamW по адаптерам, обрезка градиента по общей норме, расписание шага
//! обучения (линейный разогрев и косинусный спад).

use synaptix_core::backend::train::TrainOp;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::model::TrainModel;
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct AdamW {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u64,
}

impl Default for AdamW {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, step: 0 }
    }
}

fn sq_sum(t: &Tensor) -> Result<f64> {
    Ok(t.sqr()?.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64)
}

/// Общая L2-норма накопленных градиентов адаптеров.
pub fn grad_norm(model: &TrainModel<'_>) -> Result<f32> {
    let mut s = 0f64;
    for l in model.layers() {
        if let Some(g) = &l.ga {
            s += sq_sum(g)?;
        }
        if let Some(g) = &l.gb {
            s += sq_sum(g)?;
        }
    }
    Ok(s.sqrt() as f32)
}

impl AdamW {
    /// Шаг по всем адаптерам с накопленным градиентом; градиент умножается на
    /// `grad_mul` (обрезка нормы). Градиенты после шага сбрасываются.
    pub fn step(&mut self, model: &mut TrainModel<'_>, lr: f32, grad_mul: f32) -> Result<()> {
        self.step += 1;
        let t = self.step as i32;
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);
        let op = TrainOp::AdamW {
            lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            bc1,
            bc2,
        };
        let upd = |p: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
            let g = if grad_mul != 1.0 { g.mul_scalar(grad_mul)? } else { g.clone() };
            let d = p.dims().to_vec();
            let mut o = Tensor::train_op(op, &[p, &g, m, v], &[(d.clone(), DType::F32), (d.clone(), DType::F32), (d, DType::F32)])?;
            let v = o.pop().unwrap();
            let m = o.pop().unwrap();
            let p = o.pop().unwrap();
            Ok((p, m, v))
        };
        for l in model.layers_mut() {
            if let Some(g) = l.ga.take() {
                (l.a, l.ma, l.va) = upd(&l.a, &g, &l.ma, &l.va)?;
            }
            if let Some(g) = l.gb.take() {
                (l.b, l.mb, l.vb) = upd(&l.b, &g, &l.mb, &l.vb)?;
            }
        }
        Ok(())
    }
}

/// Шаг обучения на шаге `step` (с нуля) из `total`: линейный разогрев
/// `warmup` шагов, затем косинус до `min_ratio · lr`.
pub fn lr_at(lr: f32, step: usize, total: usize, warmup: usize, min_ratio: f32) -> f32 {
    if warmup > 0 && step < warmup {
        return lr * (step + 1) as f32 / warmup as f32;
    }
    let decay = total.saturating_sub(warmup).max(1);
    let p = ((step - warmup.min(step)) as f32 / decay as f32).min(1.0);
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
    lr * (min_ratio + (1.0 - min_ratio) * cos)
}
