use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct KtoConfig {
    pub lr: f64,
    pub beta: f64,
}
impl Default for KtoConfig {
    fn default() -> Self {
        Self { lr: 1e-5, beta: 0.1 }
    }
}

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// KTO loss (упрощённый). Логратио `lr = policy − ref` для желательных (chosen)
/// и нежелательных (rejected) примеров; KL-член = `max(0, mean(all lr))`.
///   `loss_c = 1 − σ(β·(lr_c − KL))`,  `loss_r = 1 − σ(β·(KL − lr_r))`;
///   `loss = mean([loss_c, loss_r])`. Возвращает скаляр `[1]`.
pub fn compute_loss(
    policy_chosen: &Tensor,
    ref_chosen: &Tensor,
    policy_rejected: &Tensor,
    ref_rejected: &Tensor,
    beta: f32,
) -> Result<Tensor> {
    let pc = f32v(policy_chosen)?;
    let rc = f32v(ref_chosen)?;
    let pr = f32v(policy_rejected)?;
    let rr = f32v(ref_rejected)?;
    let nc = pc.len();
    let nr = pr.len();

    let chosen_lr: Vec<f32> = (0..nc).map(|i| pc[i] - rc[i]).collect();
    let rejected_lr: Vec<f32> = (0..nr).map(|i| pr[i] - rr[i]).collect();

    // KL-член: clamp(mean(all logratios), 0, +inf)
    let kl_sum: f32 = chosen_lr.iter().sum::<f32>() + rejected_lr.iter().sum::<f32>();
    let kl = (kl_sum / (nc + nr) as f32).max(0.0);

    let mut acc = 0.0f32;
    for &lr in &chosen_lr {
        acc += 1.0 - sigmoid(beta * (lr - kl));
    }
    for &lr in &rejected_lr {
        acc += 1.0 - sigmoid(beta * (kl - lr));
    }
    let loss = acc / (nc + nr) as f32;
    Ok(Tensor::from_vec::<_, f32>(vec![loss], vec![1], policy_chosen.device())?)
}
