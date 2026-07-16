use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct DpoConfig {
    pub lr: f64,
    pub beta: f64,
}
impl Default for DpoConfig {
    fn default() -> Self {
        Self { lr: 1e-5, beta: 0.1 }
    }
}

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

/// `−log σ(x)` численно устойчиво (= softplus(−x)).
fn neg_logsigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        (1.0 + (-x).exp()).ln()
    } else {
        -x + (1.0 + x.exp()).ln()
    }
}

/// DPO loss. Входы — per-пример sequence log-probs `[N]`:
///   `logits = β·((policy_chosen − policy_rejected) − (ref_chosen − ref_rejected))`;
///   `loss = mean(−log σ(logits))`. Возвращает скаляр `[1]`.
pub fn compute_loss(
    policy_chosen: &Tensor,
    policy_rejected: &Tensor,
    ref_chosen: &Tensor,
    ref_rejected: &Tensor,
    beta: f32,
) -> Result<Tensor> {
    let pc = f32v(policy_chosen)?;
    let pr = f32v(policy_rejected)?;
    let rc = f32v(ref_chosen)?;
    let rr = f32v(ref_rejected)?;
    let n = pc.len();
    let mut acc = 0.0f32;
    for i in 0..n {
        let logit = beta * ((pc[i] - pr[i]) - (rc[i] - rr[i]));
        acc += neg_logsigmoid(logit);
    }
    let loss = acc / n as f32;
    Ok(Tensor::from_vec::<_, f32>(vec![loss], vec![1], policy_chosen.device())?)
}
