use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct OrpoConfig {
    pub lr: f64,
    pub lambda: f64,
}
impl Default for OrpoConfig {
    fn default() -> Self {
        Self { lr: 1e-5, lambda: 0.1 }
    }
}

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

fn neg_logsigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        (1.0 + (-x).exp()).ln()
    } else {
        -x + (1.0 + x.exp()).ln()
    }
}

/// ORPO loss = SFT NLL + λ·odds-ratio. Входы — per-пример sequence log-probs `[N]`
/// (значения ≤ 0):
///   `log_odds = (lp_c − log(1−e^{lp_c})) − (lp_r − log(1−e^{lp_r}))`;
///   `loss = mean(−lp_c) + λ·mean(−log σ(log_odds))`. Возвращает скаляр `[1]`.
pub fn compute_loss(
    chosen_logps: &Tensor,
    rejected_logps: &Tensor,
    lambda: f32,
) -> Result<Tensor> {
    let c = f32v(chosen_logps)?;
    let r = f32v(rejected_logps)?;
    let n = c.len();
    let mut sft = 0.0f32;
    let mut or_loss = 0.0f32;
    for i in 0..n {
        sft += -c[i];
        let log_odds = (c[i] - (1.0 - c[i].exp()).ln()) - (r[i] - (1.0 - r[i].exp()).ln());
        or_loss += neg_logsigmoid(log_odds);
    }
    let loss = sft / n as f32 + lambda * (or_loss / n as f32);
    Ok(Tensor::from_vec::<_, f32>(vec![loss], vec![1], chosen_logps.device())?)
}
