use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Auxiliary load-balance loss (Switch Transformer):
///   `loss = E · Σ_e f_e · P_e`, где `f_e` — доля токенов, направленных к эксперту e
/// (по top-1 `expert_indices`), `P_e` — средняя router-вероятность эксперта e.
/// `router_probs:[N,E]`, `expert_indices:[N]`. Возвращает скаляр `[1]`.
pub fn auxiliary_loss(router_probs: &Tensor, expert_indices: &Tensor) -> Result<Tensor> {
    if router_probs.rank() != 2 {
        return Err(SynaptixError::Unsupported("auxiliary_loss: router_probs must be [N,E]"));
    }
    let (n, e) = (router_probs.dims()[0], router_probs.dims()[1]);
    let probs = f32v(router_probs)?;
    let idx = f32v(expert_indices)?;
    if idx.len() != n {
        return Err(SynaptixError::Unsupported("auxiliary_loss: expert_indices.len() != N"));
    }
    let mut f = vec![0.0f32; e];
    let mut p = vec![0.0f32; e];
    for i in 0..n {
        let ei = idx[i].round() as usize;
        if ei < e {
            f[ei] += 1.0;
        }
        for j in 0..e {
            p[j] += probs[i * e + j];
        }
    }
    let inv_n = 1.0 / n as f32;
    let mut loss = 0.0f32;
    for j in 0..e {
        loss += (f[j] * inv_n) * (p[j] * inv_n);
    }
    loss *= e as f32;
    Tensor::from_vec::<_, f32>(vec![loss], vec![1], router_probs.device())?
        .to_dtype(router_probs.dtype())
}

/// Router z-loss (стабилизация логитов): `loss = mean_n( logsumexp_e(logits[n,:])² )`.
/// `router_logits:[N,E]`. Возвращает скаляр `[1]`.
pub fn z_loss(router_logits: &Tensor) -> Result<Tensor> {
    if router_logits.rank() != 2 {
        return Err(SynaptixError::Unsupported("z_loss: router_logits must be [N,E]"));
    }
    let (n, e) = (router_logits.dims()[0], router_logits.dims()[1]);
    let l = f32v(router_logits)?;
    let mut acc = 0.0f32;
    for i in 0..n {
        let row = &l[i * e..i * e + e];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        let lse = max + sum.ln();
        acc += lse * lse;
    }
    let loss = acc / n as f32;
    Tensor::from_vec::<_, f32>(vec![loss], vec![1], router_logits.device())?
        .to_dtype(router_logits.dtype())
}
