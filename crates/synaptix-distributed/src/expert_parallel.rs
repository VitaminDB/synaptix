//! Expert parallelism (MoE): токены маршрутизируются по экспертам. Локальная
//! математика диспетчеризации — группировка строк `x` по `expert_ids`.
//! Реальное распределение экспертов по устройствам/рангам — поверх этого.

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};

/// Сгруппировать строки `x` (`[n_tokens, dim]`) по назначенному эксперту.
/// Возвращает `num_experts` тензоров; эксперт без токенов получает `[0, dim]`.
pub fn dispatch_tokens(x: &Tensor, expert_ids: &[u32], num_experts: usize) -> Result<Vec<Tensor>> {
    let (n_tokens, dim) = x.dims2().map_err(DistError::Core)?;
    if expert_ids.len() != n_tokens {
        return Err(DistError::Other(format!(
            "dispatch_tokens: {} expert_ids != {} tokens",
            expert_ids.len(),
            n_tokens
        )));
    }
    // Индексы строк на каждого эксперта (стабильный порядок токенов).
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
    for (i, &e) in expert_ids.iter().enumerate() {
        if (e as usize) < num_experts {
            buckets[e as usize].push(i);
        }
    }
    let mut out = Vec::with_capacity(num_experts);
    for rows in buckets {
        if rows.is_empty() {
            out.push(Tensor::zeros(vec![0, dim], x.dtype(), x.device()).map_err(DistError::Core)?);
            continue;
        }
        let row_tensors: Vec<Tensor> = rows
            .iter()
            .map(|&i| x.narrow(0, i, 1).and_then(|t| t.contiguous()))
            .collect::<std::result::Result<_, _>>()
            .map_err(DistError::Core)?;
        let refs: Vec<&Tensor> = row_tensors.iter().collect();
        out.push(Tensor::cat(&refs, 0).map_err(DistError::Core)?);
    }
    Ok(out)
}
