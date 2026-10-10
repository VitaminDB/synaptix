//! CTC-loss (Graves et al., 2006) с градиентом по log-вероятностям.
//!
//! Прямой/обратный проходы считаются на хосте в лог-пространстве (f64) —
//! решётка `T × (2L+1)` мала по сравнению с энкодером. Градиент
//! `∂L/∂lp[t,k] = −Σ_{s: метка(s)=k} α_t(s)·β_t(s) / P` (занятость состояний)
//! подключается к графу собственным `GradFn`; дальше он проходит через
//! log-softmax обычным autograd — итог по логитам `softmax − занятость`.

use std::sync::Arc;

use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::grad::{GradFn, GradMeta};
use synaptix_core::tensor::Tensor;

const NEG: f64 = f64::NEG_INFINITY;

#[inline]
fn lse(a: f64, b: f64) -> f64 {
    if a == NEG {
        return b;
    }
    if b == NEG {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Результат CTC на одном примере (хост).
#[derive(Debug, Clone)]
pub struct CtcHost {
    /// −log P(labels | x); `inf`, если выравнивание невозможно.
    pub nll: f64,
    /// ∂nll/∂lp, `[T, V]` построчно.
    pub grad: Vec<f32>,
}

/// CTC на хосте: `lp` — `[T, V]` log-вероятности построчно.
pub fn ctc_host(lp: &[f32], frames: usize, vocab: usize, labels: &[u32], blank: u32) -> CtcHost {
    let s_len = 2 * labels.len() + 1;
    let ext = |s: usize| if s % 2 == 0 { blank } else { labels[s / 2] };
    let skip = |s: usize| s % 2 == 1 && s >= 3 && labels[s / 2] != labels[s / 2 - 1];
    let at = |t: usize, k: u32| lp[t * vocab + k as usize] as f64;
    let mut grad = vec![0f32; frames * vocab];
    if frames == 0 {
        return CtcHost { nll: f64::INFINITY, grad };
    }
    let mut alpha = vec![NEG; frames * s_len];
    let mut beta = vec![NEG; frames * s_len];
    alpha[0] = at(0, blank);
    if s_len > 1 {
        alpha[1] = at(0, labels[0]);
    }
    for t in 1..frames {
        for s in 0..s_len {
            let mut v = alpha[(t - 1) * s_len + s];
            if s >= 1 {
                v = lse(v, alpha[(t - 1) * s_len + s - 1]);
            }
            if skip(s) {
                v = lse(v, alpha[(t - 1) * s_len + s - 2]);
            }
            if v != NEG {
                alpha[t * s_len + s] = v + at(t, ext(s));
            }
        }
    }
    let last = (frames - 1) * s_len;
    beta[last + s_len - 1] = 0.0;
    if s_len > 1 {
        beta[last + s_len - 2] = 0.0;
    }
    for t in (0..frames - 1).rev() {
        for s in 0..s_len {
            let e = |x: usize| at(t + 1, ext(x)) + beta[(t + 1) * s_len + x];
            let mut v = e(s);
            if s + 1 < s_len {
                v = lse(v, e(s + 1));
            }
            if s + 2 < s_len && skip(s + 2) {
                v = lse(v, e(s + 2));
            }
            beta[t * s_len + s] = v;
        }
    }
    let log_p = if s_len > 1 { lse(alpha[last + s_len - 1], alpha[last + s_len - 2]) } else { alpha[last] };
    if !log_p.is_finite() {
        return CtcHost { nll: f64::INFINITY, grad };
    }
    for t in 0..frames {
        for s in 0..s_len {
            let occ = alpha[t * s_len + s] + beta[t * s_len + s] - log_p;
            if occ > -60.0 {
                grad[t * vocab + ext(s) as usize] -= occ.exp() as f32;
            }
        }
    }
    CtcHost { nll: -log_p, grad }
}

struct CtcGradFn {
    parents: [Tensor; 1],
    /// ∂loss/∂lp той же формы, что вход.
    grad: Tensor,
}

impl GradFn for CtcGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.to_dtype(self.grad.dtype())?.reshape(vec![1usize; self.grad.rank()])?;
        Ok(vec![Some(self.grad.broadcast_mul(&g)?)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "CtcGradFn"
    }
}

/// Как сводить потери батча.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtcReduction {
    /// Сумма по примерам.
    Sum,
    /// Среднее по примерам.
    Mean,
    /// Среднее по примерам, каждая потеря делится на длину разметки (как
    /// `reduction="mean"` в PyTorch).
    MeanPerLabel,
}

/// CTC-loss батча.
///
/// * `log_probs` — `[B, T, V]` (или `[T, V]` для одного примера), log-softmax
///   по словарю; градиент идёт в него.
/// * `input_lengths` — число валидных кадров каждого примера.
/// * `labels` — разметка каждого примера (без blank).
/// * `zero_infinity` — невозможные выравнивания дают 0 вместо `inf`.
///
/// Возвращает скаляр `[1]` с графом.
pub fn ctc_loss(
    log_probs: &Tensor,
    input_lengths: &[usize],
    labels: &[Vec<u32>],
    blank: u32,
    reduction: CtcReduction,
    zero_infinity: bool,
) -> Result<Tensor> {
    let squeeze = log_probs.rank() == 2;
    let lp3 = if squeeze { log_probs.unsqueeze(0)? } else { log_probs.clone() };
    if lp3.rank() != 3 {
        return Err(SynaptixError::Unsupported("ctc_loss: log_probs [B,T,V] или [T,V]"));
    }
    let (b, t_max, vocab) = (lp3.dims()[0], lp3.dims()[1], lp3.dims()[2]);
    if input_lengths.len() != b || labels.len() != b {
        return Err(SynaptixError::Other(format!(
            "ctc_loss: батч {b}, длин {}, разметок {}",
            input_lengths.len(),
            labels.len()
        )));
    }
    let host = lp3.to_dtype(DType::F32)?.to_device(synaptix_core::device::Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut grad = vec![0f32; b * t_max * vocab];
    let mut total = 0f64;
    for i in 0..b {
        let frames = input_lengths[i].min(t_max);
        let lp = &host[i * t_max * vocab..i * t_max * vocab + frames * vocab];
        let r = ctc_host(lp, frames, vocab, &labels[i], blank);
        let (nll, g) = if r.nll.is_finite() {
            (r.nll, r.grad)
        } else if zero_infinity {
            (0.0, vec![0f32; frames * vocab])
        } else {
            (f64::INFINITY, r.grad)
        };
        let w = match reduction {
            CtcReduction::Sum => 1.0,
            CtcReduction::Mean => 1.0 / b as f64,
            CtcReduction::MeanPerLabel => 1.0 / (b as f64 * labels[i].len().max(1) as f64),
        };
        total += nll * w;
        for (dst, src) in grad[i * t_max * vocab..].iter_mut().zip(g.iter()) {
            *dst = src * w as f32;
        }
    }
    let mut grad_t = Tensor::from_vec(grad, vec![b, t_max, vocab], lp3.device())?;
    if squeeze {
        grad_t = grad_t.squeeze(0)?;
    }
    let grad_t = grad_t.to_dtype(log_probs.dtype())?;
    let mut loss = Tensor::from_vec(vec![total as f32], vec![1usize], log_probs.device())?;
    if log_probs.requires_grad() && synaptix_core::grad::is_grad_enabled() {
        loss.set_grad_meta(Some(GradMeta::intermediate(Arc::new(CtcGradFn {
            parents: [log_probs.clone()],
            grad: grad_t,
        }))));
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % 1_000_000) as f64 / 1_000_000.0
    }

    fn log_softmax_rows(x: &[f64], v: usize) -> Vec<f32> {
        x.chunks(v)
            .flat_map(|r| {
                let m = r.iter().cloned().fold(NEG, f64::max);
                let z = r.iter().map(|a| (a - m).exp()).sum::<f64>().ln() + m;
                r.iter().map(move |a| (a - z) as f32).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Градиент по lp совпадает с конечными разностями.
    #[test]
    fn grad_matches_finite_differences() {
        let (t, v) = (9, 5);
        let mut seed = 11;
        let logits: Vec<f64> = (0..t * v).map(|_| rng(&mut seed) * 4.0).collect();
        let lp = log_softmax_rows(&logits, v);
        let labels = [1u32, 2, 2, 3];
        let r = ctc_host(&lp, t, v, &labels, 0);
        assert!(r.nll.is_finite());
        let eps = 1e-3f32;
        for idx in [0usize, 7, 13, 22, 31, 44] {
            let mut p = lp.clone();
            p[idx] += eps;
            let up = ctc_host(&p, t, v, &labels, 0).nll;
            p[idx] -= 2.0 * eps;
            let dn = ctc_host(&p, t, v, &labels, 0).nll;
            let fd = (up - dn) / (2.0 * eps as f64);
            assert!((fd - r.grad[idx] as f64).abs() < 1e-3, "idx {idx}: fd {fd} vs {}", r.grad[idx]);
        }
    }

    /// Сумма занятостей по словарю на каждом кадре = −1 (градиент по lp).
    #[test]
    fn occupancy_sums_to_one() {
        let (t, v) = (12, 6);
        let mut seed = 5;
        let logits: Vec<f64> = (0..t * v).map(|_| rng(&mut seed) * 3.0).collect();
        let lp = log_softmax_rows(&logits, v);
        let r = ctc_host(&lp, t, v, &[3, 1, 4], 0);
        for row in r.grad.chunks(v) {
            let s: f32 = row.iter().sum();
            assert!((s + 1.0).abs() < 1e-4, "{s}");
        }
    }

    #[test]
    fn impossible_alignment_is_infinite() {
        let lp = vec![-1.0f32; 2 * 4];
        assert!(ctc_host(&lp, 2, 4, &[1, 1, 1], 0).nll.is_infinite());
    }
}
