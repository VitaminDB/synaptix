use synaptix_core::tensor::Tensor;

use crate::compare::tensor_to_f64;
use crate::error::{DebugError, Result};

#[derive(Debug, Clone, Copy, Default)]
pub struct FiniteStats {
    pub numel: usize,
    pub nan_count: usize,
    pub pos_inf_count: usize,
    pub neg_inf_count: usize,
    pub first_nan_at: Option<usize>,
    pub first_inf_at: Option<usize>,
}

impl FiniteStats {
    pub fn is_clean(&self) -> bool {
        self.nan_count == 0 && self.pos_inf_count == 0 && self.neg_inf_count == 0
    }
}

pub fn scan_finite(t: &Tensor) -> Result<FiniteStats> {
    let v = tensor_to_f64(t)?;
    let mut stats = FiniteStats { numel: v.len(), ..Default::default() };
    for (i, x) in v.iter().enumerate() {
        if x.is_nan() {
            stats.nan_count += 1;
            if stats.first_nan_at.is_none() {
                stats.first_nan_at = Some(i);
            }
        } else if *x == f64::INFINITY {
            stats.pos_inf_count += 1;
            if stats.first_inf_at.is_none() {
                stats.first_inf_at = Some(i);
            }
        } else if *x == f64::NEG_INFINITY {
            stats.neg_inf_count += 1;
            if stats.first_inf_at.is_none() {
                stats.first_inf_at = Some(i);
            }
        }
    }
    Ok(stats)
}

pub fn check_finite(t: &Tensor) -> Result<()> {
    let s = scan_finite(t)?;
    if s.is_clean() {
        return Ok(());
    }
    if s.nan_count > 0 {
        return Err(DebugError::NonFinite {
            position: s.first_nan_at.unwrap_or(0),
            kind: "NaN",
        });
    }
    Err(DebugError::NonFinite {
        position: s.first_inf_at.unwrap_or(0),
        kind: "Inf",
    })
}

pub fn nan_inf_hook(label: impl Into<String>) -> impl Fn(&Tensor) -> Option<Tensor> {
    let label = label.into();
    move |t| {
        match scan_finite(t) {
            Ok(s) if !s.is_clean() => {
                eprintln!(
                    "[nan_inf_hook:{label}] NaN={} +Inf={} -Inf={} numel={}",
                    s.nan_count, s.pos_inf_count, s.neg_inf_count, s.numel
                );
            }
            _ => {}
        }
        None
    }
}
