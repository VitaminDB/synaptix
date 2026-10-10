//! Диагональные GMM (`AmDiagGmm`): лог-правдоподобие кадра для pdf.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiagGmms {
    pub dim: usize,
    /// Начало гауссиан pdf `i` — `offsets[i]..offsets[i+1]`.
    pub offsets: Vec<u32>,
    /// `gconst` каждой гауссианы (с log весом).
    pub gconsts: Vec<f32>,
    /// `μ·Σ⁻¹`, `[gauss × dim]`.
    pub means_invvars: Vec<f32>,
    /// `Σ⁻¹`, `[gauss × dim]`.
    pub inv_vars: Vec<f32>,
}

impl DiagGmms {
    pub fn num_pdfs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// log p(x | pdf) = logsumexp_g (gconst_g + μΣ⁻¹_g·x − ½ Σ⁻¹_g·x²).
    pub fn loglike(&self, pdf: usize, x: &[f32]) -> f32 {
        let d = self.dim;
        let (a, b) = (self.offsets[pdf] as usize, self.offsets[pdf + 1] as usize);
        let mut best = f32::NEG_INFINITY;
        let mut vals = [0f32; 256];
        let n = b - a;
        let mut store: Vec<f32>;
        let buf: &mut [f32] = if n <= vals.len() {
            &mut vals[..n]
        } else {
            store = vec![0f32; n];
            &mut store
        };
        for (gi, g) in (a..b).enumerate() {
            let mi = &self.means_invvars[g * d..(g + 1) * d];
            let iv = &self.inv_vars[g * d..(g + 1) * d];
            let mut v = self.gconsts[g];
            for k in 0..d {
                v += mi[k] * x[k] - 0.5 * iv[k] * x[k] * x[k];
            }
            buf[gi] = v;
            best = best.max(v);
        }
        if !best.is_finite() {
            return best;
        }
        best + buf.iter().map(|v| (v - best).exp()).sum::<f32>().ln()
    }
}
