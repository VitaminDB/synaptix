use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

/// Orthogonal Fine-Tuning.
///
/// Эффективная матрица — `W_eff = R · W`, где `R` — ортогональная матрица,
/// полученная Cayley-преобразованием от skew-symmetric `Q`:
/// `R = (I + Q) · (I − Q)^{−1}`.
///
/// Подсчёт `R` требует matrix inverse и происходит один раз при загрузке весов
/// (`cayley_orthogonal_cpu`) — runtime forward работает уже с готовым `R`.
pub struct OftLinear {
    pub base: Linear,
    pub r_matrix: Parameter,
}

impl OftLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let identity = identity_tensor(out_features, dtype, device)?;
        Ok(Self {
            base: Linear::from_init(
                in_features, out_features, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            r_matrix: Parameter::new(identity),
        })
    }

    pub fn from_weights(base_w: Tensor, r_matrix: Tensor) -> Result<Self> {
        if r_matrix.rank() != 2 || r_matrix.dims()[0] != r_matrix.dims()[1] {
            return Err(SynaptixError::Unsupported("OFT: R must be a square 2D matrix"));
        }
        if r_matrix.dims()[0] != base_w.dims()[0] {
            return Err(SynaptixError::shape_mismatch(base_w.dims(), r_matrix.dims()));
        }
        Ok(Self {
            base: Linear::new(base_w, None)?,
            r_matrix: Parameter::new(r_matrix),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.r_matrix.tensor();
        let w = self.base.weight();
        let w_eff = r.matmul(&w)?;
        let w_eff_t = w_eff.transpose(0, 1)?.contiguous()?;
        x.matmul(&w_eff_t)
    }
}

fn identity_tensor(n: usize, dtype: DType, device: Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::from_slice(&data, &[n, n], device)?.to_dtype(dtype)
}

/// Cayley `R = (I + Q)(I − Q)^{−1}` для skew-symmetric `Q = q_raw − q_rawᵀ`.
/// CPU-only, F32 — вызывается один раз при загрузке весов.
pub fn cayley_orthogonal_cpu(q_raw: &Tensor) -> Result<Tensor> {
    if !q_raw.device().is_cpu() {
        return Err(SynaptixError::Unsupported("cayley_orthogonal_cpu: CPU only"));
    }
    if q_raw.rank() != 2 || q_raw.dims()[0] != q_raw.dims()[1] {
        return Err(SynaptixError::Unsupported("cayley_orthogonal_cpu: input must be square 2D"));
    }
    let n = q_raw.dims()[0];
    let q_raw_f32 = q_raw.to_dtype(DType::F32)?.contiguous()?;
    let q_raw_v = q_raw_f32.to_vec2::<f32>()?;

    let mut q = vec![0.0f32; n * n];
    let mut i_minus_q = vec![0.0f32; n * n];
    let mut i_plus_q = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let s = q_raw_v[i][j] - q_raw_v[j][i];
            q[i * n + j] = s;
        }
    }
    for i in 0..n {
        for j in 0..n {
            i_minus_q[i * n + j] = if i == j { 1.0 } else { 0.0 } - q[i * n + j];
            i_plus_q[i * n + j] = if i == j { 1.0 } else { 0.0 } + q[i * n + j];
        }
    }
    let inv = invert_square_cpu(&i_minus_q, n)?;
    let mut r = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += i_plus_q[i * n + k] * inv[k * n + j];
            }
            r[i * n + j] = acc;
        }
    }
    Tensor::from_vec(r, (n, n), q_raw.device())?.to_dtype(q_raw.dtype())
}

/// Gauss-Jordan elimination с partial pivoting; row-major `n×n`.
pub(crate) fn invert_square_cpu(m: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut a = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = m[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * 2 * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return Err(SynaptixError::Unsupported("invert_square_cpu: singular matrix"));
        }
        if pivot != col {
            for j in 0..(2 * n) {
                a.swap(col * 2 * n + j, pivot * 2 * n + j);
            }
        }
        let pv = a[col * 2 * n + col];
        for j in 0..(2 * n) {
            a[col * 2 * n + j] /= pv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                a[row * 2 * n + j] -= factor * a[col * 2 * n + j];
            }
        }
    }
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}
