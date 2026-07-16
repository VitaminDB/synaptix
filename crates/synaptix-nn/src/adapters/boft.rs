use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::adapters::oft::{cayley_orthogonal_cpu, invert_square_cpu};
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

/// Butterfly Orthogonal Fine-Tuning (упрощённый block-diagonal вариант).
///
/// Эффективная матрица — `W_eff = R · W`, где `R` — блочно-диагональная
/// ортогональная матрица из `num_blocks` блоков размера `block_size`. Каждый
/// блок строится Cayley-преобразованием от skew-symmetric `Q_i = q_i − q_iᵀ`.
/// Полный `R` собирается на CPU при загрузке весов (`assemble_block_diag_r_cpu`).
pub struct BoftLinear {
    pub base: Linear,
    pub r_matrix: Parameter,
    pub num_blocks: usize,
    pub block_size: usize,
}

impl BoftLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_blocks: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        if out_features % num_blocks != 0 {
            return Err(SynaptixError::Unsupported(
                "BOFT: out_features must be divisible by num_blocks",
            ));
        }
        let block_size = out_features / num_blocks;
        let identity = identity_tensor(out_features, dtype, device)?;
        Ok(Self {
            base: Linear::from_init(
                in_features, out_features, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            r_matrix: Parameter::new(identity),
            num_blocks,
            block_size,
        })
    }

    pub fn from_weights(base_w: Tensor, r_matrix: Tensor, num_blocks: usize) -> Result<Self> {
        if r_matrix.rank() != 2 || r_matrix.dims()[0] != r_matrix.dims()[1] {
            return Err(SynaptixError::Unsupported("BOFT: R must be a square 2D matrix"));
        }
        if r_matrix.dims()[0] != base_w.dims()[0] {
            return Err(SynaptixError::shape_mismatch(base_w.dims(), r_matrix.dims()));
        }
        if r_matrix.dims()[0] % num_blocks != 0 {
            return Err(SynaptixError::Unsupported("BOFT: out_features must be divisible by num_blocks"));
        }
        let block_size = r_matrix.dims()[0] / num_blocks;
        Ok(Self {
            base: Linear::new(base_w, None)?,
            r_matrix: Parameter::new(r_matrix),
            num_blocks,
            block_size,
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

/// Сборка block-diagonal ортогональной матрицы из `num_blocks` skew-raw блоков
/// `q_blocks[k]` формы `[block_size, block_size]`. CPU-only, F32.
pub fn assemble_block_diag_r_cpu(q_blocks: &[Tensor]) -> Result<Tensor> {
    if q_blocks.is_empty() {
        return Err(SynaptixError::Unsupported("BOFT: empty block list"));
    }
    let first = &q_blocks[0];
    if first.rank() != 2 || first.dims()[0] != first.dims()[1] {
        return Err(SynaptixError::Unsupported("BOFT: each block must be square 2D"));
    }
    let block_size = first.dims()[0];
    let dtype = first.dtype();
    let device = first.device();
    for q in q_blocks.iter().skip(1) {
        if q.dims() != first.dims() {
            return Err(SynaptixError::shape_mismatch(first.dims(), q.dims()));
        }
        if q.dtype() != dtype || q.device() != device {
            return Err(SynaptixError::Unsupported("BOFT: blocks must share dtype/device"));
        }
    }
    let num_blocks = q_blocks.len();
    let n = num_blocks * block_size;
    let mut full = vec![0.0f32; n * n];
    for (k, q_raw) in q_blocks.iter().enumerate() {
        let r_k = cayley_orthogonal_cpu(q_raw)?.to_dtype(synaptix_core::dtype::DType::F32)?;
        let r_v = r_k.to_vec2::<f32>()?;
        let offset = k * block_size;
        for i in 0..block_size {
            for j in 0..block_size {
                full[(offset + i) * n + (offset + j)] = r_v[i][j];
            }
        }
    }
    Tensor::from_vec(full, (n, n), device)?.to_dtype(dtype)
}

/// Re-export для удобства использования инверсии в тестах.
#[doc(hidden)]
pub fn _invert_square_cpu(m: &[f32], n: usize) -> Result<Vec<f32>> {
    invert_square_cpu(m, n)
}
