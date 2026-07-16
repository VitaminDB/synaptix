use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::parameter::Parameter;

/// Residual VQ — стек `num_codebooks` codebook'ов, каждый `[codebook_size, dim]`.
/// Encode: для каждого codebook ищем L2-ближайший embedding к residual,
/// сохраняем index, вычитаем embedding из residual; на следующем codebook
/// работает обновлённый residual. Decode: сумма embeddings по индексам.
/// Эталон: lucidrains `vector-quantize-pytorch` (ResidualVQ), `meta-encodec`.
pub struct ResidualVQ {
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub dim: usize,
    pub codebooks: Vec<Parameter>,
}

impl ResidualVQ {
    pub fn new(
        num_codebooks: usize, codebook_size: usize, dim: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(num_codebooks);
        for i in 0..num_codebooks {
            let cb = crate::init::init_tensor(
                &[codebook_size, dim],
                InitMethod::Normal { mean: 0.0, std: 0.02 },
                dtype, i as u64, device,
            )?;
            codebooks.push(Parameter::new(cb));
        }
        Ok(Self { num_codebooks, codebook_size, dim, codebooks })
    }

    pub fn from_codebooks(codebooks: Vec<Tensor>) -> Result<Self> {
        if codebooks.is_empty() {
            return Err(SynaptixError::Unsupported("RVQ: пустой список codebook'ов"));
        }
        let dim = codebooks[0].dims()[1];
        let codebook_size = codebooks[0].dims()[0];
        for cb in &codebooks {
            if cb.dims() != [codebook_size, dim] {
                return Err(SynaptixError::shape_mismatch(&[codebook_size, dim], cb.dims()));
            }
        }
        let num_codebooks = codebooks.len();
        let params: Vec<Parameter> = codebooks.into_iter().map(Parameter::new).collect();
        Ok(Self { num_codebooks, codebook_size, dim, codebooks: params })
    }

    /// `x: [..., D]` → `indices: [..., num_codebooks]` (I64).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let last = x.rank().checked_sub(1)
            .ok_or(SynaptixError::Unsupported("RVQ::encode: scalar input"))?;
        if x.dims()[last] != self.dim {
            return Err(SynaptixError::shape_mismatch(&[self.dim], &[x.dims()[last]]));
        }
        let device = x.device();
        let x_flat = x.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let outer = x_flat.len() / self.dim;

        let codebooks_f32 = self.read_codebooks()?;

        let mut indices = vec![0_i64; outer * self.num_codebooks];
        let mut residual = vec![0.0_f32; self.dim];
        for o in 0..outer {
            residual.copy_from_slice(&x_flat[o * self.dim..(o + 1) * self.dim]);
            for c in 0..self.num_codebooks {
                let cb = &codebooks_f32[c];
                let mut best_idx: usize = 0;
                let mut best_dist = f32::INFINITY;
                for e in 0..self.codebook_size {
                    let row = &cb[e * self.dim..(e + 1) * self.dim];
                    let mut d = 0.0_f32;
                    for i in 0..self.dim {
                        let diff = residual[i] - row[i];
                        d += diff * diff;
                    }
                    if d < best_dist {
                        best_dist = d;
                        best_idx = e;
                    }
                }
                indices[o * self.num_codebooks + c] = best_idx as i64;
                let row = &cb[best_idx * self.dim..(best_idx + 1) * self.dim];
                for i in 0..self.dim {
                    residual[i] -= row[i];
                }
            }
        }
        let mut shape: Vec<usize> = x.dims().to_vec();
        let last_idx = shape.len() - 1;
        shape[last_idx] = self.num_codebooks;
        Tensor::from_vec(indices, shape, device)
    }

    /// `indices: [..., num_codebooks]` (I64) → `recon: [..., D]` в `dtype`.
    pub fn decode(&self, indices: &Tensor, dtype: DType) -> Result<Tensor> {
        if indices.dtype() != DType::I64 {
            return Err(SynaptixError::dtype_mismatch(DType::I64, indices.dtype()));
        }
        let last = indices.rank().checked_sub(1)
            .ok_or(SynaptixError::Unsupported("RVQ::decode: scalar indices"))?;
        if indices.dims()[last] != self.num_codebooks {
            return Err(SynaptixError::shape_mismatch(
                &[self.num_codebooks],
                &[indices.dims()[last]],
            ));
        }
        let device = indices.device();
        let idx_flat = indices.contiguous()?.flatten_all()?.to_vec1::<i64>()?;
        let outer = idx_flat.len() / self.num_codebooks;

        let codebooks_f32 = self.read_codebooks()?;

        let mut out = vec![0.0_f32; outer * self.dim];
        for o in 0..outer {
            for c in 0..self.num_codebooks {
                let idx = idx_flat[o * self.num_codebooks + c] as usize;
                if idx >= self.codebook_size {
                    return Err(SynaptixError::Unsupported(
                        "RVQ::decode: index выходит за codebook_size",
                    ));
                }
                let row = &codebooks_f32[c][idx * self.dim..(idx + 1) * self.dim];
                for i in 0..self.dim {
                    out[o * self.dim + i] += row[i];
                }
            }
        }
        let mut shape: Vec<usize> = indices.dims().to_vec();
        let last_idx = shape.len() - 1;
        shape[last_idx] = self.dim;
        Tensor::from_vec(out, shape, device)?.to_dtype(dtype)
    }

    /// Forward = encode → decode (quantized reconstruction).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let indices = self.encode(x)?;
        self.decode(&indices, x.dtype())
    }

    fn read_codebooks(&self) -> Result<Vec<Vec<f32>>> {
        self.codebooks
            .iter()
            .map(|p| {
                let t = p.tensor().to_dtype(DType::F32)?.contiguous()?.flatten_all()?;
                t.to_vec1::<f32>()
            })
            .collect()
    }
}
