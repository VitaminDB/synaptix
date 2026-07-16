use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

/// FSQ — Finite Scalar Quantization (Mentzer et al. 2023).
///
/// Каждый из `dim = levels.len()` каналов квантуется в `levels[i]` дискретных
/// уровней через `round(tanh(z) · half)` (с поправкой `offset` на чётность).
/// Индексы — смешанная позиционная нотация в base `(levels[0], levels[1], ...)`.
/// Codebook размер = `prod(levels)`. Эталон семантики: lucidrains
/// `vector-quantize-pytorch` (FSQ).
pub struct FiniteScalarQuantizer {
    pub levels: Vec<usize>,
    pub codebook_size: usize,
    pub dim: usize,
}

impl FiniteScalarQuantizer {
    pub fn new(levels: Vec<usize>) -> Self {
        let codebook_size = levels.iter().product();
        let dim = levels.len();
        Self { levels, codebook_size, dim }
    }

    /// `z: [..., D]` → `(codes: [..., D], indices: [...])`.
    /// `codes = round(tanh(z) * half - offset)` — F32 промежуточно, возвращаются
    /// в dtype входа. `indices` — base-D encoding в I64.
    pub fn quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let last = z.rank().checked_sub(1)
            .ok_or(SynaptixError::Unsupported("FSQ: scalar input"))?;
        if z.dims()[last] != self.dim {
            return Err(SynaptixError::shape_mismatch(&[self.dim], &[z.dims()[last]]));
        }
        let dtype_in = z.dtype();
        let device = z.device();
        let z_flat = z.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let outer = z_flat.len() / self.dim;

        let (halves, offsets) = self.params();
        let basis = self.basis();
        let mut codes_real = vec![0.0_f32; z_flat.len()];
        let mut indices = vec![0_i64; outer];

        for o in 0..outer {
            let mut idx: i64 = 0;
            for i in 0..self.dim {
                let h = halves[i];
                let off = offsets[i];
                let tanh_v = z_flat[o * self.dim + i].tanh();
                let scaled = tanh_v * h - off;
                let lo = -h - off;
                let hi = h - off;
                let rounded = scaled.round().clamp(lo, hi);
                codes_real[o * self.dim + i] = rounded;
                let shifted = (rounded + h + off).round() as i64;
                idx += shifted * basis[i] as i64;
            }
            indices[o] = idx;
        }
        let codes_shape: Vec<usize> = z.dims().to_vec();
        let codes_t = Tensor::from_vec(codes_real, codes_shape, device)?
            .to_dtype(dtype_in)?;
        let mut indices_shape: Vec<usize> = z.dims().to_vec();
        indices_shape.pop();
        let indices_t = Tensor::from_vec(indices, indices_shape, device)?;
        Ok((codes_t, indices_t))
    }

    /// `indices: [...]` (I64) → `codes: [..., D]` в указанном `dtype`.
    pub fn dequantize(&self, indices: &Tensor, dtype: DType) -> Result<Tensor> {
        if indices.dtype() != DType::I64 {
            return Err(SynaptixError::dtype_mismatch(DType::I64, indices.dtype()));
        }
        let device = indices.device();
        let idx_flat = indices.contiguous()?.flatten_all()?.to_vec1::<i64>()?;
        let outer = idx_flat.len();
        let (halves, offsets) = self.params();
        let mut codes_flat = vec![0.0_f32; outer * self.dim];
        for o in 0..outer {
            let mut idx = idx_flat[o];
            for i in 0..self.dim {
                let level = self.levels[i] as i64;
                let shifted = idx.rem_euclid(level);
                idx /= level;
                let raw = shifted as f32 - halves[i] - offsets[i];
                codes_flat[o * self.dim + i] = raw;
            }
        }
        let mut shape: Vec<usize> = indices.dims().to_vec();
        shape.push(self.dim);
        Tensor::from_vec(codes_flat, shape, device)?.to_dtype(dtype)
    }

    fn params(&self) -> (Vec<f32>, Vec<f32>) {
        let mut halves = Vec::with_capacity(self.dim);
        let mut offsets = Vec::with_capacity(self.dim);
        for &l in &self.levels {
            let lf = l as f32;
            halves.push((lf - 1.0) * 0.5);
            offsets.push(if l % 2 == 0 { 0.5 } else { 0.0 });
        }
        (halves, offsets)
    }

    fn basis(&self) -> Vec<usize> {
        let mut basis = Vec::with_capacity(self.dim);
        let mut acc: usize = 1;
        for &l in &self.levels {
            basis.push(acc);
            acc *= l;
        }
        basis
    }
}
