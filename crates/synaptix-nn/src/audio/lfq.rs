use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

/// LFQ — Lookup-Free Quantization (MagViT2, Yu et al. 2023).
///
/// Каждый канал квантуется в {-1, +1} через `sign(z)`. Index — бит-маска по
/// `dim` каналам (codebook размер = 2^dim). Decode: `±1` без LUT (lookup-free).
pub struct Lfq {
    pub codebook_size: usize,
    pub dim: usize,
}

impl Lfq {
    pub fn new(codebook_size: usize, dim: usize) -> Self {
        debug_assert_eq!(codebook_size, 1usize << dim, "LFQ: codebook_size должен быть 2^dim");
        Self { codebook_size, dim }
    }

    /// `z: [..., D]` → `(codes: [..., D] в {-1,+1}, indices: [...] бит-индекс)`.
    pub fn quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let last = z.rank().checked_sub(1)
            .ok_or(SynaptixError::Unsupported("LFQ: scalar input"))?;
        if z.dims()[last] != self.dim {
            return Err(SynaptixError::shape_mismatch(&[self.dim], &[z.dims()[last]]));
        }
        if self.dim > 63 {
            return Err(SynaptixError::Unsupported(
                "LFQ: dim>63 не помещается в i64 индекс",
            ));
        }
        let dtype_in = z.dtype();
        let device = z.device();
        let z_flat = z.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let outer = z_flat.len() / self.dim;

        let mut codes = vec![0.0_f32; z_flat.len()];
        let mut indices = vec![0_i64; outer];
        for o in 0..outer {
            let mut idx: i64 = 0;
            for i in 0..self.dim {
                let v = z_flat[o * self.dim + i];
                let s: f32 = if v >= 0.0 { 1.0 } else { -1.0 };
                codes[o * self.dim + i] = s;
                if s > 0.0 { idx |= 1i64 << i; }
            }
            indices[o] = idx;
        }
        let codes_t = Tensor::from_vec(codes, z.dims().to_vec(), device)?.to_dtype(dtype_in)?;
        let mut idx_shape: Vec<usize> = z.dims().to_vec();
        idx_shape.pop();
        let indices_t = Tensor::from_vec(indices, idx_shape, device)?;
        Ok((codes_t, indices_t))
    }

    /// `indices: [...]` (I64) → `codes: [..., D]` в {-1, +1} (lookup-free).
    pub fn dequantize(&self, indices: &Tensor, dtype: DType) -> Result<Tensor> {
        if indices.dtype() != DType::I64 {
            return Err(SynaptixError::dtype_mismatch(DType::I64, indices.dtype()));
        }
        let device = indices.device();
        let idx_flat = indices.contiguous()?.flatten_all()?.to_vec1::<i64>()?;
        let outer = idx_flat.len();
        let mut codes = vec![0.0_f32; outer * self.dim];
        for o in 0..outer {
            let idx = idx_flat[o];
            for i in 0..self.dim {
                codes[o * self.dim + i] = if (idx >> i) & 1 == 1 { 1.0 } else { -1.0 };
            }
        }
        let mut shape: Vec<usize> = indices.dims().to_vec();
        shape.push(self.dim);
        Tensor::from_vec(codes, shape, device)?.to_dtype(dtype)
    }

    /// Forward: round-trip `quantize`, возвращает только `codes` (±1 коды).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (codes, _) = self.quantize(x)?;
        Ok(codes)
    }
}
