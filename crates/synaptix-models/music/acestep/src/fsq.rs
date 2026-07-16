
use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::loader::CompLoader;
use crate::AceError;

pub const FSQ_LEVELS: [u32; 6] = [8, 8, 8, 5, 5, 5];

pub struct Fsq {
    project_out: Linear,
    basis: [u32; 6],
    dim: usize,
    device: Device,
}

impl Fsq {
    pub fn load(ck: &CompLoader, prefix: &str) -> Result<Self, AceError> {
        let w = ck.f32(&format!("{prefix}.project_out.weight"))?;
        let b = ck.f32(&format!("{prefix}.project_out.bias"))?;
        let dim = w.dims()[0];
        let project_out = Linear::new(w, Some(b)).map_err(AceError::Tensor)?;
        let mut basis = [1u32; 6];
        for j in 1..6 {
            basis[j] = basis[j - 1] * FSQ_LEVELS[j - 1];
        }
        Ok(Self { project_out, basis, dim, device: ck.device() })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn code_vec(&self, n: u32) -> [f32; 6] {
        let mut c = [0f32; 6];
        for j in 0..6 {
            let digit = ((n / self.basis[j]) % FSQ_LEVELS[j]) as f32;
            c[j] = digit * (2.0 / (FSQ_LEVELS[j] as f32 - 1.0)) - 1.0;
        }
        c
    }

    pub fn get_output_from_indices(&self, indices: &[u32]) -> Result<Tensor, AceError> {
        let t = indices.len();
        let mut flat = Vec::with_capacity(t * 6);
        for &n in indices {
            flat.extend_from_slice(&self.code_vec(n));
        }
        let codes = Tensor::from_vec(flat, vec![t, 6usize], self.device)?;
        let out = self.project_out.forward(&codes).map_err(AceError::Tensor)?;
        Ok(out.reshape((1usize, t, self.dim))?)
    }
}
