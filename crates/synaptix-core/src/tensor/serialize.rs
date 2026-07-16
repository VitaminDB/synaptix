use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::storage::Storage;

impl Tensor {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if !self.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        match &*self.storage {
            Storage::Cpu(b) => Ok(b.as_bytes().to_vec()),
            _ => Err(SynaptixError::Unsupported("to_bytes: cpu-only in MVP")),
        }
    }
}
