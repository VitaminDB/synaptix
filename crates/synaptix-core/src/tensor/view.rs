use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;

impl Tensor {
    pub fn narrow(&self, dim: usize, off: usize, len: usize) -> Result<Self> {
        let layout = self.layout.narrow(dim, off, len)?;
        Ok(self.with_layout(layout))
    }

    pub fn slice(&self, dim: usize, off: usize, len: usize) -> Result<Self> {
        self.narrow(dim, off, len)
    }

    pub fn repeat_interleave(&self, dim: usize, repeats: usize) -> Result<Self> {
        let rank = self.rank();
        if dim >= rank {
            return Err(SynaptixError::DimOutOfRange { dim, rank });
        }
        if repeats == 0 {
            return Err(SynaptixError::Unsupported("repeat_interleave: repeats must be > 0"));
        }
        if repeats == 1 {
            return Ok(self.clone());
        }
        let mut expand_dims = self.dims().to_vec();
        expand_dims.insert(dim + 1, repeats);
        let unsq = self.unsqueeze(dim + 1)?;
        let expanded = unsq.expand(expand_dims)?;
        let contig = expanded.contiguous()?;
        let mut new_dims = self.dims().to_vec();
        new_dims[dim] *= repeats;
        contig.reshape(new_dims)
    }
}
