use synaptix_core::tensor::Tensor;
use synaptix_core::dtype::DType;
use synaptix_core::device::Device;
use crate::error::{RagError, Result};

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Tensor>;
    fn embed_one(&self, text: &str) -> Result<Tensor> {
        let t = self.embed(&[text.to_string()])?;
        t.narrow(0, 0, 1).map_err(RagError::Core)
    }
    fn dim(&self) -> usize;
}

pub struct MockEmbedder { pub dim: usize }

impl Embedder for MockEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Tensor> {
        Tensor::zeros(vec![texts.len(), self.dim], DType::F32, Device::Cpu)
            .map_err(RagError::Core)
    }
    fn dim(&self) -> usize { self.dim }
}
