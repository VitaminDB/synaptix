pub mod arithmetic;
pub mod broadcast;
pub mod compare;
pub mod conversion;
pub mod creation;
pub mod debug;
pub mod indexing;
pub mod layout;
pub mod ops;
pub mod quant;
pub mod random;
pub mod reduce;
pub mod reshape;
pub mod serialize;
pub mod shape;
pub mod storage;
pub mod strides;
pub mod view;

use std::sync::Arc;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::grad::{GradFn, GradMeta};
use crate::tensor::layout::Layout;
use crate::tensor::shape::Shape;
use crate::tensor::storage::Storage;
use crate::tensor::strides::Strides;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) layout: Layout,
    pub(crate) grad_meta: Option<Arc<GradMeta>>,
}

impl Tensor {
    #[allow(dead_code)]
    pub(crate) fn from_parts(storage: Arc<Storage>, layout: Layout) -> Self {
        Self { storage, layout, grad_meta: None }
    }

    #[allow(dead_code)]
    pub(crate) fn with_layout(&self, layout: Layout) -> Self {
        Self { storage: self.storage.clone(), layout, grad_meta: None }
    }

    pub fn grad_meta(&self) -> Option<Arc<GradMeta>> {
        self.grad_meta.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.grad_meta.as_ref().map(|m| m.requires_grad()).unwrap_or(false)
    }

    pub fn requires_grad_(mut self, value: bool) -> Self {
        match &self.grad_meta {
            Some(meta) if meta.is_leaf() => {
                meta.set_requires_grad(value);
            }
            Some(_) => {}
            None => {
                self.grad_meta = Some(GradMeta::leaf(value));
            }
        }
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.grad_meta.as_ref().map(|m| m.is_leaf()).unwrap_or(true)
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn>> {
        self.grad_meta.as_ref().and_then(|m| m.grad_fn().cloned())
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad_meta.as_ref().and_then(|m| m.grad())
    }

    pub fn zero_grad(&self) {
        if let Some(meta) = self.grad_meta.as_ref() {
            meta.zero_grad();
        }
    }

    pub fn detach(&self) -> Tensor {
        Self { storage: self.storage.clone(), layout: self.layout.clone(), grad_meta: None }
    }

    pub fn backward(&self) -> Result<()> {
        crate::grad::backward(self)
    }

    pub fn backward_with(&self, gradient: Tensor) -> Result<()> {
        crate::grad::backward_with(self, gradient)
    }

    pub fn set_grad_meta(&mut self, meta: Option<Arc<GradMeta>>) {
        self.grad_meta = meta;
    }

    pub fn register_hook<F>(&self, hook: F)
    where
        F: Fn(&Tensor) -> Option<Tensor> + Send + Sync + 'static,
    {
        if let Some(meta) = self.grad_meta.as_ref() {
            meta.register_hook(std::sync::Arc::new(hook));
        }
    }

    pub fn accumulate_grad(&self, incoming: Tensor) -> Result<()> {
        let meta = self
            .grad_meta
            .as_ref()
            .ok_or_else(|| SynaptixError::Other("accumulate_grad on tensor without grad_meta".into()))?;
        meta.accumulate(incoming)
    }

    pub fn shape(&self) -> &Shape { self.layout.shape() }
    pub fn strides(&self) -> &Strides { self.layout.strides() }
    pub fn dims(&self) -> &[usize] { self.layout.dims() }
    pub fn rank(&self) -> usize { self.layout.rank() }
    pub fn numel(&self) -> usize { self.layout.numel() }
    pub fn dtype(&self) -> DType { self.layout.dtype() }
    pub fn device(&self) -> Device { self.storage.device() }
    pub fn is_contiguous(&self) -> bool { self.layout.is_contiguous() }
    pub fn layout(&self) -> &Layout { &self.layout }
    pub fn storage(&self) -> &Storage { &self.storage }
    pub fn storage_arc(&self) -> Arc<Storage> { self.storage.clone() }
    pub fn storage_and_layout(&self) -> (&Storage, &Layout) { (&self.storage, &self.layout) }

    pub fn from_storage(storage: Arc<Storage>, layout: Layout) -> Self {
        Self { storage, layout, grad_meta: None }
    }

    pub fn dim(&self, i: usize) -> Result<usize> { self.shape().dim(i) }

    pub fn dims1(&self) -> Result<usize> {
        let d = self.dims();
        if d.len() == 1 {
            Ok(d[0])
        } else {
            Err(crate::error::SynaptixError::RankMismatch { expected: 1, got: d.len() })
        }
    }

    pub fn dims2(&self) -> Result<(usize, usize)> {
        let d = self.dims();
        if d.len() == 2 {
            Ok((d[0], d[1]))
        } else {
            Err(crate::error::SynaptixError::RankMismatch { expected: 2, got: d.len() })
        }
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let d = self.dims();
        if d.len() == 3 {
            Ok((d[0], d[1], d[2]))
        } else {
            Err(crate::error::SynaptixError::RankMismatch { expected: 3, got: d.len() })
        }
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let d = self.dims();
        if d.len() == 4 {
            Ok((d[0], d[1], d[2], d[3]))
        } else {
            Err(crate::error::SynaptixError::RankMismatch { expected: 4, got: d.len() })
        }
    }

    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        let d = self.dims();
        if d.len() == 5 {
            Ok((d[0], d[1], d[2], d[3], d[4]))
        } else {
            Err(crate::error::SynaptixError::RankMismatch { expected: 5, got: d.len() })
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let layout = self.layout.unsqueeze(dim)?;
        let mut out = self.with_layout(layout);
        crate::grad::try_attach_grad_fn(crate::grad::GradOp::Unsqueeze { input: self, dim }, &mut out)?;
        Ok(out)
    }
}
