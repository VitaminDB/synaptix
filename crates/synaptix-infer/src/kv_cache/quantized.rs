use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{InferError, Result};
use super::KvCache;
use super::full::FullKvCache;

pub struct QuantizedKvCache {
    inner: FullKvCache,
    scale_k: Vec<f32>,
    scale_v: Vec<f32>,
    num_layers: usize,
}

impl QuantizedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
        device: Device,
    ) -> Result<Self> {
        let inner = FullKvCache::new(num_layers, num_heads, head_dim, capacity, device, DType::F16)?;
        Ok(Self {
            inner,
            scale_k: vec![1.0f32; num_layers],
            scale_v: vec![1.0f32; num_layers],
            num_layers,
        })
    }

    fn quantize(t: &Tensor) -> Result<Tensor> {
        t.to_dtype(DType::F16).map_err(InferError::Core)
    }

    fn dequantize(t: &Tensor) -> Result<Tensor> {
        t.to_dtype(DType::F32).map_err(InferError::Core)
    }
}

impl KvCache for QuantizedKvCache {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    fn num_heads(&self) -> usize {
        self.inner.num_heads()
    }

    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        let k16 = Self::quantize(key)?;
        let v16 = Self::quantize(value)?;
        self.inner.append(layer, &k16, &v16)
    }

    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.inner.get(layer)
    }

    fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn clear(&mut self) {
        self.inner.clear();
        for s in &mut self.scale_k {
            *s = 1.0;
        }
        for s in &mut self.scale_v {
            *s = 1.0;
        }
    }

    fn reset_to(&mut self, len: usize) {
        self.inner.reset_to(len);
    }
}
