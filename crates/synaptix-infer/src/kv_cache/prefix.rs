use std::sync::Arc;
use synaptix_core::tensor::Tensor;
use crate::error::{InferError, Result};
use super::KvCache;

pub struct PrefixKvCache {
    prefix: Option<Arc<dyn KvCache>>,
    tail: Box<dyn KvCache>,
    prefix_len: usize,
    concat_cache: Vec<Option<(Tensor, Tensor)>>,
}

impl PrefixKvCache {
    pub fn new(tail: Box<dyn KvCache>) -> Self {
        let n = tail.num_layers();
        Self { prefix: None, tail, prefix_len: 0, concat_cache: vec![None; n] }
    }

    pub fn with_prefix(prefix: Arc<dyn KvCache>, tail: Box<dyn KvCache>) -> Self {
        let prefix_len = prefix.seq_len();
        let n = tail.num_layers();
        Self { prefix: Some(prefix), tail, prefix_len, concat_cache: vec![None; n] }
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    fn rebuild(&mut self, layer: usize) -> Result<()> {
        let prefix_kv = self.prefix.as_ref().and_then(|p| p.get(layer)).map(|(k, v)| (k.clone(), v.clone()));
        let tail_kv = self.tail.get(layer).map(|(k, v)| (k.clone(), v.clone()));
        let combined = match (prefix_kv, tail_kv) {
            (Some((pk, pv)), Some((tk, tv))) => {
                let k = Tensor::cat(&[&pk, &tk], 2).map_err(InferError::Core)?;
                let v = Tensor::cat(&[&pv, &tv], 2).map_err(InferError::Core)?;
                Some((k, v))
            }
            (Some(pkv), None) => Some(pkv),
            (None, Some(tkv)) => Some(tkv),
            (None, None) => None,
        };
        self.concat_cache[layer] = combined;
        Ok(())
    }
}

impl KvCache for PrefixKvCache {
    fn num_layers(&self) -> usize {
        self.tail.num_layers()
    }

    fn head_dim(&self) -> usize {
        self.tail.head_dim()
    }

    fn num_heads(&self) -> usize {
        self.tail.num_heads()
    }

    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        self.tail.append(layer, key, value)?;
        self.rebuild(layer)
    }

    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.concat_cache.get(layer)?.as_ref().map(|(k, v)| (k, v))
    }

    fn seq_len(&self) -> usize {
        self.prefix_len + self.tail.seq_len()
    }

    fn capacity(&self) -> usize {
        self.prefix_len + self.tail.capacity()
    }

    fn clear(&mut self) {
        self.tail.clear();
        self.prefix = None;
        self.prefix_len = 0;
        for entry in &mut self.concat_cache {
            *entry = None;
        }
    }

    fn reset_to(&mut self, len: usize) {
        if len <= self.prefix_len {
            self.tail.clear();
        } else {
            self.tail.reset_to(len - self.prefix_len);
        }
        let n = self.concat_cache.len();
        for layer in 0..n {
            let _ = self.rebuild(layer);
        }
    }
}
