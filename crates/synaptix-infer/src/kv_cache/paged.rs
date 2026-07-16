use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{InferError, Result};
use super::KvCache;

pub struct KvBlock {
    pub k_chunks: Vec<Tensor>,
    pub v_chunks: Vec<Tensor>,
    pub seq_len: usize,
}

pub struct PagedKvCache {
    blocks: Vec<Vec<Option<KvBlock>>>,
    free_blocks: Vec<usize>,
    allocated: Vec<Vec<usize>>,
    concat_cache: Vec<Option<(Tensor, Tensor)>>,
    block_size: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    total_seq_len: usize,
    device: Device,
    dtype: DType,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let blocks: Vec<Vec<Option<KvBlock>>> = (0..num_layers)
            .map(|_| (0..max_blocks).map(|_| None).collect())
            .collect();
        let free_blocks: Vec<usize> = (0..max_blocks).rev().collect();
        let allocated: Vec<Vec<usize>> = (0..num_layers).map(|_| Vec::new()).collect();
        Ok(Self {
            blocks,
            free_blocks,
            allocated,
            concat_cache: vec![None; num_layers],
            block_size,
            num_layers,
            num_heads,
            head_dim,
            total_seq_len: 0,
            device,
            dtype,
        })
    }

    fn allocate_block(&mut self) -> Result<usize> {
        self.free_blocks
            .pop()
            .ok_or_else(|| InferError::Oom("no free kv blocks".into()))
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_allocated_blocks(&self, layer: usize) -> usize {
        self.allocated.get(layer).map(|v| v.len()).unwrap_or(0)
    }

    fn rebuild_concat(&mut self, layer: usize) -> Result<()> {
        self.concat_cache[layer] = None;
        let mut all_k: Vec<Tensor> = Vec::new();
        let mut all_v: Vec<Tensor> = Vec::new();
        for &bid in &self.allocated[layer] {
            if let Some(block) = &self.blocks[layer][bid] {
                all_k.extend(block.k_chunks.iter().cloned());
                all_v.extend(block.v_chunks.iter().cloned());
            }
        }
        if !all_k.is_empty() {
            let k_refs: Vec<&Tensor> = all_k.iter().collect();
            let v_refs: Vec<&Tensor> = all_v.iter().collect();
            let k_cat = Tensor::cat(&k_refs, 2).map_err(InferError::Core)?;
            let v_cat = Tensor::cat(&v_refs, 2).map_err(InferError::Core)?;
            self.concat_cache[layer] = Some((k_cat, v_cat));
        }
        Ok(())
    }
}

impl KvCache for PagedKvCache {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        if layer >= self.num_layers {
            return Err(InferError::KvCache(format!("layer {} out of range", layer)));
        }
        let new_tokens = key.dims().get(2).copied().unwrap_or(1);
        for tok in 0..new_tokens {
            let needs_new_block = match self.allocated[layer].last() {
                None => true,
                Some(&bid) => {
                    self.blocks[layer][bid]
                        .as_ref()
                        .map(|b| b.seq_len >= self.block_size)
                        .unwrap_or(true)
                }
            };
            if needs_new_block {
                let bid = self.allocate_block()?;
                self.blocks[layer][bid] = Some(KvBlock {
                    k_chunks: Vec::with_capacity(self.block_size),
                    v_chunks: Vec::with_capacity(self.block_size),
                    seq_len: 0,
                });
                self.allocated[layer].push(bid);
            }
            let bid = *self.allocated[layer].last().unwrap();
            let block = self.blocks[layer][bid].as_mut().unwrap();
            let k_tok = key.narrow(2, tok, 1).and_then(|t| t.contiguous()).map_err(InferError::Core)?;
            let v_tok = value.narrow(2, tok, 1).and_then(|t| t.contiguous()).map_err(InferError::Core)?;
            block.k_chunks.push(k_tok);
            block.v_chunks.push(v_tok);
            block.seq_len += 1;
            if layer == 0 && tok == new_tokens - 1 {
            }
        }
        if layer == 0 {
            self.total_seq_len += new_tokens;
        }
        let _ = self.device;
        let _ = self.dtype;
        self.rebuild_concat(layer)?;
        Ok(())
    }

    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.concat_cache.get(layer)?.as_ref().map(|(k, v)| (k, v))
    }

    fn seq_len(&self) -> usize {
        self.total_seq_len
    }

    fn capacity(&self) -> usize {
        self.blocks.first().map(|v| v.len()).unwrap_or(0) * self.block_size
    }

    fn clear(&mut self) {
        for (layer, alloc) in self.allocated.iter_mut().enumerate() {
            for &bid in alloc.iter() {
                self.blocks[layer][bid] = None;
                self.free_blocks.push(bid);
            }
            alloc.clear();
            self.concat_cache[layer] = None;
        }
        self.total_seq_len = 0;
    }

    fn reset_to(&mut self, len: usize) {
        if len == 0 {
            self.clear();
            return;
        }
        if len >= self.total_seq_len {
            return;
        }
        let num_layers = self.num_layers;
        for layer in 0..num_layers {
            let mut remaining = len;
            let alloc_copy = self.allocated[layer].clone();
            let mut new_alloc: Vec<usize> = Vec::new();
            for bid in alloc_copy {
                if remaining == 0 {
                    self.blocks[layer][bid] = None;
                    self.free_blocks.push(bid);
                    continue;
                }
                let block_seq = self.blocks[layer][bid].as_ref().map(|b| b.seq_len).unwrap_or(0);
                if block_seq <= remaining {
                    new_alloc.push(bid);
                    remaining -= block_seq;
                } else {
                    if let Some(block) = self.blocks[layer][bid].as_mut() {
                        block.k_chunks.truncate(remaining);
                        block.v_chunks.truncate(remaining);
                        block.seq_len = remaining;
                    }
                    new_alloc.push(bid);
                    remaining = 0;
                }
            }
            self.allocated[layer] = new_alloc;
            let _ = self.rebuild_concat(layer);
        }
        self.total_seq_len = len;
    }
}
