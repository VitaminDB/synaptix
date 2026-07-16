use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{InferError, Result};
use super::KvCache;

pub struct FullKvCache {
    k_slices: Vec<Vec<Tensor>>,
    v_slices: Vec<Vec<Tensor>>,
    concat_cache: Vec<Option<(Tensor, Tensor)>>,
    seq_len: usize,
    capacity: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
    dtype: DType,
}

impl FullKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            k_slices: vec![Vec::new(); num_layers],
            v_slices: vec![Vec::new(); num_layers],
            concat_cache: vec![None; num_layers],
            seq_len: 0,
            capacity,
            num_heads,
            head_dim,
            device,
            dtype,
        })
    }

    fn rebuild_concat(&mut self, layer: usize) {
        self.concat_cache[layer] = None;
        if self.k_slices[layer].is_empty() {
            return;
        }
        let k_refs: Vec<&Tensor> = self.k_slices[layer].iter().collect();
        let v_refs: Vec<&Tensor> = self.v_slices[layer].iter().collect();
        if let (Ok(k_cat), Ok(v_cat)) = (Tensor::cat(&k_refs, 2), Tensor::cat(&v_refs, 2)) {
            self.concat_cache[layer] = Some((k_cat, v_cat));
        }
    }

    /// Снять `n` самых старых токенов (с начала) у каждого слоя и вернуть их
    /// как по одному склеенному `(k, v)` на слой. Хвост (остальные токены)
    /// остаётся. Зеркало `reset_to` (которое режет хвост). Используется
    /// многоуровневым кэшом для вытеснения gpu→cpu.
    pub fn evict_front(&mut self, n: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let n = n.min(self.seq_len);
        if n == 0 {
            return Ok(Vec::new());
        }
        let num_layers = self.k_slices.len();
        let mut removed: Vec<(Tensor, Tensor)> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let slices_k = std::mem::take(&mut self.k_slices[layer]);
            let slices_v = std::mem::take(&mut self.v_slices[layer]);
            let (mut front_k, mut front_v) = (Vec::new(), Vec::new());
            let (mut keep_k, mut keep_v) = (Vec::new(), Vec::new());
            let mut taken = 0usize;
            for (k, v) in slices_k.into_iter().zip(slices_v.into_iter()) {
                let slen = k.dims().get(2).copied().unwrap_or(1);
                if taken >= n {
                    keep_k.push(k);
                    keep_v.push(v);
                } else if taken + slen <= n {
                    front_k.push(k);
                    front_v.push(v);
                    taken += slen;
                } else {
                    let split = n - taken;
                    front_k.push(k.narrow(2, 0, split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    front_v.push(v.narrow(2, 0, split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    keep_k.push(k.narrow(2, split, slen - split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    keep_v.push(v.narrow(2, split, slen - split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    taken = n;
                }
            }
            self.k_slices[layer] = keep_k;
            self.v_slices[layer] = keep_v;
            self.rebuild_concat(layer);
            let fk = Tensor::cat(&front_k.iter().collect::<Vec<_>>(), 2).map_err(InferError::Core)?;
            let fv = Tensor::cat(&front_v.iter().collect::<Vec<_>>(), 2).map_err(InferError::Core)?;
            removed.push((fk, fv));
        }
        self.seq_len -= n;
        Ok(removed)
    }

    /// Снять `n` самых новых токенов (с конца) у каждого слоя и вернуть их как
    /// по одному `(k, v)` на слой. Префикс остаётся. Используется для промоушена
    /// cpu→gpu (берём недавно вытесненные токены обратно).
    pub fn evict_back(&mut self, n: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let n = n.min(self.seq_len);
        if n == 0 {
            return Ok(Vec::new());
        }
        let keep = self.seq_len - n;
        let num_layers = self.k_slices.len();
        let mut removed: Vec<(Tensor, Tensor)> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let slices_k = std::mem::take(&mut self.k_slices[layer]);
            let slices_v = std::mem::take(&mut self.v_slices[layer]);
            let (mut keep_k, mut keep_v) = (Vec::new(), Vec::new());
            let (mut back_k, mut back_v) = (Vec::new(), Vec::new());
            let mut seen = 0usize;
            for (k, v) in slices_k.into_iter().zip(slices_v.into_iter()) {
                let slen = k.dims().get(2).copied().unwrap_or(1);
                if seen + slen <= keep {
                    keep_k.push(k);
                    keep_v.push(v);
                    seen += slen;
                } else if seen >= keep {
                    back_k.push(k);
                    back_v.push(v);
                } else {
                    let split = keep - seen;
                    keep_k.push(k.narrow(2, 0, split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    keep_v.push(v.narrow(2, 0, split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    back_k.push(k.narrow(2, split, slen - split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    back_v.push(v.narrow(2, split, slen - split).and_then(|t| t.contiguous()).map_err(InferError::Core)?);
                    seen = keep;
                }
            }
            self.k_slices[layer] = keep_k;
            self.v_slices[layer] = keep_v;
            self.rebuild_concat(layer);
            let bk = Tensor::cat(&back_k.iter().collect::<Vec<_>>(), 2).map_err(InferError::Core)?;
            let bv = Tensor::cat(&back_v.iter().collect::<Vec<_>>(), 2).map_err(InferError::Core)?;
            removed.push((bk, bv));
        }
        self.seq_len -= n;
        Ok(removed)
    }

    /// Вставить срезы `(k, v)` (по одному на слой) в начало кэша. Длина по seq
    /// берётся из 3-й оси первого слоя. Зеркало `evict_front`.
    pub fn prepend(&mut self, fronts: &[(Tensor, Tensor)]) -> Result<()> {
        if fronts.is_empty() {
            return Ok(());
        }
        let n = fronts[0].0.dims().get(2).copied().unwrap_or(0);
        if n == 0 {
            return Ok(());
        }
        if self.seq_len + n > self.capacity {
            return Err(InferError::KvCache("prepend: capacity exceeded".into()));
        }
        for (layer, (k, v)) in fronts.iter().enumerate() {
            if layer >= self.k_slices.len() {
                break;
            }
            self.k_slices[layer].insert(0, k.clone());
            self.v_slices[layer].insert(0, v.clone());
            self.rebuild_concat(layer);
        }
        self.seq_len += n;
        Ok(())
    }
}

impl KvCache for FullKvCache {
    fn num_layers(&self) -> usize {
        self.k_slices.len()
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        if layer >= self.k_slices.len() {
            return Err(InferError::KvCache(format!("layer {} out of range", layer)));
        }
        let new_len = key.dims().get(2).copied().unwrap_or(1);
        if self.seq_len + new_len > self.capacity {
            return Err(InferError::KvCache("kv cache capacity exceeded".into()));
        }
        self.k_slices[layer].push(key.clone());
        self.v_slices[layer].push(value.clone());
        let k_refs: Vec<&Tensor> = self.k_slices[layer].iter().collect();
        let v_refs: Vec<&Tensor> = self.v_slices[layer].iter().collect();
        let k_cat = Tensor::cat(&k_refs, 2).map_err(InferError::Core)?;
        let v_cat = Tensor::cat(&v_refs, 2).map_err(InferError::Core)?;
        self.concat_cache[layer] = Some((k_cat, v_cat));
        if layer == 0 {
            self.seq_len += new_len;
        }
        Ok(())
    }

    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.concat_cache.get(layer)?.as_ref().map(|(k, v)| (k, v))
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn clear(&mut self) {
        for slices in &mut self.k_slices {
            slices.clear();
        }
        for slices in &mut self.v_slices {
            slices.clear();
        }
        for entry in &mut self.concat_cache {
            *entry = None;
        }
        self.seq_len = 0;
    }

    fn reset_to(&mut self, len: usize) {
        if len == 0 {
            self.clear();
            return;
        }
        if len >= self.seq_len {
            return;
        }
        let num_layers = self.k_slices.len();
        for layer in 0..num_layers {
            let mut remaining = len;
            let mut keep_k: Vec<Tensor> = Vec::new();
            let mut keep_v: Vec<Tensor> = Vec::new();
            for (k, v) in self.k_slices[layer].iter().zip(self.v_slices[layer].iter()) {
                if remaining == 0 {
                    break;
                }
                let slice_len = k.dims().get(2).copied().unwrap_or(1);
                if slice_len <= remaining {
                    keep_k.push(k.clone());
                    keep_v.push(v.clone());
                    remaining -= slice_len;
                } else {
                    if let (Ok(k_part), Ok(v_part)) = (k.narrow(2, 0, remaining), v.narrow(2, 0, remaining)) {
                        if let (Ok(k_ct), Ok(v_ct)) = (k_part.contiguous(), v_part.contiguous()) {
                            keep_k.push(k_ct);
                            keep_v.push(v_ct);
                        }
                    }
                    remaining = 0;
                }
            }
            self.k_slices[layer] = keep_k;
            self.v_slices[layer] = keep_v;
            self.concat_cache[layer] = None;
            if !self.k_slices[layer].is_empty() {
                let k_refs: Vec<&Tensor> = self.k_slices[layer].iter().collect();
                let v_refs: Vec<&Tensor> = self.v_slices[layer].iter().collect();
                if let (Ok(k_cat), Ok(v_cat)) = (Tensor::cat(&k_refs, 2), Tensor::cat(&v_refs, 2)) {
                    self.concat_cache[layer] = Some((k_cat, v_cat));
                }
            }
        }
        self.seq_len = len;
    }
}
