use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, WeightSource};

use crate::config::PleConfig;

const MASK64: u64 = u64::MAX;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    (v ^ (v >> 31)) & MASK64
}

pub fn layer_multipliers(unigram_vocab: usize, ngram_size: usize, ple_index: usize, seed: u64) -> Vec<i64> {
    let max_long = i64::MAX as u64;
    let multiplier_max = max_long / (unigram_vocab.max(1) as u64);
    let half_bound = (multiplier_max / 2).max(1);
    let base = seed.wrapping_add(PRIME_1.wrapping_mul(ple_index as u64));
    (0..ngram_size)
        .map(|i| {
            let v = base.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i as u64 + 1));
            (2 * (splitmix64(v) % half_bound) + 1) as i64
        })
        .collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

pub fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

pub fn head_vocabs(cfg: &PleConfig, ple_index: usize) -> (Vec<i64>, Vec<i64>, u64) {
    let heads = cfg.ngram_heads();
    let mut sizes = Vec::with_capacity(heads);
    let mut offsets = Vec::with_capacity(heads);
    let mut total = 0u64;
    for h in 0..heads {
        let global = ple_index * heads + h;
        let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1);
        sizes.push(size as i64);
        offsets.push(total as i64);
        total += size;
    }
    (sizes, offsets, total)
}

pub trait NGramRows: Send + Sync {
    fn dim(&self) -> usize;
    fn rows(&self) -> usize;
    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError>;
}

pub fn decode_rows(bytes: &[u8], dtype: DType, out: &mut [f32]) {
    match dtype {
        DType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let raw = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                *o = f32::from_bits((raw as u32) << 16);
            }
        }
        DType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let raw = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                *o = f16_to_f32(raw);
            }
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                let b = &bytes[i * 4..i * 4 + 4];
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
    }
}

fn f16_to_f32(raw: u16) -> f32 {
    let sign = ((raw >> 15) as u32) << 31;
    let exp = ((raw >> 10) & 0x1f) as u32;
    let frac = (raw & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e = -1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((127 - 15 + e + 1) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub struct ShardedRows {
    shards: Vec<Arc<Vec<u8>>>,
    rows_per_shard: usize,
    dim: usize,
    dtype: DType,
}

impl ShardedRows {
    pub fn new(shards: Vec<Arc<Vec<u8>>>, rows_per_shard: usize, dim: usize, dtype: DType) -> Self {
        Self { shards, rows_per_shard, dim, dtype }
    }
}

impl NGramRows for ShardedRows {
    fn dim(&self) -> usize {
        self.dim
    }
    fn rows(&self) -> usize {
        self.rows_per_shard * self.shards.len()
    }
    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError> {
        let row_bytes = self.dtype.bytes_for_numel(self.dim);
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            let shard = id / self.rows_per_shard;
            let row = id % self.rows_per_shard;
            let data = self
                .shards
                .get(shard)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: шард {shard} отсутствует")))?;
            let start = row * row_bytes;
            let slice = data
                .get(start..start + row_bytes)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: строка {id} вне шарда")))?;
            decode_rows(slice, self.dtype, &mut out[i * self.dim..(i + 1) * self.dim]);
        }
        Ok(())
    }
}

pub struct TensorRows {
    data: Vec<f32>,
    rows: usize,
    dim: usize,
}

impl TensorRows {
    pub fn from_tensor(t: &Tensor) -> Result<Self, ModelError> {
        let dims = t.dims().to_vec();
        if dims.len() != 2 {
            return Err(ModelError::Load(format!("n-gram таблица: форма {dims:?}")));
        }
        let data = t
            .to_device(Device::Cpu)
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(|e| ModelError::Load(e.to_string()))?;
        Ok(Self { data, rows: dims[0], dim: dims[1] })
    }
}

impl NGramRows for TensorRows {
    fn dim(&self) -> usize {
        self.dim
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError> {
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            let start = id * self.dim;
            let row = self
                .data
                .get(start..start + self.dim)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: строка {id} вне таблицы")))?;
            out[i * self.dim..(i + 1) * self.dim].copy_from_slice(row);
        }
        Ok(())
    }
}

pub struct CachedRows {
    inner: Box<dyn NGramRows>,
    cache: Mutex<RowCache>,
}

struct RowCache {
    slot_of: HashMap<i64, usize>,
    key_of: Vec<i64>,
    data: Vec<f32>,
    next: usize,
    dim: usize,
    hits: u64,
    misses: u64,
}

impl RowCache {
    fn new(rows: usize, dim: usize) -> Self {
        Self {
            slot_of: HashMap::with_capacity(rows),
            key_of: vec![-1; rows],
            data: vec![0.0; rows * dim],
            next: 0,
            dim,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, id: i64, out: &mut [f32]) -> bool {
        match self.slot_of.get(&id) {
            Some(slot) => {
                let start = slot * self.dim;
                out.copy_from_slice(&self.data[start..start + self.dim]);
                self.hits += 1;
                true
            }
            None => {
                self.misses += 1;
                false
            }
        }
    }

    fn put(&mut self, id: i64, row: &[f32]) {
        if self.key_of.is_empty() {
            return;
        }
        let slot = self.next;
        self.next = (self.next + 1) % self.key_of.len();
        let old = self.key_of[slot];
        if old >= 0 {
            self.slot_of.remove(&old);
        }
        self.key_of[slot] = id;
        self.slot_of.insert(id, slot);
        let start = slot * self.dim;
        self.data[start..start + self.dim].copy_from_slice(row);
    }
}

impl CachedRows {
    pub fn new(inner: Box<dyn NGramRows>, cache_bytes: usize) -> Self {
        let dim = inner.dim();
        let rows = cache_bytes / (dim * std::mem::size_of::<f32>()).max(1);
        Self { inner, cache: Mutex::new(RowCache::new(rows, dim)) }
    }

    pub fn stats(&self) -> (u64, u64) {
        let c = self.cache.lock().unwrap();
        (c.hits, c.misses)
    }
}

impl NGramRows for CachedRows {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError> {
        let dim = self.inner.dim();
        let mut missing: Vec<usize> = Vec::new();
        {
            let mut cache = self.cache.lock().map_err(|_| ModelError::Forward("кэш n-gram отравлен".into()))?;
            for (i, id) in ids.iter().enumerate() {
                if !cache.get(*id, &mut out[i * dim..(i + 1) * dim]) {
                    missing.push(i);
                }
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let fetched: Result<Vec<Vec<f32>>, ModelError> = missing
            .par_iter()
            .map(|i| {
                let mut row = vec![0.0f32; dim];
                self.inner.gather_into(&ids[*i..*i + 1], &mut row)?;
                Ok(row)
            })
            .collect();
        let fetched = fetched?;
        let mut cache = self.cache.lock().map_err(|_| ModelError::Forward("кэш n-gram отравлен".into()))?;
        for (slot, i) in missing.iter().enumerate() {
            let row = &fetched[slot];
            out[i * dim..(i + 1) * dim].copy_from_slice(row);
            cache.put(ids[*i], row);
        }
        Ok(())
    }
}

pub struct NGramEmbedding {
    pub multipliers: Vec<i64>,
    pub head_sizes: Vec<i64>,
    pub head_offsets: Vec<i64>,
    ngram_size: usize,
    heads_per_ngram: usize,
    context_len: usize,
    eos: u32,
    table: Box<dyn NGramRows>,
    device: Device,
    compute: DType,
}

impl NGramEmbedding {
    pub fn new(
        cfg: &PleConfig,
        ple_index: usize,
        vocab_size: usize,
        eos: u32,
        table: Box<dyn NGramRows>,
        buffers: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>,
        device: Device,
        compute: DType,
    ) -> Result<Self, ModelError> {
        let (multipliers, head_sizes, head_offsets) = match buffers {
            Some(b) => b,
            None => {
                let (sizes, offsets, _) = head_vocabs(cfg, ple_index);
                (
                    layer_multipliers(vocab_size, cfg.ngram_size, ple_index, cfg.seed),
                    sizes,
                    offsets,
                )
            }
        };
        if multipliers.len() != cfg.ngram_size {
            return Err(ModelError::Load(format!(
                "layer_multipliers: {} значений, ожидалось {}",
                multipliers.len(),
                cfg.ngram_size
            )));
        }
        if head_sizes.len() != cfg.ngram_heads() || head_offsets.len() != cfg.ngram_heads() {
            return Err(ModelError::Load("ngram_heads_*: неверная длина".into()));
        }
        if table.dim() != cfg.head_dim() {
            return Err(ModelError::Load(format!(
                "n-gram таблица: ширина {}, ожидалась {}",
                table.dim(),
                cfg.head_dim()
            )));
        }
        Ok(Self {
            multipliers,
            head_sizes,
            head_offsets,
            ngram_size: cfg.ngram_size,
            heads_per_ngram: cfg.heads_per_ngram,
            context_len: cfg.context_len(),
            eos,
            table,
            device,
            compute,
        })
    }

    pub fn heads(&self) -> usize {
        self.head_sizes.len()
    }

    pub fn dim(&self) -> usize {
        self.table.dim()
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    fn shift_right(&self, history: &[u32], shift: usize) -> Vec<u32> {
        let n = history.len();
        if shift == 0 {
            return history.to_vec();
        }
        let mut out = vec![self.eos; n];
        let mut previous_eos: i64 = -1;
        for t in 0..n {
            let segment_start = previous_eos + 1;
            let position_in_segment = t as i64 - segment_start;
            let source = t as i64 - shift as i64;
            out[t] = if position_in_segment >= shift as i64 && source >= 0 {
                history[source as usize]
            } else {
                self.eos
            };
            if history[t] == self.eos {
                previous_eos = t as i64;
            }
        }
        out
    }

    pub fn ids_for(&self, history: &[u32], tokens: usize) -> Vec<i64> {
        let n = history.len();
        let shifted: Vec<Vec<u32>> = (0..self.ngram_size).map(|s| self.shift_right(history, s)).collect();
        let heads = self.heads();
        let mut ids = vec![0i64; tokens * heads];
        let start = n - tokens;
        for (row, t) in (start..n).enumerate() {
            for ngram in 2..=self.ngram_size {
                let head_start = (ngram - 2) * self.heads_per_ngram;
                let mut mixed = (shifted[0][t] as i64).wrapping_mul(self.multipliers[0]);
                for pos in 1..ngram {
                    mixed ^= (shifted[pos][t] as i64).wrapping_mul(self.multipliers[pos]);
                }
                for h in 0..self.heads_per_ngram {
                    let idx = head_start + h;
                    ids[row * heads + idx] =
                        mixed.rem_euclid(self.head_sizes[idx]) + self.head_offsets[idx];
                }
            }
        }
        ids
    }

    pub fn forward(&self, history: &[u32], tokens: usize) -> Result<Tensor, ModelError> {
        let ids = self.ids_for(history, tokens);
        let dim = self.table.dim();
        let mut buf = vec![0f32; ids.len() * dim];
        self.table.gather_into(&ids, &mut buf)?;
        Tensor::from_vec(buf, vec![tokens, self.heads() * dim], self.device)
            .and_then(|t| t.to_dtype(self.compute))
            .map_err(|e| ModelError::Forward(e.to_string()))
    }
}

pub fn read_i64_buffer(
    weights: &dyn WeightSource,
    key: &str,
    device: Device,
) -> Option<Vec<i64>> {
    let _ = device;
    let t = weights.tensor(key, Device::Cpu, DType::F32).ok()?;
    let v = t.flatten_all().ok()?.to_vec1::<f32>().ok()?;
    Some(v.into_iter().map(|x| x as i64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipliers_are_odd_and_bounded() {
        let m = layer_multipliers(248320, 3, 0, 1234);
        assert_eq!(m.len(), 3);
        for v in &m {
            assert!(v % 2 == 1);
            assert!(*v > 0);
            assert!((*v as i128) * 248320 < i64::MAX as i128);
        }
    }

    #[test]
    fn primes_after_base() {
        assert_eq!(nth_prime_after(9, 1), 11);
        assert_eq!(nth_prime_after(9, 2), 13);
        let p = nth_prime_after(20_000_000 - 1, 1);
        assert!(p >= 20_000_000 && is_prime(p));
    }

    #[test]
    fn shift_respects_eos_boundaries() {
        let cfg = PleConfig {
            layer_ids: vec![1],
            embed_dim: 8,
            conv_kernel_size: 4,
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_size_base: 101,
            make_vocab_divisible_by: 8,
            seed: 1234,
            split_parts: 1,
        };
        let table = TensorRows { data: vec![0.0; 1024], rows: 512, dim: 2 };
        let emb = NGramEmbedding::new(&cfg, 0, 100, 7, Box::new(table), None, Device::Cpu, DType::F32)
            .unwrap();
        let hist = vec![1u32, 2, 7, 3, 4];
        assert_eq!(emb.shift_right(&hist, 1), vec![7, 1, 2, 7, 3]);
        assert_eq!(emb.shift_right(&hist, 2), vec![7, 7, 1, 7, 7]);
    }
}
