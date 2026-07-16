use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::Result;
use super::KvCache;
use super::full::FullKvCache;

/// Двухуровневый KV-кэш: `gpu_cache` держит свежее окно (последние
/// `gpu_capacity` токенов), `cpu_cache` — вытесненный «хвост истории».
/// Логическая последовательность = `[cpu_cache][gpu_cache]` (cpu старее).
///
/// При переполнении gpu самые старые токены **вытесняются** в конец cpu
/// (`append` → `evict_front` → cpu). [`promote`] возвращает недавно вытесненные
/// токены обратно в начало gpu. На CPU обе памяти — на CPU-устройстве, поэтому
/// «перенос между уровнями» = клонирование тензоров (через `narrow`/`cat`).
pub struct TieredKvCache {
    gpu_cache: FullKvCache,
    cpu_cache: FullKvCache,
    gpu_capacity: usize,
    total_capacity: usize,
    /// Сколько токенов сейчас лежит в cpu-уровне (== `cpu_cache.seq_len`).
    eviction_cursor: usize,
}

impl TieredKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        gpu_cap: usize,
        cpu_cap: usize,
        gpu_device: Device,
        cpu_device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Внутренние кэши получают суммарную ёмкость с запасом — лимит уровня
        // gpu обеспечивает сам TieredKvCache через вытеснение, а не внутренняя
        // проверка FullKvCache.
        let inner_cap = gpu_cap + cpu_cap;
        let gpu_cache = FullKvCache::new(num_layers, num_heads, head_dim, inner_cap, gpu_device, dtype)?;
        let cpu_cache = FullKvCache::new(num_layers, num_heads, head_dim, inner_cap, cpu_device, dtype)?;
        Ok(Self { gpu_cache, cpu_cache, gpu_capacity: gpu_cap, total_capacity: inner_cap, eviction_cursor: 0 })
    }

    /// Токенов в быстром (gpu) уровне.
    pub fn gpu_len(&self) -> usize { self.gpu_cache.seq_len() }
    /// Токенов в медленном (cpu) уровне.
    pub fn cpu_len(&self) -> usize { self.cpu_cache.seq_len() }

    /// Вернуть недавно вытесненные токены из cpu обратно в начало gpu — столько,
    /// сколько влезает в свободное место gpu-уровня. Возвращает число
    /// перенесённых токенов.
    pub fn promote(&mut self) -> Result<usize> {
        let free = self.gpu_capacity.saturating_sub(self.gpu_cache.seq_len());
        let n = free.min(self.cpu_cache.seq_len());
        if n == 0 {
            return Ok(0);
        }
        let backs = self.cpu_cache.evict_back(n)?;
        self.gpu_cache.prepend(&backs)?;
        self.eviction_cursor = self.cpu_cache.seq_len();
        Ok(n)
    }
}

impl KvCache for TieredKvCache {
    fn num_layers(&self) -> usize {
        self.gpu_cache.num_layers()
    }

    fn head_dim(&self) -> usize {
        self.gpu_cache.head_dim()
    }

    fn num_heads(&self) -> usize {
        self.gpu_cache.num_heads()
    }

    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        let new_len = key.dims().get(2).copied().unwrap_or(1);
        // Вытеснение оцениваем один раз за токен-шаг — на слое 0 (только он
        // двигает seq_len во FullKvCache), сразу для всех слоёв.
        if layer == 0 {
            let projected = self.gpu_cache.seq_len() + new_len;
            if projected > self.gpu_capacity {
                let overflow = (projected - self.gpu_capacity).min(self.gpu_cache.seq_len());
                if overflow > 0 {
                    let fronts = self.gpu_cache.evict_front(overflow)?;
                    for (l, (k, v)) in fronts.iter().enumerate() {
                        self.cpu_cache.append(l, k, v)?;
                    }
                    self.eviction_cursor = self.cpu_cache.seq_len();
                }
            }
        }
        self.gpu_cache.append(layer, key, value)
    }

    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.gpu_cache.get(layer)
    }

    fn seq_len(&self) -> usize {
        self.gpu_cache.seq_len() + self.cpu_cache.seq_len()
    }

    fn capacity(&self) -> usize {
        self.total_capacity
    }

    fn clear(&mut self) {
        self.gpu_cache.clear();
        self.cpu_cache.clear();
        self.eviction_cursor = 0;
    }

    fn reset_to(&mut self, len: usize) {
        let cpu_len = self.cpu_cache.seq_len();
        if len >= cpu_len {
            self.gpu_cache.reset_to(len - cpu_len);
        } else {
            self.gpu_cache.clear();
            self.cpu_cache.reset_to(len);
            self.eviction_cursor = self.cpu_cache.seq_len();
        }
    }
}
