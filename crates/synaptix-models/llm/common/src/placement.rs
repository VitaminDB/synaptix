//! Где лежат эксперты MoE во время работы (карта / закреплённая RAM / бандл на
//! диске) и работает ли спекуляция — для виджетов службы: по ним видно, от
//! чего зависит скорость декода. Читается из другого потока без замка
//! пайплайна (тот занят генерацией).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::moe::ExpertCache;

/// Снимок раскладки экспертов.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertPlacement {
    pub layers: u32,
    pub experts_per_layer: u32,
    /// Все эксперты (слои × эксперты) и их байты.
    pub total: u32,
    pub total_bytes: u64,
    /// В кэше на карте сейчас и потолок кэша.
    pub vram: u32,
    pub vram_bytes: u64,
    pub vram_capacity: u64,
    /// В закреплённом зеркале RAM: готово и сколько поднимется (бюджет).
    pub ram: u32,
    pub ram_bytes: u64,
    pub ram_target_bytes: u64,
    /// Вне зеркала — промах читается из бандла (NVMe или page cache).
    pub disk: u32,
    pub disk_bytes: u64,
    /// Кэш на карте с загрузки.
    pub hits: u64,
    pub misses: u64,
}

/// Счётчики спекуляции с загрузки модели.
#[derive(Debug, Default)]
pub struct SpecCounters {
    pub drafted: AtomicU64,
    pub accepted: AtomicU64,
}

impl SpecCounters {
    pub fn add(&self, drafted: usize, accepted: usize) {
        self.drafted.fetch_add(drafted as u64, Ordering::Relaxed);
        self.accepted.fetch_add(accepted as u64, Ordering::Relaxed);
    }

    pub fn get(&self) -> (u64, u64) {
        (self.drafted.load(Ordering::Relaxed), self.accepted.load(Ordering::Relaxed))
    }
}

/// Зонд раскладки: всё нужное держит у себя (кэш — общий `Arc`), так что
/// снимок не трогает пайплайн.
#[derive(Clone)]
pub struct PlacementProbe {
    pub layers: u32,
    pub experts_per_layer: u32,
    pub total_bytes: u64,
    pub cache: Option<Arc<ExpertCache>>,
    /// Зеркало этой модели: бюджет, байт. `None` — зеркала нет.
    pub mirror_budget: Option<u64>,
}

impl PlacementProbe {
    pub fn snapshot(&self) -> ExpertPlacement {
        let total = self.layers * self.experts_per_layer;
        let per = self.total_bytes / total.max(1) as u64;
        let count = |bytes: u64| if per == 0 { 0 } else { (bytes / per).min(total as u64) as u32 };
        let (vram, vram_bytes, vram_capacity, hits, misses) = match &self.cache {
            Some(c) => {
                let st = c.stats();
                (st.resident as u32, st.bytes as u64, c.capacity_bytes() as u64, st.hits, st.misses)
            }
            None => (0, 0, 0, 0, 0),
        };
        let (ram_bytes, ram_target_bytes) = match self.mirror_budget {
            Some(budget) => {
                let ready = synaptix_core::device::cuda::offload_pin_cache_progress().map(|(r, _)| r as u64).unwrap_or(0);
                (ready.min(budget), budget)
            }
            None => (0, 0),
        };
        let ram = count(ram_bytes);
        let disk_bytes = self.total_bytes.saturating_sub(ram_bytes);
        ExpertPlacement {
            layers: self.layers,
            experts_per_layer: self.experts_per_layer,
            total,
            total_bytes: self.total_bytes,
            vram,
            vram_bytes,
            vram_capacity,
            ram,
            ram_bytes,
            ram_target_bytes,
            disk: total - ram,
            disk_bytes,
            hits,
            misses,
        }
    }
}
