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
    /// Раскладка, меняющаяся на ходу (эксперты уезжают с карты в RAM при
    /// нехватке памяти): если задана, кэш и зеркало берутся отсюда, а без
    /// кэша эксперты считаются целиком на карте.
    pub live: Option<Arc<LivePlacement>>,
}

/// Изменчивая часть раскладки модели на общем декодере.
#[derive(Default)]
pub struct LivePlacement {
    cache: std::sync::Mutex<Option<Arc<ExpertCache>>>,
    /// Бюджет зеркала, байт; 0 — зеркала нет.
    mirror_budget: AtomicU64,
}

impl LivePlacement {
    /// Эксперты в RAM: кэш на карте и бюджет зеркала (`None` — без зеркала).
    pub fn set_offloaded(&self, cache: Arc<ExpertCache>, mirror_budget: Option<u64>) {
        *self.cache.lock().unwrap() = Some(cache);
        self.mirror_budget.store(mirror_budget.unwrap_or(0), Ordering::Relaxed);
    }
}

impl PlacementProbe {
    pub fn snapshot(&self) -> ExpertPlacement {
        if let Some(live) = &self.live {
            let cache = live.cache.lock().unwrap().clone();
            let total = self.layers * self.experts_per_layer;
            let Some(cache) = cache else {
                // Эксперты целиком на карте.
                return ExpertPlacement {
                    layers: self.layers,
                    experts_per_layer: self.experts_per_layer,
                    total,
                    total_bytes: self.total_bytes,
                    vram: total,
                    vram_bytes: self.total_bytes,
                    // 0 — отдельного кэша нет: эксперты входят в веса модели.
                    vram_capacity: 0,
                    ..Default::default()
                };
            };
            let budget = live.mirror_budget.load(Ordering::Relaxed);
            let p = PlacementProbe { live: None, cache: Some(cache), mirror_budget: (budget > 0).then_some(budget), ..self.clone() };
            return p.snapshot();
        }
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

/// Слои модели по месту: сколько блоков на карте (остальные — на хосте в RAM,
/// приезжают на каждом проходе). Счётчик меняется на ходу.
#[derive(Clone)]
pub struct LayerProbe {
    pub total: u32,
    pub resident: Arc<AtomicU64>,
}

impl LayerProbe {
    /// Все слои на устройстве (модель без перекладки блоков).
    pub fn all_resident(total: usize) -> Self {
        Self { total: total as u32, resident: Arc::new(AtomicU64::new(total as u64)) }
    }

    /// `(всего, на карте)`.
    pub fn snapshot(&self) -> (u32, u32) {
        (self.total, (self.resident.load(Ordering::Relaxed) as u32).min(self.total))
    }
}
