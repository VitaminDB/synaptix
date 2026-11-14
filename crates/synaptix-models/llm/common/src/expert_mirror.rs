//! Эксперты MoE прямо из квантованного бандла — для любой модели, чьи
//! эксперты не помещаются на карту:
//!
//! - ленивое чтение: эксперт при промахе кэша читается срезом стопки
//!   (`<…>.experts.gate_up_proj` / `down_proj`), а не держится заранее в
//!   обычной RAM ([`BundleExperts`], [`crate::moe::MoeFfn::load_lazy`]);
//! - закреплённое зеркало стопок в RAM ([`host_mirror`]): подкачка из mmap
//!   идёт «страничный кэш → копия в pinned → DMA» и упирается в 8–12 ГБ/с, из
//!   зеркала DMA идёт напрямую (~40 ГБ/с); NVFP4 зеркало хранит сразу в
//!   перемешанной раскладке ядер;
//! - адреса одноблобных экспертов (IQ3_S-128, SQ, ggml) в зеркале — для
//!   декода без синхронизаций (zero-copy, [`crate::moe_zc`]).
//!
//! Модель описывает, где её стопки, трейтом [`BundleStacks`].

use std::sync::Arc;

use synaptix_core::device::cuda::{MirrorRange, OffloadPinCacheGuard};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use crate::weights::QLinear;

use crate::model::ModelError;
use crate::moe::ExpertSource;

/// Кусок зеркала: ≤ 64 МБ и целое число матриц. `cuMemHostAlloc` держит
/// драйверную блокировку на всё время закрепления страниц, и один кусок в
/// 800 МБ останавливал загрузку весов и подкачку на сотни миллисекунд.
const MIRROR_PIECE: usize = 64 << 20;

/// Где лежат стопки экспертов модели в бандле.
pub trait BundleStacks: Send + Sync {
    /// Квантованный бандл (клон загрузчика весов — тот же mmap).
    fn bundle(&self) -> &SynBundleLoader;
    /// Слоёв с экспертами, включая головы MTP (номерами после слоёв модели).
    fn expert_layers(&self) -> usize;
    /// Имя стопки `name` (`gate_up_proj` | `down_proj`) слоя `layer` в бандле;
    /// `None` — у слоя нет экспертов.
    fn stack_key(&self, layer: usize, name: &str) -> Option<String>;
}

/// Стопки слоя лежат квантованными — эксперта можно читать срезом.
pub fn has_lazy_stacks(src: &dyn BundleStacks, layer: usize) -> bool {
    src.stack_key(layer, "gate_up_proj").is_some_and(|k| src.bundle().quant_dims(&k).is_some())
}

/// Эксперты лежат одноблобным форматом (масштабы внутри блоба) — годятся
/// для чтения ядрами прямо из зеркала (zero-copy).
pub fn experts_one_blob(src: &dyn BundleStacks) -> bool {
    (0..src.expert_layers())
        .find_map(|l| src.stack_key(l, "gate_up_proj"))
        .and_then(|k| src.bundle().quant_kind(&k))
        .is_some_and(|k| !k.has_scales())
}

/// Диапазоны mmap всех стопок (packed + scales) в порядке слоёв — под
/// зеркало. NVFP4 на карте с FP4 MMA зеркало хранит перемешанным, как его
/// читают ядра; без FP4 MMA и у прочих форматов — байт в байт.
pub fn mirror_ranges<'a>(src: &'a dyn BundleStacks, device: Device) -> Vec<MirrorRange<'a>> {
    let l = src.bundle();
    let nvfp4_native = synaptix_core::backend::registry::backend_for(device)
        .map(|b| b.quant_native(DType::NVFP4, device))
        .unwrap_or(false);
    let mut out = Vec::new();
    for layer in 0..src.expert_layers() {
        for name in ["gate_up_proj", "down_proj"] {
            let Some(key) = src.stack_key(layer, name) else { continue };
            let Some(Ok((packed, scales))) = l.quant_blob_slices(&key) else { continue };
            let Some((slices, n, k)) = l.quant_dims(&key) else { continue };
            let repack = match l.quant_kind(&key) {
                Some(synaptix_bundle::inspect::QuantKind::Nvfp4) if nvfp4_native => Some((n, k)),
                _ => None,
            };
            // Эксперт не должен разрываться между кусками: zero-copy
            // адресует его блоб одним куском зеркала.
            let per = (packed.len() / slices.max(1)).max(1);
            let slice = repack.map(|(n, k)| n * k / 2).unwrap_or(per).max(1);
            let piece = (MIRROR_PIECE / slice).max(1) * slice;
            for b in packed.chunks(piece) {
                out.push(MirrorRange { bytes: b, nvfp4_repack: repack });
            }
            let scales_per = (scales.len() / slices.max(1)).max(1);
            let piece = (MIRROR_PIECE / scales_per).max(1) * scales_per;
            for b in scales.chunks(piece) {
                out.push(MirrorRange { bytes: b, nvfp4_repack: None });
            }
        }
    }
    out
}

fn env_f64(names: &[&str]) -> Option<f64> {
    names.iter().find_map(|n| std::env::var(n).ok()).and_then(|v| v.trim().parse::<f64>().ok())
}

fn meminfo(key: &str) -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with(key))?;
    Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
}

/// Сколько байт RAM свободно (MemAvailable).
pub fn mem_available_bytes() -> Option<usize> {
    meminfo("MemAvailable:")
}

/// Запас RAM сверх зеркала: зеркало закреплено, его не вытеснить в своп, и
/// система выталкивает туда всё остальное. max(12 ГБ, ⅛ RAM): при 16 ГБ на
/// 96 ГБ зеркалу NVFP4 qwen3.8-flash-next недоставало 0,4 ГБ до всех 63,3 ГБ
/// экспертов. `SYN_MIRROR_RESERVE_GB` (или прежнее `SYN_QWEN4EXP_MIRROR_RESERVE_GB`).
pub fn mirror_reserve_bytes() -> usize {
    if let Some(gb) = env_f64(&["SYN_MIRROR_RESERVE_GB", "SYN_QWEN4EXP_MIRROR_RESERVE_GB"]) {
        return (gb.max(0.0) * (1u64 << 30) as f64) as usize;
    }
    (12usize << 30).max(meminfo("MemTotal:").unwrap_or(0) / 8)
}

/// Поднять зеркало `ranges` (старт на паузе — заполнение читает весь бандл и
/// отняло бы NVMe у загрузки весов; продолжить — `guard.resume()`, когда
/// модель поднята). Бюджет — `SYN_HOST_MIRROR_GB` (0 — выключить; прежнее имя
/// `SYN_QWEN4EXP_HOST_MIRROR_GB`), по умолчанию MemAvailable минус
/// [`mirror_reserve_bytes`]. Скопированные страницы mmap отдаются системе, так
/// что RAM в сумме растёт лишь на то, чего в страничном кэше не было.
/// Возвращает зеркало и его бюджет, байт; `tag` — префикс строк журнала.
pub fn host_mirror(ranges: &[MirrorRange<'_>], tag: &str) -> Option<(OffloadPinCacheGuard, usize)> {
    use synaptix_core::device::cuda::offload_pin_cache_active;
    const GB: f64 = (1u64 << 30) as f64;
    if ranges.is_empty() || offload_pin_cache_active() {
        return None;
    }
    let total: usize = ranges.iter().map(|r| r.bytes.len()).sum();
    let avail = mem_available_bytes().unwrap_or(0);
    let budget = match env_f64(&["SYN_HOST_MIRROR_GB", "SYN_QWEN4EXP_HOST_MIRROR_GB"]) {
        Some(gb) if gb <= 0.0 => return None,
        Some(gb) => (gb * GB) as usize,
        None => avail.saturating_sub(mirror_reserve_bytes()),
    };
    let budget = budget.min(total);
    if budget < (2usize << 30).min(total) {
        eprintln!("[{tag}] зеркало экспертов в RAM не поднято: доступно {:.1} ГБ, стопки {:.1} ГБ", avail as f64 / GB, total as f64 / GB);
        return None;
    }
    let workers = env_f64(&["SYN_HOST_MIRROR_WORKERS", "SYN_QWEN4EXP_HOST_MIRROR_WORKERS"]).map(|v| v as usize).unwrap_or(4).clamp(1, 16);
    eprintln!(
        "[{tag}] зеркало экспертов в RAM: до {:.1} ГБ из {:.1} ГБ (RAM доступно {:.1} ГБ), копировщиков {workers}",
        budget as f64 / GB,
        total as f64 / GB,
        avail as f64 / GB
    );
    Some((OffloadPinCacheGuard::new_pooled_paused(ranges, workers, budget, true), budget))
}

/// Источник экспертов поверх квантованного бандла: пара `gate_up`/`down`
/// читается срезом стопки в момент промаха кэша (из зеркала, если оно
/// покрывает диапазон).
pub struct BundleExperts<S: BundleStacks> {
    stacks: Arc<S>,
}

impl<S: BundleStacks> BundleExperts<S> {
    pub fn new(stacks: Arc<S>) -> Self {
        Self { stacks }
    }

    /// Блоб эксперта (срез mmap) — только одноблобные форматы.
    fn blob_slice(&self, key: &str, expert: usize) -> Option<&[u8]> {
        let l = self.stacks.bundle();
        if l.quant_kind(key)?.has_scales() {
            return None;
        }
        let (slices, _, _) = l.quant_dims(key)?;
        let Ok((packed, _)) = l.quant_blob_slices(key)? else { return None };
        let per = packed.len() / slices.max(1);
        packed.get(expert * per..(expert + 1) * per)
    }
}

impl<S: BundleStacks + 'static> ExpertSource for BundleExperts<S> {
    fn blob_addrs(&self, layer: usize, expert: usize) -> Option<(u64, u64)> {
        let gu = self.blob_slice(&self.stacks.stack_key(layer, "gate_up_proj")?, expert)?;
        let dn = self.blob_slice(&self.stacks.stack_key(layer, "down_proj")?, expert)?;
        Some((synaptix_core::device::cuda::mirror_addr(gu)?, synaptix_core::device::cuda::mirror_addr(dn)?))
    }

    fn fetch(&self, layer: usize, expert: usize, device: Device) -> Result<(QLinear, QLinear), ModelError> {
        let one = |name: &str| -> Result<QLinear, ModelError> {
            let key = self
                .stacks
                .stack_key(layer, name)
                .ok_or_else(|| ModelError::Load(format!("слой {layer}: нет стопки экспертов {name}")))?;
            let w = self
                .stacks
                .bundle()
                .load_quant_expert(&key, expert, device)
                .ok_or_else(|| ModelError::Load(format!("{key}: стопка не квантована")))?
                .map_err(|e| ModelError::Load(e.to_string()))?;
            Ok(QLinear::Quant(w))
        };
        Ok((one("gate_up_proj")?, one("down_proj")?))
    }
}

/// Готовое описание стопок: клон загрузчика весов (тот же mmap) и имена
/// стопок `(gate_up, down)` по слоям (`None` — у слоя нет экспертов; головы
/// MTP — номерами после слоёв модели).
pub struct ExpertStacks {
    loader: SynBundleLoader,
    keys: Vec<Option<(String, String)>>,
}

impl ExpertStacks {
    pub fn new(loader: SynBundleLoader, keys: Vec<Option<(String, String)>>) -> Self {
        Self { loader, keys }
    }

    /// Стопки по префиксу блока: слой `l` → `prefix(l)` + `.experts.{gate_up,down}_proj`.
    pub fn from_prefixes(loader: SynBundleLoader, prefixes: impl IntoIterator<Item = Option<String>>) -> Self {
        let keys = prefixes
            .into_iter()
            .map(|p| p.map(|p| (format!("{p}.experts.gate_up_proj"), format!("{p}.experts.down_proj"))))
            .collect();
        Self { loader, keys }
    }

    /// Байт во всех стопках (packed + scales).
    pub fn total_bytes(&self, device: Device) -> u64 {
        mirror_ranges(self, device).iter().map(|r| r.bytes.len() as u64).sum()
    }
}

impl BundleStacks for ExpertStacks {
    fn bundle(&self) -> &SynBundleLoader {
        &self.loader
    }
    fn expert_layers(&self) -> usize {
        self.keys.len()
    }
    fn stack_key(&self, layer: usize, name: &str) -> Option<String> {
        let (gu, dn) = self.keys.get(layer)?.as_ref()?;
        match name {
            "gate_up_proj" => Some(gu.clone()),
            "down_proj" => Some(dn.clone()),
            _ => None,
        }
    }
}
