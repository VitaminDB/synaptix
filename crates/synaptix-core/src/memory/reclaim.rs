//! Реестр отдаваемой памяти устройства: кэши, которые на OOM обязаны
//! подвинуться.
//!
//! Зачем: аллокатор видит только «не хватило N байт» и умеет ровно два
//! приёма — sync всех стримов и `cuMemPoolTrimTo`. Оба бессильны, когда
//! память ЖИВАЯ и держит её кэш, который в любой момент готов её отдать
//! (кэш экспертов MoE перечитывается из бандла, кэш перемешанных копий
//! строится заново). Раньше такой кэш умел ужиматься только изнутри своего
//! слоя — `MoeFfn::forward` ловил OOM и резал кэш вдвое, — поэтому падение на
//! аллокации ВНЕ MoE (`слой 0 mlp_hc: alloc_uninit(219 МБ)`) уже ничем не
//! лечилось, хотя рядом стояло 12 ГБ отдаваемого кэша.
//!
//! Здесь кэш регистрируется один раз, а `reclaim` вызывается аллокатором
//! между ретраями — до того, как он сдастся.

use std::sync::{Mutex, OnceLock, Weak};

use crate::device::Device;

/// Кэш, готовый отдать память устройства под чужую аллокацию.
pub trait Reclaimable: Send + Sync {
    /// Освободить не меньше `want` байт на `device`; вернуть, сколько отдано.
    /// Вызывается из аллокатора на OOM — реализация не должна блокироваться
    /// (её же мьютекс может быть занят тем, кто и упёрся в OOM: берите
    /// `try_lock` и возвращайте 0).
    fn reclaim(&self, device: Device, want: usize) -> usize;

    /// Сколько байт кэш готов отдать, если попросят. Нужно планировщику: без
    /// этого «свободная VRAM» занижена на весь размер кэша, и бюджет KV
    /// выглядит нулевым рядом с десятью отдаваемыми гигабайтами. По умолчанию
    /// `0` — кэш, который не умеет отвечать, просто не участвует в оценке.
    fn reclaimable_bytes(&self, _device: Device) -> usize {
        0
    }
}

type Slot = Weak<dyn Reclaimable>;

fn registry() -> &'static Mutex<Vec<Slot>> {
    static R: OnceLock<Mutex<Vec<Slot>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// Зарегистрировать кэш. Держим слабую ссылку: выгруженная модель уходит из
/// реестра сама, без парного `unregister`.
pub fn register(who: &std::sync::Arc<dyn Reclaimable>) {
    if let Ok(mut g) = registry().lock() {
        g.retain(|w| w.strong_count() > 0);
        g.push(std::sync::Arc::downgrade(who));
    }
}

/// Попросить зарегистрированные кэши освободить `want` байт на `device`.
/// Возвращает, сколько байт отдано суммарно.
pub fn reclaim(device: Device, want: usize) -> usize {
    let live: Vec<std::sync::Arc<dyn Reclaimable>> = {
        let Ok(mut g) = registry().lock() else { return 0 };
        g.retain(|w| w.strong_count() > 0);
        g.iter().filter_map(|w| w.upgrade()).collect()
    };
    let mut freed = 0usize;
    for r in live {
        if freed >= want {
            break;
        }
        freed += r.reclaim(device, want - freed);
    }
    freed
}

/// Сколько байт зарегистрированные кэши готовы отдать на `device`.
pub fn reclaimable(device: Device) -> usize {
    let live: Vec<std::sync::Arc<dyn Reclaimable>> = {
        let Ok(mut g) = registry().lock() else { return 0 };
        g.retain(|w| w.strong_count() > 0);
        g.iter().filter_map(|w| w.upgrade()).collect()
    };
    live.iter().map(|r| r.reclaimable_bytes(device)).sum()
}

/// Есть ли кому отдавать (дешёвая проверка перед дорогим ретраем).
pub fn any_registered() -> bool {
    registry().lock().map(|g| g.iter().any(|w| w.strong_count() > 0)).unwrap_or(false)
}
