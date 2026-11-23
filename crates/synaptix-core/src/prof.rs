//! Лёгкий профиль по участкам (`SYNAPTIX_PROFILE=1`): суммарное время и число вызовов
//! именованных участков кода. Выключенный — одна загрузка атомика на участок.
//! Нужен там, где системный профайлер недоступен (телефон без `perf_event`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static STATE: AtomicU8 = AtomicU8::new(0); // 0 — не проверено, 1 — выкл., 2 — вкл.
static TABLE: Mutex<Option<HashMap<&'static str, (f64, u64)>>> = Mutex::new(None);

/// Включён ли профиль (переменная `SYNAPTIX_PROFILE`).
#[inline]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var_os("SYNAPTIX_PROFILE").is_some_and(|v| v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Добавить `secs` к участку `name`.
pub fn add(name: &'static str, secs: f64) {
    let mut t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let e = t.get_or_insert_with(HashMap::new).entry(name).or_insert((0.0, 0));
    e.0 += secs;
    e.1 += 1;
}

/// Выполнить `f` и учесть его время под именем `name` (если профиль включён).
#[inline]
pub fn span<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    add(name, t.elapsed().as_secs_f64());
    r
}

/// Таблица участков по убыванию времени; очищает накопленное.
pub fn take_report() -> String {
    let mut t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = t.take() else { return String::new() };
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    let mut s = String::new();
    for (name, (secs, n)) in v {
        s.push_str(&format!("{:>10.1} мс {:>8} × {:>9.3} мс  {name}\n", secs * 1e3, n, secs * 1e3 / n.max(1) as f64));
    }
    s
}
