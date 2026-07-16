//! Prometheus metrics exporter.
//!
//! `init_prometheus(port)`:
//!  - `port == 0` — установить ТОЛЬКО recorder (без HTTP-listener'a). Метрики копятся в
//!    `metrics`-recorder и могут быть прочитаны через `render()`.
//!  - `port != 0` — recorder + HTTP listener на `0.0.0.0:port` (GET `/metrics`). Требует
//!    активный tokio runtime — `metrics-exporter-prometheus::PrometheusBuilder::install()`
//!    spawnит экспортер на текущем reactor'е. Если runtime не запущен — экспортер сам
//!    создаст одноразовый thread с runtime'ом.
//!
//! `record_counter`/`gauge`/`histogram` — простые обвязки над `metrics::*!`. Если recorder
//! не установлен — макросы просто no-op (это семантика `metrics` crate).
//!
//! `render()` (фича recorder-only) — возвращает текущее состояние метрик в Prometheus text
//! format; полезно для unit-тестов и embed-сценариев без HTTP.

use metrics_exporter_prometheus::{BuildError, PrometheusBuilder, PrometheusHandle};
use parking_lot::Mutex as PMutex;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};

/// Хэндл `metrics-exporter-prometheus` (`PrometheusHandle`) для рендера снапшота через
/// `render()`. Устанавливается из `init_prometheus`.
static HANDLE: OnceLock<PMutex<PrometheusHandle>> = OnceLock::new();
/// Гард, защищающий критическую секцию install — иначе параллельные тесты будут пытаться
/// `set_global_recorder` одновременно и получать «already initialized».
static INSTALL_GATE: Mutex<()> = Mutex::new(());

/// Поднимает Prometheus recorder. Если `port != 0` — добавляет HTTP listener на
/// `0.0.0.0:port` (требует tokio). Idempotent: повторный вызов с recorder, уже установленным,
/// возвращает `Ok` без re-install.
pub fn init_prometheus(port: u16) -> Result<(), String> {
    if HANDLE.get().is_some() {
        return Ok(());
    }

    let _gate = INSTALL_GATE.lock().map_err(|e| format!("install gate poisoned: {e}"))?;
    if HANDLE.get().is_some() {
        return Ok(());
    }

    if port == 0 {
        let handle = PrometheusBuilder::new()
            .install_recorder()
            .map_err(|e: BuildError| format!("prometheus install_recorder: {e}"))?;
        let _ = HANDLE.set(PMutex::new(handle));
        return Ok(());
    }

    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .map_err(|e| format!("prometheus listener addr: {e}"))?;

    // `install()` спавнит HTTP listener на current tokio runtime ИЛИ создаёт отдельный
    // thread+runtime если current отсутствует.
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()
        .map_err(|e: BuildError| format!("prometheus install: {e}"))?;
    // В HTTP-режиме рендер делает listener — мы не получаем handle от builder.install().
    // `render()` тогда вернёт пустую строку (recorder уже стоит, но HANDLE остаётся None).
    Ok(())
}

/// Снапшот всех метрик в Prometheus text format. Если recorder не установлен или установлен
/// в HTTP-режиме — вернёт пустую строку.
pub fn render() -> String {
    HANDLE
        .get()
        .map(|h| h.lock().render())
        .unwrap_or_default()
}

/// Идемпотентно установить recorder в локальный режим (без HTTP). Удобно для тестов:
/// несколько `record_*` могут писать в один и тот же recorder между тестами.
pub fn ensure_local_recorder() -> Result<(), String> {
    init_prometheus(0)
}

/// Записать increment на counter.
pub fn record_counter(name: &str, value: u64) {
    metrics::counter!(name.to_string()).increment(value);
}

/// Установить gauge.
pub fn record_gauge(name: &str, value: f64) {
    metrics::gauge!(name.to_string()).set(value);
}

/// Записать observation в histogram.
pub fn record_histogram(name: &str, value: f64) {
    metrics::histogram!(name.to_string()).record(value);
}
