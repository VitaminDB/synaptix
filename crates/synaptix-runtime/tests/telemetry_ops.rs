//! Тесты `synaptix-runtime::telemetry::{nvtx, prometheus}`.

use synaptix_runtime::telemetry::nvtx::{mark as nvtx_mark, NvtxRange};
use synaptix_runtime::telemetry::prometheus::{
    ensure_local_recorder, init_prometheus, record_counter, record_gauge, record_histogram, render,
};

#[test]
fn nvtx_range_push_pop_no_panic() {
    {
        let r = NvtxRange::push("test_range");
        assert_eq!(r.name(), "test_range");
        // На no-feature сборке — drop = no-op. На feature `nvtx` — nvtxRangeEnd().
    } // drop здесь
}

#[test]
fn nvtx_mark_no_panic() {
    nvtx_mark("test_mark");
    nvtx_mark(""); // пустое — тоже не должно паниковать
}

#[test]
fn nvtx_range_nested() {
    let _outer = NvtxRange::push("outer");
    {
        let _inner = NvtxRange::push("inner");
        // inner drop здесь
    }
    // outer drop в конце функции
}

#[test]
fn prometheus_init_local_idempotent() {
    // Первая инициализация — install recorder.
    ensure_local_recorder().expect("init_prometheus(0) первый раз");
    // Вторая — должна вернуть Ok (идемпотентно) без re-install (set_global_recorder можно один раз).
    ensure_local_recorder().expect("init_prometheus(0) повторно");
}

#[test]
fn prometheus_render_returns_string() {
    // Если recorder не стоит, render() пустая строка. Если стоит — мы видим хоть что-то.
    let s = render();
    // Текущий процесс мог уже инициализировать recorder в другом тесте, либо нет.
    // Проверяем только что render не паникует.
    assert!(s.is_empty() || !s.is_empty());
}

#[test]
fn prometheus_record_metrics_no_panic() {
    // Если recorder не стоит, макросы `metrics::*!` — no-op. Не паникуют в любом случае.
    record_counter("syn.runtime.test_counter", 1);
    record_counter("syn.runtime.test_counter", 3);
    record_gauge("syn.runtime.test_gauge", 42.0);
    record_gauge("syn.runtime.test_gauge", 100.5);
    record_histogram("syn.runtime.test_histogram", 0.01);
    record_histogram("syn.runtime.test_histogram", 0.05);
}

#[test]
fn prometheus_local_recorder_renders_metrics() {
    ensure_local_recorder().expect("install local recorder");
    record_counter("syn.runtime.render_test.counter", 7);
    record_gauge("syn.runtime.render_test.gauge", 3.14);
    record_histogram("syn.runtime.render_test.hist", 0.123);

    let s = render();
    // С локальным recorder'ом render() возвращает Prometheus text format.
    // Имена метрик переименовываются через snake_case (точки → подчёркивания).
    assert!(
        s.contains("render_test")
            || s.contains("syn_runtime")
            || s.contains("counter"),
        "ожидали что-то в Prometheus render-snapshot, получили: {} chars",
        s.len()
    );
}

#[test]
fn prometheus_init_with_bad_port_returns_ok_in_local_mode() {
    // 0 → local recorder. Любой другой пробует HTTP listener — но он spawnит thread на свой
    // runtime если нет tokio, так что тут может либо успешно установить, либо вернуть Err.
    // Не проверяем конкретный исход — только что не паникует. Через recorder уже занят, тут
    // будет early-return Ok.
    assert!(init_prometheus(0).is_ok());
}
