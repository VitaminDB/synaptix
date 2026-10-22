//! Профиль шага обучения по участкам: `SYN_TRAIN_PROF=1` — каждый участок
//! синхронизирует карту и копит время; отчёт — [`report_and_clear`].

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

use synaptix_core::device::Device;

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYN_TRAIN_PROF").is_ok_and(|v| v != "0"))
}

thread_local! {
    static ACC: RefCell<BTreeMap<&'static str, (f64, u64)>> = const { RefCell::new(BTreeMap::new()) };
}

fn sync(dev: Device) {
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
    }
}

pub fn prof<T>(dev: Device, name: &'static str, f: impl FnOnce() -> T) -> T {
    if !on() {
        return f();
    }
    sync(dev);
    let t0 = Instant::now();
    let r = f();
    sync(dev);
    let dt = t0.elapsed().as_secs_f64();
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let e = a.entry(name).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    });
    r
}

/// Участок до конца области видимости (как [`prof`], но охранником).
pub struct Span {
    dev: Device,
    name: &'static str,
    t0: Option<Instant>,
}

impl Span {
    pub fn new(dev: Device, name: &'static str) -> Self {
        let t0 = on().then(|| {
            sync(dev);
            Instant::now()
        });
        Self { dev, name, t0 }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some(t0) = self.t0 else { return };
        sync(self.dev);
        let dt = t0.elapsed().as_secs_f64();
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            let e = a.entry(self.name).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
        });
    }
}

pub fn enabled() -> bool {
    on()
}

pub fn report_and_clear() -> String {
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let total: f64 = a.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = a.iter().map(|(k, v)| (*k, *v)).collect();
        rows.sort_by(|x, y| y.1 .0.partial_cmp(&x.1 .0).unwrap());
        let mut s = String::new();
        for (k, (t, n)) in rows {
            s.push_str(&format!("{k:>16} {:>8.1} мс {:>5.1} % ×{n}\n", t * 1e3, 100.0 * t / total.max(1e-9)));
        }
        a.clear();
        s
    })
}
