//! Грубый профиль этапов MoE: включается `SYN_MOE_PROFILE`.
//!
//! Замеры считают время до возврата из вызова, а очередь CUDA синхронизируется
//! не здесь, а на ближайшей выгрузке на хост (роутер, top-k), поэтому доли
//! стоит читать как порядок величины, а не как точный тайминг ядер.

static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static STAGES: std::sync::Mutex<Vec<(&'static str, u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn profiling() -> bool {
    *ON.get_or_init(|| std::env::var("SYN_MOE_PROFILE").is_ok())
}

pub fn stage<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !profiling() {
        return f();
    }
    let started = std::time::Instant::now();
    let out = f();
    let nanos = started.elapsed().as_nanos() as u64;
    if let Ok(mut stages) = STAGES.lock() {
        match stages.iter_mut().find(|(n, _, _)| *n == name) {
            Some(entry) => {
                entry.1 += nanos;
                entry.2 += 1;
            }
            None => stages.push((name, nanos, 1)),
        }
    }
    out
}

pub fn report() -> String {
    let Ok(mut stages) = STAGES.lock() else {
        return String::new();
    };
    stages.sort_by_key(|(_, nanos, _)| std::cmp::Reverse(*nanos));
    let total: u64 = stages.iter().map(|(_, n, _)| *n).sum();
    let mut out = String::new();
    for (name, nanos, calls) in stages.iter() {
        out.push_str(&format!(
            "  {name}: {:.2} с ({:.1}%), вызовов {calls}\n",
            *nanos as f64 / 1e9,
            100.0 * *nanos as f64 / total.max(1) as f64,
        ));
    }
    stages.clear();
    out
}
