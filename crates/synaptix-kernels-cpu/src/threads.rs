use std::sync::OnceLock;

static POOL: OnceLock<usize> = OnceLock::new();

pub fn pool() -> usize {
    *POOL.get_or_init(|| {
        let want = std::env::var("SYNAPTIX_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(want)
            .thread_name(|i| format!("synaptix-cpu-{i}"))
            .build_global();
        rayon::current_num_threads().max(1)
    })
}
