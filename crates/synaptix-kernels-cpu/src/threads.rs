use std::sync::OnceLock;

static POOL: OnceLock<usize> = OnceLock::new();

pub fn pool() -> usize {
    *POOL.get_or_init(|| {
        let want = std::env::var("SYNAPTIX_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .or_else(fast_cores)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(want)
            .thread_name(|i| format!("synaptix-cpu-{i}"))
            .build_global();
        rayon::current_num_threads().max(1)
    })
}

fn fast_cores() -> Option<usize> {
    let mut freqs = Vec::new();
    for i in 0..256 {
        let p = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
        match std::fs::read_to_string(&p) {
            Ok(t) => freqs.push(t.trim().parse::<u64>().ok()?),
            Err(_) if i == 0 => return None,
            Err(_) => break,
        }
    }
    let max = *freqs.iter().max()?;
    let fast = freqs.iter().filter(|&&f| f * 10 >= max * 7).count();
    (fast < freqs.len()).then_some(fast.max(1))
}
