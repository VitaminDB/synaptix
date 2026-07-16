use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn set_num_threads(n: usize) { NUM_THREADS.store(n, Ordering::Relaxed); }

pub fn num_threads() -> usize {
    let n = NUM_THREADS.load(Ordering::Relaxed);
    if n == 0 { available_parallelism() } else { n }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}
