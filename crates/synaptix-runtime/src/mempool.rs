use std::sync::atomic::{AtomicUsize, Ordering};

static POOL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static POOL_PEAK: AtomicUsize = AtomicUsize::new(0);

pub fn record_alloc(bytes: usize) {
    let prev = POOL_ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
    let new = prev + bytes;
    let mut peak = POOL_PEAK.load(Ordering::Relaxed);
    while new > peak {
        match POOL_PEAK.compare_exchange_weak(peak, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(cur) => peak = cur,
        }
    }
}

pub fn record_free(bytes: usize) {
    POOL_ALLOCATED.fetch_sub(bytes.min(POOL_ALLOCATED.load(Ordering::Relaxed)), Ordering::Relaxed);
}

pub fn allocated_bytes() -> usize { POOL_ALLOCATED.load(Ordering::Relaxed) }
pub fn peak_bytes() -> usize { POOL_PEAK.load(Ordering::Relaxed) }
pub fn reset_peak() { POOL_PEAK.store(POOL_ALLOCATED.load(Ordering::Relaxed), Ordering::Relaxed); }
pub fn allocated_mb() -> f64 { allocated_bytes() as f64 / 1_048_576.0 }
pub fn peak_mb() -> f64 { peak_bytes() as f64 / 1_048_576.0 }
