use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

static CUDA_TRIM_POOL: AtomicBool = AtomicBool::new(false);
static DEBUG_SYNC: AtomicBool = AtomicBool::new(false);
static BENCH_MODE: AtomicBool = AtomicBool::new(false);
static LOG_LEVEL: AtomicU32 = AtomicU32::new(1);

pub fn cuda_trim_pool() -> bool { CUDA_TRIM_POOL.load(Ordering::Relaxed) }
pub fn set_cuda_trim_pool(v: bool) { CUDA_TRIM_POOL.store(v, Ordering::Relaxed); }
pub fn debug_sync() -> bool { DEBUG_SYNC.load(Ordering::Relaxed) }
pub fn set_debug_sync(v: bool) { DEBUG_SYNC.store(v, Ordering::Relaxed); }
pub fn bench_mode() -> bool { BENCH_MODE.load(Ordering::Relaxed) }
pub fn set_bench_mode(v: bool) { BENCH_MODE.store(v, Ordering::Relaxed); }
pub fn log_level() -> u32 { LOG_LEVEL.load(Ordering::Relaxed) }
pub fn set_log_level(v: u32) { LOG_LEVEL.store(v, Ordering::Relaxed); }
