use std::sync::atomic::{AtomicBool, Ordering};

static MEM_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_mem_trace_enabled(on: bool) {
    MEM_TRACE_ENABLED.store(on, Ordering::Relaxed);
}

pub fn mem_trace_enabled() -> bool {
    MEM_TRACE_ENABLED.load(Ordering::Relaxed)
}
