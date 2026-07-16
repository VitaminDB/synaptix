use std::sync::atomic::{AtomicBool, Ordering};

static GRAPH_DECODE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_graph_decode_enabled(on: bool) {
    GRAPH_DECODE_ENABLED.store(on, Ordering::Relaxed);
}

pub fn graph_decode_enabled() -> bool {
    GRAPH_DECODE_ENABLED.load(Ordering::Relaxed)
}
