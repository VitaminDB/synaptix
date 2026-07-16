pub fn init_tracing(level: tracing::Level) {
    let _ = tracing_subscriber::fmt()
        .with_max_level(level)
        .try_init();
}

pub fn init_default() {
    init_tracing(tracing::Level::INFO);
}
