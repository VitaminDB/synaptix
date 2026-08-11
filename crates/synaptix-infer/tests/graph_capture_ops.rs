use synaptix_infer::graph_capture::{policy::CaptureConfig, GraphCapturer, GraphReplayer};

#[test]
fn capturer_starts_empty() {
    let c = GraphCapturer::new(3);
    assert_eq!(c.warmup_steps, 3);
    assert!(!c.is_captured());
}

#[test]
fn replayer_cannot_replay_without_graph() {
    let cfg = CaptureConfig::default();
    let r = GraphReplayer::new(cfg);
    assert!(!r.can_replay(1, 128));
}
