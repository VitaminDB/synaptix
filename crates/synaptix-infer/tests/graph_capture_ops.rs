//! Unit-тесты `graph_capture` без CUDA (default features).
//!
//! Реальный capture/replay через cudarc проверяется в `cuda_graph_capture.rs`
//! (`#[cfg(feature = "cuda")]`).

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

#[cfg(not(feature = "cuda"))]
#[test]
fn capture_returns_error_without_cuda_feature() {
    let mut c = GraphCapturer::new(0);
    let res = c.capture();
    assert!(res.is_err(), "capture без cuda фичи должен возвращать ошибку");
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("CUDA graph capture requires `cuda` feature"),
        "ожидали внятное сообщение про missing-feature, получили: {msg}"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn replay_returns_error_without_cuda_feature() {
    let r = GraphReplayer::new(CaptureConfig::default());
    let res = r.replay();
    assert!(res.is_err());
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("CUDA graph replay requires `cuda` feature"),
        "ожидали внятное сообщение про missing-feature, получили: {msg}"
    );
}
