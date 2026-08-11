//! Реальный CUDA graph capture/replay через cudarc.
//!
//! Тест работает только при `--features cuda` и наличии GPU. Без GPU тест пропускается
//! через `Option::None` из `setup()`, аналогично остальным CUDA-тестам workspace.
//!
//! В captured-step мы используем `cuLaunchHostFunc` (host callback), потому что это
//! единственная операция, которая *гарантированно* поддерживается graph capture в любых
//! режимах: `cuMemsetD8Async`/`cuMemcpyDtoDAsync` требуют либо mempool-памяти, либо
//! kernel-launch, а компилировать ядро ради smoke-теста — overkill.


use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use synaptix_infer::graph_capture::{policy::CaptureConfig, GraphCapturer, GraphReplayer};

/// Каждому тесту нужен СВОЙ stream: `cuStreamBeginCapture` блокирует операции на stream
/// до `cuStreamEndCapture`, а параллельные тесты на одном stream сразу дадут
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = ctx.new_stream().ok()?;
    Some((ctx, stream))
}

static COUNTER_A: AtomicU64 = AtomicU64::new(0);
static COUNTER_B: AtomicU64 = AtomicU64::new(0);
static COUNTER_C: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn inc_counter(arg: *mut std::ffi::c_void) {
    let cnt = &*(arg as *const AtomicU64);
    cnt.fetch_add(1, Ordering::SeqCst);
}

/// Воткнуть в текущий stream `cuLaunchHostFunc` который инкрементирует counter.
fn enqueue_host_inc(stream: &CudaStream, counter: &'static AtomicU64) -> Result<(), String> {
    let raw = stream.cu_stream();
    unsafe {
        cudarc::driver::result::stream::launch_host_function(
            raw,
            inc_counter,
            counter as *const AtomicU64 as *mut std::ffi::c_void,
        )
        .map_err(|e| format!("launch_host_function: {e}"))
    }
}

#[test]
fn capture_and_replay_counts_host_callbacks() {
    let (_ctx, stream) = match setup() {
        Some(p) => p,
        None => {
            eprintln!("CUDA не доступна — пропускаем тест");
            return;
        }
    };

    COUNTER_A.store(0, Ordering::SeqCst);

    let mut capturer = GraphCapturer::new(2);
    let graph_res = capturer.capture_with(&stream, |s| {
        enqueue_host_inc(s, &COUNTER_A)
            .map_err(synaptix_infer::error::InferError::Other)
    });
    let graph = match graph_res {
        Ok(g) => g,
        Err(e) => panic!("capture failed: {e}"),
    };
    assert!(capturer.is_captured());

    // Во время capture host_func ТОЛЬКО ЗАПИСЫВАЕТСЯ в граф — не выполняется. Поэтому после
    // begin_capture..end_capture counter не растёт. Реально работают warmup_steps=2 прохода.
    stream.synchronize().expect("sync after capture");
    assert_eq!(
        COUNTER_A.load(Ordering::SeqCst),
        2,
        "ожидали 2 host_func из warmup, capture-pass только записывается в граф"
    );

    let replayer = GraphReplayer::from_graph(CaptureConfig::default(), graph, 1, 1);
    assert_eq!(replayer.batch_size, Some(1));
    assert_eq!(replayer.seq_len, Some(1));

    // Каждый replay реально исполняет записанный host_func.
    for _ in 0..4 {
        if let Err(e) = replayer.replay() {
            panic!("replay failed: {e}");
        }
    }
    stream.synchronize().expect("sync after replay");
    assert_eq!(
        COUNTER_A.load(Ordering::SeqCst),
        2 + 4,
        "после 4 replay'ев counter = 2 (warmup) + 4 (replays) = 6"
    );
}

#[test]
fn replayer_upload_before_first_launch() {
    let (_ctx, stream) = match setup() {
        Some(p) => p,
        None => return,
    };

    COUNTER_B.store(0, Ordering::SeqCst);

    let mut capturer = GraphCapturer::new(1);
    let graph_res = capturer.capture_with(&stream, |s| {
        enqueue_host_inc(s, &COUNTER_B)
            .map_err(synaptix_infer::error::InferError::Other)
    });
    let graph = match graph_res {
        Ok(g) => g,
        Err(e) => panic!("capture failed: {e}"),
    };

    let mut replayer = GraphReplayer::new(CaptureConfig::default());
    replayer.set_graph(graph, 1, 1);
    replayer.upload().expect("upload");
    if let Err(e) = replayer.replay() {
        panic!("replay failed: {e}");
    }
    stream.synchronize().expect("sync");
    // warmup=1 (real) + capture-pass (запись) + replay=1 (real) == 2
    assert_eq!(COUNTER_B.load(Ordering::SeqCst), 2);
}

/// Граф можно re-launch'ить много раз без re-capture.
#[test]
fn graph_is_reusable_many_times() {
    let (_ctx, stream) = match setup() {
        Some(p) => p,
        None => return,
    };

    COUNTER_C.store(0, Ordering::SeqCst);

    let mut capturer = GraphCapturer::new(0);
    let graph_res = capturer.capture_with(&stream, |s| {
        enqueue_host_inc(s, &COUNTER_C)
            .map_err(synaptix_infer::error::InferError::Other)
    });
    let graph = match graph_res {
        Ok(g) => g,
        Err(e) => panic!("capture failed: {e}"),
    };
    stream.synchronize().expect("sync after capture");
    // warmup=0 + capture-pass (запись, не исполнение) == 0
    assert_eq!(COUNTER_C.load(Ordering::SeqCst), 0);

    let replayer = GraphReplayer::from_graph(CaptureConfig::default(), graph, 1, 1);
    for _ in 0..100 {
        replayer.replay().expect("replay");
    }
    stream.synchronize().expect("sync after 100 replays");
    assert_eq!(COUNTER_C.load(Ordering::SeqCst), 100);
}
