
use std::time::Instant;

use synaptix_asr_whisper::loader::WhisperWeights;
use synaptix_asr_whisper::model::{WhisperDecodeState, WhisperModel};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::tensor::Tensor;
use synaptix_infer::graph_capture::GraphCapturer;
use synaptix_infer::InferError;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/whisper-large-v3-turbo.syn".to_string());
    let device = Device::Cuda(0);
    let dtype = DType::F16;
    let n: usize = std::env::var("SYN_REPLAY_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200);

    let w = WhisperWeights::open(&path, device, dtype).expect("open .syn");
    let model = WhisperModel::load(&w).expect("load model");
    let vocab = w.config.vocab_size;
    let max_target = w.config.max_target_positions;
    let d_model = w.config.d_model;
    let enc_len = w.config.max_source_positions; // 1500

    let enc_out = Tensor::randn(vec![1, enc_len, d_model], Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_device(device)
        .unwrap();

    let mut cache = no_grad(|| model.decoder.make_dev_cache(&enc_out, max_target, device, dtype)).unwrap();
    let mut state = WhisperDecodeState::new(device, dtype, vocab).unwrap();
    state.update(1000, 4).unwrap();

    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let mut capturer = GraphCapturer::new(3);
    let graph = {
        let m = &model;
        let st = &mut state;
        let cc = &mut cache;
        no_grad(|| {
            capturer.capture_with(&stream, |_s| {
                m.decoder.forward_decode_dev(st, cc).map_err(|e| InferError::Other(e.to_string()))
            })
        })
    }
    .expect("capture");
    let _ = graph.upload();

    for _ in 0..10 {
        graph.launch().unwrap();
    }
    stream.synchronize().unwrap();

    let t = Instant::now();
    for i in 0..n {
        state.update(1000 + (i % 50) as u32, (5 + i) as u32).unwrap();
        graph.launch().unwrap();
        stream.synchronize().unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("graph replay (decode step): {ms:.3} ms/step  ({n} replays)");
}
