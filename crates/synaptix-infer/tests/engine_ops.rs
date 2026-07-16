//! A1/A2: SimpleEngine::step и InferPipeline::step_batch через mock-ForwardFn.

use std::any::Any;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_infer::batch::InferBatch;
use synaptix_infer::engine::{ForwardFn, InferenceEngine, SimpleEngine};
use synaptix_infer::pipeline::{InferPipeline, InferPipelineConfig};
use synaptix_infer::session::{InferRequest, InferSession, SamplingParams};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::rng::Philox4x32;

fn setup() {
    ensure_registered();
}

/// Детерминированная «модель»: для каждой позиции логиты дают argmax =
/// `(input_token + 1) % vocab`. Возвращает `[seq, vocab]` — step_session берёт
/// последнюю строку.
struct CounterLm {
    vocab: usize,
}

impl ForwardFn for CounterLm {
    fn forward(&self, input_ids: &Tensor, _position_ids: &Tensor, _kv: &mut dyn Any) -> synaptix_infer::Result<Tensor> {
        let toks: Vec<u32> = input_ids.to_vec1::<u32>().unwrap();
        let n = toks.len();
        let mut data = vec![0f32; n * self.vocab];
        for (i, &t) in toks.iter().enumerate() {
            let nxt = ((t as usize) + 1) % self.vocab;
            data[i * self.vocab + nxt] = 10.0;
        }
        Ok(Tensor::from_vec::<_, f32>(data, vec![n, self.vocab], Device::Cpu).unwrap())
    }
}

fn greedy(max_new: usize, stop: Vec<u32>) -> SamplingParams {
    SamplingParams { max_new_tokens: max_new, stop_token_ids: stop, ..SamplingParams::greedy() }
}

#[test]
fn t30_1_engine_greedy_until_max_len() {
    setup();
    let mut eng = SimpleEngine::new(Box::new(CounterLm { vocab: 16 }), 16, 4, Device::Cpu, 0);
    let id = eng.submit(InferRequest::new(vec![1, 2, 3], greedy(4, vec![]))).unwrap();

    let mut all = Vec::new();
    for _ in 0..10 {
        let toks = eng.step().unwrap();
        for t in toks {
            assert_eq!(t.request_id, id);
            all.push(t.clone());
        }
        if eng.is_idle() {
            break;
        }
    }
    let ids: Vec<u32> = all.iter().map(|t| t.token_id).collect();
    assert_eq!(ids, vec![4, 5, 6, 7], "incrementing greedy from last prompt token 3");
    let last = all.last().unwrap();
    assert!(last.is_last);
    assert_eq!(last.stop_reason, Some(synaptix_infer::sampling::stop_criteria::StopReason::MaxLength));
    assert!(eng.is_idle());
}

#[test]
fn t30_2_engine_stops_on_eos_token() {
    setup();
    let mut eng = SimpleEngine::new(Box::new(CounterLm { vocab: 16 }), 16, 4, Device::Cpu, 0);
    let _ = eng.submit(InferRequest::new(vec![1, 2, 3], greedy(100, vec![6]))).unwrap();

    let mut ids = Vec::new();
    for _ in 0..20 {
        for t in eng.step().unwrap() {
            ids.push(t.token_id);
            if t.is_last {
                assert_eq!(t.stop_reason, Some(synaptix_infer::sampling::stop_criteria::StopReason::EosToken));
            }
        }
        if eng.is_idle() {
            break;
        }
    }
    assert_eq!(ids, vec![4, 5, 6], "stop right after emitting EOS token 6");
    assert!(eng.is_idle());
}

#[test]
fn t30_3_engine_batch_of_two() {
    setup();
    let mut eng = SimpleEngine::new(Box::new(CounterLm { vocab: 32 }), 32, 4, Device::Cpu, 0);
    let id_a = eng.submit(InferRequest::new(vec![10], greedy(3, vec![]))).unwrap();
    let id_b = eng.submit(InferRequest::new(vec![20], greedy(3, vec![]))).unwrap();

    let mut per_req: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
    for _ in 0..10 {
        let toks = eng.step().unwrap();
        for t in toks {
            per_req.entry(t.request_id).or_default().push(t.token_id);
        }
        if eng.is_idle() {
            break;
        }
    }
    assert_eq!(per_req[&id_a], vec![11, 12, 13]);
    assert_eq!(per_req[&id_b], vec![21, 22, 23]);
}

#[test]
fn t30_4_engine_cancel_active() {
    setup();
    let mut eng = SimpleEngine::new(Box::new(CounterLm { vocab: 32 }), 32, 4, Device::Cpu, 0);
    let id_a = eng.submit(InferRequest::new(vec![10], greedy(100, vec![]))).unwrap();
    let id_b = eng.submit(InferRequest::new(vec![20], greedy(100, vec![]))).unwrap();

    // Один шаг — обе активны.
    let toks = eng.step().unwrap();
    assert_eq!(toks.len(), 2);
    assert_eq!(eng.pending(), 2);

    // Отменяем A — должна остаться только B.
    eng.cancel(id_a);
    assert_eq!(eng.pending(), 1);
    let toks = eng.step().unwrap();
    assert_eq!(toks.len(), 1);
    assert_eq!(toks[0].request_id, id_b);
}

#[test]
fn t30_5_pipeline_step_batch_same_path() {
    setup();
    let cfg = InferPipelineConfig {
        num_layers: 1,
        num_heads: 1,
        head_dim: 8,
        vocab_size: 16,
        max_seq_len: 64,
        device: Device::Cpu,
        dtype: DType::F32,
    };
    let pipe = InferPipeline::new(cfg, Box::new(CounterLm { vocab: 16 }));
    let mut batch = InferBatch::new();
    batch.add(InferSession::new(InferRequest::new(vec![1, 2, 3], greedy(2, vec![]))));
    batch.add(InferSession::new(InferRequest::new(vec![5], greedy(2, vec![]))));
    let mut rng = Philox4x32::new(0);

    // Шаг 1 (prefill): 3→4 и 5→6.
    let t1 = pipe.step_batch(&mut batch, &mut rng).unwrap();
    assert_eq!(t1.iter().map(|t| t.token_id).collect::<Vec<_>>(), vec![4, 6]);
    // Шаг 2 (decode): 4→5 (max reached) и 6→7 (max reached).
    let t2 = pipe.step_batch(&mut batch, &mut rng).unwrap();
    assert_eq!(t2.iter().map(|t| t.token_id).collect::<Vec<_>>(), vec![5, 7]);
    assert!(t2.iter().all(|t| t.is_last));
}
