use synaptix_infer::scheduler::disaggregated::DisaggregatedScheduler;
use synaptix_infer::scheduler::{ChunkedPrefillScheduler, ContinuousBatchScheduler, FcfsScheduler, Scheduler};
use synaptix_infer::session::{InferRequest, InferSession, SamplingParams};

fn mk_session(n_prompt: usize, max_new: usize) -> InferSession {
    let prompt: Vec<u32> = (0..n_prompt as u32).collect();
    let sp = SamplingParams { max_new_tokens: max_new, ..SamplingParams::greedy() };
    InferSession::new(InferRequest::new(prompt, sp))
}

#[test]
fn t21_1_fcfs_basic() {
    let mut sched = FcfsScheduler::new(2, 100);
    sched.add_request(mk_session(10, 10));
    sched.add_request(mk_session(15, 5));
    sched.add_request(mk_session(8, 8));
    assert_eq!(sched.pending_count(), 3);
    let batch = sched.schedule().unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(sched.pending_count(), 1);
}

#[test]
fn t21_2_continuous_batch_admit_within_budget() {
    let mut sched = ContinuousBatchScheduler::new(4, 100);
    sched.add_request(mk_session(30, 10));
    sched.add_request(mk_session(30, 10));
    sched.add_request(mk_session(30, 10));
    sched.add_request(mk_session(30, 10));
    assert_eq!(sched.pending_count(), 4);
    let batch = sched.schedule().unwrap();
    assert_eq!(batch.len(), 3);
}

#[test]
fn t21_3_chunked_prefill() {
    let mut sched = ChunkedPrefillScheduler::new(64, 4);
    sched.add_request(mk_session(200, 5));
    sched.add_request(mk_session(50, 5));
    assert_eq!(sched.pending_count(), 2);
    let batch = sched.schedule().unwrap();
    assert_eq!(batch.len(), 2);
}

#[test]
fn t21_4_disaggregated_routing_and_capacity() {
    let mut s = DisaggregatedScheduler::new(2, 2);
    assert_eq!(s.route_prefill().unwrap(), 0);
    assert_eq!(s.route_prefill().unwrap(), 1);
    assert!(s.route_prefill().is_err(), "оба prefill-воркера заняты (cap=1)");
    assert_eq!(s.prefill_inflight(), 2);
    assert_eq!(s.prefill_capacity(), 2);
}

#[test]
fn t21_5_disaggregated_least_loaded() {
    let mut s = DisaggregatedScheduler::with_capacity(2, 2, 2, 2);
    let seq: Vec<usize> = (0..4).map(|_| s.route_prefill().unwrap()).collect();
    assert_eq!(seq, vec![0, 1, 0, 1], "round-robin по наименее загруженному");
    assert_eq!(s.prefill_load(), &[2, 2]);
    assert!(s.route_prefill().is_err());
}

#[test]
fn t21_6_disaggregated_migrate_prefill_to_decode() {
    let mut s = DisaggregatedScheduler::new(1, 1);
    let pw = s.route_prefill().unwrap();
    assert_eq!(pw, 0);
    let dw = s.migrate(pw).unwrap();
    assert_eq!(dw, 0);
    assert_eq!(s.prefill_inflight(), 0, "prefill-слот освобождён");
    assert_eq!(s.decode_inflight(), 1);
}

#[test]
fn t21_7_disaggregated_complete_frees_slot() {
    let mut s = DisaggregatedScheduler::new(1, 1);
    s.route_decode().unwrap();
    assert!(s.route_decode().is_err());
    s.complete_decode(0);
    assert!(s.route_decode().is_ok(), "слот освобождён после complete");
}
