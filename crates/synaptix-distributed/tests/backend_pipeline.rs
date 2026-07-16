//! In-memory backend + pipeline_parallel: multi-thread, single-process.
//!
//! Все тесты звонят `destroy_process_group()` в начале (на случай если предыдущий тест
//! оставил state) — backend singleton, поэтому тесты идут **серийно**:
//! `--test-threads=1`. Это можно было бы обойти разделением OnceLock по группам, но для
//! текущих 2 TODO это overkill. Серийность задана через `serial_test`-эмулятор: все тесты
//! берут одну глобальную мьютекс `TEST_LOCK`.

use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_distributed::init::{
    barrier, destroy_process_group, init_process_group, is_initialized, local_rank, world_size,
};
use synaptix_distributed::pipeline_parallel::PipelineStage;
use synaptix_kernels_cpu::ensure_registered;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn reset() {
    ensure_registered();
    destroy_process_group();
}

fn t1(v: &[f32]) -> Tensor {
    Tensor::from_vec::<_, f32>(v.to_vec(), vec![v.len()], Device::Cpu).unwrap()
}

#[test]
fn init_local_backend_sets_world_state() {
    let _g = TEST_LOCK.lock();
    reset();
    init_process_group("local", 0, 1).unwrap();
    assert!(is_initialized());
    assert_eq!(world_size(), Some(1));
    assert_eq!(local_rank(), Some(0));
    destroy_process_group();
    assert!(!is_initialized());
    assert!(local_rank().is_none());
}

#[test]
fn init_unknown_backend_errors() {
    let _g = TEST_LOCK.lock();
    reset();
    let r = init_process_group("nccl", 0, 1);
    assert!(r.is_err());
    let r = init_process_group("xyz", 0, 1);
    assert!(r.is_err());
}

#[test]
fn init_rank_out_of_range() {
    let _g = TEST_LOCK.lock();
    reset();
    let r = init_process_group("local", 4, 4);
    assert!(r.is_err());
}

#[test]
fn pipeline_first_stage_cannot_recv() {
    let _g = TEST_LOCK.lock();
    reset();
    init_process_group("local", 0, 2).unwrap();
    let st = PipelineStage::new(0, 2);
    assert!(st.is_first());
    let r = st.recv_forward();
    assert!(r.is_err());
}

#[test]
fn pipeline_two_stage_forward_pass() {
    let _g = TEST_LOCK.lock();
    reset();

    // 2 потока: stage 0 шлёт две activation, stage 1 их получает и складывает.
    let h0 = thread::spawn(|| {
        init_process_group("local", 0, 2).unwrap();
        let st = PipelineStage::new(0, 2);
        st.send_forward(&t1(&[1.0, 2.0, 3.0])).unwrap();
        st.send_forward(&t1(&[10.0, 20.0, 30.0])).unwrap();
        barrier().unwrap();
    });
    let h1 = thread::spawn(|| {
        init_process_group("local", 1, 2).unwrap();
        let st = PipelineStage::new(1, 2);
        let a = st.recv_forward().unwrap();
        let b = st.recv_forward().unwrap();
        barrier().unwrap();
        let av = a.to_vec1::<f32>().unwrap();
        let bv = b.to_vec1::<f32>().unwrap();
        assert_eq!(av, vec![1.0, 2.0, 3.0]);
        assert_eq!(bv, vec![10.0, 20.0, 30.0]);
    });
    h0.join().unwrap();
    h1.join().unwrap();
    destroy_process_group();
}

#[test]
fn pipeline_recv_timeout_when_no_sender() {
    let _g = TEST_LOCK.lock();
    reset();
    // 2 потока: только stage 1, без отправителя — recv с timeout должен вернуть Err.
    init_process_group("local", 1, 2).unwrap();
    let st = PipelineStage::new(1, 2);
    let r = st.recv_forward_timeout(Duration::from_millis(50));
    assert!(r.is_err(), "ожидали timeout-err когда нет sender'а");
    destroy_process_group();
}

#[test]
fn barrier_synchronizes_three_threads() {
    let _g = TEST_LOCK.lock();
    reset();

    // 3 потока подходят к barrier с разными задержками; после barrier counter == 3.
    let counter = Arc::new(parking_lot::Mutex::new(0usize));
    let mut handles = Vec::new();
    for rank in 0..3 {
        let c = counter.clone();
        handles.push(thread::spawn(move || {
            init_process_group("local", rank, 3).unwrap();
            // разное время «работы» до barrier'а:
            thread::sleep(Duration::from_millis(10 * rank as u64));
            barrier().unwrap();
            *c.lock() += 1;
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*counter.lock(), 3);
    destroy_process_group();
}

#[test]
fn three_stage_pipeline_forward_chain() {
    let _g = TEST_LOCK.lock();
    reset();

    // stage 0 → stage 1 → stage 2: каждый stage добавляет +1.0 к received tensor (loopback +1).
    let h0 = thread::spawn(|| {
        init_process_group("local", 0, 3).unwrap();
        let st = PipelineStage::new(0, 3);
        st.send_forward(&t1(&[1.0, 1.0])).unwrap();
        barrier().unwrap();
    });
    let h1 = thread::spawn(|| {
        init_process_group("local", 1, 3).unwrap();
        let st = PipelineStage::new(1, 3);
        let r = st.recv_forward().unwrap();
        let r_plus = r.add_scalar(1.0).unwrap();
        st.send_forward(&r_plus).unwrap();
        barrier().unwrap();
    });
    let h2 = thread::spawn(|| {
        init_process_group("local", 2, 3).unwrap();
        let st = PipelineStage::new(2, 3);
        let r = st.recv_forward().unwrap();
        let r_plus = r.add_scalar(1.0).unwrap();
        barrier().unwrap();
        let v = r_plus.to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![3.0, 3.0], "1 + 1 + 1 = 3 по всем компонентам");
    });
    h0.join().unwrap();
    h1.join().unwrap();
    h2.join().unwrap();
    destroy_process_group();
}

#[test]
fn last_stage_send_is_noop() {
    let _g = TEST_LOCK.lock();
    reset();
    init_process_group("local", 1, 2).unwrap();
    let st = PipelineStage::new(1, 2);
    // is_last == true. Не должно паниковать и не должно бросать ошибку, даже если
    // вообще никто не получит этот tensor.
    let r = st.send_forward(&t1(&[42.0]));
    assert!(r.is_ok());
    destroy_process_group();
}

#[test]
fn init_idempotent_for_same_world_size() {
    let _g = TEST_LOCK.lock();
    reset();
    init_process_group("local", 0, 4).unwrap();
    init_process_group("local", 0, 4).unwrap(); // повторный вызов того же rank'а — Ok.
    destroy_process_group();
}

#[test]
fn init_mismatched_world_size_errors() {
    let _g = TEST_LOCK.lock();
    reset();
    init_process_group("local", 0, 4).unwrap();
    let r = init_process_group("local", 0, 8); // другой world_size — Err.
    assert!(r.is_err());
    destroy_process_group();
}
