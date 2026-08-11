//! CPU/CUDA тесты для `synaptix-core::memory::{cuda_pool, pinned}`.

use synaptix_core::memory::cuda_pool::{
    cuda_allocated_bytes, cuda_peak_bytes, record_cuda_alloc, record_cuda_free, reset_peak,
    set_trim_threshold, trim_cuda_mempool, trim_threshold,
};
use synaptix_core::memory::pinned::PinnedBuf;

#[test]
fn alloc_counters_track_bytes_and_peak() {
    // Сохраняем-восстанавливаем стартовое состояние, т.к. счётчики глобальные.
    let start_alloc = cuda_allocated_bytes();
    let start_peak = cuda_peak_bytes();

    record_cuda_alloc(1024);
    record_cuda_alloc(2048);
    assert!(cuda_allocated_bytes() >= start_alloc + 3072);
    assert!(cuda_peak_bytes() >= start_peak.max(start_alloc + 3072));

    record_cuda_free(1024);
    assert!(cuda_allocated_bytes() >= start_alloc + 2048);
    // peak не падает после free
    assert!(cuda_peak_bytes() >= start_alloc + 3072);

    record_cuda_free(2048);
}

#[test]
fn reset_peak_clamps_to_current_allocated() {
    record_cuda_alloc(4096);
    let current = cuda_allocated_bytes();
    reset_peak();
    assert_eq!(cuda_peak_bytes(), current);
    record_cuda_free(4096);
}

#[test]
fn trim_threshold_roundtrip() {
    let prev = trim_threshold();
    set_trim_threshold(64 * 1024 * 1024);
    assert_eq!(trim_threshold(), 64 * 1024 * 1024);
    set_trim_threshold(prev);
}

#[test]
fn trim_cuda_mempool_is_safe_without_cuda() {
    // На CPU-сборке = no-op. На CUDA-сборке без устройства = silent error swallow.
    // В обоих случаях НЕ должно паниковать.
    trim_cuda_mempool();
}

#[test]
fn pinned_buf_read_write() {
    let mut buf = PinnedBuf::new(128);
    assert_eq!(buf.len(), 128);
    assert!(!buf.is_empty());
    // zero-init
    assert!(buf.as_slice().iter().all(|&b| b == 0));

    for (i, b) in buf.as_mut_slice().iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7);
    }
    for (i, &b) in buf.as_slice().iter().enumerate() {
        assert_eq!(b, (i as u8).wrapping_mul(7));
    }
}

#[test]
fn pinned_buf_empty_is_valid() {
    let buf = PinnedBuf::new(0);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    // as_slice на len=0 buf должен быть валидным пустым slice
    assert_eq!(buf.as_slice().len(), 0);
}

#[test]
fn pinned_buf_drop_does_not_panic() {
    for _ in 0..32 {
        let _b = PinnedBuf::new(1024);
    }
}

#[test]
fn pinned_buf_uses_cuda_when_available() {
    // На системе с CUDA — буфер должен быть page-locked.
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA не доступна — пропускаем тест");
        return;
    }
    let buf = PinnedBuf::new(4096);
    assert!(
        buf.is_pinned(),
        "при наличии CUDA PinnedBuf должен использовать cuMemHostAlloc"
    );
}

#[test]
fn trim_cuda_mempool_device_returns_ok() {
    use synaptix_core::memory::cuda_pool::trim_cuda_mempool_device;

    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA не доступна — пропускаем тест");
        return;
    }
    set_trim_threshold(0);
    trim_cuda_mempool_device(0).expect("trim_cuda_mempool_device должен вернуть Ok на исправном GPU");
}

#[test]
fn pinned_buf_can_be_used_with_cuda_htod() {
    // Не идеальный тест (нужен CudaStream), но хотя бы проверяем что pointer not null.
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let mut buf = PinnedBuf::new(256);
    buf.as_mut_slice()[0] = 42;
    assert!(!buf.as_ptr().is_null());
    assert_eq!(buf.as_slice()[0], 42);
}
