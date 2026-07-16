//! B4: TieredKvCache — вытеснение gpu→cpu при переполнении и промоушен cpu→gpu.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_infer::kv_cache::tiered::TieredKvCache;
use synaptix_infer::kv_cache::KvCache;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::assert_allclose;

fn setup() { ensure_registered(); }

fn tok(seed: usize, num_heads: usize, head_dim: usize) -> (Tensor, Tensor) {
    let total = num_heads * head_dim;
    let k: Vec<f32> = (0..total).map(|i| (i + seed * 1000) as f32 * 0.01).collect();
    let v: Vec<f32> = (0..total).map(|i| (i + seed * 1000 + 500) as f32 * 0.01).collect();
    let k = Tensor::from_vec::<_, f32>(k, vec![1, num_heads, 1, head_dim], Device::Cpu).unwrap();
    let v = Tensor::from_vec::<_, f32>(v, vec![1, num_heads, 1, head_dim], Device::Cpu).unwrap();
    (k, v)
}

#[test]
fn t41_1_overflow_evicts_oldest_gpu_to_cpu() {
    setup();
    let mut cache = TieredKvCache::new(2, 2, 8, 4, 8, Device::Cpu, Device::Cpu, DType::F32).unwrap();
    for i in 0..8 {
        let (k, v) = tok(i, 2, 8);
        cache.append(0, &k, &v).unwrap();
        cache.append(1, &k, &v).unwrap();
    }
    assert_eq!(cache.gpu_len(), 4, "gpu holds recent window");
    assert_eq!(cache.cpu_len(), 4, "older tokens spilled to cpu");
    assert_eq!(cache.seq_len(), 8);
    let (k, _) = cache.get(0).unwrap();
    assert_eq!(k.dims(), &[1, 2, 4, 8]);
}

#[test]
fn t41_2_gpu_window_is_last_tokens() {
    setup();
    let mut cache = TieredKvCache::new(1, 2, 8, 4, 8, Device::Cpu, Device::Cpu, DType::F32).unwrap();
    for i in 0..7 {
        let (k, v) = tok(i, 2, 8);
        cache.append(0, &k, &v).unwrap();
    }
    // gpu должен держать токены 3,4,5,6 (последние 4).
    let expected_k: Vec<Tensor> = (3..7).map(|i| tok(i, 2, 8).0).collect();
    let refs: Vec<&Tensor> = expected_k.iter().collect();
    let expected = Tensor::cat(&refs, 2).unwrap();
    let (k, _) = cache.get(0).unwrap();
    assert_eq!(k.dims(), &[1, 2, 4, 8]);
    assert_allclose(k, &expected, 1e-6, 1e-6);
}

#[test]
fn t41_3_promote_moves_cpu_back_to_gpu() {
    setup();
    let mut cache = TieredKvCache::new(1, 2, 8, 4, 8, Device::Cpu, Device::Cpu, DType::F32).unwrap();
    for i in 0..8 {
        let (k, v) = tok(i, 2, 8);
        cache.append(0, &k, &v).unwrap();
    }
    assert_eq!((cache.gpu_len(), cache.cpu_len()), (4, 4));

    // Сжимаем до 2 логических токенов: оба окажутся в cpu, gpu пуст.
    cache.reset_to(2);
    assert_eq!((cache.gpu_len(), cache.cpu_len(), cache.seq_len()), (0, 2, 2));

    // Промоушен возвращает их в gpu (есть место).
    let moved = cache.promote().unwrap();
    assert_eq!(moved, 2);
    assert_eq!((cache.gpu_len(), cache.cpu_len(), cache.seq_len()), (2, 0, 2));

    // Это первые два токена последовательности (0,1) — порядок сохранён.
    let expected_k: Vec<Tensor> = (0..2).map(|i| tok(i, 2, 8).0).collect();
    let refs: Vec<&Tensor> = expected_k.iter().collect();
    let expected = Tensor::cat(&refs, 2).unwrap();
    let (k, _) = cache.get(0).unwrap();
    assert_allclose(k, &expected, 1e-6, 1e-6);
}

#[test]
fn t41_4_promote_full_gpu_noop() {
    setup();
    let mut cache = TieredKvCache::new(1, 2, 8, 4, 8, Device::Cpu, Device::Cpu, DType::F32).unwrap();
    for i in 0..6 {
        let (k, v) = tok(i, 2, 8);
        cache.append(0, &k, &v).unwrap();
    }
    assert_eq!((cache.gpu_len(), cache.cpu_len()), (4, 2));
    // gpu заполнен — промоутить некуда.
    assert_eq!(cache.promote().unwrap(), 0);
    assert_eq!((cache.gpu_len(), cache.cpu_len()), (4, 2));
}
