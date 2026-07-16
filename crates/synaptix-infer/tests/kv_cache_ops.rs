use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_infer::kv_cache::{FullKvCache, KvCache, PagedKvCache, PrefixKvCache};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::assert_allclose;

fn setup() { ensure_registered(); }

fn make_tok(seed: u64, num_heads: usize, head_dim: usize, n: usize) -> (Tensor, Tensor) {
    let _ = seed;
    let total = num_heads * n * head_dim;
    let kv: Vec<f32> = (0..total).map(|i| (i + seed as usize) as f32 * 0.01).collect();
    let kv2: Vec<f32> = (0..total).map(|i| (i + seed as usize + 1000) as f32 * 0.01).collect();
    let k = Tensor::from_vec::<_, f32>(kv, vec![num_heads, n, head_dim], Device::Cpu).unwrap();
    let v = Tensor::from_vec::<_, f32>(kv2, vec![num_heads, n, head_dim], Device::Cpu).unwrap();
    let k = k.unsqueeze(0).unwrap();
    let v = v.unsqueeze(0).unwrap();
    (k, v)
}

#[test]
fn t20_1_full_append_and_get() {
    setup();
    let mut cache = FullKvCache::new(2, 4, 16, 32, Device::Cpu, DType::F32).unwrap();
    let (k1, v1) = make_tok(0, 4, 16, 3);
    cache.append(0, &k1, &v1).unwrap();
    cache.append(1, &k1, &v1).unwrap();
    assert_eq!(cache.seq_len(), 3);

    let (k_got, v_got) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 3, 16]);
    assert_allclose(k_got, &k1, 1e-6, 1e-6);
    assert_allclose(v_got, &v1, 1e-6, 1e-6);

    let (k2, v2) = make_tok(10, 4, 16, 2);
    cache.append(0, &k2, &v2).unwrap();
    cache.append(1, &k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 5);
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 5, 16]);
}

#[test]
fn t20_2_full_reset_to_partial() {
    setup();
    let mut cache = FullKvCache::new(1, 4, 16, 32, Device::Cpu, DType::F32).unwrap();
    let (k1, v1) = make_tok(0, 4, 16, 3);
    cache.append(0, &k1, &v1).unwrap();
    let (k2, v2) = make_tok(10, 4, 16, 4);
    cache.append(0, &k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 7);

    cache.reset_to(5);
    assert_eq!(cache.seq_len(), 5);
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 5, 16]);

    cache.reset_to(2);
    assert_eq!(cache.seq_len(), 2);
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 2, 16]);
}

#[test]
fn t20_3_paged_append_get_full_concat() {
    setup();
    let mut cache = PagedKvCache::new(2, 4, 16, 4, 8, Device::Cpu, DType::F32).unwrap();
    let (k1, v1) = make_tok(0, 4, 16, 2);
    cache.append(0, &k1, &v1).unwrap();
    cache.append(1, &k1, &v1).unwrap();
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 2, 16]);
    assert_allclose(k_got, &k1, 1e-6, 1e-6);

    let (k2, v2) = make_tok(50, 4, 16, 5);
    cache.append(0, &k2, &v2).unwrap();
    cache.append(1, &k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 7);
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 7, 16]);
    assert!(cache.num_allocated_blocks(0) == 2);
}

#[test]
fn t20_4_paged_reset_to_partial() {
    setup();
    let mut cache = PagedKvCache::new(1, 4, 16, 4, 4, Device::Cpu, DType::F32).unwrap();
    for i in 0..10 {
        let (k, v) = make_tok(i, 4, 16, 1);
        cache.append(0, &k, &v).unwrap();
    }
    assert_eq!(cache.seq_len(), 10);

    cache.reset_to(6);
    assert_eq!(cache.seq_len(), 6);
    let (k_got, _) = cache.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 6, 16]);
    assert_eq!(cache.num_allocated_blocks(0), 2);

    cache.reset_to(2);
    assert_eq!(cache.seq_len(), 2);
    assert_eq!(cache.num_allocated_blocks(0), 1);
}

#[test]
fn t20_5_prefix_kv_cache() {
    setup();
    let mut prefix_inner = FullKvCache::new(1, 4, 16, 32, Device::Cpu, DType::F32).unwrap();
    let (pk, pv) = make_tok(100, 4, 16, 3);
    prefix_inner.append(0, &pk, &pv).unwrap();

    let prefix: Arc<dyn KvCache> = Arc::new(prefix_inner);
    let tail: Box<dyn KvCache> = Box::new(FullKvCache::new(1, 4, 16, 32, Device::Cpu, DType::F32).unwrap());
    let mut combined = PrefixKvCache::with_prefix(prefix, tail);

    let (tk, tv) = make_tok(200, 4, 16, 2);
    combined.append(0, &tk, &tv).unwrap();
    assert_eq!(combined.seq_len(), 5);
    assert_eq!(combined.prefix_len(), 3);

    let (k_got, _) = combined.get(0).unwrap();
    assert_eq!(k_got.dims(), &[1, 4, 5, 16]);
}
