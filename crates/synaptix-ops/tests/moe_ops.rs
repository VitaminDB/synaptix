//! Аналитические unit-тесты для дискретных/index MoE-операций (без torch).

use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::ffn::moe::dispatch::{gather_tokens, scatter_tokens};
use synaptix_ops::ffn::moe::ep_all_to_all;
use synaptix_ops::ffn::moe::router::hash::hash_router;
use synaptix_ops::ffn::moe::router::mod_routing::mod_router;

fn setup() {
    ensure_registered();
}

fn dev() -> synaptix_core::device::Device {
    synaptix_core::device::Device::Cpu
}

#[test]
fn scatter_gather_roundtrip() {
    setup();
    // x[4,2], perm индексы → scatter, затем gather восстанавливает исходник
    let x = Tensor::from_vec::<_, f32>(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![4, 2],
        dev(),
    )
    .unwrap();
    let idx = Tensor::from_vec::<_, f32>(vec![2.0, 0.0, 3.0, 1.0], vec![4], dev()).unwrap();
    let scattered = scatter_tokens(&x, &idx).unwrap();
    // out[i] = x[idx[i]] → строка 0 = x[2] = [4,5]
    let s = scattered.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(&s[0..2], &[4.0, 5.0]);
    assert_eq!(&s[2..4], &[0.0, 1.0]);
    // gather по тем же индексам обращает перестановку
    let restored = gather_tokens(&scattered, &idx).unwrap();
    let r = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(r, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn hash_router_modulo() {
    setup();
    let ids = Tensor::from_vec::<_, f32>(vec![0.0, 1.0, 5.0, 7.0, 8.0], vec![5], dev()).unwrap();
    let out = hash_router(&ids, 3).unwrap();
    let o = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // token_id % 3
    assert_eq!(o, vec![0.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn mod_router_threshold() {
    setup();
    let logits = Tensor::from_vec::<_, f32>(vec![0.2, 0.8, 0.5, 0.1], vec![4], dev()).unwrap();
    let (mask, any) = mod_router(&logits, 0.5).unwrap();
    let m = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // >= 0.5 → 1.0
    assert_eq!(m, vec![0.0, 1.0, 1.0, 0.0]);
    assert!(any);

    let (_, any2) = mod_router(&logits, 1.5).unwrap();
    assert!(!any2, "ни один токен не превышает порог 1.5");
}

#[test]
fn ep_all_to_all_identity() {
    setup();
    let x = Tensor::from_vec::<_, f32>(vec![1.0, 2.0, 3.0], vec![3], dev()).unwrap();
    let out = ep_all_to_all(&x).unwrap();
    let o = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(o, vec![1.0, 2.0, 3.0]);
}
