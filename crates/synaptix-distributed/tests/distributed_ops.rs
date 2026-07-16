//! synaptix-distributed — локальная математика примитивов (single-process).

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_distributed::collectives::all_gather::all_gather;
use synaptix_distributed::collectives::all_reduce::{reduce_shards, ReduceOp};
use synaptix_distributed::collectives::reduce_scatter::reduce_scatter;
use synaptix_distributed::context_parallel::{gather_kv, scatter_kv};
use synaptix_distributed::expert_parallel::dispatch_tokens;
use synaptix_distributed::sequence_parallel::{gather_sequence, scatter_sequence};
use synaptix_distributed::tensor_parallel::{column_parallel_linear, row_parallel_linear};
use synaptix_distributed::world::shard_range;
use synaptix_distributed::zero::{gather_flat, partition_flat};
use synaptix_distributed::zero::zero3::Zero3;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::assert_allclose;

fn setup() { ensure_registered(); }

fn t1(v: &[f32]) -> Tensor {
    Tensor::from_vec::<_, f32>(v.to_vec(), vec![v.len()], Device::Cpu).unwrap()
}
fn mat(rows: usize, cols: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32).collect();
    Tensor::from_vec::<_, f32>(data, vec![rows, cols], Device::Cpu).unwrap()
}

#[test]
fn t60_1_shard_range_partition() {
    // 10 на 3 ранга → 4,3,3; покрытие без пересечений.
    let parts: Vec<(usize, usize)> = (0..3).map(|r| shard_range(10, r, 3)).collect();
    assert_eq!(parts, vec![(0, 4), (4, 3), (7, 3)]);
    let total: usize = parts.iter().map(|(_, l)| l).sum();
    assert_eq!(total, 10);
    // Ровное деление.
    assert_eq!(shard_range(8, 1, 4), (2, 2));
    // Граничные.
    assert_eq!(shard_range(5, 9, 3), (0, 0));
}

#[test]
fn t60_2_reduce_shards_ops() {
    setup();
    let a = t1(&[1.0, 2.0, 3.0]);
    let b = t1(&[4.0, 0.0, 6.0]);
    let c = t1(&[2.0, 5.0, 1.0]);
    let refs = [&a, &b, &c];
    assert_allclose(&reduce_shards(&refs, ReduceOp::Sum).unwrap(), &t1(&[7.0, 7.0, 10.0]), 1e-6, 1e-6);
    assert_allclose(&reduce_shards(&refs, ReduceOp::Mean).unwrap(), &t1(&[7.0 / 3.0, 7.0 / 3.0, 10.0 / 3.0]), 1e-5, 1e-5);
    assert_allclose(&reduce_shards(&refs, ReduceOp::Max).unwrap(), &t1(&[4.0, 5.0, 6.0]), 1e-6, 1e-6);
    assert_allclose(&reduce_shards(&refs, ReduceOp::Min).unwrap(), &t1(&[1.0, 0.0, 1.0]), 1e-6, 1e-6);
}

#[test]
fn t60_3_all_gather_then_reduce_scatter() {
    setup();
    let s0 = t1(&[1.0, 2.0, 3.0, 4.0]);
    let s1 = t1(&[10.0, 20.0, 30.0, 40.0]);
    // all_gather по dim 0.
    let g = all_gather(&[&s0, &s1], 0).unwrap();
    assert_eq!(g.dims(), &[8]);
    // reduce_scatter Sum по dim 0: сумма = [11,22,33,44], split на 2 → [11,22],[33,44].
    let parts = reduce_scatter(&[&s0, &s1], ReduceOp::Sum, 0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_allclose(&parts[0], &t1(&[11.0, 22.0]), 1e-6, 1e-6);
    assert_allclose(&parts[1], &t1(&[33.0, 44.0]), 1e-6, 1e-6);
}

#[test]
fn t60_4_column_parallel_equals_full() {
    setup();
    let x = mat(2, 4, 1.0);
    let w = mat(4, 6, 0.5);
    let full = x.matmul(&w).unwrap();
    let cp = column_parallel_linear(&x, &w, 3).unwrap();
    assert_eq!(cp.dims(), full.dims());
    assert_allclose(&cp, &full, 1e-4, 1e-4);
}

#[test]
fn t60_5_row_parallel_equals_full() {
    setup();
    let x = mat(2, 4, 1.0);
    let w = mat(4, 6, 0.5);
    let full = x.matmul(&w).unwrap();
    let rp = row_parallel_linear(&x, &w, 3).unwrap();
    assert_eq!(rp.dims(), full.dims());
    assert_allclose(&rp, &full, 1e-4, 1e-4);
}

#[test]
fn t60_6_sequence_scatter_gather_roundtrip() {
    setup();
    let x = mat(7, 3, 1.0); // ось seq = 0, длина 7
    let world = 3;
    let shards: Vec<Tensor> = (0..world).map(|r| scatter_sequence(&x, 0, r, world).unwrap()).collect();
    assert_eq!(shards[0].dims(), &[3, 3]);
    assert_eq!(shards[2].dims(), &[2, 3]);
    let refs: Vec<&Tensor> = shards.iter().collect();
    let back = gather_sequence(&refs, 0).unwrap();
    assert_allclose(&back, &x, 1e-6, 1e-6);
}

#[test]
fn t60_7_context_kv_scatter_gather_roundtrip() {
    setup();
    let kv = mat(6, 4, 2.0);
    let world = 2;
    let shards: Vec<Tensor> = (0..world).map(|r| scatter_kv(&kv, 0, r, world).unwrap()).collect();
    let refs: Vec<&Tensor> = shards.iter().collect();
    assert_allclose(&gather_kv(&refs, 0).unwrap(), &kv, 1e-6, 1e-6);
}

#[test]
fn t60_8_dispatch_tokens_grouping() {
    setup();
    // 4 токена, dim 2: эксперты [0,1,0,2], num_experts=3.
    let x = mat(4, 2, 0.0); // строки: [0,1],[2,3],[4,5],[6,7]
    let groups = dispatch_tokens(&x, &[0, 1, 0, 2], 3).unwrap();
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0].dims(), &[2, 2]); // токены 0 и 2
    assert_eq!(groups[1].dims(), &[1, 2]); // токен 1
    assert_eq!(groups[2].dims(), &[1, 2]); // токен 3
    assert_allclose(&groups[0], &Tensor::from_vec::<_, f32>(vec![0.0, 1.0, 4.0, 5.0], vec![2, 2], Device::Cpu).unwrap(), 1e-6, 1e-6);
}

#[test]
fn t60_9_zero_partition_gather_roundtrip() {
    setup();
    let param = mat(3, 4, 1.0); // numel 12
    let world = 4;
    let shards: Vec<Tensor> = (0..world).map(|r| partition_flat(&param, r, world).unwrap()).collect();
    assert_eq!(shards.iter().map(|s| s.dims()[0]).collect::<Vec<_>>(), vec![3, 3, 3, 3]);
    let refs: Vec<&Tensor> = shards.iter().collect();
    let flat = gather_flat(&refs).unwrap();
    assert_allclose(&flat, &param.flatten_all().unwrap(), 1e-6, 1e-6);

    // Через Zero3 API.
    let z = Zero3::new(world);
    let zs: Vec<Tensor> = (0..world).map(|r| z.shard(&param, r).unwrap()).collect();
    let zrefs: Vec<&Tensor> = zs.iter().collect();
    assert_allclose(&z.all_gather_param(&zrefs).unwrap(), &param.flatten_all().unwrap(), 1e-6, 1e-6);
}
