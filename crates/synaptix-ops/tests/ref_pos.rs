use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::pos::{
    alibi_bias, apply_rope_interleaved, apply_rope_split, sinusoidal::sinusoidal_positional_embedding,
    t5_relative_position_bucket, RopeCache,
};
use synaptix_test_utils::{assert_allclose, assert_exact_eq, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t05_1_rope_split() {
    setup();
    let t = load_case("pos", "rope_split");
    let q = &t["q"];
    let k = &t["k"];
    let head_dim = q.dims()[3];
    let seq = q.dims()[2];
    let cache = RopeCache::new(head_dim, seq, 10000.0, Device::Cpu).unwrap();
    let q_out = apply_rope_split(q, &cache, None).unwrap();
    let k_out = apply_rope_split(k, &cache, None).unwrap();
    assert_allclose(&q_out, &t["q_out"], 1e-5, 1e-5);
    assert_allclose(&k_out, &t["k_out"], 1e-5, 1e-5);
}

#[test]
fn t05_2_rope_interleaved() {
    setup();
    let t = load_case("pos", "rope_interleaved");
    let q = &t["q"];
    let k = &t["k"];
    let head_dim = q.dims()[3];
    let seq = q.dims()[2];
    let cache = RopeCache::new(head_dim, seq, 10000.0, Device::Cpu).unwrap();
    let q_out = apply_rope_interleaved(q, &cache, None).unwrap();
    let k_out = apply_rope_interleaved(k, &cache, None).unwrap();
    assert_allclose(&q_out, &t["q_out"], 1e-5, 1e-5);
    assert_allclose(&k_out, &t["k_out"], 1e-5, 1e-5);
}

#[test]
fn t05_5_alibi() {
    setup();
    let t = load_case("pos", "alibi");
    let slopes_expected = &t["slopes"];
    let bias_expected = &t["bias"];
    let n_heads = slopes_expected.dims()[0];
    let seq = bias_expected.dims()[bias_expected.rank() - 1];
    let slopes_ours = synaptix_ops::pos::alibi::alibi_slopes(n_heads);
    let slopes_tensor = Tensor::from_vec(slopes_ours, (n_heads,), Device::Cpu).unwrap();
    assert_allclose(&slopes_tensor, slopes_expected, 1e-6, 1e-6);

    let bias_ours = alibi_bias(n_heads, seq, Device::Cpu).unwrap();
    let bias_reshaped = bias_ours.reshape((n_heads, seq, seq)).unwrap();
    assert_allclose(&bias_reshaped, bias_expected, 1e-6, 1e-6);
}

#[test]
fn t05_6_sinusoidal() {
    setup();
    let t = load_case("pos", "sinusoidal");
    let expected = &t["output"];
    let seq = expected.dims()[0];
    let dim = expected.dims()[1];
    let result = sinusoidal_positional_embedding(seq, dim, Device::Cpu).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t05_7_t5_relative() {
    setup();
    let t = load_case("pos", "t5_relative");
    let expected = &t["relative_buckets"];
    let seq_q = expected.dims()[0];
    let seq_k = expected.dims()[1];
    let buckets = t5_relative_position_bucket(seq_q, seq_k, 32, 128, true);
    let buckets_tensor = Tensor::from_vec(
        buckets.into_iter().map(|v| v as i32).collect::<Vec<_>>(),
        (seq_q, seq_k),
        Device::Cpu,
    )
    .unwrap();
    assert_exact_eq(&buckets_tensor, expected);
}
