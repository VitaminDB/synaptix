use synaptix_core::device::Device;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::attention::softmax::{
    cross_attention, gqa_attention, mqa_attention, scaled_dot_attention, sliding_window_attention,
};
use synaptix_ops::mask::causal_mask;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

fn make_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

#[test]
fn t07_1_scaled_dot_no_mask() {
    setup();
    let t = load_case("attention", "scaled_dot_no_mask");
    let q = &t["q"];
    let scale = make_scale(q.dims()[3]);
    let result = scaled_dot_attention(q, &t["k"], &t["v"], scale, None).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t07_2_scaled_dot_causal() {
    setup();
    let t = load_case("attention", "scaled_dot_causal");
    let q = &t["q"];
    let seq = q.dims()[2];
    let scale = make_scale(q.dims()[3]);
    let mask = causal_mask(seq, Device::Cpu).unwrap();
    let result = scaled_dot_attention(q, &t["k"], &t["v"], scale, Some(&mask)).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t07_3_gqa() {
    setup();
    let t = load_case("attention", "gqa");
    let q = &t["q"];
    let seq = q.dims()[2];
    let scale = make_scale(q.dims()[3]);
    let mask = causal_mask(seq, Device::Cpu).unwrap();
    let result = gqa_attention(q, &t["k"], &t["v"], scale, Some(&mask)).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t07_4_mqa() {
    setup();
    let t = load_case("attention", "mqa");
    let q = &t["q"];
    let seq = q.dims()[2];
    let scale = make_scale(q.dims()[3]);
    let mask = causal_mask(seq, Device::Cpu).unwrap();
    let result = mqa_attention(q, &t["k"], &t["v"], scale, Some(&mask)).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t07_5_sliding_window() {
    setup();
    let t = load_case("attention", "sliding_window");
    let q = &t["q"];
    let scale = make_scale(q.dims()[3]);
    let result = sliding_window_attention(q, &t["k"], &t["v"], scale, 8, None).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t07_6_cross_attention() {
    setup();
    let t = load_case("attention", "cross_attention");
    let q = &t["q"];
    let scale = make_scale(q.dims()[3]);
    let result = cross_attention(q, &t["k"], &t["v"], scale, None).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}
