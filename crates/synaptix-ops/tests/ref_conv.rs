use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::conv::{causal_conv3d, conv1d, conv2d, conv3d, transposed_conv};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t13_1_conv1d_basic() {
    setup();
    let t = load_case("conv", "conv1d_basic");
    let out = conv1d(&t["x"], &t["w"], Some(&t["bias"]), 1, 1).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_2_conv1d_stride2() {
    setup();
    let t = load_case("conv", "conv1d_stride2");
    let out = conv1d(&t["x"], &t["w"], None, 2, 1).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_3_conv2d_basic() {
    setup();
    let t = load_case("conv", "conv2d_basic");
    let out = conv2d(&t["x"], &t["w"], Some(&t["bias"]), (1, 1), (1, 1), (1, 1)).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_4_conv2d_stride2() {
    setup();
    let t = load_case("conv", "conv2d_stride2");
    let out = conv2d(&t["x"], &t["w"], None, (2, 2), (1, 1), (1, 1)).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_5_conv2d_dilated() {
    setup();
    let t = load_case("conv", "conv2d_dilated");
    let out = conv2d(&t["x"], &t["w"], None, (1, 1), (2, 2), (2, 2)).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_6_conv3d_basic() {
    setup();
    let t = load_case("conv", "conv3d_basic");
    let out = conv3d(&t["x"], &t["w"], Some(&t["bias"]), (1, 1, 1), (1, 1, 1), (1, 1, 1)).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_7_conv3d_stride() {
    setup();
    let t = load_case("conv", "conv3d_stride");
    let out = conv3d(&t["x"], &t["w"], None, (2, 2, 2), (1, 1, 1), (1, 1, 1)).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_8_transposed_conv_basic() {
    setup();
    let t = load_case("conv", "transposed_conv_basic");
    let out = transposed_conv(&t["x"], &t["w"], Some(&t["bias"]), 1, 0).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_9_transposed_conv_stride2() {
    setup();
    let t = load_case("conv", "transposed_conv_stride2");
    let out = transposed_conv(&t["x"], &t["w"], None, 2, 1).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_10_causal_conv3d_basic() {
    setup();
    let t = load_case("conv", "causal_conv3d_basic");
    let out = causal_conv3d(&t["x"], &t["weight"], Some(&t["bias"]), 1).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t13_11_causal_conv3d_stride2() {
    setup();
    let t = load_case("conv", "causal_conv3d_stride2");
    let out = causal_conv3d(&t["x"], &t["weight"], None, 2).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}
