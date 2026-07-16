use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, assert_exact_eq, load_case};

fn setup() {
    ensure_registered();
}

#[test]
fn t01_1_matmul_2d() {
    setup();
    let t = load_case("core_tensors", "matmul_2d");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.matmul(b).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_2_matmul_batch() {
    setup();
    let t = load_case("core_tensors", "matmul_batch");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.matmul(b).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_3_reduce_sum_dim1() {
    setup();
    let t = load_case("core_tensors", "reduce_sum_dim1");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.sum([1]).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_4_reduce_sum_all() {
    setup();
    let t = load_case("core_tensors", "reduce_sum_all");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.sum_all().unwrap().reshape([1]).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_5_reduce_mean() {
    setup();
    let t = load_case("core_tensors", "reduce_mean");
    let x = &t["input"];
    let expected = &t["output"];
    let n = x.shape().dims()[1] as f32;
    let result = x.sum([1]).unwrap().mul_scalar(1.0 / n).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_6_argmax() {
    setup();
    let t = load_case("core_tensors", "argmax");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.argmax(1).unwrap().to_dtype(DType::I64).unwrap();
    assert_exact_eq(&result, expected);
}

#[test]
fn t01_7_broadcast_add() {
    setup();
    let t = load_case("core_tensors", "broadcast_add");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.add(b).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_8_gather_2d() {
    setup();
    let t = load_case("core_tensors", "gather_2d");
    let x = &t["input"];
    let indices = &t["indices"];
    let expected = &t["output"];
    let result = x.gather(indices, 1).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_9_masked_fill() {
    setup();
    let t = load_case("core_tensors", "masked_fill");
    let x = &t["input"];
    let mask = &t["mask"];
    let expected = &t["output"];
    let result = x.masked_fill(mask, -1e9).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_10_cat_dim0() {
    setup();
    let t = load_case("core_tensors", "cat_dim0");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = Tensor::cat(&[a, b], 0).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t01_11_cast_f32_bf16_f32() {
    setup();
    let t = load_case("core_tensors", "cast_f32_bf16_f32");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap();
    assert_allclose(&result, expected, 5e-2, 5e-2);
}
