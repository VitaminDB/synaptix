use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
}

#[test]
fn t02_1_unary_sqrt_f32() {
    setup();
    let t = load_case("kernels_cpu", "unary_sqrt_f32");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.sqrt().unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t02_2_unary_silu_f32() {
    setup();
    let t = load_case("kernels_cpu", "unary_silu_f32");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.silu().unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t02_3_binary_broadcast_add() {
    setup();
    let t = load_case("kernels_cpu", "binary_bcast_add");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.add(b).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

#[test]
fn t02_4_gemm_f32() {
    setup();
    let t = load_case("kernels_cpu", "gemm_f32");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.matmul(b).unwrap();
    assert_allclose(&result, expected, 2e-4, 2e-4);
}

#[test]
fn t02_5_gemm_bf16() {
    setup();
    let t = load_case("kernels_cpu", "gemm_bf16");
    let a = &t["input_a"];
    let b = &t["input_b"];
    let expected = &t["output"];
    let result = a.matmul(b).unwrap();
    assert_allclose(&result, expected, 5e-2, 5e-2);
}

#[test]
fn t02_6_reduction_sum_dim0() {
    setup();
    let t = load_case("kernels_cpu", "reduce_sum_dim0");
    let x = &t["input"];
    let expected = &t["output"];
    let result = x.sum([0]).unwrap();
    assert_allclose(&result, expected, 1e-5, 1e-5);
}

