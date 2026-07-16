use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::norm::{
    adaln, batch_norm_inference, group_norm, instance_norm, layer_norm,
    pixel_norm, rms_norm, rms_norm_gated, rms_norm_qwen, rms_norm_silu_gated,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
}

#[test]
fn t04_1_rms_norm() {
    setup();
    let t = load_case("norm", "rms_norm");
    let result = rms_norm(&t["input"], &t["weight"], 1e-6).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_2_rms_norm_qwen() {
    setup();
    let t = load_case("norm", "rms_norm_qwen");
    let result = rms_norm_qwen(&t["input"], &t["weight"], 1e-6).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_3_rms_norm_silu_gated() {
    setup();
    // reference (gen_norm.case_rms_norm_gated) делает F.silu(gate) перед умножением →
    // тестируем нашу `rms_norm_silu_gated` (silu внутри).
    let t = load_case("norm", "rms_norm_gated");
    let result = rms_norm_silu_gated(&t["input"], &t["gate"], &t["weight"], 1e-6).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_3b_rms_norm_gated_without_silu() {
    setup();
    // Тот же reference, но gate активируем silu заранее, потом передаём в новый
    // `rms_norm_gated` (без silu) — должно совпасть.
    let t = load_case("norm", "rms_norm_gated");
    let silu_gate = t["gate"].silu().unwrap();
    let result = rms_norm_gated(&t["input"], &silu_gate, &t["weight"], 1e-6).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_4_layer_norm() {
    setup();
    let t = load_case("norm", "layer_norm");
    let result = layer_norm(&t["input"], Some(&t["weight"]), Some(&t["bias"]), 1e-5).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_5_group_norm() {
    setup();
    let t = load_case("norm", "group_norm");
    let result = group_norm(&t["input"], Some(&t["weight"]), Some(&t["bias"]), 8, 1e-5).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_6_batch_norm_inference() {
    setup();
    let t = load_case("norm", "batch_norm_inference");
    let result = batch_norm_inference(
        &t["input"],
        &t["running_mean"],
        &t["running_var"],
        Some(&t["weight"]),
        Some(&t["bias"]),
        1e-5,
    )
    .unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_7_instance_norm() {
    setup();
    let t = load_case("norm", "instance_norm");
    let result = instance_norm(&t["input"], Some(&t["weight"]), Some(&t["bias"]), 1e-5).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_8_adaln_zero() {
    setup();
    let t = load_case("norm", "adaln_zero");
    let result = adaln(&t["input"], &t["scale"], &t["shift"], 1e-6).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t04_9_pixel_norm() {
    setup();
    let t = load_case("norm", "pixel_norm");
    let result = pixel_norm(&t["input"], 1e-8).unwrap();
    assert_allclose(&result, &t["output"], 1e-6, 1e-6);
}
