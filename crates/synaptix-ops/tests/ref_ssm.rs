use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::ssm::{
    h3_forward, liquid_step, mamba_scan, mamba_step, mlstm_step, monarch_ssm, rwkv_time_mix,
    rwkv_wkv, s4_forward, s5_forward, slstm_step, titans_memory_step, ttt_layer, MambaState,
};
use synaptix_core::tensor::Tensor;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t12_1_mamba_scan() {
    setup();
    let t = load_case("ssm", "mamba_scan");
    let out = mamba_scan(&t["x"], &t["a"], &t["b"], &t["c"], &t["d"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t12_2_mamba_step() {
    setup();
    let t = load_case("ssm", "mamba_step");
    let h_in = t["h_in"].clone();
    let conv_buf = Tensor::zeros(vec![1], h_in.dtype(), h_in.device()).unwrap();
    let mut state = MambaState { h: h_in, conv_buf };
    let y = mamba_step(&t["x"], &mut state, &t["a"], &t["b"], &t["c"], &t["dt"]).unwrap();
    assert_allclose(&y, &t["y"], 1e-5, 1e-5);
    assert_allclose(&state.h, &t["h_out"], 1e-5, 1e-5);
}

#[test]
fn t12_3_rwkv_wkv() {
    setup();
    let t = load_case("ssm", "rwkv_wkv");
    let out = rwkv_wkv(&t["k"], &t["v"], &t["r"], &t["time_decay"], &t["time_first"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t12_4_rwkv_time_mix() {
    setup();
    let t = load_case("ssm", "rwkv_time_mix");
    let out = rwkv_time_mix(
        &t["x"], &t["x_prev"],
        &t["mix_k"], &t["mix_v"], &t["mix_r"],
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

// ───────────────────────── расширенное SSM-семейство ─────────────────────────

#[test]
fn t12_5_s4() {
    setup();
    let t = load_case("ssm", "s4");
    let out = s4_forward(&t["x"], &t["a"], &t["b"], &t["c"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_6_s5() {
    setup();
    let t = load_case("ssm", "s5");
    let out = s5_forward(&t["x"], &t["lambda"], &t["b"], &t["c"], &t["d"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_7_h3() {
    setup();
    let t = load_case("ssm", "h3");
    let out = h3_forward(&t["x"], &t["k"], &t["q"], &t["a"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_8_ttt() {
    setup();
    let t = load_case("ssm", "ttt");
    let out = ttt_layer(&t["x"], &t["w"], 0.1).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_9_liquid() {
    setup();
    let t = load_case("ssm", "liquid");
    let out = liquid_step(&t["x"], &t["state"], &t["tau"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_10_titans() {
    setup();
    let t = load_case("ssm", "titans");
    let out = titans_memory_step(&t["x"], &t["mem"], &t["surprise"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_11_slstm() {
    setup();
    let t = load_case("ssm", "slstm");
    let out = slstm_step(&t["x"], &t["h"], &t["c"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_12_mlstm() {
    setup();
    let t = load_case("ssm", "mlstm");
    let out = mlstm_step(&t["x"], &t["h"], &t["c"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t12_13_monarch() {
    setup();
    let t = load_case("ssm", "monarch");
    let out = monarch_ssm(&t["x"], &t["m1"], &t["m2"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}
