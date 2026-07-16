use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::hybrid::{FalconMamba, GriffinBlock, Hymba, Jamba, Samba, Zamba};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_falcon_mamba_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "falcon_mamba");
    let m = FalconMamba::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_griffin_block_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "griffin_block");
    let m = GriffinBlock::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["fc_in_w"].clone(), Some(t["fc_in_b"].clone()),
        t["fc_out_w"].clone(), Some(t["fc_out_b"].clone()),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_hymba_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "hymba");
    let m = Hymba::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["attn_proj_w"].clone(), Some(t["attn_proj_b"].clone()),
        t["ssm_proj_w"].clone(), Some(t["ssm_proj_b"].clone()),
        t["fuse_w"].clone(), Some(t["fuse_b"].clone()),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_jamba_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "jamba");
    let m = Jamba::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["gate_w"].clone(),
        t["expert0_w"].clone(), Some(t["expert0_b"].clone()),
        t["expert1_w"].clone(), Some(t["expert1_b"].clone()),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_samba_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "samba");
    let m = Samba::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["fc_in_w"].clone(), Some(t["fc_in_b"].clone()),
        t["fc_out_w"].clone(), Some(t["fc_out_b"].clone()),
        t["window_gate"].clone(),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_zamba_vs_python() {
    setup();
    let t = load_case("nn_hybrid", "zamba");
    let m = Zamba::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["mamba_w"].clone(), Some(t["mamba_b"].clone()),
        t["shared_attn_w"].clone(), Some(t["shared_attn_b"].clone()),
        t["out_w"].clone(), Some(t["out_b"].clone()),
        1e-5,
    ).unwrap();
    let y = m.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
