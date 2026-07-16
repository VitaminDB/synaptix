use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::adapters::{
    BoftLinear, GaloreLinear, OftLinear, PTuningV2, PromptTuning, QLoraLinear, ReftAdapter,
    VeraLinear,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_qlora_vs_python() {
    setup();
    let t = load_case("nn_adapters", "qlora");
    let r = t["lora_a"].dims()[0];
    let scaling = 8.0_f32 / r as f32;
    let q = QLoraLinear::from_weights(
        t["base_w"].clone(),
        t["lora_a"].clone(),
        t["lora_b"].clone(),
        scaling,
    ).unwrap();
    let y = q.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_vera_vs_python() {
    setup();
    let t = load_case("nn_adapters", "vera");
    let v = VeraLinear::from_weights(
        t["base_w"].clone(),
        t["a_shared"].clone(),
        t["b_shared"].clone(),
        t["lambda_d"].clone(),
        t["lambda_b"].clone(),
    ).unwrap();
    let y = v.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_oft_vs_python_precomputed_r() {
    setup();
    let t = load_case("nn_adapters", "oft");
    let oft = OftLinear::from_weights(t["base_w"].clone(), t["r_matrix"].clone()).unwrap();
    let y = oft.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_oft_cayley_matches_python() {
    setup();
    let t = load_case("nn_adapters", "oft");
    let r = synaptix_nn::adapters::oft::cayley_orthogonal_cpu(&t["q_raw"]).unwrap();
    assert_allclose(&r, &t["r_matrix"], 1e-5, 1e-5);
}

#[test]
fn t_boft_vs_python_precomputed_r() {
    setup();
    let t = load_case("nn_adapters", "boft");
    let boft = BoftLinear::from_weights(t["base_w"].clone(), t["r_matrix"].clone(), 2).unwrap();
    let y = boft.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_boft_assemble_block_diag_matches_python() {
    setup();
    let t = load_case("nn_adapters", "boft");
    let r = synaptix_nn::adapters::boft::assemble_block_diag_r_cpu(&[
        t["q_block0"].clone(),
        t["q_block1"].clone(),
    ]).unwrap();
    assert_allclose(&r, &t["r_matrix"], 1e-5, 1e-5);
}

#[test]
fn t_galore_vs_python_passthrough() {
    setup();
    let t = load_case("nn_adapters", "galore");
    let g = GaloreLinear::from_weights(t["base_w"].clone(), 4, 0.25).unwrap();
    let y = g.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_reft_vs_python() {
    setup();
    let t = load_case("nn_adapters", "reft");
    let reft = ReftAdapter::from_weights(
        t["r_proj"].clone(),
        t["w"].clone(),
        t["b"].clone(),
    ).unwrap();
    let y = reft.forward(&t["h"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_prompt_tuning_prepend_vs_python() {
    setup();
    let t = load_case("nn_adapters", "prompt_tuning");
    let pt = PromptTuning::from_weights(t["soft_prompts"].clone()).unwrap();
    let y = pt.prepend(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_p_tuning_v2_full_vs_python() {
    setup();
    let t = load_case("nn_adapters", "p_tuning_v2");
    let p = PTuningV2::from_weights(
        t["embeddings"].clone(),
        t["reparam_w"].clone(),
        3,
    ).unwrap();
    let full = p.forward().unwrap();
    assert_allclose(&full, &t["full"], 1e-5, 1e-5);
}

#[test]
fn t_p_tuning_v2_layer_kv_vs_python() {
    setup();
    let t = load_case("nn_adapters", "p_tuning_v2");
    let p = PTuningV2::from_weights(
        t["embeddings"].clone(),
        t["reparam_w"].clone(),
        3,
    ).unwrap();
    let full = p.forward().unwrap();
    let (k, v) = p.layer_kv(&full, 1).unwrap();
    assert_allclose(&k, &t["layer_k"], 1e-5, 1e-5);
    assert_allclose(&v, &t["layer_v"], 1e-5, 1e-5);
}
