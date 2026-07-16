use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::multimodal::{CrossModalAttention, MlpProjector, VlmBlock};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t25_1_mlp_projector_gelu_exact_vs_python() {
    setup();
    let t = load_case("nn_multimodal", "mlp_projector");
    let proj = MlpProjector::from_weights(
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
    ).unwrap();
    let y = proj.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t25_2_cross_modal_attention_vs_python() {
    setup();
    let t = load_case("nn_multimodal", "cross_modal_attention");
    let attn = CrossModalAttention::from_weights(
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        2,
    ).unwrap();
    let y = attn.forward(&t["x"], &t["context"], None).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t25_3_vlm_block_vs_python() {
    setup();
    let t = load_case("nn_multimodal", "vlm_block");
    let block = VlmBlock::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["q_w"].clone(), t["kv_w"].clone(), t["o_w"].clone(),
        2, 1e-5,
    ).unwrap();
    let y = block.forward(&t["x"], &t["context"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
