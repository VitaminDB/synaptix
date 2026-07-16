use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::conformer::{
    attention_module::AttentionModule, conv_module::ConvModule, ff_module::FeedForwardModule,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_ff_module_vs_torchaudio() {
    setup();
    let t = load_case("nn_conformer", "ff_module");
    let ff = FeedForwardModule::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
        1e-5,
    ).unwrap();
    let y = ff.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_conv_module_vs_torchaudio() {
    setup();
    let t = load_case("nn_conformer", "conv_module");
    let conv = ConvModule::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["pw1_w"].clone(), None,
        t["dw_w"].clone(), None,
        t["bn_mean"].clone(), t["bn_var"].clone(),
        Some(t["bn_w"].clone()), Some(t["bn_b"].clone()),
        t["pw2_w"].clone(), None,
        1e-5, 1e-5,
    ).unwrap();
    let y = conv.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_attention_module_vs_torch_mha() {
    setup();
    let t = load_case("nn_conformer", "attention_module");
    let attn = AttentionModule::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["q_w"].clone(), Some(t["q_b"].clone()),
        t["k_w"].clone(), Some(t["k_b"].clone()),
        t["v_w"].clone(), Some(t["v_b"].clone()),
        t["o_w"].clone(), Some(t["o_b"].clone()),
        2, 1e-5,
    ).unwrap();
    let y = attn.forward(&t["x"], None, None).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_attention_module_rel_pos_bias_vs_torch() {
    setup();
    let t = load_case("nn_conformer", "attention_module_relpos");
    let attn = AttentionModule::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["q_w"].clone(), Some(t["q_b"].clone()),
        t["k_w"].clone(), Some(t["k_b"].clone()),
        t["v_w"].clone(), Some(t["v_b"].clone()),
        t["o_w"].clone(), Some(t["o_b"].clone()),
        2, 1e-5,
    ).unwrap();
    // rel_bias [nh, S, S] — additive on scores; AttentionModule принимает
    // его как broadcast-bias через scaled_dot_attention mask-аргумент.
    let y = attn.forward(&t["x"], Some(&t["rel_bias"]), None).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
