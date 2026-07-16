use synaptix_nn::dit::{DitBlock, FinalLayer, Patchify};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t16_1_dit_block() {
    setup();
    let t = load_case("nn_dit", "block");
    let block = DitBlock::from_weights(
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["ff1_w"].clone(), Some(t["ff1_b"].clone()),
        t["ff2_w"].clone(), Some(t["ff2_b"].clone()),
        t["adaln_w"].clone(), Some(t["adaln_b"].clone()),
        4,
    ).unwrap();
    let y = block.forward(&t["x"], &t["cond"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t16_2_final_layer() {
    setup();
    let t = load_case("nn_dit", "final_layer");
    let fl = FinalLayer::from_weights(
        t["linear_w"].clone(), Some(t["linear_b"].clone()),
        t["adaln_w"].clone(), Some(t["adaln_b"].clone()),
    ).unwrap();
    let y = fl.forward(&t["x"], &t["cond"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t16_3_patchify_forward() {
    setup();
    let t = load_case("nn_dit", "patchify");
    let patch = Patchify::from_weights(2, 3, t["patch_weight"].clone(), Some(t["patch_bias"].clone())).unwrap();
    let y = patch.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["tokens_output"], 1e-4, 1e-4);
}

#[test]
fn t16_4_patchify_unpatchify() {
    setup();
    let t = load_case("nn_dit", "patchify");
    let out_ch = 3;
    let patch = Patchify::from_weights(2, out_ch, t["patch_weight"].clone(), Some(t["patch_bias"].clone())).unwrap();
    let img = patch.unpatchify(&t["pre_unpatch"], 8, 8).unwrap();
    assert_allclose(&img, &t["img_output"], 1e-4, 1e-4);
}
