use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::vision::{ViTBlock, VisionTransformer};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t17_1_vit_single_block() {
    setup();
    let t = load_case("nn_vit", "vit");
    let block = ViTBlock::from_weights(
        t["n1_w"].clone(), t["n1_b"].clone(),
        t["n2_w"].clone(), t["n2_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["ff1_w"].clone(), Some(t["ff1_b"].clone()),
        t["ff2_w"].clone(), Some(t["ff2_b"].clone()),
        4,
    ).unwrap();
    let vit = VisionTransformer::from_weights(
        3, 2,
        t["patch_w"].clone(), Some(t["patch_b"].clone()),
        vec![block],
        t["final_w"].clone(), t["final_b"].clone(),
        4,
    ).unwrap();
    let y = vit.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
