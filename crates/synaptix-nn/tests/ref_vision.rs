use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::linear::Linear;
use synaptix_nn::vision::clip_vision::ClipVision;
use synaptix_nn::vision::vit::{ViTBlock, VisionTransformer};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t24_1_clip_vision_minimal_vs_python() {
    setup();
    let t = load_case("nn_vision", "clip_vision_minimal");
    let block = ViTBlock::from_weights(
        t["n1_w"].clone(), t["n1_b"].clone(),
        t["n2_w"].clone(), t["n2_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["ff1_w"].clone(), Some(t["ff1_b"].clone()),
        t["ff2_w"].clone(), Some(t["ff2_b"].clone()),
        4,
    ).unwrap();
    let vit = VisionTransformer::from_weights(
        3, 4,
        t["patch_w"].clone(), Some(t["patch_b"].clone()),
        vec![block],
        t["norm_w"].clone(), t["norm_b"].clone(),
        4,
    ).unwrap();
    let visual_projection = Linear::new(t["proj_w"].clone(), None).unwrap();
    let clip = ClipVision::from_parts(vit, visual_projection).unwrap();
    let y = clip.forward(&t["image"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
