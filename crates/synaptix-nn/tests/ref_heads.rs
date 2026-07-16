use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::heads::{
    BboxHead, CtcHead, KeypointHead, MlmHead, QaHead, RegressionActivation, RegressionHead,
    RnnTHead, SegmentationHead,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t14_9_ctc_head() {
    setup();
    let t = load_case("nn_heads", "ctc_head");
    let head = CtcHead::from_weights(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_10_mlm_head() {
    setup();
    let t = load_case("nn_heads", "mlm_head");
    let head = MlmHead::from_weights(
        t["dense_w"].clone(), Some(t["dense_b"].clone()),
        Some(t["ln_w"].clone()), Some(t["ln_b"].clone()), 1e-12,
        t["out_w"].clone(), Some(t["out_b"].clone()),
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t14_11_qa_head() {
    setup();
    let t = load_case("nn_heads", "qa_head");
    let head = QaHead::from_weights(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
    let (start, end) = head.forward_split(&t["x"]).unwrap();
    let start_c = start.contiguous().unwrap();
    let end_c = end.contiguous().unwrap();
    assert_allclose(&start_c, &t["start"], 1e-5, 1e-5);
    assert_allclose(&end_c, &t["end"], 1e-5, 1e-5);
}

#[test]
fn t14_12_segmentation_head() {
    setup();
    let t = load_case("nn_heads", "segmentation_head");
    let head = SegmentationHead::from_weights(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let y = head.forward_bchw(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t14_13_bbox_head_sigmoid() {
    setup();
    let t = load_case("nn_heads", "bbox_head");
    let num_classes = t["output_sigmoid"].dims()[t["output_sigmoid"].rank() - 2];
    let head = BboxHead::from_weights(
        t["weight"].clone(), Some(t["bias"].clone()), num_classes, true,
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output_sigmoid"], 1e-5, 1e-5);
}

#[test]
fn t14_14_bbox_head_raw() {
    setup();
    let t = load_case("nn_heads", "bbox_head");
    let num_classes = t["output_raw"].dims()[t["output_raw"].rank() - 2];
    let head = BboxHead::from_weights(
        t["weight"].clone(), Some(t["bias"].clone()), num_classes, false,
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output_raw"], 1e-5, 1e-5);
}

#[test]
fn t14_15_regression_head() {
    setup();
    let t = load_case("nn_heads", "regression_head");
    let head = RegressionHead::from_weights(
        t["dense_w"].clone(), Some(t["dense_b"].clone()),
        t["out_w"].clone(), Some(t["out_b"].clone()),
        RegressionActivation::Tanh,
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_16_keypoint_head_sigmoid() {
    setup();
    let t = load_case("nn_heads", "keypoint_head");
    let num_keypoints = t["output_sigmoid"].dims()[t["output_sigmoid"].rank() - 2];
    let head = KeypointHead::from_weights(
        t["weight"].clone(), Some(t["bias"].clone()), num_keypoints, true,
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output_sigmoid"], 1e-5, 1e-5);
}

#[test]
fn t14_17_keypoint_head_raw() {
    setup();
    let t = load_case("nn_heads", "keypoint_head");
    let num_keypoints = t["output_raw"].dims()[t["output_raw"].rank() - 2];
    let head = KeypointHead::from_weights(
        t["weight"].clone(), Some(t["bias"].clone()), num_keypoints, false,
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output_raw"], 1e-5, 1e-5);
}

#[test]
fn t14_18_rnn_t_head() {
    setup();
    let t = load_case("nn_heads", "rnn_t_head");
    let head = RnnTHead::from_weights(
        t["enc_w"].clone(), Some(t["enc_b"].clone()),
        t["pred_w"].clone(), Some(t["pred_b"].clone()),
        t["out_w"].clone(), Some(t["out_b"].clone()),
    ).unwrap();
    let y = head.forward(&t["enc"], &t["pred"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}
