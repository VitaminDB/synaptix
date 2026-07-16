use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::adapters::{DoraLinear, Ia3Linear, LoraLinear, PrefixTuning};
use synaptix_nn::heads::{ClassificationHead, LmHead, RewardHead, TokenClsHead};
use synaptix_nn::Linear;
use synaptix_nn::parameter::Parameter;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t14_1_lm_head() {
    setup();
    let t = load_case("nn_heads", "lm_head");
    let head = LmHead::from_weights(t["weight"].clone(), None).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_2_cls_head() {
    setup();
    let t = load_case("nn_heads", "cls_head");
    let head = ClassificationHead::from_weights(
        t["dense_w"].clone(), Some(t["dense_b"].clone()),
        t["out_w"].clone(), Some(t["out_b"].clone()),
    ).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_3_token_cls_head() {
    setup();
    let t = load_case("nn_heads", "token_cls_head");
    let head = TokenClsHead::from_weights(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_4_reward_head() {
    setup();
    let t = load_case("nn_heads", "reward_head");
    let head = RewardHead::from_weights(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let y = head.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_5_lora() {
    setup();
    let t = load_case("nn_heads", "lora");
    let r = t["lora_a"].dims()[0];
    let alpha = 8.0_f32;
    let scaling = alpha / r as f32;
    let base = Linear::new(t["base_w"].clone(), None).unwrap();
    let lora_a = Linear::new(t["lora_a"].clone(), None).unwrap();
    let lora_b = Linear::new(t["lora_b"].clone(), None).unwrap();
    let lora = LoraLinear { base, lora_a, lora_b, scaling };
    let y = lora.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_6_dora() {
    setup();
    let t = load_case("nn_heads", "dora");
    let r = t["lora_a"].dims()[0];
    let alpha = 8.0_f32;
    let scaling = alpha / r as f32;
    let dora = DoraLinear::from_weights(
        t["base_w"].clone(),
        t["lora_a"].clone(),
        t["lora_b"].clone(),
        t["magnitude"].clone(),
        scaling,
    ).unwrap();
    let y = dora.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_7_ia3() {
    setup();
    let t = load_case("nn_heads", "ia3");
    let ia3 = Ia3Linear::from_weights(t["base_w"].clone(), t["scale"].clone()).unwrap();
    let y = ia3.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t14_8_prefix_tuning() {
    setup();
    let t = load_case("nn_heads", "prefix_tuning");
    let pk = t["prefix_keys"].clone();
    let pv = t["prefix_values"].clone();
    let num_layers = pk.dims()[0];
    let prefix_len = pk.dims()[1];
    let prefix = PrefixTuning {
        prefix_keys: Parameter::new(pk),
        prefix_values: Parameter::new(pv),
        prefix_len,
        num_layers,
    };
    let (k, v) = prefix.get_prefix(1).unwrap();
    assert_allclose(&k, &t["layer_k"], 1e-5, 1e-5);
    assert_allclose(&v, &t["layer_v"], 1e-5, 1e-5);
}
