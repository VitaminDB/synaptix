use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::ffn::{d_gate_net, geglu, kan_forward, mlp, monarch_mixer, reglu, swiglu, Activation};
use synaptix_ops::ffn::moe::{
    auxiliary_loss, expert_choice_router, fine_grained_moe, shared_expert_forward, soft_router,
    top_k_router, z_loss, Expert,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t08_1_mlp_gelu() {
    setup();
    let t = load_case("ffn", "mlp_gelu");
    let result = mlp(
        &t["input"],
        &t["w1"],
        Some(&t["b1"]),
        &t["w2"],
        Some(&t["b2"]),
        Activation::Gelu,
    )
    .unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_2_swiglu() {
    setup();
    let t = load_case("ffn", "swiglu");
    let result = swiglu(&t["input"], &t["w_gate"], &t["w_up"], &t["w_down"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_3_geglu() {
    setup();
    let t = load_case("ffn", "geglu");
    let w_combined = &t["w_combined"];
    let intermediate = w_combined.dims()[0] / 2;
    let w_up = w_combined.narrow(0, 0, intermediate).unwrap().contiguous().unwrap();
    let w_gate = w_combined.narrow(0, intermediate, intermediate).unwrap().contiguous().unwrap();
    let result = geglu(&t["input"], &w_gate, &w_up, &t["w_down"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_4_reglu() {
    setup();
    let t = load_case("ffn", "reglu");
    let w_combined = &t["w_combined"];
    let intermediate = w_combined.dims()[0] / 2;
    let w_up = w_combined.narrow(0, 0, intermediate).unwrap().contiguous().unwrap();
    let w_gate = w_combined.narrow(0, intermediate, intermediate).unwrap().contiguous().unwrap();
    let result = reglu(&t["input"], &w_gate, &w_up, &t["w_down"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

// ───────────────────────── ffn/moe расширение ─────────────────────────

#[test]
fn t08_5_d_gate_net() {
    setup();
    let t = load_case("ffn", "d_gate_net");
    let out = d_gate_net(&t["x"], &t["gate_weight"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_6_monarch_mixer() {
    setup();
    let t = load_case("ffn", "monarch_mixer");
    let out = monarch_mixer(&t["x"], &t["m1"], &t["m2"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t08_7_kan() {
    setup();
    let t = load_case("ffn", "kan");
    let out = kan_forward(&t["x"], &t["grid"], &t["coeff"], 3).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t08_8_expert() {
    setup();
    let t = load_case("ffn", "expert");
    let expert = Expert { fc1: t["fc1"].clone(), fc2: t["fc2"].clone() };
    let out = expert.forward(&t["x"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t08_9_shared_expert() {
    setup();
    let t = load_case("ffn", "shared_expert");
    let out = shared_expert_forward(&t["x"], &t["fc1"], &t["fc2"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t08_10_fine_grained_moe() {
    setup();
    let t = load_case("ffn", "fine_grained_moe");
    let out = fine_grained_moe(
        &t["x"], &t["router_w"], &t["experts_fc1"], &t["experts_fc2"], 2,
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t08_11_soft_router() {
    setup();
    let t = load_case("ffn", "soft_router");
    let out = soft_router(&t["logits"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_12_top_k_router() {
    setup();
    let t = load_case("ffn", "top_k_router");
    let (idx, w) = top_k_router(&t["logits"], 3).unwrap();
    assert_allclose(&idx, &t["indices"], 1e-5, 1e-5);
    assert_allclose(&w, &t["weights"], 1e-5, 1e-5);
}

#[test]
fn t08_13_expert_choice_router() {
    setup();
    let t = load_case("ffn", "expert_choice_router");
    let out = expert_choice_router(&t["logits"], 3).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_14_auxiliary_loss() {
    setup();
    let t = load_case("ffn", "auxiliary_loss");
    let out = auxiliary_loss(&t["router_probs"], &t["expert_indices"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t08_15_z_loss() {
    setup();
    let t = load_case("ffn", "z_loss");
    let out = z_loss(&t["router_logits"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}
