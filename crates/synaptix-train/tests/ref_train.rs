use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, load_case};
use synaptix_train::losses::{cross_entropy, l1_loss, mse_loss, smooth_l1_loss, Reduction};
use synaptix_train::optimizer::adafactor::{Adafactor, AdafactorConfig};
use synaptix_train::optimizer::adam8bit::{Adam8bit, Adam8bitConfig};
use synaptix_train::optimizer::adamw::{AdamW, AdamWConfig};
use synaptix_train::optimizer::adem_amix::{AdemAmix, AdemAmixConfig};
use synaptix_train::optimizer::grad_clip::{clip_grad_norm, clip_grad_value};
use synaptix_train::optimizer::lion::{Lion, LionConfig};
use synaptix_train::optimizer::muon::{Muon, MuonConfig};
use synaptix_train::optimizer::sophia::{Sophia, SophiaConfig};
use synaptix_train::rlhf::{dpo, kto, orpo};

fn setup() { ensure_registered(); }

fn vec1f(t: &synaptix_core::tensor::Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn t31_1_cross_entropy_2d() {
    setup();
    let t = load_case("train", "cross_entropy");
    let logits = &t["logits"];
    let targets = &t["targets"];
    let l_none = cross_entropy(logits, targets, None, Reduction::None).unwrap();
    assert_allclose(&l_none, &t["loss_none"], 1e-5, 1e-5);
    let l_mean = cross_entropy(logits, targets, None, Reduction::Mean).unwrap();
    assert!((vec1f(&l_mean)[0] - vec1f(&t["loss_mean"])[0]).abs() < 1e-5);
    let l_sum = cross_entropy(logits, targets, None, Reduction::Sum).unwrap();
    assert!((vec1f(&l_sum)[0] - vec1f(&t["loss_sum"])[0]).abs() < 1e-5);
}

#[test]
fn t31_2_cross_entropy_3d_with_ignore() {
    setup();
    let t = load_case("train", "cross_entropy_3d");
    let l_mean = cross_entropy(&t["logits"], &t["targets"], Some(-100), Reduction::Mean).unwrap();
    assert!((vec1f(&l_mean)[0] - vec1f(&t["loss_mean"])[0]).abs() < 1e-5);
}

#[test]
fn t31_3_mse_l1() {
    setup();
    let t = load_case("train", "mse_l1");
    let m_mean = mse_loss(&t["input"], &t["target"], Reduction::Mean).unwrap();
    assert!((vec1f(&m_mean)[0] - vec1f(&t["mse_mean"])[0]).abs() < 1e-5);
    let m_sum = mse_loss(&t["input"], &t["target"], Reduction::Sum).unwrap();
    assert!((vec1f(&m_sum)[0] - vec1f(&t["mse_sum"])[0]).abs() < 1e-4);
    let l1 = l1_loss(&t["input"], &t["target"], Reduction::Mean).unwrap();
    assert!((vec1f(&l1)[0] - vec1f(&t["l1_mean"])[0]).abs() < 1e-5);
    let sl1 = smooth_l1_loss(&t["input"], &t["target"], 1.0, Reduction::Mean).unwrap();
    assert!((vec1f(&sl1)[0] - vec1f(&t["smooth_l1_mean"])[0]).abs() < 1e-5);
}

#[test]
fn t31_4_adamw_three_steps() {
    setup();
    let t = load_case("train", "adamw");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = AdamW::new(AdamWConfig {
        lr: 0.01, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.01,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}

#[test]
fn t31_5_lion_three_steps() {
    setup();
    let t = load_case("train", "lion");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = Lion::new(LionConfig {
        lr: 0.001, betas: (0.9, 0.99), weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-5, 1e-5);
}

#[test]
fn t31_6_adafactor_three_steps() {
    setup();
    let t = load_case("train", "adafactor");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = Adafactor::new(AdafactorConfig {
        lr: 0.01,
        eps1: 1e-30,
        clip_threshold: 1.0,
        decay_rate: -0.8,
        beta1: None,
        weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}

// ───────────── расширение: оптимизаторы AdEMAMix / Sophia / Adam8bit ─────────────

#[test]
fn t31_8_adem_amix_three_steps() {
    setup();
    let t = load_case("train", "adem_amix");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = AdemAmix::new(AdemAmixConfig {
        lr: 0.01, betas: (0.9, 0.999, 0.9999), alpha: 2.0, eps: 1e-8, weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}

#[test]
fn t31_9_sophia_three_steps() {
    setup();
    let t = load_case("train", "sophia");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = Sophia::new(SophiaConfig {
        lr: 0.01, betas: (0.96, 0.99), rho: 0.04, eps: 1e-12, weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}

#[test]
fn t31_10_adam8bit_three_steps() {
    setup();
    let t = load_case("train", "adam8bit");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = Adam8bit::new(Adam8bitConfig {
        lr: 0.01, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}

// ───────────── расширение: grad_clip ─────────────

#[test]
fn t31_11_clip_grad_norm() {
    setup();
    let t = load_case("train", "grad_clip_norm");
    let mut grads = vec![t["g0"].clone(), t["g1"].clone()];
    let total = clip_grad_norm(&mut grads, 1.0).unwrap();
    let expected_total = t["total_norm"].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
    assert!((total as f32 - expected_total).abs() < 1e-3);
    assert_allclose(&grads[0], &t["g0_clipped"], 1e-4, 1e-4);
    assert_allclose(&grads[1], &t["g1_clipped"], 1e-4, 1e-4);
}

#[test]
fn t31_12_clip_grad_value() {
    setup();
    let t = load_case("train", "grad_clip_value");
    let mut grads = vec![t["g"].clone()];
    clip_grad_value(&mut grads, 0.5).unwrap();
    assert_allclose(&grads[0], &t["clipped"], 1e-5, 1e-5);
}

// ───────────── расширение: RLHF losses DPO / ORPO / KTO ─────────────

#[test]
fn t31_13_dpo() {
    setup();
    let t = load_case("train", "dpo");
    let out = dpo::compute_loss(
        &t["policy_chosen"], &t["policy_rejected"], &t["ref_chosen"], &t["ref_rejected"], 0.1,
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t31_14_orpo() {
    setup();
    let t = load_case("train", "orpo");
    let out = orpo::compute_loss(&t["chosen_logps"], &t["rejected_logps"], 0.1).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t31_15_kto() {
    setup();
    let t = load_case("train", "kto");
    let out = kto::compute_loss(
        &t["policy_chosen"], &t["ref_chosen"], &t["policy_rejected"], &t["ref_rejected"], 0.1,
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t31_7_muon_three_steps() {
    setup();
    let t = load_case("train", "muon");
    let mut params = vec![t["params_init"].clone()];
    let mut opt = Muon::new(MuonConfig {
        lr: 0.02,
        momentum: 0.95,
        nesterov: true,
        ns_steps: 5,
        weight_decay: 0.0,
    });
    for step in 0..3 {
        let grads = vec![t[&format!("grad_{step}")].clone()];
        opt.step_params(&mut params, &grads).unwrap();
    }
    assert_allclose(&params[0], &t["params_final"], 1e-4, 1e-4);
}
