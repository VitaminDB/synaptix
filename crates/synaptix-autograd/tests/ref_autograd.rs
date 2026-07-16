use synaptix_core::dtype::DType;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
    synaptix_autograd::init().unwrap();
}

#[test]
fn t11_1_matmul_grad() {
    setup();
    let t = load_case("autograd", "matmul_grad");
    let a = t["a"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let b = t["b"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let out = a.matmul(&b).unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let grad_a = a.grad().expect("grad_a").to_dtype(DType::F32).unwrap();
    let grad_b = b.grad().expect("grad_b").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad_a, &t["grad_a_analytical"], 1e-5, 1e-5);
    assert_allclose(&grad_b, &t["grad_b_analytical"], 1e-5, 1e-5);
}

#[test]
fn t11_3_gelu_tanh_grad() {
    setup();
    let t = load_case("autograd", "gelu_grad");
    let x = t["input"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let out = x.gelu_tanh().unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let grad = x.grad().expect("grad").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad, &t["grad_analytical"], 1e-4, 1e-4);
}

#[test]
fn t11_7_broadcast_grad() {
    setup();
    let t = load_case("autograd", "broadcast_grad");
    let a = t["a"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let b = t["b"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let out = a.add(&b).unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let grad_a = a.grad().expect("grad_a").to_dtype(DType::F32).unwrap();
    let grad_b = b.grad().expect("grad_b").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad_a, &t["grad_a"], 1e-5, 1e-5);
    assert_allclose(&grad_b, &t["grad_b"], 1e-5, 1e-5);
}

#[test]
fn t11_2_rms_norm_grad() {
    setup();
    let t = load_case("autograd", "rms_norm_grad");
    let x = t["input"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let weight = t["weight"].to_dtype(DType::F64).unwrap();
    let last = x.rank() - 1;
    let var = x.sqr().unwrap().mean_keepdim(last).unwrap();
    let inv = var.add_scalar(1e-6).unwrap().sqrt().unwrap().recip().unwrap();
    let x_norm = x.broadcast_mul(&inv).unwrap();
    let out = x_norm.broadcast_mul(&weight).unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let grad = x.grad().expect("grad").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad, &t["grad_x_analytical"], 1e-4, 1e-4);
}

#[test]
fn t11_4_softmax_ce_grad() {
    setup();
    let t = load_case("autograd", "softmax_ce_grad");
    let logits = t["logits"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let targets = &t["targets"];

    let log_probs = synaptix_ops::attention::log_softmax_dim(&logits, 1).unwrap();
    let targets_u32 = targets.to_dtype(DType::U32).unwrap();
    let target_idx = targets_u32.unsqueeze(1).unwrap();
    let picked = log_probs.gather(&target_idx, 1).unwrap();
    let loss = picked
        .sum_all()
        .unwrap()
        .mul_scalar(-1.0 / picked.dims()[0] as f32)
        .unwrap();
    loss.backward().unwrap();
    let grad = logits.grad().expect("grad").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad, &t["grad_analytical"], 1e-4, 1e-4);
}

#[test]
fn t11_5_mlp_training() {
    setup();
    let t = load_case("autograd", "mlp_training");
    let loss_curve = t["loss_curve"].to_vec1::<f32>().unwrap();
    let initial = loss_curve[0];
    let final_loss = loss_curve[99];
    assert!(initial > 0.0);
    assert!(final_loss < initial, "loss must decrease: {} -> {}", initial, final_loss);
    assert!(final_loss < 3.0, "final loss too high: {}", final_loss);
}

#[test]
fn t11_6_chain_rule() {
    setup();
    let t = load_case("autograd", "chain_rule");
    let x = t["input"].to_dtype(DType::F64).unwrap().requires_grad_(true);
    let weight = t["weight"].to_dtype(DType::F64).unwrap();
    let weight_rms = t["weight_rms"].to_dtype(DType::F64).unwrap();

    let last = x.rank() - 1;
    let var = x.sqr().unwrap().mean_keepdim(last).unwrap();
    let inv = var.add_scalar(1e-6).unwrap().sqrt().unwrap().recip().unwrap();
    let normed = x.broadcast_mul(&inv).unwrap().broadcast_mul(&weight_rms).unwrap();
    let lin = normed.matmul(&weight.transpose(0, 1).unwrap().contiguous().unwrap()).unwrap();
    let out = lin.relu().unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let grad = x.grad().expect("grad").to_dtype(DType::F32).unwrap();
    assert_allclose(&grad, &t["grad_analytical"], 1e-4, 1e-4);
}
