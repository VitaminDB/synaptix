use synaptix_core::tensor::Tensor;
use synaptix_core::error::Result as SynResult;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::activation::{elu, gelu_exact, gelu_tanh, hardswish, leaky_relu, mish, softplus};
use synaptix_ops::attention::{log_softmax_dim, softmax_dim};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
}

fn check_act<F>(name: &str, f: F, atol_f32: f64, atol_f16: f64)
where
    F: Fn(&Tensor) -> SynResult<Tensor>,
{
    for (suffix, atol) in [("f32", atol_f32), ("f16", atol_f16), ("bf16", 5e-2)] {
        let t = load_case("activation", &format!("{}_{}", name, suffix));
        let result = f(&t["input"]).unwrap();
        assert_allclose(&result, &t["output"], atol, atol);
    }
}

#[test]
fn t03_01_relu() { setup(); check_act("relu", |x| x.relu(), 1e-6, 5e-3); }

#[test]
fn t03_02_gelu_tanh() { setup(); check_act("gelu_tanh", gelu_tanh, 1e-5, 5e-3); }

#[test]
fn t03_03_gelu_exact() { setup(); check_act("gelu_exact", gelu_exact, 1e-6, 5e-3); }

#[test]
fn t03_04_silu() { setup(); check_act("silu", |x| x.silu(), 1e-6, 5e-3); }

#[test]
fn t03_05_mish() { setup(); check_act("mish", mish, 1e-5, 5e-3); }

#[test]
fn t03_06_elu() { setup(); check_act("elu", |x| elu(x, 1.0), 1e-5, 5e-3); }

#[test]
fn t03_07_leaky_relu() { setup(); check_act("leaky_relu", |x| leaky_relu(x, 0.01), 1e-6, 5e-3); }

#[test]
fn t03_08_hardswish() { setup(); check_act("hardswish", hardswish, 1e-5, 5e-3); }

#[test]
fn t03_09_sigmoid() { setup(); check_act("sigmoid", |x| x.sigmoid(), 1e-6, 5e-3); }

#[test]
fn t03_10_tanh() { setup(); check_act("tanh", |x| x.tanh(), 1e-6, 5e-3); }

#[test]
fn t03_11_softmax() {
    setup();
    check_act("softmax", |x| softmax_dim(x, x.rank() - 1), 1e-6, 5e-3);
}

#[test]
fn t03_12_log_softmax() {
    setup();
    check_act("log_softmax", |x| log_softmax_dim(x, x.rank() - 1), 1e-6, 5e-3);
}

#[test]
fn t03_13_softplus() {
    setup();
    check_act("softplus", |x| softplus(x, 1.0, 20.0), 1e-5, 5e-3);
}
