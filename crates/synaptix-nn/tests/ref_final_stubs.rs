use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::squeezeformer::Squeezeformer;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_squeezeformer_even_vs_python() {
    setup();
    let t = load_case("nn_final", "squeezeformer_even");
    let s = Squeezeformer::from_weights(t["proj_w"].clone(), Some(t["proj_b"].clone())).unwrap();
    let y = s.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_squeezeformer_odd_vs_python() {
    setup();
    let t = load_case("nn_final", "squeezeformer_odd");
    let s = Squeezeformer::from_weights(t["proj_w"].clone(), Some(t["proj_b"].clone())).unwrap();
    let y = s.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
