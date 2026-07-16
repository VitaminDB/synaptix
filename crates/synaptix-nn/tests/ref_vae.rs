use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::vae::{kl_divergence, reparameterize_with_eps, PerChannelStats, PixelNorm};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t19_1_pixel_norm() {
    setup();
    let t = load_case("nn_vae", "pixel_norm");
    let pn = PixelNorm::new(1e-8);
    let y = pn.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t19_2_per_channel_stats_normalize() {
    setup();
    let t = load_case("nn_vae", "per_channel_stats");
    let mean: Vec<f32> = t["mean"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let std: Vec<f32> = t["std"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let stats = PerChannelStats::new(mean, std);
    let normed = stats.normalize(&t["x"]).unwrap();
    assert_allclose(&normed, &t["normalized"], 1e-5, 1e-5);
}

#[test]
fn t19_3_per_channel_stats_denormalize() {
    setup();
    let t = load_case("nn_vae", "per_channel_stats");
    let mean: Vec<f32> = t["mean"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let std: Vec<f32> = t["std"].flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let stats = PerChannelStats::new(mean, std);
    let denormed = stats.denormalize(&t["normalized"]).unwrap();
    assert_allclose(&denormed, &t["denormalized"], 1e-5, 1e-5);
}

#[test]
fn t19_4_reparameterize() {
    setup();
    let t = load_case("nn_vae", "reparameterize");
    let z = reparameterize_with_eps(&t["mean"], &t["logvar"], Some(&t["eps"])).unwrap();
    assert_allclose(&z, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t19_5_kl_divergence() {
    setup();
    let t = load_case("nn_vae", "kl_divergence");
    let kl = kl_divergence(&t["mean"], &t["logvar"]).unwrap();
    assert_allclose(&kl, &t["output"], 1e-5, 1e-5);
}
