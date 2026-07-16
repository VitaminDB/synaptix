use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::diffusion::{Apg, Cfg, ControlNet, Gligen, IpAdapter, Pag, T2iAdapter};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_cfg_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "cfg");
    let cfg = Cfg::new(7.5);
    let y = cfg.apply(&t["cond"], &t["uncond"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_pag_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "pag");
    let pag = Pag::new(3.0);
    let y = pag.apply(&t["cond"], &t["perturbed"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_apg_orthogonal_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "apg");
    let apg = Apg::new(7.5, 0.0).with_norm_threshold(2.5);
    let y = apg.apply(&t["cond"], &t["uncond"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_apg_rescale_active_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "apg_rescale_active");
    let apg = Apg::new(4.0, 0.0).with_norm_threshold(1.0);
    let y = apg.apply(&t["cond"], &t["uncond"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_controlnet_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "controlnet");
    let cn = ControlNet::from_weights(t["proj_w"].clone(), Some(t["proj_b"].clone()), 0.75).unwrap();
    let y = cn.forward(&t["x"], &t["control"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_t2i_adapter_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "t2i_adapter");
    let ad = T2iAdapter::from_weights(t["proj_w"].clone(), Some(t["proj_b"].clone()), 1.2).unwrap();
    let y = ad.forward(&t["x"], &t["condition"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_ip_adapter_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "ip_adapter");
    let ip = IpAdapter::from_weights(t["proj_w"].clone(), Some(t["proj_b"].clone()), 0.6).unwrap();
    let y = ip.forward(&t["x"], &t["image_emb"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_gligen_vs_python() {
    setup();
    let t = load_case("nn_diffusion", "gligen");
    let g = Gligen::from_weights(
        t["entity_w"].clone(), Some(t["entity_b"].clone()),
        t["box_w"].clone(), Some(t["box_b"].clone()),
        t["gate"].clone(),
        1.5,
    ).unwrap();
    let y = g.forward(&t["x"], &t["boxes"], &t["entity_emb"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}
