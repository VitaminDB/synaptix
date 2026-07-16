use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::unet::{
    sinusoidal_timestep_embedding, ResNetBlock, TimeEmbedding, UNet2d, UNet3d,
    UNetAttnBlock, UNetCrossAttnBlock,
};
use synaptix_nn::Linear;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t_sinusoidal_only_vs_python() {
    setup();
    let t = load_case("nn_unet", "sinusoidal_only");
    let y = sinusoidal_timestep_embedding(&t["timesteps"], 16).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t_time_embedding_vs_python() {
    setup();
    let t = load_case("nn_unet", "time_embedding");
    let te = TimeEmbedding::from_weights(
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
    ).unwrap();
    let y = te.forward(&t["timesteps"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_resnet_block_vs_python() {
    setup();
    let t = load_case("nn_unet", "resnet_block");
    let r = ResNetBlock::from_weights(
        t["norm1_w"].clone(), t["norm1_b"].clone(),
        t["conv1_w"].clone(), Some(t["conv1_b"].clone()),
        t["norm2_w"].clone(), t["norm2_b"].clone(),
        t["conv2_w"].clone(), Some(t["conv2_b"].clone()),
        t["time_emb_proj_w"].clone(), Some(t["time_emb_proj_b"].clone()),
        Some(t["shortcut_w"].clone()),
        1e-5,
    ).unwrap();
    let y = r.forward(&t["x"], &t["time_emb"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_attn_block_vs_python() {
    setup();
    let t = load_case("nn_unet", "attn_block");
    let attn = UNetAttnBlock::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        2, 1e-5,
    ).unwrap();
    let y = attn.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_cross_attn_block_vs_python() {
    setup();
    let t = load_case("nn_unet", "cross_attn_block");
    let attn = UNetCrossAttnBlock::from_weights(
        t["norm_w"].clone(), t["norm_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        2, 1e-5,
    ).unwrap();
    let y = attn.forward(&t["x"], &t["context"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_unet_2d_vs_python() {
    setup();
    let t = load_case("nn_unet", "unet_2d");
    let mut u = UNet2d::new(6, 6, 8, 2, 12, 8, 16, synaptix_core::device::Device::Cpu, synaptix_core::dtype::DType::F32).unwrap();

    u.conv_in = Linear::new(t["conv_in_w"].clone(), Some(t["conv_in_b"].clone())).unwrap();
    u.time_embedding = TimeEmbedding::from_weights(
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
    ).unwrap();
    u.resnet = ResNetBlock::from_weights(
        t["n1w"].clone(), t["n1b"].clone(),
        t["c1w"].clone(), Some(t["c1b"].clone()),
        t["n2w"].clone(), t["n2b"].clone(),
        t["c2w"].clone(), Some(t["c2b"].clone()),
        t["tew"].clone(), Some(t["teb"].clone()),
        None, 1e-5,
    ).unwrap();
    u.attn = UNetAttnBlock::from_weights(
        t["a_nw"].clone(), t["a_nb"].clone(),
        t["a_qw"].clone(), t["a_kw"].clone(), t["a_vw"].clone(), t["a_ow"].clone(),
        2, 1e-5,
    ).unwrap();
    u.cross_attn = UNetCrossAttnBlock::from_weights(
        t["c_nw"].clone(), t["c_nb"].clone(),
        t["c_qw"].clone(), t["c_kw"].clone(), t["c_vw"].clone(), t["c_ow"].clone(),
        2, 1e-5,
    ).unwrap();
    u.conv_out = Linear::new(t["conv_out_w"].clone(), Some(t["conv_out_b"].clone())).unwrap();

    let y = u.forward(&t["x"], &t["timesteps"], &t["text_ctx"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t_unet_3d_vs_python() {
    setup();
    let t = load_case("nn_unet", "unet_3d");
    let mut u = UNet3d::new(6, 6, 8, 2, 8, 16, synaptix_core::device::Device::Cpu, synaptix_core::dtype::DType::F32).unwrap();

    u.conv_in = Linear::new(t["conv_in_w"].clone(), Some(t["conv_in_b"].clone())).unwrap();
    u.time_embedding = TimeEmbedding::from_weights(
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
    ).unwrap();
    u.resnet = ResNetBlock::from_weights(
        t["n1w"].clone(), t["n1b"].clone(),
        t["c1w"].clone(), Some(t["c1b"].clone()),
        t["n2w"].clone(), t["n2b"].clone(),
        t["c2w"].clone(), Some(t["c2b"].clone()),
        t["tew"].clone(), Some(t["teb"].clone()),
        None, 1e-5,
    ).unwrap();
    u.temporal_attn = UNetAttnBlock::from_weights(
        t["a_nw"].clone(), t["a_nb"].clone(),
        t["a_qw"].clone(), t["a_kw"].clone(), t["a_vw"].clone(), t["a_ow"].clone(),
        2, 1e-5,
    ).unwrap();
    u.conv_out = Linear::new(t["conv_out_w"].clone(), Some(t["conv_out_b"].clone())).unwrap();

    let y = u.forward(&t["x"], &t["timesteps"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
