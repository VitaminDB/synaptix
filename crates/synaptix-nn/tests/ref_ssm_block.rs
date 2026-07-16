use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::ssm_block::mamba2_block::Mamba2Block;
use synaptix_nn::ssm_block::xlstm_block::{XLstmBlock, XLstmKind};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t22_1_mamba2_block_vs_python() {
    setup();
    let t = load_case("nn_ssm_block", "mamba2_block");
    let hidden_size = t["x"].dims()[2];
    let d_inner = t["conv_w"].dims()[0];
    let num_heads = t["a_log"].dims()[0];
    let head_dim = d_inner / num_heads;
    let d_state = (t["in_proj_w"].dims()[0] - 2 * d_inner - num_heads) / 2;
    let d_conv = t["conv_w"].dims()[2];

    let block = Mamba2Block::from_weights(
        t["in_proj_w"].clone(),
        t["conv_w"].clone(),
        Some(t["conv_b"].clone()),
        t["out_proj_w"].clone(),
        t["a_log"].clone(),
        t["d"].clone(),
        t["dt_bias"].clone(),
        t["norm_w"].clone(),
        hidden_size, d_state, num_heads, head_dim, d_conv, 1e-5,
    ).unwrap();
    let y = block.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t22_2_xlstm_slstm_vs_python() {
    setup();
    let t = load_case("nn_ssm_block", "xlstm_slstm");
    let block = XLstmBlock::from_weights(
        t["gate_w"].clone(), Some(t["gate_b"].clone()),
        t["out_w"].clone(), None, XLstmKind::SLstm,
    ).unwrap();
    let y = block.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t22_3_xlstm_mlstm_vs_python() {
    setup();
    let t = load_case("nn_ssm_block", "xlstm_mlstm");
    let block = XLstmBlock::from_weights(
        t["gate_w"].clone(), Some(t["gate_b"].clone()),
        t["out_w"].clone(), None, XLstmKind::MLstm,
    ).unwrap();
    let y = block.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-5, 1e-5);
}
