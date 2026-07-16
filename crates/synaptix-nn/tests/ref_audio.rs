use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::audio::{ConformerBlock, ConformerEnc, WhisperEnc};
use synaptix_nn::transformer::TransformerBlock;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t18_1_whisper_enc() {
    setup();
    let t = load_case("nn_audio", "whisper");
    let block = TransformerBlock::from_weights(
        t["n1_w"].clone(), t["n1_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["n2_w"].clone(), t["n2_b"].clone(),
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
        4,
    ).unwrap();
    let enc = WhisperEnc::from_weights(
        t["c1_w"].clone(), t["c1_b"].clone(),
        t["c2_w"].clone(), t["c2_b"].clone(),
        vec![block],
        t["final_w"].clone(), t["final_b"].clone(),
    ).unwrap();
    let y = enc.forward(&t["mel"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t18_2_conformer_block() {
    setup();
    let t = load_case("nn_audio", "conformer");
    let block = ConformerBlock::from_weights(
        t["ff1_n_w"].clone(), t["ff1_n_b"].clone(),
        t["ff1_in_w"].clone(), t["ff1_in_b"].clone(),
        t["ff1_out_w"].clone(), t["ff1_out_b"].clone(),
        t["attn_n_w"].clone(), t["attn_n_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["ff2_n_w"].clone(), t["ff2_n_b"].clone(),
        t["ff2_in_w"].clone(), t["ff2_in_b"].clone(),
        t["ff2_out_w"].clone(), t["ff2_out_b"].clone(),
        t["final_n_w"].clone(), t["final_n_b"].clone(),
        4,
    ).unwrap();
    let enc = ConformerEnc::from_weights(vec![block]).unwrap();
    let y = enc.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}
