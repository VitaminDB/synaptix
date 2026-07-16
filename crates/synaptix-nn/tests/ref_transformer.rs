use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::transformer::TransformerBlock;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t15_1_transformer_block() {
    setup();
    let t = load_case("nn_transformer", "block");
    let block = TransformerBlock::from_weights(
        t["n1_w"].clone(), t["n1_b"].clone(),
        t["q_w"].clone(), t["k_w"].clone(), t["v_w"].clone(), t["o_w"].clone(),
        t["n2_w"].clone(), t["n2_b"].clone(),
        t["fc1_w"].clone(), Some(t["fc1_b"].clone()),
        t["fc2_w"].clone(), Some(t["fc2_b"].clone()),
        4,
    ).unwrap();
    let y = block.forward(&t["x"]).unwrap();
    assert_allclose(&y, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t15_2_transformer_encoder_2layers() {
    setup();
    let t = load_case("nn_transformer", "encoder_2layers");
    let num_heads = 2;
    let mut cur = t["x"].clone();
    for l in 0..2 {
        let block = TransformerBlock::from_weights(
            t[&format!("l{l}_n1_w")].clone(), t[&format!("l{l}_n1_b")].clone(),
            t[&format!("l{l}_q_w")].clone(), t[&format!("l{l}_k_w")].clone(),
            t[&format!("l{l}_v_w")].clone(), t[&format!("l{l}_o_w")].clone(),
            t[&format!("l{l}_n2_w")].clone(), t[&format!("l{l}_n2_b")].clone(),
            t[&format!("l{l}_fc1_w")].clone(), Some(t[&format!("l{l}_fc1_b")].clone()),
            t[&format!("l{l}_fc2_w")].clone(), Some(t[&format!("l{l}_fc2_b")].clone()),
            num_heads,
        ).unwrap();
        cur = block.forward(&cur).unwrap();
    }
    assert_allclose(&cur, &t["output"], 1e-4, 1e-4);
}
