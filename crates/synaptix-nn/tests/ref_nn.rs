use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::adapters::lora::LoraLinear;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::ffn::swiglu;
use synaptix_ops::norm::layer_norm;
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() {
    ensure_registered();
}

#[test]
fn t09_1_linear_forward() {
    setup();
    let t = load_case("nn", "linear_forward");
    let lin = Linear::new(t["weight"].clone(), Some(t["bias"].clone())).unwrap();
    let result = lin.forward(&t["input"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t09_2_linear_no_bias() {
    setup();
    let t = load_case("nn", "linear_no_bias");
    let lin = Linear::new(t["weight"].clone(), None).unwrap();
    let result = lin.forward(&t["input"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t09_3_sequential_forward() {
    setup();
    let t = load_case("nn", "sequential_forward");
    let lin1 = Linear::new(t["lin1_weight"].clone(), Some(t["lin1_bias"].clone())).unwrap();
    let lin2 = Linear::new(t["lin2_weight"].clone(), Some(t["lin2_bias"].clone())).unwrap();
    let h1 = lin1.forward(&t["input"]).unwrap();
    let h2 = gelu_tanh(&h1).unwrap();
    let h3 = layer_norm(&h2, Some(&t["ln_weight"]), Some(&t["ln_bias"]), 1e-5).unwrap();
    let result = lin2.forward(&h3).unwrap();
    assert_allclose(&result, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t09_4_transformer_block() {
    setup();
    let t = load_case("nn", "transformer_block");
    let x = &t["input"];

    let q_proj = Linear::new(t["q_proj_weight"].clone(), None).unwrap();
    let k_proj = Linear::new(t["k_proj_weight"].clone(), None).unwrap();
    let v_proj = Linear::new(t["v_proj_weight"].clone(), None).unwrap();
    let o_proj = Linear::new(t["o_proj_weight"].clone(), None).unwrap();

    let r = layer_norm(x, Some(&t["norm1_weight"]), Some(&t["norm1_bias"]), 1e-5).unwrap();
    let dims = r.dims().to_vec();
    let (batch, seq, hidden) = (dims[0], dims[1], dims[2]);
    let n_heads = 8usize;
    let head_dim = hidden / n_heads;

    let to_heads = |t: &Tensor| -> Tensor {
        t.reshape((batch, seq, n_heads, head_dim))
            .unwrap()
            .permute(vec![0, 2, 1, 3])
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let q = to_heads(&q_proj.forward(&r).unwrap());
    let k = to_heads(&k_proj.forward(&r).unwrap());
    let v = to_heads(&v_proj.forward(&r).unwrap());

    let mask = synaptix_ops::mask::causal_mask(seq, synaptix_core::device::Device::Cpu).unwrap();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn = scaled_dot_attention(&q, &k, &v, scale, Some(&mask)).unwrap();
    let attn = attn
        .permute(vec![0, 2, 1, 3])
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((batch, seq, hidden))
        .unwrap();
    let x1 = x.add(&o_proj.forward(&attn).unwrap()).unwrap();

    let r2 = layer_norm(&x1, Some(&t["norm2_weight"]), Some(&t["norm2_bias"]), 1e-5).unwrap();
    let ffn = swiglu(&r2, &t["w_gate_weight"], &t["w_up_weight"], &t["w_down_weight"]).unwrap();
    let result = x1.add(&ffn).unwrap();

    assert_allclose(&result, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t09_5_lora_forward() {
    setup();
    let t = load_case("nn", "lora_forward");
    let base = Linear::new(t["base_weight"].clone(), None).unwrap();
    let lora_a = Linear::new(t["lora_a"].clone(), None).unwrap();
    let lora_b = Linear::new(t["lora_b"].clone(), None).unwrap();
    let scaling = t["scale"].to_scalar::<f32>().unwrap();
    let lora = LoraLinear { base, lora_a, lora_b, scaling };
    let result = lora.forward(&t["input"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-5, 1e-5);
}

#[test]
fn t09_6_lora_merge() {
    setup();
    let t = load_case("nn", "lora_merge");
    let base = Linear::new(t["base_weight"].clone(), None).unwrap();
    let lora_a = Linear::new(t["lora_a"].clone(), None).unwrap();
    let lora_b = Linear::new(t["lora_b"].clone(), None).unwrap();
    let scaling = t["scale"].to_scalar::<f32>().unwrap();
    let lora = LoraLinear { base, lora_a, lora_b, scaling };

    let unmerged = lora.forward(&t["input"]).unwrap();
    assert_allclose(&unmerged, &t["output_unmerged"], 1e-5, 1e-5);

    let merged_weight = lora.merge_weights().unwrap();
    assert_allclose(&merged_weight, &t["merged_weight"], 1e-5, 1e-5);

    let merged_linear = Linear::new(merged_weight, None).unwrap();
    let merged_result = merged_linear.forward(&t["input"]).unwrap();
    assert_allclose(&merged_result, &t["output_merged"], 1e-5, 1e-5);
}
