use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::attention::softmax::{
    differential_attention, flash_attention_v2, flash_decode,
    lightning_attention, mla_attention, nsa_attention, ring_attention_local,
    streaming_sink_attention, stripe_attention, NsaConfig, SinkConfig,
};
use synaptix_ops::attention::softmax::sparse::{
    bigbird_attention, blockwise_attention, longformer_attention, reformer_lsh_attention,
    strided_attention, BigBirdConfig,
};
use synaptix_ops::attention::linear::{
    abc_attention, based_attention, chunk_scan, cosformer_attention, delta_net_attention,
    gated_delta_net_attention, gla_attention, hyena_attention, linformer_attention,
    naive_linear_attention, performer_attention, retnet_attention, synthesizer_attention,
    tnn_attention,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t11_1_mla() {
    setup();
    let t = load_case("attention_advanced", "mla");
    let q_nope = &t["q_nope"];
    let q_rope = &t["q_rope"];
    let d_nope = q_nope.dims()[3];
    let d_rope = q_rope.dims()[3];
    let scale = 1.0 / ((d_nope + d_rope) as f32).sqrt();
    let out = mla_attention(q_nope, q_rope, &t["k_nope"], &t["k_rope"], &t["v"], scale, None).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_2_lightning_no_causal() {
    setup();
    let t = load_case("attention_advanced", "lightning_no_causal");
    let out = lightning_attention(&t["q"], &t["k"], &t["v"], None, false).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_3_lightning_causal() {
    setup();
    let t = load_case("attention_advanced", "lightning_causal");
    let out = lightning_attention(&t["q"], &t["k"], &t["v"], Some(&t["slope"]), true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_4_ring_no_causal() {
    setup();
    let t = load_case("attention_advanced", "ring_no_causal");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = ring_attention_local(&t["q"], &t["k"], &t["v"], scale, 4, false).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_5_ring_causal() {
    setup();
    let t = load_case("attention_advanced", "ring_causal");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = ring_attention_local(&t["q"], &t["k"], &t["v"], scale, 4, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_6_nsa() {
    setup();
    let t = load_case("attention_advanced", "nsa");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let cfg = NsaConfig { block_size: 4, window_size: 4 };
    let out = nsa_attention(&t["q"], &t["k"], &t["v"], scale, &cfg, None).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_7_differential() {
    setup();
    let t = load_case("attention_advanced", "differential");
    let d = t["q1"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = differential_attention(
        &t["q1"], &t["q2"], &t["k1"], &t["k2"], &t["v"], scale, 0.5, None,
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_8_streaming_sink() {
    setup();
    let t = load_case("attention_advanced", "streaming_sink");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let cfg = SinkConfig { num_sink_tokens: 2, window_size: 4 };
    let out = streaming_sink_attention(&t["q"], &t["k"], &t["v"], scale, &cfg).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_9_stripe() {
    setup();
    let t = load_case("attention_advanced", "stripe");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = stripe_attention(&t["q"], &t["k"], &t["v"], scale, 3, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_10_blockwise() {
    setup();
    let t = load_case("attention_advanced", "blockwise");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = blockwise_attention(&t["q"], &t["k"], &t["v"], scale, 4, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_11_longformer() {
    setup();
    let t = load_case("attention_advanced", "longformer");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = longformer_attention(&t["q"], &t["k"], &t["v"], scale, 2, &[0, 5], true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_12_strided() {
    setup();
    let t = load_case("attention_advanced", "strided");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = strided_attention(&t["q"], &t["k"], &t["v"], scale, 4, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_13_bigbird() {
    setup();
    let t = load_case("attention_advanced", "bigbird");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let cfg = BigBirdConfig { window: 2, num_global: 2, random_per_row: 0, seed: 0 };
    let out = bigbird_attention(&t["q"], &t["k"], &t["v"], scale, &cfg, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_14_reformer_lsh() {
    setup();
    let t = load_case("attention_advanced", "reformer_lsh");
    let d = t["q"].dims()[3];
    let scale = 1.0 / (d as f32).sqrt();
    let out = reformer_lsh_attention(&t["q"], &t["k"], &t["v"], &t["buckets"], scale, true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

// ───────────────────────── linear-attention family ─────────────────────────

#[test]
fn t11_15_naive_linear() {
    setup();
    let t = load_case("attention_advanced", "naive_linear");
    let out = naive_linear_attention(&t["q"], &t["k"], &t["v"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_16_retnet() {
    setup();
    let t = load_case("attention_advanced", "retnet");
    let out = retnet_attention(&t["q"], &t["k"], &t["v"], 0.9).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_17_gla() {
    setup();
    let t = load_case("attention_advanced", "gla");
    let out = gla_attention(&t["q"], &t["k"], &t["v"], &t["gate"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_18_delta_net() {
    setup();
    let t = load_case("attention_advanced", "delta_net");
    let out = delta_net_attention(&t["q"], &t["k"], &t["v"], 0.5).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_19_gated_delta_net() {
    setup();
    let t = load_case("attention_advanced", "gated_delta_net");
    let dk = t["q"].dims()[3];
    let q_scale = (dk as f32).powf(-0.5);
    let out = gated_delta_net_attention(
        &t["q"], &t["k"], &t["v"], &t["g"], &t["beta"], q_scale,
    ).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_20_chunk_scan() {
    setup();
    let t = load_case("attention_advanced", "chunk_scan");
    // chunk_size=4 над s=8 → проверяем эквивалентность рекуррентному causal-скану
    let out = chunk_scan(&t["q"], &t["k"], &t["v"], 4).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_21_based() {
    setup();
    let t = load_case("attention_advanced", "based");
    let out = based_attention(&t["q"], &t["k"], &t["v"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_22_cosformer() {
    setup();
    let t = load_case("attention_advanced", "cosformer");
    let out = cosformer_attention(&t["q"], &t["k"], &t["v"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_23_performer() {
    setup();
    let t = load_case("attention_advanced", "performer");
    let out = performer_attention(&t["q"], &t["k"], &t["v"], &t["proj"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_24_linformer() {
    setup();
    let t = load_case("attention_advanced", "linformer");
    let out = linformer_attention(&t["q"], &t["k"], &t["v"], &t["e_proj"], &t["f_proj"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_25_tnn() {
    setup();
    let t = load_case("attention_advanced", "tnn");
    let out = tnn_attention(&t["q"], &t["k"], &t["v"], &t["rel_kernel"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_26_synthesizer_no_causal() {
    setup();
    let t = load_case("attention_advanced", "synthesizer_no_causal");
    let out = synthesizer_attention(&t["q"], &t["k"], &t["v"], &t["synth"], false).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_27_synthesizer_causal() {
    setup();
    let t = load_case("attention_advanced", "synthesizer_causal");
    let out = synthesizer_attention(&t["q"], &t["k"], &t["v"], &t["synth"], true).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_28_hyena() {
    setup();
    let t = load_case("attention_advanced", "hyena");
    let out = hyena_attention(&t["q"], &t["k"], &t["v"], &t["filt"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_29_abc() {
    setup();
    let t = load_case("attention_advanced", "abc");
    let out = abc_attention(&t["q"], &t["k"], &t["v"], &t["slot_proj"]).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

// FlashAttention (online-softmax, bit-exact со стандартным attention) + flash-decode.

#[test]
fn t11_30_flash_v2_nomask() {
    setup();
    let t = load_case("attention_advanced", "flash_v2_nomask");
    let out = flash_attention_v2(&t["q"], &t["k"], &t["v"], None).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_31_flash_v2_causal() {
    setup();
    let t = load_case("attention_advanced", "flash_v2_causal");
    let out = flash_attention_v2(&t["q"], &t["k"], &t["v"], Some(&t["mask"])).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}

#[test]
fn t11_33_flash_decode() {
    setup();
    let t = load_case("attention_advanced", "flash_decode");
    let out = flash_decode(&t["q"], &t["k_cache"], &t["v_cache"], None).unwrap();
    assert_allclose(&out, &t["output"], 1e-4, 1e-4);
}
