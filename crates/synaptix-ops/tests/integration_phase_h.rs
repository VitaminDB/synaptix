use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::ffn::swiglu;
use synaptix_ops::mask::causal_mask;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::{RopeCache, RopeLayout, apply_rope};

#[test]
fn mini_transformer_block_runs() {
    synaptix_kernels_cpu::ensure_registered();
    let batch = 1usize;
    let heads = 2usize;
    let seq = 4usize;
    let head_dim = 8usize;
    let model_dim = heads * head_dim;
    let hidden = 32usize;

    let total_x = batch * seq * model_dim;
    let x_data: Vec<f32> = (0..total_x).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
    let x = Tensor::from_vec(x_data, (batch, seq, model_dim), Device::Cpu).unwrap();

    let make_w = |out_features: usize, in_features: usize, offset: usize| -> Tensor {
        let total = out_features * in_features;
        let data: Vec<f32> = (0..total)
            .map(|i| (((i + offset) % 17) as f32) * 0.02 - 0.15)
            .collect();
        Tensor::from_vec(data, (out_features, in_features), Device::Cpu).unwrap()
    };

    let w_q = make_w(model_dim, model_dim, 0);
    let w_k = make_w(model_dim, model_dim, 1);
    let w_v = make_w(model_dim, model_dim, 2);
    let w_o = make_w(model_dim, model_dim, 3);

    let rms_w = Tensor::from_vec(vec![1.0_f32; model_dim], (model_dim,), Device::Cpu).unwrap();

    let normed_in = rms_norm(&x, &rms_w, 1e-6).unwrap();

    let normed_2d = normed_in.reshape((batch * seq, model_dim)).unwrap();
    let q = normed_2d
        .matmul(&w_q.transpose(0, 1).unwrap().contiguous().unwrap())
        .unwrap()
        .reshape((batch, seq, heads, head_dim))
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let k = normed_2d
        .matmul(&w_k.transpose(0, 1).unwrap().contiguous().unwrap())
        .unwrap()
        .reshape((batch, seq, heads, head_dim))
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let v = normed_2d
        .matmul(&w_v.transpose(0, 1).unwrap().contiguous().unwrap())
        .unwrap()
        .reshape((batch, seq, heads, head_dim))
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();

    let rope_cache = RopeCache::new(head_dim, seq, 10000.0, Device::Cpu).unwrap();
    let q_rope = apply_rope(&q, &rope_cache, None, RopeLayout::Split).unwrap();
    let k_rope = apply_rope(&k, &rope_cache, None, RopeLayout::Split).unwrap();

    let mask = causal_mask(seq, Device::Cpu).unwrap();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn = scaled_dot_attention(&q_rope, &k_rope, &v, scale, Some(&mask)).unwrap();
    let attn_merged = attn
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((batch, seq, model_dim))
        .unwrap();
    let attn_proj_2d = attn_merged
        .reshape((batch * seq, model_dim))
        .unwrap()
        .matmul(&w_o.transpose(0, 1).unwrap().contiguous().unwrap())
        .unwrap()
        .reshape((batch, seq, model_dim))
        .unwrap();

    let after_attn = x.add(&attn_proj_2d).unwrap();

    let normed_ffn = rms_norm(&after_attn, &rms_w, 1e-6).unwrap();
    let w_gate = make_w(hidden, model_dim, 4);
    let w_up = make_w(hidden, model_dim, 5);
    let w_down = make_w(model_dim, hidden, 6);

    let ffn_2d = normed_ffn.reshape((batch * seq, model_dim)).unwrap();
    let ffn_out_2d = swiglu(&ffn_2d, &w_gate, &w_up, &w_down).unwrap();
    let ffn_out = ffn_out_2d.reshape((batch, seq, model_dim)).unwrap();
    let block_out = after_attn.add(&ffn_out).unwrap();

    assert_eq!(block_out.dims(), &[batch, seq, model_dim]);
    let v: Vec<f32> = block_out
        .reshape((batch * seq * model_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut finite = 0usize;
    for x in &v {
        if x.is_finite() {
            finite += 1;
        }
    }
    assert_eq!(finite, v.len(), "all elements should be finite");
}
