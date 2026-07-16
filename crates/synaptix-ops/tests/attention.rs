use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::{log_softmax_dim, softmax_dim};
use synaptix_ops::attention::softmax::{
    cross_attention, gqa_attention, mqa_attention, scaled_dot_attention,
    sliding_window_attention,
};
use synaptix_ops::mask::causal_mask;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(*a, *b, tol), "{label} idx {i}: {a} vs {b}");
    }
}

fn ref_softmax(x: &[f64], dims: &[usize], dim: usize) -> Vec<f64> {
    let inner: usize = dims[dim + 1..].iter().product::<usize>().max(1);
    let outer: usize = dims[..dim].iter().product::<usize>().max(1);
    let k = dims[dim];
    let mut out = vec![0.0f64; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut max = f64::NEG_INFINITY;
            for r in 0..k {
                let idx = (o * k + r) * inner + i;
                if x[idx] > max { max = x[idx]; }
            }
            let mut sum = 0.0f64;
            for r in 0..k {
                let idx = (o * k + r) * inner + i;
                sum += (x[idx] - max).exp();
            }
            for r in 0..k {
                let idx = (o * k + r) * inner + i;
                out[idx] = (x[idx] - max).exp() / sum;
            }
        }
    }
    out
}

#[test]
fn softmax_stable_to_extreme_values() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![1e30, 1e30, 1e30, 1.0, 2.0, 3.0];
    let xt = Tensor::from_vec(x.clone(), (2, 3), Device::Cpu).unwrap();
    let y = softmax_dim(&xt, 1).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for r in 0..2 {
        let row_sum: f32 = (0..3).map(|c| v[r * 3 + c]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "row {r} sum = {row_sum}");
    }
    assert!((v[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((v[1] - 1.0 / 3.0).abs() < 1e-5);
    assert!((v[2] - 1.0 / 3.0).abs() < 1e-5);
}

#[test]
fn softmax_matches_reference_3d() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 1.2).collect();
    let xt = Tensor::from_vec(x.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let y = softmax_dim(&xt, 2).unwrap();
    let flat: Vec<f32> = y.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let r = ref_softmax(&xf, &[2, 3, 4], 2);
    let rf: Vec<f32> = r.iter().map(|&v| v as f32).collect();
    assert_close(&flat, &rf, 1e-5, "softmax");
}

#[test]
fn log_softmax_consistent_with_softmax() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let xt = Tensor::from_vec(x.clone(), (1, 4), Device::Cpu).unwrap();
    let lp = log_softmax_dim(&xt, 1).unwrap();
    let p = softmax_dim(&xt, 1).unwrap();
    let lpv: Vec<f32> = lp.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let pv: Vec<f32> = p.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for (a, b) in lpv.iter().zip(pv.iter()) {
        let diff = (a - b.ln()).abs();
        assert!(diff < 1e-5, "lp={a} vs log(p)={}", b.ln());
    }
}

#[test]
fn scaled_dot_attention_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
    let k: Vec<f32> = (0..16).map(|i| (i as f32) * 0.07 - 0.5).collect();
    let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
    let qt = Tensor::from_vec(q.clone(), (1, 2, 2, 4), Device::Cpu).unwrap();
    let kt = Tensor::from_vec(k.clone(), (1, 2, 2, 4), Device::Cpu).unwrap();
    let vt = Tensor::from_vec(v.clone(), (1, 2, 2, 4), Device::Cpu).unwrap();
    let scale = 0.5_f32;
    let y = scaled_dot_attention(&qt, &kt, &vt, scale, None).unwrap();
    let flat: Vec<f32> = y
        .to_vec3::<f32>()
        .ok()
        .map(|x| x.into_iter().flatten().flatten().collect())
        .unwrap_or_else(|| {
            y.contiguous().unwrap().reshape((16usize,)).unwrap().to_vec1::<f32>().unwrap()
        });

    let mut expected = vec![0.0f32; 16];
    for b in 0..1 {
        for h in 0..2 {
            let mut q_mat = [[0.0f64; 4]; 2];
            let mut k_mat = [[0.0f64; 4]; 2];
            let mut v_mat = [[0.0f64; 4]; 2];
            for s in 0..2 {
                for d in 0..4 {
                    let idx = ((b * 2 + h) * 2 + s) * 4 + d;
                    q_mat[s][d] = q[idx] as f64;
                    k_mat[s][d] = k[idx] as f64;
                    v_mat[s][d] = v[idx] as f64;
                }
            }
            let mut scores = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let mut acc = 0.0f64;
                    for d in 0..4 {
                        acc += q_mat[i][d] * k_mat[j][d];
                    }
                    scores[i][j] = acc * scale as f64;
                }
            }
            let mut probs = [[0.0f64; 2]; 2];
            for i in 0..2 {
                let max = scores[i][0].max(scores[i][1]);
                let e0 = (scores[i][0] - max).exp();
                let e1 = (scores[i][1] - max).exp();
                let s = e0 + e1;
                probs[i][0] = e0 / s;
                probs[i][1] = e1 / s;
            }
            for i in 0..2 {
                for d in 0..4 {
                    let acc = probs[i][0] * v_mat[0][d] + probs[i][1] * v_mat[1][d];
                    let out_idx = ((b * 2 + h) * 2 + i) * 4 + d;
                    expected[out_idx] = acc as f32;
                }
            }
        }
    }
    assert_close(&flat, &expected, 1e-5, "scaled_dot");
}

#[test]
fn scaled_dot_with_causal_mask() {
    synaptix_kernels_cpu::ensure_registered();
    let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
    let k: Vec<f32> = (0..16).map(|i| (i as f32) * 0.07).collect();
    let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let qt = Tensor::from_vec(q, (1, 2, 2, 4), Device::Cpu).unwrap();
    let kt = Tensor::from_vec(k, (1, 2, 2, 4), Device::Cpu).unwrap();
    let vt = Tensor::from_vec(v, (1, 2, 2, 4), Device::Cpu).unwrap();
    let mask = causal_mask(2, Device::Cpu).unwrap();
    let y = scaled_dot_attention(&qt, &kt, &vt, 0.25, Some(&mask)).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 4]);
}

#[test]
fn mqa_broadcasts_kv_heads() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![0.1_f32; 1 * 4 * 3 * 2], (1, 4, 3, 2), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.2_f32; 1 * 1 * 3 * 2], (1, 1, 3, 2), Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![0.3_f32; 1 * 1 * 3 * 2], (1, 1, 3, 2), Device::Cpu).unwrap();
    let y = mqa_attention(&q, &k, &v, 0.7, None).unwrap();
    assert_eq!(y.dims(), &[1, 4, 3, 2]);
}

#[test]
fn gqa_repeats_kv_heads() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![0.1_f32; 1 * 4 * 3 * 2], (1, 4, 3, 2), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.2_f32; 1 * 2 * 3 * 2], (1, 2, 3, 2), Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![0.3_f32; 1 * 2 * 3 * 2], (1, 2, 3, 2), Device::Cpu).unwrap();
    let y = gqa_attention(&q, &k, &v, 0.7, None).unwrap();
    assert_eq!(y.dims(), &[1, 4, 3, 2]);
}

#[test]
fn sliding_window_attention_works() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![0.1_f32; 1 * 1 * 4 * 2], (1, 1, 4, 2), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.2_f32; 1 * 1 * 4 * 2], (1, 1, 4, 2), Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![0.3_f32; 1 * 1 * 4 * 2], (1, 1, 4, 2), Device::Cpu).unwrap();
    let y = sliding_window_attention(&q, &k, &v, 0.5, 2, None).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 2]);
}

#[test]
fn cross_attention_supports_h_q_equals_h_kv() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![0.1_f32; 1 * 2 * 3 * 4], (1, 2, 3, 4), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.2_f32; 1 * 2 * 5 * 4], (1, 2, 5, 4), Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![0.3_f32; 1 * 2 * 5 * 4], (1, 2, 5, 4), Device::Cpu).unwrap();
    let y = cross_attention(&q, &k, &v, 0.5, None).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3, 4]);
}
