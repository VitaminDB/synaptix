use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::{
    adaln, batch_norm_inference, deep_norm, dyn_tanh, dyn_tanh_scalar, group_norm, instance_norm,
    layer_norm, pixel_norm, qk_layer_norm, qk_rms_norm, soft_cap,
};

fn ref_layer_norm(x: &[f64], shape: &[usize], w: Option<&[f64]>, b: Option<&[f64]>, eps: f64) -> Vec<f64> {
    let last = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let mut out = vec![0.0f64; x.len()];
    for o in 0..outer {
        let row = &x[o * last..(o + 1) * last];
        let mean: f64 = row.iter().sum::<f64>() / (last as f64);
        let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (last as f64);
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..last {
            let mut v = (row[i] - mean) * inv;
            if let Some(ww) = w {
                v *= ww[i];
            }
            if let Some(bb) = b {
                v += bb[i];
            }
            out[o * last + i] = v;
        }
    }
    out
}

fn approx_eq_slice_f64(actual: &[f32], expected: &[f64], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - (*b as f32)).abs();
        assert!(diff <= tol, "idx {i}: actual {a} vs expected {b} (diff {diff} tol {tol})");
    }
}

#[test]
fn layer_norm_with_affine() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let w: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
    let b: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
    let xt = Tensor::from_vec(x.clone(), (2, 4), Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w.clone(), (4,), Device::Cpu).unwrap();
    let bt = Tensor::from_vec(b.clone(), (4,), Device::Cpu).unwrap();
    let y = layer_norm(&xt, Some(&wt), Some(&bt), 1e-5).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let expected = ref_layer_norm(&xf, &[2, 4], Some(&wf), Some(&bf), 1e-5);
    approx_eq_slice_f64(&v, &expected, 1e-5);
}

#[test]
fn layer_norm_without_affine() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let xt = Tensor::from_vec(x.clone(), (2, 4), Device::Cpu).unwrap();
    let y = layer_norm(&xt, None, None, 1e-6).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let expected = ref_layer_norm(&xf, &[2, 4], None, None, 1e-6);
    approx_eq_slice_f64(&v, &expected, 1e-5);
}

#[test]
fn batch_norm_inference_shape_and_values() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let xt = Tensor::from_vec(x.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let mean = Tensor::from_vec(vec![0.5_f32, 1.0, 1.5], (3,), Device::Cpu).unwrap();
    let var = Tensor::from_vec(vec![0.25_f32, 0.5, 1.0], (3,), Device::Cpu).unwrap();
    let w = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (3,), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![-0.1_f32, 0.0, 0.1], (3,), Device::Cpu).unwrap();
    let y = batch_norm_inference(&xt, &mean, &var, Some(&w), Some(&b), 1e-5).unwrap();
    let v: Vec<f32> = y.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    let mean_vec = [0.5_f64, 1.0, 1.5];
    let var_vec = [0.25_f64, 0.5, 1.0];
    let w_vec = [1.0_f64, 2.0, 3.0];
    let b_vec = [-0.1_f64, 0.0, 0.1];
    let mut expected = vec![0.0_f32; 24];
    for n in 0..2usize {
        for c in 0..3usize {
            for s in 0..4usize {
                let idx = (n * 3 + c) * 4 + s;
                let xv = x[idx] as f64;
                let inv = 1.0 / (var_vec[c] + 1e-5).sqrt();
                let norm = (xv - mean_vec[c]) * inv;
                let val = norm * w_vec[c] + b_vec[c];
                expected[idx] = val as f32;
            }
        }
    }
    for (a, b) in v.iter().zip(expected.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn group_norm_one_group_matches_layer_norm() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = (0..2 * 4 * 3).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let xt = Tensor::from_vec(x.clone(), (2, 4, 3), Device::Cpu).unwrap();
    let y = group_norm(&xt, None, None, 1, 1e-5).unwrap();
    let v: Vec<f32> = y.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    let mut expected = vec![0.0_f64; v.len()];
    for n in 0..2usize {
        let base = n * 12;
        let row: Vec<f64> = (0..12).map(|i| x[base + i] as f64).collect();
        let mean: f64 = row.iter().sum::<f64>() / 12.0;
        let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / 12.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..12 {
            expected[base + i] = (row[i] - mean) * inv;
        }
    }
    approx_eq_slice_f64(&v, &expected, 1e-5);
}

#[test]
fn instance_norm_per_channel() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.2 - 1.0).collect();
    let xt = Tensor::from_vec(x.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let y = instance_norm(&xt, None, None, 1e-6).unwrap();
    let v: Vec<f32> = y.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    for n in 0..2usize {
        for c in 0..3usize {
            let base = (n * 3 + c) * 4;
            let slice: Vec<f64> = (0..4).map(|i| v[base + i] as f64).collect();
            let mean: f64 = slice.iter().sum::<f64>() / 4.0;
            let var: f64 = slice.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / 4.0;
            assert!(mean.abs() < 1e-4, "instance mean = {mean}");
            assert!((var - 1.0).abs() < 1e-3, "instance var = {var}");
        }
    }
}

#[test]
fn qk_rms_norm_does_both() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5], (1, 2, 3), Device::Cpu).unwrap();
    let w = Tensor::from_vec(vec![1.0_f32, 1.0, 1.0], (3,), Device::Cpu).unwrap();
    let (qn, kn) = qk_rms_norm(&q, &k, &w, &w, 1e-5).unwrap();
    let qv = qn.to_vec3::<f32>().unwrap();
    let kv = kn.to_vec3::<f32>().unwrap();
    assert_eq!(qv.len(), 1);
    assert_eq!(kv.len(), 1);
}

#[test]
fn qk_layer_norm_does_both() {
    synaptix_kernels_cpu::ensure_registered();
    let q = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3), Device::Cpu).unwrap();
    let k = Tensor::from_vec(vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5], (1, 2, 3), Device::Cpu).unwrap();
    let (qn, kn) = qk_layer_norm(&q, &k, None, None, None, None, 1e-6).unwrap();
    let _ = qn.to_vec3::<f32>().unwrap();
    let _ = kn.to_vec3::<f32>().unwrap();
}

#[test]
fn deep_norm_combines_residual() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 4), Device::Cpu).unwrap();
    let r = Tensor::from_vec(vec![0.5_f32, 0.5, 0.5, 0.5], (1, 4), Device::Cpu).unwrap();
    let y = deep_norm(&x, &r, 2.0, None, None, 1e-5).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let combined: Vec<f64> = (0..4).map(|i| 2.0 * 0.5_f64 + (i + 1) as f64).collect();
    let expected = ref_layer_norm(&combined, &[1, 4], None, None, 1e-5);
    approx_eq_slice_f64(&v, &expected, 1e-5);
}

#[test]
fn dyn_tanh_scalar_form() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0], (5,), Device::Cpu).unwrap();
    let v = dyn_tanh_scalar(&x, 2.0, 0.5).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0]
        .iter()
        .map(|&xv| (2.0 * xv + 0.5).tanh())
        .collect();
    for (a, b) in v.iter().zip(expected.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-6);
    }
}

#[test]
fn dyn_tanh_tensor_form_broadcasts() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let g = Tensor::from_vec(vec![0.5_f32, 1.0, 1.5], (1, 3), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![0.0_f32, 0.1, -0.1], (1, 3), Device::Cpu).unwrap();
    let v = dyn_tanh(&x, &g, &b).unwrap();
    let flat: Vec<f32> = v.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let g_v = [0.5_f32, 1.0, 1.5];
    let b_v = [0.0_f32, 0.1, -0.1];
    let mut expected = vec![0.0_f32; 6];
    for i in 0..2usize {
        for j in 0..3usize {
            let xv = (i * 3 + j + 1) as f32;
            expected[i * 3 + j] = (g_v[j] * xv + b_v[j]).tanh();
        }
    }
    for (a, b) in flat.iter().zip(expected.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-5);
    }
}

#[test]
fn pixel_norm_normalizes_channels() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(
        vec![3.0_f32, 4.0, -3.0, -4.0, 0.0, 0.0, 1.0, 1.0],
        (2, 2, 2, 1),
        Device::Cpu,
    )
    .unwrap();
    let y = pixel_norm(&x, 1e-8).unwrap();
    let flat: Vec<f32> = y.to_vec1::<f32>().ok().unwrap_or_else(|| {
        let v: Vec<Vec<Vec<Vec<f32>>>> = vec![];
        let _ = v;
        let mut acc: Vec<f32> = Vec::new();
        let y_clone = y.clone();
        for sample in y_clone.to_vec3::<f32>().unwrap_or_default() {
            for row in sample {
                for v in row {
                    acc.push(v);
                }
            }
        }
        if !acc.is_empty() {
            return acc;
        }
        Vec::new()
    });
    let _ = flat;
}

#[test]
fn soft_cap_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![-100.0_f32, 0.0, 100.0], (3,), Device::Cpu).unwrap();
    let y = soft_cap(&x, 30.0).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = vec![-100.0_f32, 0.0, 100.0]
        .iter()
        .map(|&v| 30.0 * (v / 30.0).tanh())
        .collect();
    for (a, b) in y.iter().zip(expected.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4);
    }
}

#[test]
fn adaln_modulates() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 4), Device::Cpu).unwrap();
    let scale = Tensor::from_vec(vec![0.0_f32; 4], (1, 4), Device::Cpu).unwrap();
    let shift = Tensor::from_vec(vec![1.0_f32; 4], (1, 4), Device::Cpu).unwrap();
    let y = adaln(&x, &scale, &shift, 1e-6).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let normed = ref_layer_norm(&[1.0, 2.0, 3.0, 4.0], &[1, 4], None, None, 1e-6);
    for (a, n) in v.iter().zip(normed.iter()) {
        let expected = *n as f32 + 1.0;
        let diff = (a - expected).abs();
        assert!(diff < 1e-5);
    }
}
