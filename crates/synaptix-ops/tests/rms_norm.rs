use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::{rms_norm, rms_norm_gated, rms_norm_qwen, rms_norm_silu_gated};

fn ref_rms_norm_f32(x: &[f32], w: &[f32], shape: &[usize], eps: f32, gain_plus_one: bool) -> Vec<f32> {
    let last = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let mut out = vec![0.0f32; x.len()];
    for o in 0..outer {
        let row = &x[o * last..(o + 1) * last];
        let var: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (last as f64);
        let inv = 1.0f64 / (var + eps as f64).sqrt();
        for i in 0..last {
            let g = if gain_plus_one { w[i] + 1.0 } else { w[i] };
            out[o * last + i] = ((row[i] as f64) * inv * (g as f64)) as f32;
        }
    }
    out
}

fn ref_rms_norm_silu_gated_f32(
    x: &[f32],
    gate: &[f32],
    w: &[f32],
    shape: &[usize],
    eps: f32,
) -> Vec<f32> {
    let last = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let mut out = vec![0.0f32; x.len()];
    for o in 0..outer {
        let gated: Vec<f64> = (0..last).map(|i| {
            let xi = x[o * last + i] as f64;
            let g = gate[o * last + i] as f64;
            let silu_g = g / (1.0 + (-g).exp());
            xi * silu_g
        }).collect();
        let var: f64 = gated.iter().map(|&v| v * v).sum::<f64>() / (last as f64);
        let inv = 1.0f64 / (var + eps as f64).sqrt();
        for i in 0..last {
            out[o * last + i] = (gated[i] * inv * w[i] as f64) as f32;
        }
    }
    out
}

fn ref_rms_norm_gated_no_silu_f32(
    x: &[f32],
    gate: &[f32],
    w: &[f32],
    shape: &[usize],
    eps: f32,
) -> Vec<f32> {
    let last = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let mut out = vec![0.0f32; x.len()];
    for o in 0..outer {
        let gated: Vec<f64> = (0..last).map(|i| {
            let xi = x[o * last + i] as f64;
            let g = gate[o * last + i] as f64;
            xi * g
        }).collect();
        let var: f64 = gated.iter().map(|&v| v * v).sum::<f64>() / (last as f64);
        let inv = 1.0f64 / (var + eps as f64).sqrt();
        for i in 0..last {
            out[o * last + i] = (gated[i] * inv * w[i] as f64) as f32;
        }
    }
    out
}

#[test]
fn rms_norm_f32_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [2usize, 3usize, 4usize];
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let w: Vec<f32> = vec![0.5, -0.25, 1.0, 2.0];
    let eps = 1e-6f32;

    let xt = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w.clone(), (4usize,), Device::Cpu).unwrap();

    let y = rms_norm(&xt, &wt, eps).unwrap();
    assert_eq!(y.dims(), &shape);
    let y_3d: Vec<Vec<Vec<f32>>> = y.to_vec3().unwrap();
    let y_flat: Vec<f32> = y_3d.into_iter().flatten().flatten().collect();

    let r = ref_rms_norm_f32(&x, &w, &shape, eps, false);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
    }
}

#[test]
fn rms_norm_qwen_f32_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [2usize, 5usize];
    let x: Vec<f32> = (0..10).map(|i| (i as f32) * 0.2 - 0.7).collect();
    let w: Vec<f32> = vec![0.0, 0.1, -0.2, 0.3, -0.4];
    let eps = 1e-5f32;

    let xt = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w.clone(), (5usize,), Device::Cpu).unwrap();

    let y = rms_norm_qwen(&xt, &wt, eps).unwrap();
    let y_flat: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let r = ref_rms_norm_f32(&x, &w, &shape, eps, true);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
    }
}

#[test]
fn rms_norm_silu_gated_f32_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [3usize, 4usize];
    let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.15 - 0.5).collect();
    let gate: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = vec![0.5, 0.25, 1.0, 0.75];
    let eps = 1e-6f32;

    let xt = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let gt = Tensor::from_vec(gate.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w.clone(), (4usize,), Device::Cpu).unwrap();

    let y = rms_norm_silu_gated(&xt, &gt, &wt, eps).unwrap();
    let y_flat: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let r = ref_rms_norm_silu_gated_f32(&x, &gate, &w, &shape, eps);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
    }
}

#[test]
fn rms_norm_gated_no_silu_f32_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [3usize, 4usize];
    let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.15 - 0.5).collect();
    let gate: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = vec![0.5, 0.25, 1.0, 0.75];
    let eps = 1e-6f32;

    let xt = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let gt = Tensor::from_vec(gate.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w.clone(), (4usize,), Device::Cpu).unwrap();

    let y = rms_norm_gated(&xt, &gt, &wt, eps).unwrap();
    let y_flat: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let r = ref_rms_norm_gated_no_silu_f32(&x, &gate, &w, &shape, eps);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
    }
}

#[test]
fn rms_norm_bf16_roundtrip() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [2usize, 6usize];
    let xf: Vec<f32> = (0..12).map(|i| (i as f32) * 0.13 - 0.5).collect();
    let wf: Vec<f32> = vec![0.5, -0.25, 1.0, 2.0, 0.0, -1.5];
    let eps = 1e-6f32;

    let x_bf16: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_bf16: Vec<bf16> = wf.iter().map(|&v| bf16::from_f32(v)).collect();
    let xt = Tensor::from_vec(x_bf16.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w_bf16.clone(), (6usize,), Device::Cpu).unwrap();

    let y = rms_norm(&xt, &wt, eps).unwrap();
    assert_eq!(y.dtype(), DType::BF16);
    let y_flat: Vec<bf16> = y.to_vec2::<bf16>().unwrap().into_iter().flatten().collect();

    let xf2: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();
    let wf2: Vec<f32> = w_bf16.iter().map(|v| v.to_f32()).collect();
    let r = ref_rms_norm_f32(&xf2, &wf2, &shape, eps, false);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        let diff = (a.to_f32() - b).abs();
        assert!(diff < 0.05, "bf16 diff too large: {a} vs {b} (diff {diff})");
    }
}

#[test]
fn rms_norm_f16_roundtrip() {
    synaptix_kernels_cpu::ensure_registered();
    let shape = [4usize, 8usize];
    let xf: Vec<f32> = (0..32).map(|i| (i as f32) * 0.07 - 0.9).collect();
    let wf: Vec<f32> = (0..8).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let eps = 1e-5f32;

    let x_f16: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
    let w_f16: Vec<f16> = wf.iter().map(|&v| f16::from_f32(v)).collect();
    let xt = Tensor::from_vec(x_f16.clone(), shape, Device::Cpu).unwrap();
    let wt = Tensor::from_vec(w_f16.clone(), (8usize,), Device::Cpu).unwrap();

    let y = rms_norm(&xt, &wt, eps).unwrap();
    assert_eq!(y.dtype(), DType::F16);
    let y_flat: Vec<f16> = y.to_vec2::<f16>().unwrap().into_iter().flatten().collect();

    let xf2: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let wf2: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();
    let r = ref_rms_norm_f32(&xf2, &wf2, &shape, eps, false);
    for (a, b) in y_flat.iter().zip(r.iter()) {
        let diff = (a.to_f32() - b).abs();
        assert!(diff < 0.005, "f16 diff too large: {a} vs {b} (diff {diff})");
    }
}
