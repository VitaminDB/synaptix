use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::ffn::{Activation, geglu, mlp, reglu, swiglu};

fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

fn ref_silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn ref_gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[test]
fn mlp_runs_and_shape_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![0.5_f32, -0.5, 1.0, -1.0], (1, 4), Device::Cpu).unwrap();
    let w1 = Tensor::from_vec(vec![0.1_f32; 8 * 4], (8, 4), Device::Cpu).unwrap();
    let w2 = Tensor::from_vec(vec![0.05_f32; 4 * 8], (4, 8), Device::Cpu).unwrap();
    let out = mlp(&x, &w1, None, &w2, None, Activation::Silu).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn swiglu_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let x_data: Vec<f32> = vec![1.0, -1.0, 0.5, 2.0];
    let w_gate: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, -0.1, 0.0, 0.5, -0.5];
    let w_up: Vec<f32> = vec![0.5, -0.5, 0.25, 0.0, 0.1, 0.2, -0.3, 0.4];
    let w_down: Vec<f32> = vec![1.0, -1.0, 0.5, 0.25, 0.0, 0.5, -0.5, 1.0];
    let x = Tensor::from_vec(x_data.clone(), (1, 4), Device::Cpu).unwrap();
    let wg = Tensor::from_vec(w_gate.clone(), (2, 4), Device::Cpu).unwrap();
    let wu = Tensor::from_vec(w_up.clone(), (2, 4), Device::Cpu).unwrap();
    let wd = Tensor::from_vec(w_down.clone(), (4, 2), Device::Cpu).unwrap();
    let out = swiglu(&x, &wg, &wu, &wd).unwrap();
    let v: Vec<f32> = out.to_vec2::<f32>().unwrap().into_iter().flatten().collect();

    let mut gate = [0.0_f32; 2];
    let mut up = [0.0_f32; 2];
    for i in 0..2 {
        for j in 0..4 {
            gate[i] += w_gate[i * 4 + j] * x_data[j];
            up[i] += w_up[i * 4 + j] * x_data[j];
        }
    }
    let hidden = [ref_silu(gate[0]) * up[0], ref_silu(gate[1]) * up[1]];
    let mut expected = [0.0_f32; 4];
    for i in 0..4 {
        for j in 0..2 {
            expected[i] += w_down[i * 2 + j] * hidden[j];
        }
    }
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!(approx(*a, *b, 1e-5), "{a} vs {b}");
    }
}

#[test]
fn geglu_uses_gelu_activation() {
    synaptix_kernels_cpu::ensure_registered();
    let x_data: Vec<f32> = vec![0.5, 1.0, -0.5, -1.0];
    let w_gate: Vec<f32> = vec![0.2, 0.1, 0.0, -0.1, 0.4, 0.3, -0.2, 0.5];
    let w_up: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1];
    let w_down: Vec<f32> = vec![0.5, 0.5, 1.0, 0.25];
    let x = Tensor::from_vec(x_data.clone(), (1, 4), Device::Cpu).unwrap();
    let wg = Tensor::from_vec(w_gate.clone(), (2, 4), Device::Cpu).unwrap();
    let wu = Tensor::from_vec(w_up.clone(), (2, 4), Device::Cpu).unwrap();
    let wd = Tensor::from_vec(w_down.clone(), (2, 2), Device::Cpu).unwrap();
    let out = geglu(&x, &wg, &wu, &wd).unwrap();
    let v: Vec<f32> = out.to_vec2::<f32>().unwrap().into_iter().flatten().collect();

    let mut gate = [0.0_f32; 2];
    let mut up = [0.0_f32; 2];
    for i in 0..2 {
        for j in 0..4 {
            gate[i] += w_gate[i * 4 + j] * x_data[j];
            up[i] += w_up[i * 4 + j] * x_data[j];
        }
    }
    let hidden = [ref_gelu_tanh(gate[0]) * up[0], ref_gelu_tanh(gate[1]) * up[1]];
    let mut expected = [0.0_f32; 2];
    for i in 0..2 {
        for j in 0..2 {
            expected[i] += w_down[i * 2 + j] * hidden[j];
        }
    }
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!(approx(*a, *b, 1e-4), "{a} vs {b}");
    }
}

#[test]
fn reglu_returns_correct_shape() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32, -1.0, 0.5, -0.5], (1, 4), Device::Cpu).unwrap();
    let wg = Tensor::from_vec(vec![0.1_f32; 2 * 4], (2, 4), Device::Cpu).unwrap();
    let wu = Tensor::from_vec(vec![0.2_f32; 2 * 4], (2, 4), Device::Cpu).unwrap();
    let wd = Tensor::from_vec(vec![0.3_f32; 4 * 2], (4, 2), Device::Cpu).unwrap();
    let out = reglu(&x, &wg, &wu, &wd).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}
