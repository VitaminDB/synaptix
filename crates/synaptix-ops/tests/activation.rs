use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::activation::{
    gelu_exact, gelu_tanh, glu, leaky_relu, mish, prelu, quick_gelu, relu, relu_squared, silu,
    softplus, softsign, swish_beta, tanh,
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

fn assert_vec_eq(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(*a, *b, tol), "idx {i}: actual {a} vs expected {b} (tol {tol})");
    }
}

fn ref_gelu_exact(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_f64(x / std::f64::consts::SQRT_2))
}

fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

#[test]
fn gelu_tanh_matches_tensor_method() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let y = gelu_tanh(&xt).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| {
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * xv * (1.0 + (c * (xv + 0.044715 * xv * xv * xv)).tanh())
        })
        .collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn gelu_exact_matches_erf_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let y = gelu_exact(&xt).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x.iter().map(|&xv| ref_gelu_exact(xv as f64) as f32).collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn quick_gelu_matches_sigmoid_form() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, 0.0, 2.0];
    let xt = Tensor::from_vec(x.clone(), (3,), Device::Cpu).unwrap();
    let y = quick_gelu(&xt).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| {
            let s = 1.702 * xv;
            xv * (1.0 / (1.0 + (-s).exp()))
        })
        .collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn silu_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 2.5];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = silu(&xt).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x.iter().map(|&xv| xv / (1.0 + (-xv).exp())).collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn swish_beta_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let xt = Tensor::from_vec(x.clone(), (3,), Device::Cpu).unwrap();
    let beta = 2.0_f32;
    let v = swish_beta(&xt, beta).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| xv / (1.0 + (-beta * xv).exp()))
        .collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn relu_and_relu_squared() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-3.0, -0.5, 0.0, 0.5, 3.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = relu(&xt).unwrap().to_vec1::<f32>().unwrap();
    assert_vec_eq(&v, &[0.0, 0.0, 0.0, 0.5, 3.0], 1e-5);
    let v2 = relu_squared(&xt).unwrap().to_vec1::<f32>().unwrap();
    assert_vec_eq(&v2, &[0.0, 0.0, 0.0, 0.25, 9.0], 1e-5);
}

#[test]
fn leaky_relu_negative_slope() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 2.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = leaky_relu(&xt, 0.1).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> =
        x.iter().map(|&xv| if xv >= 0.0 { xv } else { 0.1 * xv }).collect();
    assert_vec_eq(&v, &expected, 1e-5);
}

#[test]
fn prelu_per_channel_weights() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 2.0, -3.0];
    let xt = Tensor::from_vec(x.clone(), (1, 2, 3), Device::Cpu).unwrap();
    let wt = Tensor::from_vec(vec![0.1, 0.25], (2,), Device::Cpu).unwrap();
    let v = prelu(&xt, &wt).unwrap().to_vec3::<f32>().unwrap();
    let flat: Vec<f32> = v.into_iter().flatten().flatten().collect();
    let w = [0.1, 0.25];
    let expected: Vec<f32> = (0..6)
        .map(|i| {
            let ch = i / 3;
            let xv = x[i];
            if xv >= 0.0 { xv } else { w[ch] * xv }
        })
        .collect();
    assert_vec_eq(&flat, &expected, 1e-5);
}

#[test]
fn mish_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = mish(&xt).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| {
            let sp = (1.0_f32 + xv.exp()).ln();
            xv * sp.tanh()
        })
        .collect();
    assert_vec_eq(&v, &expected, 1e-4);
}

#[test]
fn tanh_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = tanh(&xt).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x.iter().map(|&xv| xv.tanh()).collect();
    assert_vec_eq(&v, &expected, 1e-6);
}

#[test]
fn softplus_correct_beta1() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-30.0, -1.0, 0.0, 1.0, 30.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = softplus(&xt, 1.0, 20.0).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| xv.max(0.0) + (1.0 + (-xv.abs()).exp()).ln())
        .collect();
    for (a, b) in v.iter().zip(expected.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "softplus diff {diff}: {a} vs {b}");
    }
}

#[test]
fn softsign_correct() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    let xt = Tensor::from_vec(x.clone(), (5,), Device::Cpu).unwrap();
    let v = softsign(&xt).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = x.iter().map(|&xv| xv / (1.0 + xv.abs())).collect();
    assert_vec_eq(&v, &expected, 1e-6);
}

#[test]
fn glu_splits_and_gates() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let xt = Tensor::from_vec(x.clone(), (1, 8), Device::Cpu).unwrap();
    let v = glu(&xt, 1).unwrap().to_vec2::<f32>().unwrap();
    let flat: Vec<f32> = v.into_iter().flatten().collect();
    let expected: Vec<f32> = (0..4)
        .map(|i| {
            let a = x[i];
            let b = x[i + 4];
            a * (1.0 / (1.0 + (-b).exp()))
        })
        .collect();
    assert_vec_eq(&flat, &expected, 1e-5);
}

#[test]
fn silu_bf16_matches_f32_tolerance() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<bf16> = vec![bf16::from_f32(-1.0), bf16::from_f32(0.5), bf16::from_f32(2.0)];
    let xt = Tensor::from_vec(x.clone(), (3,), Device::Cpu).unwrap();
    let v = silu(&xt).unwrap();
    assert_eq!(v.dtype(), DType::BF16);
    let vf = v.to_vec1::<bf16>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| {
            let xf = xv.to_f32();
            xf / (1.0 + (-xf).exp())
        })
        .collect();
    for (a, b) in vf.iter().zip(expected.iter()) {
        let diff = (a.to_f32() - b).abs();
        assert!(diff < 0.05, "bf16 silu diff {diff}");
    }
}

#[test]
fn mish_f16_matches_f32() {
    synaptix_kernels_cpu::ensure_registered();
    let x: Vec<f16> = vec![f16::from_f32(-1.0), f16::from_f32(0.5), f16::from_f32(2.0)];
    let xt = Tensor::from_vec(x.clone(), (3,), Device::Cpu).unwrap();
    let v = mish(&xt).unwrap();
    assert_eq!(v.dtype(), DType::F16);
    let vf = v.to_vec1::<f16>().unwrap();
    let expected: Vec<f32> = x
        .iter()
        .map(|&xv| {
            let xf = xv.to_f32();
            let sp = (1.0 + xf.exp()).ln();
            xf * sp.tanh()
        })
        .collect();
    for (a, b) in vf.iter().zip(expected.iter()) {
        let diff = (a.to_f32() - b).abs();
        assert!(diff < 0.005, "f16 mish diff {diff}");
    }
}
