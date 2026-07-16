use std::collections::BTreeMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::{
    Dropout, InitMethod, Linear, Module, ModuleDict, ModuleList, Parameter, Sequential,
    init_tensor,
};

fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

#[test]
fn parameter_set_validates() {
    synaptix_kernels_cpu::ensure_registered();
    let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (3,), Device::Cpu).unwrap();
    let p = Parameter::new(t).with_name("w");
    let t2 = Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], (3,), Device::Cpu).unwrap();
    p.set(t2).unwrap();
    assert_eq!(p.tensor().to_vec1::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn linear_forward_with_bias() {
    synaptix_kernels_cpu::ensure_registered();
    let w = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![0.5_f32, -0.5], (2,), Device::Cpu).unwrap();
    let l = Linear::new(w, Some(b)).unwrap();
    let x = Tensor::from_vec(vec![1.0_f32, 1.0, 1.0], (1, 3), Device::Cpu).unwrap();
    let y = l.forward(&x).unwrap();
    let v: Vec<f32> = y.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert!(approx(v[0], 1.0 + 2.0 + 3.0 + 0.5, 1e-5));
    assert!(approx(v[1], 4.0 + 5.0 + 6.0 - 0.5, 1e-5));
}

#[test]
fn linear_from_init_deterministic() {
    synaptix_kernels_cpu::ensure_registered();
    let l1 = Linear::from_init(
        4,
        2,
        false,
        InitMethod::KaimingUniform { fan_in: 4, a: 0.0 },
        InitMethod::Zeros,
        Device::Cpu,
        DType::F32,
        42,
    )
    .unwrap();
    let l2 = Linear::from_init(
        4,
        2,
        false,
        InitMethod::KaimingUniform { fan_in: 4, a: 0.0 },
        InitMethod::Zeros,
        Device::Cpu,
        DType::F32,
        42,
    )
    .unwrap();
    let w1: Vec<f32> = l1.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let w2: Vec<f32> = l2.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(w1, w2);
}

#[test]
fn init_uniform_in_range() {
    synaptix_kernels_cpu::ensure_registered();
    let t = init_tensor(
        &[1000],
        InitMethod::Uniform { low: -1.0, high: 2.0 },
        DType::F32,
        7,
        Device::Cpu,
    )
    .unwrap();
    let v = t.to_vec1::<f32>().unwrap();
    for x in &v {
        assert!(*x >= -1.0 && *x < 2.0);
    }
}

#[test]
fn init_kaiming_normal_std_approx() {
    synaptix_kernels_cpu::ensure_registered();
    let fan_in = 100usize;
    let t = init_tensor(
        &[100_000],
        InitMethod::KaimingNormal { fan_in, a: 0.0 },
        DType::F32,
        11,
        Device::Cpu,
    )
    .unwrap();
    let v = t.to_vec1::<f32>().unwrap();
    let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / (v.len() as f64);
    let var: f64 = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (v.len() as f64);
    let expected_std = (2.0_f64 / fan_in as f64).sqrt();
    assert!((var.sqrt() - expected_std).abs() < 0.01);
}

#[test]
fn init_orthogonal_qq_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let t = init_tensor(
        &[5, 5],
        InitMethod::Orthogonal { gain: 1.0 },
        DType::F32,
        9,
        Device::Cpu,
    )
    .unwrap();
    let q = t.contiguous().unwrap();
    let qt = q.transpose(0, 1).unwrap().contiguous().unwrap();
    let prod = q.matmul(&qt).unwrap();
    let v: Vec<f32> = prod.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for i in 0..5 {
        for j in 0..5 {
            let val = v[i * 5 + j];
            if i == j {
                assert!((val - 1.0).abs() < 1e-3, "diag {i}: {val}");
            } else {
                assert!(val.abs() < 1e-3, "({i},{j}) = {val}");
            }
        }
    }
}

#[test]
fn sequential_chains_modules() {
    synaptix_kernels_cpu::ensure_registered();
    let l1 = Linear::from_init(
        4,
        4,
        false,
        InitMethod::Ones,
        InitMethod::Zeros,
        Device::Cpu,
        DType::F32,
        1,
    )
    .unwrap();
    let l2 = Linear::from_init(
        4,
        2,
        true,
        InitMethod::Ones,
        InitMethod::Zeros,
        Device::Cpu,
        DType::F32,
        2,
    )
    .unwrap();
    let seq = Sequential::new().add(l1).add(l2);
    let x = Tensor::from_vec(vec![1.0_f32, 1.0, 1.0, 1.0], (1, 4), Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2]);
    let params = seq.named_parameters("net");
    let names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
    assert!(names.contains(&"net.0.weight".to_string()));
    assert!(names.contains(&"net.1.weight".to_string()));
    assert!(names.contains(&"net.1.bias".to_string()));
}

#[test]
fn module_list_iter_works() {
    synaptix_kernels_cpu::ensure_registered();
    let mut ml = ModuleList::new();
    for _ in 0..3 {
        ml.push(
            Linear::from_init(
                2,
                2,
                false,
                InitMethod::Zeros,
                InitMethod::Zeros,
                Device::Cpu,
                DType::F32,
                0,
            )
            .unwrap(),
        );
    }
    assert_eq!(ml.len(), 3);
    let mut count = 0;
    for _ in ml.iter() {
        count += 1;
    }
    assert_eq!(count, 3);
}

#[test]
fn module_dict_lookup() {
    let mut dict = ModuleDict::new();
    dict.insert(
        "linear",
        Linear::from_init(
            2,
            2,
            false,
            InitMethod::Zeros,
            InitMethod::Zeros,
            Device::Cpu,
            DType::F32,
            0,
        )
        .unwrap(),
    );
    assert!(dict.contains("linear"));
    assert!(!dict.contains("missing"));
}

#[test]
fn dropout_in_eval_mode_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let d = Dropout::new(0.5);
    d.set_training(false);
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (4,), Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn dropout_deterministic_with_seed() {
    synaptix_kernels_cpu::ensure_registered();
    let d1 = Dropout::with_seed(0.3, 0xABCD);
    let d2 = Dropout::with_seed(0.3, 0xABCD);
    let x = Tensor::from_vec(vec![1.0_f32; 1000], (1000,), Device::Cpu).unwrap();
    let y1 = d1.forward(&x).unwrap().to_vec1::<f32>().unwrap();
    let y2 = d2.forward(&x).unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn dropout_zero_p_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let d = Dropout::new(0.0);
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (3,), Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
fn state_dict_roundtrip() {
    synaptix_kernels_cpu::ensure_registered();
    let l = Linear::from_init(
        2,
        2,
        true,
        InitMethod::KaimingNormal { fan_in: 2, a: 0.0 },
        InitMethod::Zeros,
        Device::Cpu,
        DType::F32,
        42,
    )
    .unwrap();
    let mut dict = BTreeMap::new();
    dict.insert(
        "weight".to_string(),
        Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap(),
    );
    dict.insert(
        "bias".to_string(),
        Tensor::from_vec(vec![0.5_f32, -0.5], (2,), Device::Cpu).unwrap(),
    );
    l.load_state_dict(&dict).unwrap();
    let new_dict = l.state_dict();
    let w = new_dict.get("weight").unwrap();
    let v: Vec<f32> = w.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
}
