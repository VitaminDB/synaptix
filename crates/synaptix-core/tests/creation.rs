use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

#[test]
fn zeros_roundtrip_f32() {
    let t = Tensor::zeros((2, 3), DType::F32, Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![0.0; 3]; 2]);
}

#[test]
fn ones_f32() {
    let t = Tensor::ones((2, 2), DType::F32, Device::Cpu).unwrap();
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1.0; 2]; 2]);
}

#[test]
fn ones_bf16() {
    let t = Tensor::ones((4,), DType::BF16, Device::Cpu).unwrap();
    let v: Vec<half::bf16> = t.to_vec1().unwrap();
    assert!(v.iter().all(|x| *x == half::bf16::ONE));
}

#[test]
fn from_vec_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data.clone(), (2, 3), Device::Cpu).unwrap();
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
}

#[test]
fn arange_basic() {
    let t = Tensor::arange::<u32>(0, 5, Device::Cpu).unwrap();
    let v: Vec<u32> = t.to_vec1().unwrap();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
}

#[test]
fn to_scalar_works() {
    let t = Tensor::from_vec(vec![42.0f32], (1,), Device::Cpu).unwrap();
    let s: f32 = t.to_scalar().unwrap();
    assert_eq!(s, 42.0);
}

#[test]
fn cat_along_dim_0() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), Device::Cpu).unwrap();
    let c = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[4, 2]);
    let v: Vec<Vec<f32>> = c.to_vec2().unwrap();
    assert_eq!(v, vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]);
}

#[test]
fn cat_along_dim_1() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), Device::Cpu).unwrap();
    let c = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
    let v: Vec<Vec<f32>> = c.to_vec2().unwrap();
    assert_eq!(v, vec![
        vec![1.0, 2.0, 5.0, 6.0],
        vec![3.0, 4.0, 7.0, 8.0],
    ]);
}

#[test]
fn dtype_mismatch_to_vec() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu).unwrap();
    assert!(t.to_vec1::<u32>().is_err());
}

#[test]
fn debug_display() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let s = format!("{}", t);
    assert!(s.contains("F32"));
    assert!(s.contains("Cpu"));
    let s = format!("{:?}", t);
    assert!(s.contains("Tensor"));
}

#[cfg(feature = "cuda")]
#[test]
fn cat_cuda_matches_cpu_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    // Несколько форм, включая cat по последней оси с большим outer (горячий
    // путь partial-rope, где старый цикл выдавал outer микро-memcpy).
    let cases: &[(Vec<usize>, Vec<usize>, usize)] = &[
        (vec![2, 3, 4], vec![2, 3, 4], 0),
        (vec![2, 3, 4], vec![2, 5, 4], 1),
        (vec![3, 7, 5], vec![3, 7, 9], 2),
        (vec![1, 24, 40, 16], vec![1, 24, 40, 48], 3),
    ];
    for (da, db, dim) in cases {
        let na: usize = da.iter().product();
        let nb: usize = db.iter().product();
        let va: Vec<f32> = (0..na).map(|i| i as f32 * 0.5 - 3.0).collect();
        let vb: Vec<f32> = (0..nb).map(|i| -(i as f32) * 0.25 + 1.0).collect();
        let a_cpu = Tensor::from_vec(va.clone(), da.clone(), Device::Cpu).unwrap();
        let b_cpu = Tensor::from_vec(vb.clone(), db.clone(), Device::Cpu).unwrap();
        let ref_cpu: Vec<f32> = Tensor::cat(&[&a_cpu, &b_cpu], *dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let a_cu = a_cpu.to_device(dev).unwrap();
        let b_cu = b_cpu.to_device(dev).unwrap();
        let out_cu = Tensor::cat(&[&a_cu, &b_cu], *dim).unwrap();
        let got: Vec<f32> = out_cu.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), ref_cpu.len(), "len mismatch dim={dim}");
        for (i, (g, r)) in got.iter().zip(ref_cpu.iter()).enumerate() {
            assert!((g - r).abs() < 1e-6, "cat cuda dim={dim} idx={i}: {g} vs {r}");
        }
    }
}
