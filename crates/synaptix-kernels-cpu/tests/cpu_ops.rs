use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() { ensure_registered(); }

#[test]
fn add_f32() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), Device::Cpu).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec2::<f32>().unwrap(), vec![vec![11.0, 22.0], vec![33.0, 44.0]]);
}

#[test]
fn broadcast_add() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![100.0f32, 200.0, 300.0], (1, 3), Device::Cpu).unwrap();
    let c = a.broadcast_add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec2::<f32>().unwrap(), vec![
        vec![101.0, 202.0, 303.0],
        vec![104.0, 205.0, 306.0],
    ]);
}

#[test]
fn matmul_f32() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), Device::Cpu).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec2::<f32>().unwrap(), vec![
        vec![19.0, 22.0],
        vec![43.0, 50.0],
    ]);
}

#[test]
fn matmul_rectangular() {
    setup();
    let a = Tensor::from_vec((1..=6).map(|x| x as f32).collect(), (2, 3), Device::Cpu).unwrap();
    let b = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), Device::Cpu).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
    let v = c.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], vec![38.0, 44.0, 50.0, 56.0]);
    assert_eq!(v[1], vec![83.0, 98.0, 113.0, 128.0]);
}

#[test]
fn unary_sqrt() {
    setup();
    let a = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), Device::Cpu).unwrap();
    let s = a.sqrt().unwrap();
    let v = s.to_vec2::<f32>().unwrap();
    assert_eq!(v, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
}

#[test]
fn unary_silu_smoke() {
    setup();
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (3,), Device::Cpu).unwrap();
    let s = a.silu().unwrap();
    let v = s.to_vec1::<f32>().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-6);
    assert!((v[1] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-6);
}

#[test]
fn reduce_sum_all_f32() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 10.0);
}

#[test]
fn reduce_sum_dim() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let s = a.sum([1usize]).unwrap();
    assert_eq!(s.dims(), &[2]);
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn reduce_mean() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu).unwrap();
    let m = a.mean().unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), 2.5);
}

#[test]
fn cast_f32_to_bf16_and_back() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.5, -3.0], (3,), Device::Cpu).unwrap();
    let b = a.to_dtype(DType::BF16).unwrap();
    assert_eq!(b.dtype(), DType::BF16);
    let c = b.to_dtype(DType::F32).unwrap();
    let v = c.to_vec1::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 1e-2);
    assert!((v[2] - (-3.0)).abs() < 1e-2);
}

#[test]
fn matmul_bf16() {
    setup();
    let a_data: Vec<half::bf16> = vec![1.0f32, 2.0, 3.0, 4.0].into_iter().map(half::bf16::from_f32).collect();
    let b_data: Vec<half::bf16> = vec![5.0f32, 6.0, 7.0, 8.0].into_iter().map(half::bf16::from_f32).collect();
    let a = Tensor::from_vec(a_data, (2, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(b_data, (2, 2), Device::Cpu).unwrap();
    let c = a.matmul(&b).unwrap();
    let v = c.to_vec2::<half::bf16>().unwrap();
    assert!((v[0][0].to_f32() - 19.0).abs() < 1.0);
    assert!((v[1][1].to_f32() - 50.0).abs() < 1.0);
}

#[test]
fn contiguous_materializes_transposed() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let t = a.transpose(0, 1).unwrap();
    let c = t.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.to_vec2::<f32>().unwrap(), vec![
        vec![1.0, 4.0],
        vec![2.0, 5.0],
        vec![3.0, 6.0],
    ]);
}

#[test]
fn argmax_along_dim() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], (2, 3), Device::Cpu).unwrap();
    let am = a.argmax(1).unwrap();
    assert_eq!(am.dtype(), DType::U32);
    assert_eq!(am.to_vec1::<u32>().unwrap(), vec![1, 0]);
}
