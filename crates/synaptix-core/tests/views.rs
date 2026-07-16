use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

#[test]
fn reshape_contig() {
    let t = Tensor::from_vec(
        (0..12).map(|x| x as f32).collect(),
        (3, 4),
        Device::Cpu,
    )
    .unwrap();
    let r = t.reshape((2, 6)).unwrap();
    assert_eq!(r.dims(), &[2, 6]);
    assert_eq!(r.to_vec2::<f32>().unwrap(), vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    ]);
}

#[test]
fn transpose_then_back_is_identity() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu).unwrap();
    let tt = t.transpose(0, 1).unwrap().transpose(0, 1).unwrap();
    assert_eq!(tt.dims(), &[2, 3]);
    assert!(tt.is_contiguous());
    assert_eq!(tt.to_vec2::<f32>().unwrap(), t.to_vec2::<f32>().unwrap());
}

#[test]
fn permute_3d() {
    let t = Tensor::from_vec(
        (0..24).map(|x| x as f32).collect(),
        (2, 3, 4),
        Device::Cpu,
    )
    .unwrap();
    let p = t.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
    assert_eq!(p.strides().as_slice(), &[1, 12, 4]);
}

#[test]
fn narrow_keeps_strides() {
    let t = Tensor::from_vec((0..20).map(|x| x as f32).collect(), (4, 5), Device::Cpu).unwrap();
    let n = t.narrow(0, 1, 2).unwrap();
    assert_eq!(n.dims(), &[2, 5]);
    assert_eq!(n.layout().offset(), 5);
}

#[test]
fn unsqueeze_then_squeeze() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu).unwrap();
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 3]);
    let s = u.squeeze(0).unwrap();
    assert_eq!(s.dims(), &[3]);
}

#[test]
fn expand_broadcasts() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu).unwrap();
    let e = t.expand((4, 3)).unwrap();
    assert_eq!(e.dims(), &[4, 3]);
    assert_eq!(e.strides().as_slice(), &[0, 1]);
}

#[test]
fn flatten_all() {
    let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), (2, 3), Device::Cpu).unwrap();
    let f = t.flatten_all().unwrap();
    assert_eq!(f.dims(), &[6]);
    assert_eq!(f.to_vec1::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn mixed_device_collection() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![3.0f32, 4.0], (2,), Device::Cpu).unwrap();
    let v: Vec<Tensor> = vec![a, b];
    assert_eq!(v.len(), 2);
    for t in &v {
        assert_eq!(t.dtype(), synaptix_core::dtype::DType::F32);
    }
}
