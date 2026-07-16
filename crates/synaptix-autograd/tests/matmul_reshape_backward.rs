use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
    autograd_init().unwrap();
}

fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu).unwrap().requires_grad_(true)
}

fn const_t(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu).unwrap()
}

fn flat(t: &Tensor) -> Vec<f32> {
    let numel: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((numel,)).unwrap().to_vec1::<f32>().unwrap()
}

fn scalar(t: &Tensor) -> f32 {
    t.reshape((1,)).unwrap().to_vec1::<f32>().unwrap()[0]
}

fn numerical_grad<F>(input: &[f32], shape: &[usize], f: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> Tensor,
{
    let eps = 1e-3f32;
    let mut g = vec![0.0f32; input.len()];
    for i in 0..input.len() {
        let mut p = input.to_vec();
        p[i] += eps;
        let yp = scalar(&f(&Tensor::from_vec(p, shape.to_vec(), Device::Cpu).unwrap()).sum_all().unwrap());
        let mut m = input.to_vec();
        m[i] -= eps;
        let ym = scalar(&f(&Tensor::from_vec(m, shape.to_vec(), Device::Cpu).unwrap()).sum_all().unwrap());
        g[i] = (yp - ym) / (2.0 * eps);
    }
    g
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let scale = a.abs().max(e.abs()).max(1.0);
        assert!(diff / scale < tol, "idx {i}: actual={a} expected={e}");
    }
}

#[test]
fn matmul_backward_2d() {
    setup();
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
    let a = leaf(a_data.clone(), &[3, 4]);
    let b = leaf(b_data.clone(), &[4, 2]);
    let c = a.matmul(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    let num_ga = numerical_grad(&a_data, &[3, 4], |t| t.matmul(&const_t(b_data.clone(), &[4, 2])).unwrap());
    let num_gb = numerical_grad(&b_data, &[4, 2], |t| const_t(a_data.clone(), &[3, 4]).matmul(t).unwrap());
    assert_close(&ga, &num_ga, 1e-2);
    assert_close(&gb, &num_gb, 1e-2);
}

#[test]
fn reshape_backward_returns_to_input_shape() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = a.reshape((3, 2)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    let s = r.sum_all().unwrap();
    s.backward().unwrap();
    let ga = a.grad().unwrap();
    assert_eq!(ga.dims(), &[2, 3]);
    assert_eq!(flat(&ga), vec![1.0; 6]);
}

#[test]
fn transpose_backward_reverses_axes() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t = a.transpose(0, 1).unwrap().contiguous().unwrap();
    assert_eq!(t.dims(), &[3, 2]);
    let s = t.mul(&const_t(vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0], &[3, 2])).unwrap().sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![1.0, 100.0, 10000.0, 10.0, 1000.0, 100000.0]);
}

#[test]
fn permute_inverse_backward() {
    setup();
    let a = leaf((1..=24).map(|x| x as f32).collect(), &[2, 3, 4]);
    let p = a.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
    let s = p.sum_all().unwrap();
    s.backward().unwrap();
    let ga = a.grad().unwrap();
    assert_eq!(ga.dims(), &[2, 3, 4]);
}

#[test]
fn squeeze_unsqueeze_backward() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let u = a.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 3]);
    let q = u.squeeze(0).unwrap();
    assert_eq!(q.dims(), &[3]);
    let s = q.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0]);
}

#[test]
fn cast_backward_returns_to_source_dtype() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = a.to_dtype(DType::F32).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = a.grad().unwrap();
    assert_eq!(ga.dtype(), DType::F32);
    assert_eq!(flat(&ga), vec![1.0; 4]);
}

#[test]
fn matmul_chain_with_reshape() {
    setup();
    let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.5).collect();
    let a = leaf(a_data.clone(), &[6]);
    let b = leaf(b_data.clone(), &[3, 4]);
    let a_mat = a.reshape((2, 3)).unwrap();
    let c = a_mat.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = a.grad().unwrap();
    assert_eq!(ga.dims(), &[6]);
}
