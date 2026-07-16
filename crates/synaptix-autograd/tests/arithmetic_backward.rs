use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
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
    let dims = t.dims().to_vec();
    let numel: usize = dims.iter().product();
    let shape = t.shape().clone();
    let _ = shape;
    let flat = t.reshape((numel,)).unwrap();
    flat.to_vec1::<f32>().unwrap()
}

fn numerical_grad<F>(input_initial: &[f32], shape: &[usize], f: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> Tensor,
{
    let eps = 1e-3f32;
    let mut grads = vec![0.0f32; input_initial.len()];
    for i in 0..input_initial.len() {
        let mut data_plus = input_initial.to_vec();
        data_plus[i] += eps;
        let t_plus = Tensor::from_vec(data_plus, shape.to_vec(), Device::Cpu).unwrap();
        let y_plus = f(&t_plus).sum_all().unwrap();
        let y_plus_val = y_plus.reshape((1,)).unwrap().to_vec1::<f32>().unwrap()[0];

        let mut data_minus = input_initial.to_vec();
        data_minus[i] -= eps;
        let t_minus = Tensor::from_vec(data_minus, shape.to_vec(), Device::Cpu).unwrap();
        let y_minus = f(&t_minus).sum_all().unwrap();
        let y_minus_val = y_minus.reshape((1,)).unwrap().to_vec1::<f32>().unwrap()[0];

        grads[i] = (y_plus_val - y_minus_val) / (2.0 * eps);
    }
    grads
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "lens differ");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let scale = a.abs().max(e.abs()).max(1.0);
        assert!(
            diff / scale < tol,
            "idx {i}: actual={a} expected={e} diff={diff}",
        );
    }
}

#[test]
fn add_backward_simple() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let c = a.add(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(flat(&ga), vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(flat(&gb), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn mul_backward_numerical() {
    setup();
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5f32, 1.5, 2.5, 0.1, 0.2, 0.3];
    let a = leaf(a_data.clone(), &[2, 3]);
    let b = leaf(b_data.clone(), &[2, 3]);
    let c = a.mul(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());

    let num_ga = numerical_grad(&a_data, &[2, 3], |t| t.mul(&const_t(b_data.clone(), &[2, 3])).unwrap());
    let num_gb = numerical_grad(&b_data, &[2, 3], |t| const_t(a_data.clone(), &[2, 3]).mul(t).unwrap());
    assert_close(&ga, &num_ga, 1e-3);
    assert_close(&gb, &num_gb, 1e-3);
}

#[test]
fn div_backward_numerical() {
    setup();
    let a_data = vec![2.0f32, 4.0, 6.0, 8.0];
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = leaf(a_data.clone(), &[2, 2]);
    let b = leaf(b_data.clone(), &[2, 2]);
    let c = a.div(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    let num_ga = numerical_grad(&a_data, &[2, 2], |t| t.div(&const_t(b_data.clone(), &[2, 2])).unwrap());
    let num_gb = numerical_grad(&b_data, &[2, 2], |t| const_t(a_data.clone(), &[2, 2]).div(t).unwrap());
    assert_close(&ga, &num_ga, 1e-2);
    assert_close(&gb, &num_gb, 1e-2);
}

#[test]
fn sub_backward_signs_correct() {
    setup();
    let a = leaf(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let b = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = a.sub(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![-1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn neg_backward() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let n = a.neg().unwrap();
    let s = n.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![-1.0, -1.0, -1.0]);
}

#[test]
fn add_scalar_backward_passes_grad_unchanged() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let c = a.add_scalar(7.0).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0]);
}

#[test]
fn mul_scalar_backward_scales_grad() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let c = a.mul_scalar(3.0).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![3.0, 3.0, 3.0]);
}

#[test]
fn affine_backward_uses_mul_only() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let c = a.affine(2.5, 9.0).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    assert_eq!(ga, vec![2.5, 2.5, 2.5]);
}

#[test]
fn add_broadcast_backward_sums_correctly() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = leaf(vec![10.0, 20.0, 30.0], &[1, 3]);
    let c = a.broadcast_add(&b).unwrap();
    let s = c.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![2.0, 2.0, 2.0]);
}

#[test]
fn mixed_chain_backward() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]);
    let c = a.mul(&b).unwrap();
    let d = c.add(&a).unwrap();
    let s = d.sum_all().unwrap();
    s.backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.5, 2.0, 2.5, 3.0]);
    assert_eq!(gb, vec![1.0, 2.0, 3.0, 4.0]);
}
