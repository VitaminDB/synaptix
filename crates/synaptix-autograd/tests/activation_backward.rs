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
        let yp = scalar(
            &f(&Tensor::from_vec(p, shape.to_vec(), Device::Cpu).unwrap()).sum_all().unwrap(),
        );
        let mut m = input.to_vec();
        m[i] -= eps;
        let ym = scalar(
            &f(&Tensor::from_vec(m, shape.to_vec(), Device::Cpu).unwrap()).sum_all().unwrap(),
        );
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
fn sigmoid_backward() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.5, 2.0];
    let a = leaf(data.clone(), &[4]);
    let y = a.sigmoid().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.sigmoid().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn tanh_backward() {
    setup();
    let data = vec![-1.5f32, -0.3, 0.7, 1.2];
    let a = leaf(data.clone(), &[4]);
    let y = a.tanh().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.tanh().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn silu_backward() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.5, 2.0];
    let a = leaf(data.clone(), &[4]);
    let y = a.silu().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.silu().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn gelu_tanh_backward() {
    setup();
    let data = vec![-1.5f32, -0.3, 0.7, 1.2];
    let a = leaf(data.clone(), &[4]);
    let y = a.gelu_tanh().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.gelu_tanh().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn gelu_exact_backward() {
    setup();
    let data = vec![-1.5f32, -0.3, 0.7, 1.2];
    let a = leaf(data.clone(), &[4]);
    let y = a.gelu_exact().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.gelu_exact().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn erf_backward() {
    setup();
    let data = vec![-1.0f32, -0.3, 0.3, 1.0];
    let a = leaf(data.clone(), &[4]);
    let y = a.erf().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.erf().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn exp_log_recip_sqrt_square_backward() {
    setup();
    let data = vec![0.5f32, 1.0, 1.5, 2.0];
    let a = leaf(data.clone(), &[4]);
    let y = a.exp().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.exp().unwrap());
    assert_close(&g, &num, 1e-3);

    a.zero_grad();
    let y = a.log().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.log().unwrap());
    assert_close(&g, &num, 1e-3);

    a.zero_grad();
    let y = a.recip().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.recip().unwrap());
    assert_close(&g, &num, 1e-2);

    a.zero_grad();
    let y = a.sqrt().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.sqrt().unwrap());
    assert_close(&g, &num, 1e-3);

    a.zero_grad();
    let y = a.sqr().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    let num = numerical_grad(&data, &[4], |t| t.sqr().unwrap());
    assert_close(&g, &num, 1e-3);
}

#[test]
fn relu_backward_masks_negative() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let a = leaf(data.clone(), &[5]);
    let y = a.relu().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn relu2_backward_two_x_on_positive() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let a = leaf(data.clone(), &[5]);
    let y = a.relu2().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![0.0, 0.0, 0.0, 1.0, 4.0]);
}

#[test]
fn leaky_relu_backward_alpha_on_negative() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.5, 2.0];
    let alpha = 0.1f32;
    let a = leaf(data.clone(), &[4]);
    let y = a.leaky_relu(alpha).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![alpha, alpha, 1.0, 1.0]);
}

#[test]
fn abs_backward_uses_sign() {
    setup();
    let data = vec![-2.0f32, -0.5, 0.5, 2.0];
    let a = leaf(data.clone(), &[4]);
    let y = a.abs().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![-1.0, -1.0, 1.0, 1.0]);
}

#[test]
fn sign_step_have_zero_grad() {
    setup();
    let data = vec![-1.0f32, 0.0, 1.0];
    let a = leaf(data.clone(), &[3]);
    let y = a.sign().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![0.0, 0.0, 0.0]);

    let b = leaf(data, &[3]);
    let y = b.step_gt_zero().unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = flat(&b.grad().unwrap());
    assert_eq!(g, vec![0.0, 0.0, 0.0]);
}

#[test]
fn rms_norm_via_composition() {
    setup();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = leaf(data.clone(), &[2, 3]);
    let sq = a.sqr().unwrap();
    let var = sq.mean_keepdim(1).unwrap();
    let rms = var.add_scalar(1e-6).unwrap().sqrt().unwrap();
    let norm = a.broadcast_div(&rms).unwrap();
    norm.sum_all().unwrap().backward().unwrap();
    let g = a.grad().unwrap();
    assert_eq!(g.dims(), &[2, 3]);
    let num = numerical_grad(&data, &[2, 3], |t| {
        let sq = t.sqr().unwrap();
        let var = sq.mean_keepdim(1).unwrap();
        let rms = var.add_scalar(1e-6).unwrap().sqrt().unwrap();
        t.broadcast_div(&rms).unwrap()
    });
    assert_close(&flat(&g), &num, 1e-2);
}
