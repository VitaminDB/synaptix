use synaptix_autograd::checkpoint::checkpoint;
use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
    autograd_init().unwrap();
}

fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu)
        .unwrap()
        .requires_grad_(true)
}

fn flat(t: &Tensor) -> Vec<f32> {
    let numel: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((numel,)).unwrap().to_vec1::<f32>().unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let diff = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(diff / scale < tol, "idx {i}: {x} vs {y} diff={diff}");
    }
}

fn deep_chain(t: &Tensor) -> synaptix_core::error::Result<Tensor> {
    let a = t.silu()?;
    let b = a.mul_scalar(2.0)?;
    let c = b.add_scalar(0.5)?;
    let d = c.sqr()?;
    let e = d.tanh()?;
    e.silu()
}

#[test]
fn checkpoint_grad_matches_direct() {
    setup();
    let data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];

    let a = leaf(data.clone(), &[6]);
    let y = deep_chain(&a).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g_direct = flat(&a.grad().unwrap());

    let a2 = leaf(data, &[6]);
    let y2 = checkpoint(&a2, |t| deep_chain(t)).unwrap();
    y2.sum_all().unwrap().backward().unwrap();
    let g_chk = flat(&a2.grad().unwrap());

    assert_close(&g_chk, &g_direct, 1e-5);
}

#[test]
fn checkpoint_output_matches_direct() {
    setup();
    let data = vec![0.1f32, 0.2, 0.3, 0.4];

    let a = leaf(data.clone(), &[4]);
    let y = deep_chain(&a).unwrap();
    let y_data = flat(&y);

    let a2 = leaf(data, &[4]);
    let y2 = checkpoint(&a2, |t| deep_chain(t)).unwrap();
    let y2_data = flat(&y2);

    assert_close(&y2_data, &y_data, 1e-5);
}

#[test]
fn checkpoint_works_without_grad_input() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3usize,), Device::Cpu).unwrap();
    let y = checkpoint(&a, |t| t.silu()).unwrap();
    assert!(y.grad_meta().is_none());
    assert_eq!(y.dims(), &[3]);
}
