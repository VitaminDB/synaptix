use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[test]
fn hook_observes_grad_without_modifying() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = calls.clone();
    a.register_hook(move |_g| {
        calls_clone.fetch_add(1, Ordering::Relaxed);
        None
    });
    let c = a.mul_scalar(2.0).unwrap();
    c.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![2.0, 2.0, 2.0]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn hook_can_modify_grad() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    a.register_hook(|g| {
        let scaled = g.affine(10.0, 0.0).unwrap();
        Some(scaled)
    });
    let c = a.mul_scalar(2.0).unwrap();
    c.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![20.0, 20.0, 20.0]);
}

#[test]
fn hook_on_intermediate_intercepts() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let b = a.mul_scalar(2.0).unwrap();
    b.register_hook(|g| Some(g.affine(0.0, 0.0).unwrap()));
    let c = b.mul_scalar(3.0).unwrap();
    c.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![0.0, 0.0, 0.0]);
}
