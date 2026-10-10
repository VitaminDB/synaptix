use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
    autograd_init().unwrap();
}

fn flat(t: &Tensor) -> Vec<f32> {
    let numel: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((numel,)).unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn narrow_backward_places_grad_in_slice() {
    setup();
    let x = Tensor::from_vec((0..12).map(|v| v as f32).collect::<Vec<_>>(), vec![3usize, 4], Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3usize, 2], Device::Cpu).unwrap();
    let y = x.narrow(1, 1, 2).unwrap().mul(&w).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    assert_eq!(flat(&x.grad().unwrap()), vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0]);
}

#[test]
fn expand_backward_sums_broadcast_axes() {
    setup();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3usize, 1], Device::Cpu).unwrap().requires_grad_(true);
    let w = Tensor::from_vec((0..12).map(|v| v as f32).collect::<Vec<_>>(), vec![2usize, 3, 2], Device::Cpu).unwrap();
    let y = x.expand(vec![2usize, 3, 2]).unwrap().contiguous().unwrap().mul(&w).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    // d/dx[i] = Σ_{b,j} w[b,i,j]
    assert_eq!(flat(&x.grad().unwrap()), vec![0.0 + 1.0 + 6.0 + 7.0, 2.0 + 3.0 + 8.0 + 9.0, 4.0 + 5.0 + 10.0 + 11.0]);
}
