use synaptix_autograd::functional::{activation, distance, linear, loss};
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

fn relinearize(t: Tensor) -> Tensor {
    t.detach().requires_grad_(true)
}

#[test]
fn mse_loss_decreases_on_linear_regression() {
    setup();
    let x = const_t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let y_true = const_t(vec![3.0, 5.0, 7.0], &[3, 1]);

    let mut w = const_t(vec![0.0, 0.0], &[2, 1]).requires_grad_(true);
    let mut b = const_t(vec![0.0], &[1, 1]).requires_grad_(true);

    let lr = 0.02;
    let mut loss_last = 0.0;
    let mut loss_first = 0.0;
    for step in 0..300 {
        let pred = linear::linear(&x, &w, Some(&b)).unwrap();
        let l = loss::mse_loss(&pred, &y_true).unwrap();
        if step == 0 {
            loss_first = scalar(&l);
        }
        loss_last = scalar(&l);
        l.backward().unwrap();
        let new_w = w.sub(&w.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        let new_b = b.sub(&b.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        w = relinearize(new_w);
        b = relinearize(new_b);
    }
    assert!(loss_last < loss_first * 0.01, "loss did not decrease enough");
}

#[test]
fn bce_with_logits_returns_positive_scalar() {
    setup();
    let logits = leaf(vec![-1.5, 0.0, 1.5, 2.5], &[4]);
    let target = const_t(vec![0.0, 0.5, 1.0, 1.0], &[4]);
    let l = loss::bce_with_logits(&logits, &target).unwrap();
    let lv = scalar(&l);
    assert!(lv > 0.0, "bce should be positive, got {lv}");
    l.backward().unwrap();
    let g = logits.grad().unwrap();
    assert_eq!(g.dims(), &[4]);
}

#[test]
fn cross_entropy_one_hot_softmax_converges() {
    setup();
    let n_classes = 4usize;
    let n_samples = 8usize;
    let mut x_data: Vec<f32> = Vec::with_capacity(n_samples * n_classes);
    let mut y_data: Vec<f32> = vec![0.0f32; n_samples * n_classes];
    for i in 0..n_samples {
        let cls = i % n_classes;
        y_data[i * n_classes + cls] = 1.0;
        for c in 0..n_classes {
            x_data.push(if c == cls { 1.0 } else { 0.0 });
        }
    }
    let x = const_t(x_data, &[n_samples, n_classes]);
    let y = const_t(y_data, &[n_samples, n_classes]);

    let mut w = Tensor::from_vec(
        vec![0.01f32; n_classes * n_classes],
        (n_classes, n_classes),
        Device::Cpu,
    )
    .unwrap()
    .requires_grad_(true);

    let lr = 0.5f32;
    let mut loss_first = 0.0;
    let mut loss_last = 0.0;
    for step in 0..200 {
        let logits = x.matmul(&w).unwrap();
        let l = loss::cross_entropy_with_one_hot(&logits, &y, 1).unwrap();
        if step == 0 {
            loss_first = scalar(&l);
        }
        loss_last = scalar(&l);
        l.backward().unwrap();
        let new_w = w.sub(&w.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        w = relinearize(new_w);
    }
    assert!(loss_last < loss_first * 0.5, "CE did not decrease: first={loss_first} last={loss_last}");
}

#[test]
fn softmax_sums_to_one() {
    setup();
    let x = const_t(vec![1.0, 2.0, 3.0, 0.5, 0.3, 0.2], &[2, 3]);
    let s = activation::softmax(&x, 1).unwrap();
    let sums = flat(&s.sum(vec![1]).unwrap());
    for v in sums {
        assert!((v - 1.0).abs() < 1e-5, "softmax row should sum to 1, got {v}");
    }
}

#[test]
fn log_softmax_eq_log_of_softmax() {
    setup();
    let x = const_t(vec![1.0, 2.0, 3.0, 0.5, 0.3, 0.2], &[2, 3]);
    let ls = activation::log_softmax(&x, 1).unwrap();
    let sm = activation::softmax(&x, 1).unwrap();
    let log_sm = sm.log().unwrap();
    let a = flat(&ls);
    let b = flat(&log_sm);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-4, "log_softmax != log(softmax): {x} vs {y}");
    }
}

#[test]
fn euclidean_distance_backward() {
    setup();
    let a = leaf(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = const_t(vec![0.0, 0.0, 1.0, 0.0], &[2, 2]);
    let d = distance::euclidean(&a, &b).unwrap();
    assert_eq!(d.dims(), &[2]);
    d.sum_all().unwrap().backward().unwrap();
    assert!(a.grad().is_some());
}

#[test]
fn cosine_distance_backward() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5], &[2, 3]);
    let b = const_t(vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5], &[2, 3]);
    let c = distance::cosine(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2]);
    c.sum_all().unwrap().backward().unwrap();
    assert!(a.grad().is_some());
}

#[test]
fn mlp_with_relu_classification_converges() {
    setup();
    let in_dim = 2usize;
    let hidden = 8usize;
    let out = 2usize;
    let n = 16usize;
    let mut x_raw = Vec::with_capacity(n * in_dim);
    let mut y_raw = vec![0.0f32; n * out];
    for i in 0..n {
        let cls = i % 2;
        let cx = if cls == 0 { -1.0 } else { 1.0 };
        let cy = if cls == 0 { -1.0 } else { 1.0 };
        x_raw.push(cx + ((i as f32) % 3.0) * 0.1);
        x_raw.push(cy + ((i as f32) % 3.0) * 0.1);
        y_raw[i * out + cls] = 1.0;
    }
    let x = const_t(x_raw, &[n, in_dim]);
    let y = const_t(y_raw, &[n, out]);

    let init = |numel: usize, scale: f32, seed: u64| -> Vec<f32> {
        let mut state = seed ^ 0xDEADBEEFCAFEBABE;
        (0..numel)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let mut w1 = Tensor::from_vec(init(in_dim * hidden, 0.5, 13), (in_dim, hidden), Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let mut b1 = Tensor::zeros((1usize, hidden), synaptix_core::dtype::DType::F32, Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let mut w2 = Tensor::from_vec(init(hidden * out, 0.5, 17), (hidden, out), Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let mut b2 = Tensor::zeros((1usize, out), synaptix_core::dtype::DType::F32, Device::Cpu)
        .unwrap()
        .requires_grad_(true);

    let lr = 0.1f32;
    let mut loss_first = 0.0;
    let mut loss_last = 0.0;
    for step in 0..400 {
        let h = linear::linear(&x, &w1, Some(&b1)).unwrap();
        let h = activation::relu(&h).unwrap();
        let logits = linear::linear(&h, &w2, Some(&b2)).unwrap();
        let l = loss::cross_entropy_with_one_hot(&logits, &y, 1).unwrap();
        if step == 0 {
            loss_first = scalar(&l);
        }
        loss_last = scalar(&l);
        l.backward().unwrap();
        let new_w1 = w1.sub(&w1.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        let new_b1 = b1.sub(&b1.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        let new_w2 = w2.sub(&w2.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        let new_b2 = b2.sub(&b2.grad().unwrap().affine(lr, 0.0).unwrap()).unwrap();
        w1 = relinearize(new_w1);
        b1 = relinearize(new_b1);
        w2 = relinearize(new_w2);
        b2 = relinearize(new_b2);
    }
    assert!(loss_last < loss_first * 0.5, "MLP+ReLU+CE did not converge: first={loss_first} last={loss_last}");
}
