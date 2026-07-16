use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
    autograd_init().unwrap();
}

fn scalar(t: &Tensor) -> f32 {
    t.reshape((1,)).unwrap().to_vec1::<f32>().unwrap()[0]
}

fn deterministic_uniform(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 32) as u32;
            let v = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v * scale
        })
        .collect()
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).unwrap().sqr().unwrap().mean().unwrap()
}

fn relinearize_leaf(t: Tensor) -> Tensor {
    t.detach().requires_grad_(true)
}

#[test]
fn mlp_two_layer_silu_mse_converges() {
    setup();
    let in_dim = 3usize;
    let hid_dim = 8usize;
    let out_dim = 1usize;
    let batch = 16usize;

    let xs_raw = deterministic_uniform(batch * in_dim, 11, 1.0);
    let xs = Tensor::from_vec(xs_raw.clone(), (batch, in_dim), Device::Cpu).unwrap();
    let true_w: Vec<f32> = vec![1.5f32, -0.7, 0.3];
    let target_raw: Vec<f32> = (0..batch)
        .map(|b| {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += xs_raw[b * in_dim + i] * true_w[i];
            }
            acc + 0.2
        })
        .collect();
    let target = Tensor::from_vec(target_raw, (batch, out_dim), Device::Cpu).unwrap();

    let mut w1 = Tensor::from_vec(
        deterministic_uniform(in_dim * hid_dim, 42, 0.4),
        (in_dim, hid_dim),
        Device::Cpu,
    )
    .unwrap()
    .requires_grad_(true);
    let mut b1 = Tensor::zeros((1usize, hid_dim), DType::F32, Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let mut w2 = Tensor::from_vec(
        deterministic_uniform(hid_dim * out_dim, 99, 0.4),
        (hid_dim, out_dim),
        Device::Cpu,
    )
    .unwrap()
    .requires_grad_(true);
    let mut b2 = Tensor::zeros((1usize, out_dim), DType::F32, Device::Cpu)
        .unwrap()
        .requires_grad_(true);

    let lr = 0.05f32;
    let steps = 400;

    let loss_first = {
        let h = xs.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().silu().unwrap();
        let y = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        scalar(&mse_loss(&y, &target))
    };

    let mut loss_last = loss_first;
    for _ in 0..steps {
        let h = xs.matmul(&w1).unwrap().broadcast_add(&b1).unwrap().silu().unwrap();
        let y = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
        let loss = mse_loss(&y, &target);
        loss_last = scalar(&loss);
        loss.backward().unwrap();

        let gw1 = w1.grad().unwrap();
        let gb1 = b1.grad().unwrap();
        let gw2 = w2.grad().unwrap();
        let gb2 = b2.grad().unwrap();

        let new_w1 = w1.sub(&gw1.affine(lr, 0.0).unwrap()).unwrap();
        let new_b1 = b1.sub(&gb1.affine(lr, 0.0).unwrap()).unwrap();
        let new_w2 = w2.sub(&gw2.affine(lr, 0.0).unwrap()).unwrap();
        let new_b2 = b2.sub(&gb2.affine(lr, 0.0).unwrap()).unwrap();

        w1 = relinearize_leaf(new_w1);
        b1 = relinearize_leaf(new_b1);
        w2 = relinearize_leaf(new_w2);
        b2 = relinearize_leaf(new_b2);
    }

    assert!(
        loss_last < loss_first * 0.5,
        "loss did not decrease: first={loss_first}, last={loss_last}",
    );
    assert!(loss_last < 0.5, "final loss too high: {loss_last}");
}
