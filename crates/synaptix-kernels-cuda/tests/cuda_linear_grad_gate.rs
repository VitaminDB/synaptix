use half::bf16;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn have_gpu() -> bool {
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(f * scale)
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

const SHAPES: &[(usize, usize, usize)] = &[(1, 1536, 1536), (1, 1536, 256), (4, 512, 640)];

#[test]
fn plain_tensors_take_fast_path_under_grad_enabled() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    assert!(
        synaptix_core::grad::is_grad_enabled(),
        "тест проверяет поведение при включённом autograd по умолчанию"
    );
    for &(m, k, n) in SHAPES {
        let xt =
            Tensor::from_vec(det(0x1111, m * k, 0.3), (m, k), Device::Cuda(0)).unwrap();
        let wt =
            Tensor::from_vec(det(0x2222, n * k, 0.3), (n, k), Device::Cuda(0)).unwrap();

        let default_path = xt.linear(&wt).unwrap();
        let fast = {
            let _ng = synaptix_core::grad::NoGradGuard::new();
            xt.linear(&wt).unwrap()
        };
        let slow = {
            let _forced = synaptix_core::tensor::ops::ForceUnfusedLinearGuard::new(true);
            xt.linear(&wt).unwrap()
        };

        assert!(
            default_path.grad_meta().is_none(),
            "{m}x{k}x{n}: у выхода без requires_grad не должно быть grad-меты"
        );
        assert_eq!(host(&default_path), host(&fast), "{m}x{k}x{n}: default != no_grad");

        let (a, b) = (host(&default_path), host(&slow));
        let mut num = 0f64;
        let mut da = 0f64;
        let mut db = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += *x as f64 * *y as f64;
            da += *x as f64 * *x as f64;
            db += *y as f64 * *y as f64;
        }
        let cos = num / (da.sqrt() * db.sqrt() + 1e-30);
        assert!(cos >= 0.999, "{m}x{k}x{n}: fast vs unfused cos={cos}");
    }
}

#[test]
fn requires_grad_still_builds_graph() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    synaptix_autograd::init().unwrap();
    let (m, k, n) = (2usize, 64usize, 32usize);
    let xt = Tensor::from_vec(det(0x3333, m * k, 0.3), (m, k), Device::Cuda(0)).unwrap();
    let wt = Tensor::from_vec(det(0x4444, n * k, 0.3), (n, k), Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    let out = xt.linear(&wt).unwrap();
    assert!(
        out.grad_meta().is_some(),
        "requires_grad на весе обязан построить граф даже с быстрым backend-путём"
    );
    assert!(out.grad_fn().is_some(), "у выхода должен быть grad_fn");
}
