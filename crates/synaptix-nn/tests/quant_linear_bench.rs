use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;

fn cuda() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let d = Device::Cuda(0);
    Tensor::zeros(vec![1], DType::F32, d).ok().map(|_| d)
}

fn rand_tensor(shape: Vec<usize>, dev: Device, scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i * 31 % 257) as f32 / 257.0 - 0.5) * scale).collect();
    Tensor::from_vec(v, shape, dev).unwrap().to_dtype(DType::BF16).unwrap()
}

#[test]
#[ignore]
fn bench_quant_matmul_shapes() {
    let Some(dev) = cuda() else { return };
    let rows = 1103usize;
    let shapes: [(usize, usize, &str); 4] = [
        (21504, 5376, "qkv"),
        (5376, 7168, "out"),
        (28672, 5376, "fc1"),
        (5376, 14336, "fc2"),
    ];
    for (n, k, name) in shapes {
        let w = rand_tensor(vec![n, k], dev, 0.2);
        let q = QuantLinear::build(w, None, DType::NVFP4, DType::BF16).unwrap();
        for amax in [8.0f32, 4096.0] {
            let x = rand_tensor(vec![rows, k], dev, amax);
            for _ in 0..3 {
                let _ = q.forward(&x).unwrap();
            }
            let _ = q.forward(&x).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap()
                .to_scalar::<f32>().unwrap();
            let iters = 20;
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = q.forward(&x).unwrap();
            }
            let _ = q.forward(&x).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap()
                .to_scalar::<f32>().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gflops = 2.0 * rows as f64 * n as f64 * k as f64 / (ms * 1e6);
            eprintln!("[bench] {name} [{n}x{k}] amax {amax:.0} · {ms:.3} мс · {gflops:.0} GFLOPS");
        }
    }
}

#[test]
#[ignore]
fn bench_prescale_pieces() {
    let Some(dev) = cuda() else { return };
    let x = rand_tensor(vec![1103, 7168], dev, 64.0);
    let sync = |t: &Tensor| {
        let _ = t.to_dtype(DType::F32).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    };
    let one = Tensor::from_vec(vec![2.0f32], vec![1], dev).unwrap().to_dtype(DType::BF16).unwrap();
    let mut probe: Vec<(&str, Box<dyn Fn() -> Tensor>)> = Vec::new();
    let xa = x.clone();
    probe.push(("abs", Box::new(move || xa.abs().unwrap())));
    let xb = x.clone();
    probe.push(("abs+max_all", Box::new(move || xb.abs().unwrap().max_all().unwrap())));
    let xc = x.clone();
    probe.push(("max_all", Box::new(move || xc.max_all().unwrap())));
    let xd = x.clone();
    let o1 = one.clone();
    probe.push(("broadcast_mul", Box::new(move || xd.broadcast_mul(&o1).unwrap())));
    let xe = x.clone();
    probe.push(("mul_scalar", Box::new(move || xe.mul_scalar(0.5).unwrap())));
    let xf = x.clone();
    probe.push(("to_f16", Box::new(move || xf.to_dtype(DType::F16).unwrap())));

    for (name, f) in probe {
        for _ in 0..3 {
            sync(&f());
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = f();
        }
        sync(&f());
        eprintln!("[piece] {name} · {:.3} мс", t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
}
