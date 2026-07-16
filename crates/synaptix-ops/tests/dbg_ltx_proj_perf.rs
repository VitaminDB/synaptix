//! Диагностика LTX qkv_proj 35ms: linear_quant nvfp4 m=3520 n=4096 k=4096,
//! вход BF16 (как LTX compute) vs F16, + выровненный m=3584.
#![cfg(feature = "cuda")]

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn ltx_proj_shapes() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    let (n, k) = (4096usize, 4096usize);
    let w = Tensor::randn(vec![n, k], Device::Cpu)
        .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
        .to_dtype(DType::F16).unwrap();
    let qw = w.quantize_to_nvfp4().unwrap();
    for (m, dt) in [(3520usize, DType::BF16), (3520, DType::F16), (3584, DType::F16), (3584, DType::BF16)] {
        let x = Tensor::randn(vec![m, k], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
            .to_dtype(dt).unwrap();
        let run = || {
            let y = if dt == DType::F16 {
                x.linear_quant(&qw).unwrap()
            } else {
                x.to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap().to_dtype(dt).unwrap()
            };
            std::hint::black_box(&y);
        };
        for _ in 0..5 { run(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..20 { run(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let dtm = t0.elapsed().as_secs_f64() / 20.0;
        let fl = 2.0 * m as f64 * n as f64 * k as f64;
        println!("m={m} {dt:?}: {:.2}ms = {:.1} TF", dtm * 1e3, fl / dtm / 1e12);
    }
}

#[test]
fn ltx_stage2_proj() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    for (m, n, k, dt) in [
        (14080usize, 4096usize, 4096usize, DType::BF16),
        (14080, 4096, 4096, DType::F16),
        (14080, 16384, 4096, DType::F16),
        (14080, 4096, 16384, DType::F16),
    ] {
        let w = Tensor::randn(vec![n, k], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
            .to_dtype(DType::F16).unwrap();
        let qw = w.quantize_to_nvfp4().unwrap();
        let x = Tensor::randn(vec![m, k], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
            .to_dtype(dt).unwrap();
        let run = || {
            let y = if dt == DType::F16 {
                x.linear_quant(&qw).unwrap()
            } else {
                x.to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap().to_dtype(dt).unwrap()
            };
            std::hint::black_box(&y);
        };
        for _ in 0..3 { run(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 { run(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let dtm = t0.elapsed().as_secs_f64() / 10.0;
        let fl = 2.0 * m as f64 * n as f64 * k as f64;
        println!("m={m} n={n} k={k} {dt:?}: {:.2}ms = {:.1} TF", dtm * 1e3, fl / dtm / 1e12);
    }
}

#[test]
fn ltx_ada_ops() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    let t = 9216usize;
    let modul = Tensor::randn(vec![1usize, t, 9, 4096], Device::Cpu)
        .unwrap().to_device(dev).unwrap().to_dtype(DType::BF16).unwrap();
    let sst = Tensor::randn(vec![9usize, 4096], Device::Cpu).unwrap().to_device(dev).unwrap();
    let sync = || synaptix_core::device::cuda::synchronize(0).unwrap();
    let time_op = |name: &str, f: &dyn Fn()| {
        for _ in 0..3 { f(); }
        sync();
        let t0 = std::time::Instant::now();
        for _ in 0..10 { f(); }
        sync();
        println!("{name}: {:.2}ms", t0.elapsed().as_secs_f64() / 10.0 * 1e3);
    };
    time_op("narrow+squeeze+contiguous", &|| {
        let ts = modul.narrow(2, 1, 1).unwrap().squeeze(2).unwrap().contiguous().unwrap();
        std::hint::black_box(&ts);
    });
    let ts = modul.narrow(2, 1, 1).unwrap().squeeze(2).unwrap().contiguous().unwrap();
    let table = sst.narrow(0, 1, 1).unwrap().contiguous().unwrap().reshape(vec![1usize, 1, 4096]).unwrap().to_dtype(DType::BF16).unwrap();
    time_op("broadcast_add", &|| {
        let y = ts.broadcast_add(&table).unwrap();
        std::hint::black_box(&y);
    });
    time_op("table_prep(F32 narrow+to_dtype)", &|| {
        let tb = sst.narrow(0, 1, 1).unwrap().contiguous().unwrap().reshape(vec![1usize, 1, 4096]).unwrap().to_dtype(DType::BF16).unwrap();
        std::hint::black_box(&tb);
    });
}
