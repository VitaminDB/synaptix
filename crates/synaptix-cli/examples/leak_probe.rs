use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

fn live() -> f64 {
    synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64 / 1e9
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let probes: Vec<(&str, Box<dyn Fn() -> ()>)> = vec![
        ("zeros+silu small", Box::new(move || {
            let a = Tensor::zeros(vec![1, 128, 1, 72, 128], DType::F32, dev).unwrap();
            let _b = a.silu().unwrap();
        })),
        ("cat small", Box::new(move || {
            let a = Tensor::zeros(vec![1, 128, 1, 72, 128], DType::F32, dev).unwrap();
            let _c = Tensor::cat(&[&a, &a], 2).unwrap();
        })),
        ("narrow+contig small", Box::new(move || {
            let a = Tensor::zeros(vec![1, 128, 4, 72, 128], DType::F32, dev).unwrap();
            let _c = a.narrow(2, 1, 2).unwrap().contiguous().unwrap();
        })),
        ("zeros big 553MB", Box::new(move || {
            let a = Tensor::zeros(vec![1, 512, 117, 36, 64], DType::F32, dev).unwrap();
            let _b = a.silu().unwrap();
        })),
    ];
    for (name, f) in &probes {
        let l0 = live();
        for _ in 0..50 {
            f();
        }
        let _ = synaptix_core::device::cuda::synchronize_all(0);
        println!("{name}: live {l0:.3} -> {:.3} GB", live());
    }
}
