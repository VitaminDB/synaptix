//! Время операций свёрточного кодировщика на CPU.
use std::time::Instant;
use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_ops::conv::conv1d;
use synaptix_ops::norm::layer_norm;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let t_in = 6399usize;
    let x = Tensor::from_vec((0..512 * t_in).map(|i| (i % 13) as f32 * 0.1).collect::<Vec<_>>(), (1, 512, t_in), Device::Cpu).unwrap();
    let w = Tensor::from_vec((0..512 * 512 * 3).map(|i| (i % 7) as f32 * 0.01).collect::<Vec<_>>(), (512, 512, 3), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![0.1f32; 512], (512,), Device::Cpu).unwrap();
    let g = Tensor::from_vec(vec![1.0f32; 512], (512,), Device::Cpu).unwrap();
    let x0 = Tensor::from_vec((0..64000).map(|i| (i % 17) as f32 * 0.01).collect::<Vec<_>>(), (1, 1, 64000), Device::Cpu).unwrap();
    let w0 = Tensor::from_vec((0..512 * 10).map(|i| (i % 7) as f32 * 0.01).collect::<Vec<_>>(), (512, 1, 10), Device::Cpu).unwrap();
    for _ in 0..2 {
        let t = Instant::now();
        let h0 = conv1d(&x0, &w0, Some(&b), 5, 0).unwrap();
        println!("conv0 {:?} -> {:?}", t.elapsed(), h0.dims());
    }
    for _ in 0..2 {
        let t = Instant::now();
        let h = conv1d(&x, &w, Some(&b), 2, 0).unwrap();
        let t1 = t.elapsed();
        let tr = h.transpose(1, 2).unwrap().contiguous().unwrap();
        let t2 = t.elapsed();
        let ln = layer_norm(&tr, Some(&g), Some(&b), 1e-5).unwrap();
        let t3 = t.elapsed();
        let back = ln.transpose(1, 2).unwrap().contiguous().unwrap();
        let t4 = t.elapsed();
        let _ = back.gelu_exact().unwrap();
        let t5 = t.elapsed();
        println!("conv {:?} transpose {:?} layer_norm {:?} transpose {:?} gelu {:?}", t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4);
    }
}
