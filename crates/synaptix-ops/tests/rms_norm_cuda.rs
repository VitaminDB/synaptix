
use half::bf16;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::{rms_norm, rms_norm_gated, rms_norm_qwen};

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn rms_norm_f32_cuda_matches_cpu() {
    if !setup() { return; }
    let shape = [2usize, 3usize, 4usize];
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let w: Vec<f32> = vec![0.5, -0.25, 1.0, 2.0];
    let eps = 1e-6f32;

    let x_cpu = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let w_cpu = Tensor::from_vec(w.clone(), (4usize,), Device::Cpu).unwrap();
    let y_cpu = rms_norm(&x_cpu, &w_cpu, eps).unwrap();

    let x_cuda = x_cpu.to_device(Device::Cuda(0)).unwrap();
    let w_cuda = w_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cuda = rms_norm(&x_cuda, &w_cuda, eps)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();

    let a: Vec<f32> = y_cpu.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    let b: Vec<f32> = y_cuda.to_vec3::<f32>().unwrap().into_iter().flatten().flatten().collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn rms_norm_qwen_f32_cuda_matches_cpu() {
    if !setup() { return; }
    let shape = [4usize, 8usize];
    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.13 - 0.7).collect();
    let w: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let eps = 1e-5f32;

    let x_cpu = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let w_cpu = Tensor::from_vec(w.clone(), (8usize,), Device::Cpu).unwrap();
    let y_cpu = rms_norm_qwen(&x_cpu, &w_cpu, eps).unwrap();

    let x_cuda = x_cpu.to_device(Device::Cuda(0)).unwrap();
    let w_cuda = w_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cuda = rms_norm_qwen(&x_cuda, &w_cuda, eps)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();

    let a: Vec<f32> = y_cpu.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let b: Vec<f32> = y_cuda.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn rms_norm_gated_f32_cuda_matches_cpu() {
    if !setup() { return; }
    let shape = [3usize, 4usize];
    let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.15 - 0.5).collect();
    let gate: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = vec![0.5, 0.25, 1.0, 0.75];
    let eps = 1e-6f32;

    let x_cpu = Tensor::from_vec(x.clone(), shape, Device::Cpu).unwrap();
    let g_cpu = Tensor::from_vec(gate.clone(), shape, Device::Cpu).unwrap();
    let w_cpu = Tensor::from_vec(w.clone(), (4usize,), Device::Cpu).unwrap();
    let y_cpu = rms_norm_gated(&x_cpu, &g_cpu, &w_cpu, eps).unwrap();

    let x_cuda = x_cpu.to_device(Device::Cuda(0)).unwrap();
    let g_cuda = g_cpu.to_device(Device::Cuda(0)).unwrap();
    let w_cuda = w_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cuda = rms_norm_gated(&x_cuda, &g_cuda, &w_cuda, eps)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();

    let a: Vec<f32> = y_cpu.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let b: Vec<f32> = y_cuda.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn rms_norm_bf16_cuda_matches_cpu() {
    if !setup() { return; }
    let shape = [2usize, 6usize];
    let xf: Vec<f32> = (0..12).map(|i| (i as f32) * 0.13 - 0.5).collect();
    let wf: Vec<f32> = vec![0.5, -0.25, 1.0, 2.0, 0.0, -1.5];
    let eps = 1e-6f32;

    let x_bf: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_bf: Vec<bf16> = wf.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_cpu = Tensor::from_vec(x_bf.clone(), shape, Device::Cpu).unwrap();
    let w_cpu = Tensor::from_vec(w_bf.clone(), (6usize,), Device::Cpu).unwrap();
    let y_cpu = rms_norm(&x_cpu, &w_cpu, eps).unwrap();

    let x_cuda = x_cpu.to_device(Device::Cuda(0)).unwrap();
    let w_cuda = w_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cuda = rms_norm(&x_cuda, &w_cuda, eps)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();

    let a: Vec<bf16> = y_cpu.to_vec2::<bf16>().unwrap().into_iter().flatten().collect();
    let b: Vec<bf16> = y_cuda.to_vec2::<bf16>().unwrap().into_iter().flatten().collect();
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x.to_f32() - y.to_f32()).abs();
        assert!(d < 0.02, "{x} vs {y}");
    }
}
