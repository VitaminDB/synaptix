
use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered as ensure_cpu;
use synaptix_kernels_cuda::ensure_registered as ensure_cuda;

fn setup() -> bool {
    ensure_cpu();
    ensure_cuda();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn unary_sqrt_f32_cuda_matches_cpu() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.5).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = cpu_t.sqrt().unwrap();
    let y_cuda = cuda_t.sqrt().unwrap().to_device(Device::Cpu).unwrap();
    let a: Vec<f32> = y_cpu
        .to_vec3::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let b: Vec<f32> = y_cuda
        .to_vec3::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn unary_silu_f32_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (21,), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = cpu_t.silu().unwrap();
    let y_cuda = cuda_t.silu().unwrap().to_device(Device::Cpu).unwrap();
    let a: Vec<f32> = y_cpu.to_vec1().unwrap();
    let b: Vec<f32> = y_cuda.to_vec1().unwrap();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn unary_affine_f32_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = cpu_t.affine(0.5, 3.0).unwrap();
    let y_cuda = cuda_t
        .affine(0.5, 3.0)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    let a: Vec<f32> = y_cpu
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let b: Vec<f32> = y_cuda
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn binary_add_f32_cuda_broadcast() {
    if !setup() {
        return;
    }
    let a_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..4).map(|i| (i as f32) * 10.0).collect();
    let a_cpu = Tensor::from_vec(a_data.clone(), (3, 4), Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), (4,), Device::Cpu).unwrap();
    let a_cuda = a_cpu.to_device(Device::Cuda(0)).unwrap();
    let b_cuda = b_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = a_cpu.broadcast_add(&b_cpu).unwrap();
    let y_cuda = a_cuda
        .broadcast_add(&b_cuda)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    let aa: Vec<f32> = y_cpu
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let bb: Vec<f32> = y_cuda
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(aa, bb);
}

#[test]
fn binary_mul_bf16_cuda() {
    if !setup() {
        return;
    }
    let a_data: Vec<bf16> = (0..16).map(|i| bf16::from_f32(i as f32 * 0.1)).collect();
    let b_data: Vec<bf16> = (0..16)
        .map(|i| bf16::from_f32((i as f32) * 0.05 + 1.0))
        .collect();
    let a_cpu = Tensor::from_vec(a_data.clone(), (4, 4), Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), (4, 4), Device::Cpu).unwrap();
    let a_cuda = a_cpu.to_device(Device::Cuda(0)).unwrap();
    let b_cuda = b_cpu.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = a_cpu.mul(&b_cpu).unwrap();
    let y_cuda = a_cuda.mul(&b_cuda).unwrap().to_device(Device::Cpu).unwrap();
    let aa: Vec<bf16> = y_cpu
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let bb: Vec<bf16> = y_cuda
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    for (x, y) in aa.iter().zip(bb.iter()) {
        let d = (x.to_f32() - y.to_f32()).abs();
        assert!(d < 0.01, "{x} vs {y} (diff {d})");
    }
}

#[test]
fn cast_f32_to_bf16_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (4, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = cpu_t.to_dtype(DType::BF16).unwrap();
    let y_cuda = cuda_t
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    let a: Vec<bf16> = y_cpu
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let b: Vec<bf16> = y_cuda
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(a, b);
}

#[test]
fn cast_bf16_to_f32_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<bf16> = (0..16)
        .map(|i| bf16::from_f32(i as f32 * 0.1 - 0.5))
        .collect();
    let cpu_t = Tensor::from_vec(data.clone(), (4, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y_cpu = cpu_t.to_dtype(DType::F32).unwrap();
    let y_cuda = cuda_t
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    let a: Vec<f32> = y_cpu
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let b: Vec<f32> = y_cuda
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(a, b);
}

#[test]
fn cast_f16_f32_roundtrip_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32 * 0.25)).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (8,), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let y = cuda_t
        .to_dtype(DType::F32)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let back = y.to_device(Device::Cpu).unwrap();
    assert_eq!(back.to_vec1::<f16>().unwrap(), data);
}

#[test]
fn reduce_sum_f32_cuda_matches_cpu() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let s_cpu: f32 = cpu_t.sum_all().unwrap().to_scalar().unwrap();
    let s_cuda: f32 = cuda_t
        .sum_all()
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!((s_cpu - s_cuda).abs() < 1e-4, "{s_cpu} vs {s_cuda}");
}

#[test]
fn reduce_mean_keepdim_f32_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (2, 3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let cpu_r = cpu_t.mean_keepdim(2).unwrap();
    let cuda_r = cuda_t
        .mean_keepdim(2)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    assert_eq!(cuda_r.dims(), &[2, 3, 1]);
    let a: Vec<f32> = cpu_r
        .to_vec3::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let b: Vec<f32> = cuda_r
        .to_vec3::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn reduce_max_keepdim_bf16_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<bf16> = (0..12)
        .map(|i| bf16::from_f32((i as f32 - 6.0) * 0.3))
        .collect();
    let cpu_t = Tensor::from_vec(data.clone(), (3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let r_cpu = cpu_t.sum_keepdim(1).unwrap();
    let r_cuda = cuda_t
        .sum_keepdim(1)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    let a: Vec<bf16> = r_cpu
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let b: Vec<bf16> = r_cuda
        .to_vec2::<bf16>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x.to_f32() - y.to_f32()).abs();
        assert!(d < 0.02, "{x} vs {y}");
    }
}

#[test]
fn argmax_f32_cuda() {
    if !setup() {
        return;
    }
    let data: Vec<f32> = vec![0.1, 0.5, 0.3, 0.9, 0.4, 0.7, 0.8, 0.2];
    let cpu_t = Tensor::from_vec(data.clone(), (2, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let cpu_r = cpu_t.argmax(1).unwrap();
    let cuda_r = cuda_t.argmax(1).unwrap().to_device(Device::Cpu).unwrap();
    assert_eq!(cuda_r.dims(), &[2]);
    assert_eq!(
        cuda_r.to_vec1::<u32>().unwrap(),
        cpu_r.to_vec1::<u32>().unwrap()
    );
    assert_eq!(cuda_r.to_vec1::<u32>().unwrap(), vec![3, 2]);
}
