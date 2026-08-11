
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
fn cpu_to_cuda_to_cpu_f32() {
    if !setup() {
        eprintln!("no cuda device, skipping");
        return;
    }
    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let cpu_t = Tensor::from_vec(data.clone(), (3, 4), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    assert_eq!(cuda_t.device(), Device::Cuda(0));
    let back = cuda_t.to_device(Device::Cpu).unwrap();
    assert_eq!(
        back.to_vec2::<f32>().unwrap(),
        cpu_t.to_vec2::<f32>().unwrap()
    );
}

#[test]
fn cpu_to_cuda_to_cpu_u32() {
    if !setup() {
        return;
    }
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let cpu_t = Tensor::from_vec(data.clone(), (5,), Device::Cpu).unwrap();
    let cuda_t = cpu_t.to_device(Device::Cuda(0)).unwrap();
    let back = cuda_t.to_device(Device::Cpu).unwrap();
    assert_eq!(back.to_vec1::<u32>().unwrap(), data);
}

#[test]
fn zeros_on_cuda_then_read() {
    if !setup() {
        return;
    }
    let t = Tensor::zeros((2, 4), DType::F32, Device::Cuda(0)).unwrap();
    let cpu = t.to_device(Device::Cpu).unwrap();
    assert_eq!(cpu.to_vec2::<f32>().unwrap(), vec![vec![0.0; 4]; 2]);
}

// Dense matmul на CUDA идёт через best_cu (F32 SIMT, F16/BF16 WMMA float-acc;
// N%8 закрыт N-pad). cutlass выпилен (decutlass).
#[test]
fn matmul_f32_cpu_vs_cuda() {
    if !setup() {
        return;
    }
    let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

    let a_cpu = Tensor::from_vec(a_data.clone(), (2, 3), Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), (3, 4), Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_cuda = Tensor::from_vec(a_data, (2, 3), Device::Cuda(0)).unwrap();
    let b_cuda = Tensor::from_vec(b_data, (3, 4), Device::Cuda(0)).unwrap();
    let c_cuda = a_cuda.matmul(&b_cuda).unwrap();
    let c_back = c_cuda.to_device(Device::Cpu).unwrap();

    let cpu_v = c_cpu.to_vec2::<f32>().unwrap();
    let cuda_v = c_back.to_vec2::<f32>().unwrap();
    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (cpu_v[i][j] - cuda_v[i][j]).abs() < 1e-3,
                "mismatch at ({i},{j}): cpu={} cuda={}",
                cpu_v[i][j],
                cuda_v[i][j]
            );
        }
    }
}

#[test]
fn matmul_bf16_cuda() {
    if !setup() {
        return;
    }
    let a: Vec<half::bf16> = (1..=4).map(|x| half::bf16::from_f32(x as f32)).collect();
    let b: Vec<half::bf16> = (5..=8).map(|x| half::bf16::from_f32(x as f32)).collect();

    let a_t = Tensor::from_vec(a, (2, 2), Device::Cuda(0)).unwrap();
    let b_t = Tensor::from_vec(b, (2, 2), Device::Cuda(0)).unwrap();
    let c = a_t.matmul(&b_t).unwrap();
    let back = c.to_device(Device::Cpu).unwrap();
    let v = back.to_vec2::<half::bf16>().unwrap();
    assert!((v[0][0].to_f32() - 19.0).abs() < 2.0);
    assert!((v[1][1].to_f32() - 50.0).abs() < 2.0);
}

#[test]
fn mixed_device_collection_compiles() {
    if !setup() {
        return;
    }
    let a = Tensor::zeros((2,), DType::F32, Device::Cpu).unwrap();
    let b = Tensor::zeros((2,), DType::F32, Device::Cuda(0)).unwrap();
    let v: Vec<Tensor> = vec![a, b];
    assert_eq!(v[0].device(), Device::Cpu);
    assert_eq!(v[1].device(), Device::Cuda(0));
}
