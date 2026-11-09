//! Кодировщик IQ3_S-128 на карте против CPU-эталона: байты совпадают.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::ggml::{quantize_iq3_s128, GgmlType};
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            let mut acc = 0f32;
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                acc += (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            acc * 0.02
        })
        .collect()
}

#[test]
fn iq3s128_gpu_matches_cpu() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    for (n, k, with_imp) in [(64usize, 640usize, false), (32, 2560, true)] {
        let x = noise(n as u64 * 31 + k as u64, n * k);
        // Через f16, как на деле: эталон получает те же округлённые значения.
        let xh: Vec<f32> = x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        let imp: Option<Vec<f32>> = with_imp.then(|| (0..k).map(|i| 0.5 + (i % 7) as f32 * 0.3).collect());
        let cpu = quantize_iq3_s128(&xh, n, k, imp.as_deref());
        let t = Tensor::from_vec(xh.clone(), vec![n, k], Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(Device::Cuda(0)).unwrap();
        let q = t.quantize_to_ggml(GgmlType::Iq3S128, imp.as_deref()).expect("квант на карте");
        let gpu = q.to_device(Device::Cpu).unwrap().packed_arc().unwrap().as_cpu().unwrap().as_bytes().to_vec();
        let diff = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
        eprintln!("IQ3_S-128 [{n}×{k}] imp={with_imp}: различающихся байт {diff} из {}", cpu.len());
        assert_eq!(diff, 0);
    }
}

#[test]
#[ignore]
fn iq3s128_gpu_speed() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let (n, k) = (1280usize, 2560usize);
    let x = noise(5, n * k);
    let t = Tensor::from_vec(x, vec![n, k], Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(Device::Cuda(0)).unwrap();
    let _ = t.quantize_to_ggml(GgmlType::Iq3S128, None).unwrap();
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let st = std::time::Instant::now();
    for _ in 0..5 {
        let _ = t.quantize_to_ggml(GgmlType::Iq3S128, None).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    eprintln!("gate_up [1280×2560]: {:.1} мс", st.elapsed().as_secs_f64() * 200.0);
}
