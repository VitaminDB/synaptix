//! Энкодер SQ на карте бит в бит с CPU-эталоном
//! `synaptix_core::quant::sq::quantize_matrix`: все SQ1…SQ8, вход F16 и
//! BF16, хвостовой неполный супер-блок, выбросы в данных (подбор шкалы
//! должен сходиться одинаково). Плюс round-trip через `QuantWeight::dequantize`
//! и `Tensor::quantize_to` по DType.

use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::sq;
use synaptix_core::tensor::Tensor;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn data(seed: u64, n: usize, k: usize) -> Vec<f32> {
    let mut r = Lcg(seed);
    (0..n * k)
        .map(|i| {
            let v = r.f32() * 0.05;
            if i % 41 == 0 { v * 6.0 } else { v }
        })
        .collect()
}

#[test]
fn gpu_encoder_matches_cpu_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for (n, k) in [(8usize, 256usize), (5, 288), (16, 1024), (3, 32)] {
        let x = data(7 + n as u64, n, k);
        for in_dt in [DType::F16, DType::BF16] {
            // Эталон считает из тех же округлённых значений, что видит ядро.
            let x_r: Vec<f32> = match in_dt {
                DType::F16 => x.iter().map(|v| f16::from_f32(*v).to_f32()).collect(),
                _ => x.iter().map(|v| bf16::from_f32(*v).to_f32()).collect(),
            };
            let t = Tensor::from_vec(x_r.clone(), vec![n, k], Device::Cpu).unwrap().to_dtype(in_dt).unwrap().to_device(dev).unwrap();
            for bits in 1..=8u8 {
                let want = sq::quantize_matrix(bits, &x_r, n, k).unwrap();
                let qw = t.quantize_to(DType::Sq { bits }).unwrap();
                assert_eq!(qw.dtype(), DType::Sq { bits });
                let got = qw.to_device(Device::Cpu).unwrap();
                let got = got.packed_arc().unwrap();
                let got = got.as_cpu().unwrap().as_bytes().to_vec();
                assert_eq!(got.len(), want.len(), "sq{bits} [{n},{k}] {in_dt:?}: размер блоба");
                if got != want {
                    let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                    panic!("sq{bits} [{n},{k}] {in_dt:?}: блоб отличается с байта {first}: {} vs {}", got[first], want[first]);
                }
                // Round-trip на карте = эталонный деквант.
                let back = qw.dequantize(DType::F16).unwrap().to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
                let back = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let ref_back = sq::dequantize_matrix(bits, &want, n, k).unwrap();
                for i in 0..n * k {
                    assert!((back[i] - f16::from_f32(ref_back[i]).to_f32()).abs() <= 1e-6, "sq{bits}: деквант [{i}]");
                }
            }
        }
    }
}

#[test]
fn nvfp4_dequant_roundtrip_for_transcoding() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (n, k) = (128usize, 256usize);
    let x = data(3, n, k);
    let t = Tensor::from_vec(x.clone(), vec![n, k], Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
    let q = t.quantize_to_nvfp4().unwrap();
    let back = q.dequantize(DType::F16).unwrap().to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
    let back = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mse: f32 = x.iter().zip(&back).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|a| a * a).sum::<f32>() / x.len() as f32;
    assert!(mse < var * 0.05, "NVFP4 деквант: mse {mse} против дисперсии {var}");
    // Перекодировка NVFP4 → SQ4 через плотный F16.
    let sq4 = q.dequantize(DType::F16).unwrap().quantize_to(DType::Sq { bits: 4 }).unwrap();
    let back2 = sq4.dequantize(DType::F16).unwrap().to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
    let back2 = back2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mse2: f32 = x.iter().zip(&back2).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / x.len() as f32;
    assert!(mse2 < var * 0.08, "NVFP4→SQ4: mse {mse2} против дисперсии {var}");
}
