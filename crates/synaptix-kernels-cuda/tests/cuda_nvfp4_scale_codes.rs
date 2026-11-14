//! NVFP4: каждый код масштаба E4M3 (1…0x7D — субнормальные и нормальные)
//! читается ядрами карты так же, как CPU-деквантом. Строка `r` веса целиком
//! на одном коде масштаба: `amax = 6·s(code)`, поэтому кодировщик выбирает
//! ровно его, и ошибку пути легко приписать коду. Пути —
//! ядра декода: GEMV с плотной активацией (`gemv_a16`) и `linear_quant` (M=1,
//! активация в NVFP4 — допуск шире).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CpuBuf, Storage};
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::quant::fp8_e4m3::decode_e4m3;
use std::sync::Arc;

const N: usize = 256;
const K: usize = 512;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).unwrap()
}

#[test]
fn every_scale_code_decodes_like_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let dev = Device::Cuda(0);
    let code_of = |r: usize| (1 + r % 0x7d) as u8;
    // Вес: строка r на масштабе s(code), в каждой группе есть ±6s.
    let z = noise(3, N * K);
    let mut w = vec![0f32; N * K];
    for r in 0..N {
        let s = decode_e4m3(code_of(r));
        for c in 0..K {
            w[r * K + c] = if c % 16 == 0 { 6.0 * s } else { z[r * K + c] * 12.0 * s };
        }
    }
    let (p, sc) = synaptix_kernels_cpu::quant::encode::quantize_nvfp4_rows(&w, N, K).unwrap();
    let cpu = QuantWeight::new(Arc::new(Storage::Cpu(CpuBuf::from_vec(p))), Arc::new(Storage::Cpu(CpuBuf::from_vec(sc))), DType::NVFP4, N, K).unwrap();
    let deq_cpu = host(&cpu.dequantize(DType::F16).unwrap());
    let gpu = cpu.to_device(dev).unwrap();
    let xh: Vec<f32> = noise(9, K).iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let x = Tensor::from_vec(xh.clone(), vec![1, K], Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
    let a16 = host(&gpu.gemv_a16(&x).unwrap());
    let lq = host(&x.linear_quant(&gpu).unwrap());
    // Эталон выхода — по CPU-кванту, в f64.
    let reference: Vec<f64> = (0..N).map(|r| (0..K).map(|c| deq_cpu[r * K + c] as f64 * xh[c] as f64).sum()).collect();
    let mut bad = Vec::new();
    for r in 0..N {
        let scale = (0..K).map(|c| (deq_cpu[r * K + c] as f64 * xh[c] as f64).abs()).sum::<f64>().max(1e-30);
        let ea = (a16[r] as f64 - reference[r]).abs() / scale;
        let el = (lq[r] as f64 - reference[r]).abs() / scale;
        if ea > 2e-2 || el > 0.15 {
            bad.push(format!("код 0x{:02x} (s={:.3e}): gemv_a16 {ea:.3}, linear_quant {el:.3}", code_of(r), decode_e4m3(code_of(r))));
        }
    }
    assert!(bad.is_empty(), "коды масштаба, которые карта читает не так:\n{}", bad.join("\n"));
}
