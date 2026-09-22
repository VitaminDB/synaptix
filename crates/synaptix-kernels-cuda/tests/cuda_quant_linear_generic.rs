//! Портируемый `linear_quant` через `Tensor` API для одноблобных форматов:
//! M = 1 (GEMV) и M = 100 (деквант полосами + плотный GEMM), F16 и BF16
//! активации, против CPU-эталона (деквант f32 · x в f64). Под
//! `SYN_FORCE_ARCH=sm_80` тем же путём идут NVFP4/MXFP8 (см. набор
//! cuda_matmul_quant_dispatch / cuda_linear_quant_mxfp8).

use std::sync::Arc;

use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::{block_row_bytes, dequant_row_f32, sq, GgmlType};
use synaptix_core::tensor::quant::QuantWeight;
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

/// Q4_K из SQ-подобных данных: делаем блоки «честными» — квантуем f32 в Q8_0
/// вручную (простой формат с точной обратимостью шкалы) и в SQ4.
fn q8_0_blob(x: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * (k / 32) * 34);
    for r in 0..n {
        for b in 0..k / 32 {
            let v = &x[r * k + b * 32..r * k + b * 32 + 32];
            let amax = v.iter().fold(0f32, |a, t| a.max(t.abs()));
            let d = amax / 127.0;
            let d16 = f16::from_f32(d);
            out.extend_from_slice(&d16.to_le_bytes());
            let dd = d16.to_f32();
            for t in v {
                let q = if dd > 0.0 { (t / dd).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                out.push(q as u8);
            }
        }
    }
    out
}

fn run_case(dtype: DType, blob: Vec<u8>, n: usize, k: usize, m: usize, act: DType, rng: &mut Lcg) {
    let rb = block_row_bytes(dtype, k).unwrap();
    let mut wf = vec![0f32; n * k];
    for r in 0..n {
        dequant_row_f32(dtype, &blob[r * rb..(r + 1) * rb], k, &mut wf[r * k..(r + 1) * k]).unwrap();
    }
    let dev = Device::Cuda(0);
    let packed = Tensor::from_raw_slice(&blob, vec![blob.len()], DType::U8, dev).unwrap();
    let qw = QuantWeight::new_block(packed.storage_arc(), dtype, n, k).unwrap();
    let x: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
    let xt = Tensor::from_vec(x.clone(), (m, k), dev).unwrap().to_dtype(act).unwrap();
    // Активация после округления к act.
    let xq: Vec<f32> = match act {
        DType::F16 => x.iter().map(|v| f16::from_f32(*v).to_f32()).collect(),
        _ => x.iter().map(|v| bf16::from_f32(*v).to_f32()).collect(),
    };
    let y = xt.linear_quant(&qw).unwrap_or_else(|e| panic!("{dtype:?} m={m} {act:?}: linear_quant: {e}"));
    assert_eq!(y.dims(), &[m, n]);
    assert_eq!(y.dtype(), act);
    let got = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut worst = 0f64;
    let mut scale = 1e-3f64;
    let mut want = vec![0f64; m * n];
    for mm in 0..m {
        for r in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                acc += wf[r * k + c] as f64 * xq[mm * k + c] as f64;
            }
            want[mm * n + r] = acc;
            scale = scale.max(acc.abs());
        }
    }
    let tol = if act == DType::BF16 { 2e-2 } else { 5e-3 } * scale + 1e-4;
    for i in 0..m * n {
        let d = (got[i] as f64 - want[i]).abs();
        worst = worst.max(d);
        assert!(d <= tol, "{dtype:?} m={m} {act:?} [{i}]: gpu {} vs {} (tol {tol})", got[i], want[i]);
    }
    let _ = worst;
}

#[test]
fn block_formats_linear_quant_gemv_and_gemm() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut rng = Lcg(2026);
    let (n, k) = (192usize, 1024usize);
    let x: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
    let cases: Vec<(DType, Vec<u8>)> = vec![
        (DType::Sq { bits: 4 }, sq::quantize_matrix(4, &x, n, k).unwrap()),
        (DType::Sq { bits: 2 }, sq::quantize_matrix(2, &x, n, k).unwrap()),
        (DType::Ggml(GgmlType::Q8_0), q8_0_blob(&x, n, k)),
    ];
    for (dtype, blob) in cases {
        for m in [1usize, 8, 100] {
            for act in [DType::F16, DType::BF16] {
                run_case(dtype, blob.clone(), n, k, m, act, &mut rng);
            }
        }
    }
}

#[test]
fn block_weight_moves_between_devices_and_stays_usable() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut rng = Lcg(9);
    let (n, k) = (64usize, 256usize);
    let x: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
    let blob = sq::quantize_matrix(4, &x, n, k).unwrap();
    let packed = Tensor::from_raw_slice(&blob, vec![blob.len()], DType::U8, Device::Cpu).unwrap();
    let host = QuantWeight::new_block(packed.storage_arc(), DType::Sq { bits: 4 }, n, k).unwrap();
    let dev = host.to_device(Device::Cuda(0)).unwrap();
    assert!(dev.scales_opt().is_none());
    let xt = Tensor::from_vec(vec![1f32; k], (1, k), Device::Cuda(0)).unwrap().to_dtype(DType::F16).unwrap();
    let y = xt.linear_quant(&dev).unwrap();
    assert_eq!(y.dims(), &[1, n]);
    let back = dev.to_device(Device::Cpu).unwrap();
    assert_eq!(back.packed_arc().unwrap().as_cpu().unwrap().as_bytes(), &blob[..]);
    let _ = Arc::new(back);
}
