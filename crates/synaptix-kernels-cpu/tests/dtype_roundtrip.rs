//! Приведение типов на CPU: F32 → BF16/F16 → F32 укладывается в точность
//! формата, особые значения (±0, ±inf, NaN) переживают круг.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn samples() -> Vec<f32> {
    let mut v: Vec<f32> = (-500..500).map(|i| i as f32 * 0.37).collect();
    v.extend([1e-3, -1e-3, 3.14159, 65504.0, -65504.0, 1.0 / 3.0]);
    v
}

fn roundtrip(data: &[f32], via: DType) -> Vec<f32> {
    synaptix_kernels_cpu::ensure_registered();
    Tensor::from_vec(data.to_vec(), vec![data.len()], Device::Cpu)
        .unwrap()
        .to_dtype(via)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .filter(|(x, _)| x.abs() > 1e-6)
        .map(|(x, y)| ((x - y) / x).abs())
        .fold(0.0, f32::max)
}

#[test]
fn bf16_roundtrip_within_precision() {
    let x = samples();
    let y = roundtrip(&x, DType::BF16);
    // 8 бит мантиссы: округление к ближайшему — не больше 2^-8.
    assert!(max_rel_err(&x, &y) <= 2f32.powi(-8), "{}", max_rel_err(&x, &y));
}

#[test]
fn f16_roundtrip_within_precision() {
    let x = samples();
    let y = roundtrip(&x, DType::F16);
    // 11 бит мантиссы в нормальном диапазоне.
    assert!(max_rel_err(&x, &y) <= 2f32.powi(-11), "{}", max_rel_err(&x, &y));
}

#[test]
fn special_values_survive() {
    let x = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    for via in [DType::BF16, DType::F16] {
        let y = roundtrip(&x, via);
        assert_eq!(y[0].to_bits(), 0.0f32.to_bits(), "{via:?}");
        assert_eq!(y[1].to_bits(), (-0.0f32).to_bits(), "{via:?}");
        assert_eq!(y[2], f32::INFINITY, "{via:?}");
        assert_eq!(y[3], f32::NEG_INFINITY, "{via:?}");
        assert!(y[4].is_nan(), "{via:?}");
    }
}

#[test]
fn f16_overflow_saturates_to_inf() {
    let y = roundtrip(&[70000.0, -70000.0], DType::F16);
    assert_eq!(y, vec![f32::INFINITY, f32::NEG_INFINITY]);
}
