use std::io::Cursor;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_debug::compare::{compare_dumps, compare_tensors};
use synaptix_debug::nan_detector::{check_finite, scan_finite};
use synaptix_debug::{
    cos_sim, dump_to_writer, l1_distance, l2_distance, load_from_reader, max_abs, rel_err,
    TensorDump,
};
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
}

#[test]
fn dump_round_trip_f32() {
    setup();
    let t = Tensor::from_vec(vec![1.0f32, -2.5, 3.25, 4.125], (2usize, 2), Device::Cpu).unwrap();
    let mut buf = Vec::new();
    dump_to_writer(&t, "weight.0", &mut buf).unwrap();
    let mut cursor = Cursor::new(&buf);
    let d = load_from_reader(&mut cursor).unwrap();
    assert_eq!(d.name, "weight.0");
    assert_eq!(d.dtype, DType::F32);
    assert_eq!(d.dims, vec![2usize, 2]);
    let v: Vec<f32> = (0..4)
        .map(|i| f32::from_le_bytes(d.data[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();
    assert_eq!(v, vec![1.0, -2.5, 3.25, 4.125]);
}

#[test]
fn dump_round_trip_bf16() {
    setup();
    let raw = vec![half::bf16::from_f32(1.5), half::bf16::from_f32(-0.25)];
    let bytes: Vec<u8> = raw.iter().flat_map(|x| x.to_le_bytes()).collect();
    let t = Tensor::from_raw_bytes(bytes, (2usize,), DType::BF16, Device::Cpu).unwrap();
    let mut buf = Vec::new();
    dump_to_writer(&t, "act.bf16", &mut buf).unwrap();
    let mut cursor = Cursor::new(&buf);
    let d = load_from_reader(&mut cursor).unwrap();
    assert_eq!(d.dtype, DType::BF16);
    assert_eq!(d.dims, vec![2]);
    assert_eq!(d.data.len(), 4);
}

#[test]
fn compare_identical_tensors() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2usize, 2), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2usize, 2), Device::Cpu).unwrap();
    let r = compare_tensors(&a, &b).unwrap();
    assert!((r.cos_sim - 1.0).abs() < 1e-12);
    assert_eq!(r.max_abs, 0.0);
    assert_eq!(r.l1, 0.0);
    assert_eq!(r.l2, 0.0);
    assert_eq!(r.numel, 4);
}

#[test]
fn compare_perturbed_tensors() {
    setup();
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4usize,), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![1.01f32, 1.99, 3.0, 4.02], (4usize,), Device::Cpu).unwrap();
    let r = compare_tensors(&a, &b).unwrap();
    assert!(r.cos_sim > 0.999);
    assert!(r.max_abs > 0.0);
    assert!(r.l1 > 0.0);
}

#[test]
fn compare_dumps_workflow() {
    setup();
    let a = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], (3usize,), Device::Cpu).unwrap();
    let b = Tensor::from_vec(vec![0.1001f32, 0.2002, 0.2998], (3usize,), Device::Cpu).unwrap();
    let mut buf_a = Vec::new();
    dump_to_writer(&a, "a", &mut buf_a).unwrap();
    let mut buf_b = Vec::new();
    dump_to_writer(&b, "b", &mut buf_b).unwrap();
    let da = load_from_reader(&mut Cursor::new(buf_a)).unwrap();
    let db = load_from_reader(&mut Cursor::new(buf_b)).unwrap();
    let r = compare_dumps(&da, &db).unwrap();
    assert!(r.cos_sim > 0.999_99);
    assert!(r.max_abs < 0.01);
}

#[test]
fn compare_slice_helpers_match_expected() {
    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![1.5f64, 1.5, 3.5];
    assert!((cos_sim(&a, &b) - 0.9794).abs() < 1e-3, "cos_sim={}", cos_sim(&a, &b));
    assert!((max_abs(&a, &b) - 0.5).abs() < 1e-12);
    assert!(rel_err(&a, &b) > 0.0);
    assert!((l1_distance(&a, &b) - 1.5).abs() < 1e-12);
    assert!((l2_distance(&a, &b) - (3.0f64 * 0.25f64).sqrt()).abs() < 1e-12);
}

#[test]
fn nan_detector_clean_tensor() {
    setup();
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3usize,), Device::Cpu).unwrap();
    let s = scan_finite(&t).unwrap();
    assert!(s.is_clean());
    check_finite(&t).unwrap();
}

#[test]
fn nan_detector_finds_nan() {
    setup();
    let nan = f32::NAN;
    let t = Tensor::from_vec(vec![1.0f32, nan, 3.0], (3usize,), Device::Cpu).unwrap();
    let s = scan_finite(&t).unwrap();
    assert_eq!(s.nan_count, 1);
    assert_eq!(s.first_nan_at, Some(1));
    assert!(check_finite(&t).is_err());
}

#[test]
fn nan_detector_finds_inf() {
    setup();
    let pinf = f32::INFINITY;
    let ninf = f32::NEG_INFINITY;
    let t = Tensor::from_vec(vec![1.0f32, pinf, ninf, 4.0], (4usize,), Device::Cpu).unwrap();
    let s = scan_finite(&t).unwrap();
    assert_eq!(s.pos_inf_count, 1);
    assert_eq!(s.neg_inf_count, 1);
    assert!(!s.is_clean());
}

#[test]
fn shape_assert_macro_passes_on_match() {
    setup();
    let t = Tensor::zeros((2usize, 3, 4), DType::F32, Device::Cpu).unwrap();
    synaptix_debug::shape_assert!(t, [2, 3, 4]);
    synaptix_debug::shape_assert_rank!(t, 3);
}

#[test]
#[should_panic(expected = "shape_assert!")]
fn shape_assert_macro_panics_on_mismatch() {
    setup();
    let t = Tensor::zeros((2usize, 3), DType::F32, Device::Cpu).unwrap();
    synaptix_debug::shape_assert!(t, [3, 2]);
}

#[test]
fn dump_struct_numel_correct() {
    let d = TensorDump {
        name: "x".into(),
        dtype: DType::F32,
        dims: vec![2, 3, 4],
        data: vec![0u8; 4 * 2 * 3 * 4],
    };
    assert_eq!(d.numel(), 24);
}
