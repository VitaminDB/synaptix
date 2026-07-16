use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

pub fn reference_data_dir() -> PathBuf {
    if let Ok(v) = std::env::var("SYNAPTIX_TEST_DATA") {
        return PathBuf::from(v);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
}

pub fn reference_data_path(module: &str, file: &str) -> PathBuf {
    reference_data_dir().join(module).join(file)
}

pub fn load_safetensors(path: &Path) -> HashMap<String, Tensor> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("load_safetensors: не могу прочитать {:?}: {}", path, e));
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("load_safetensors: ошибка парсинга {:?}: {}", path, e));
    let mut map = HashMap::new();
    for (name, view) in st.tensors() {
        let dtype = st_dtype_to_synaptix(view.dtype());
        let shape: Vec<usize> = view.shape().to_vec();
        let data: Vec<u8> = view.data().to_vec();
        let tensor = Tensor::from_raw_bytes(data, shape, dtype, Device::Cpu)
            .unwrap_or_else(|e| panic!("load_safetensors: ошибка создания тензора '{}': {}", name, e));
        map.insert(name.to_string(), tensor);
    }
    map
}

pub fn load_case(module: &str, case: &str) -> HashMap<String, Tensor> {
    let path = reference_data_path(module, &format!("{}.safetensors", case));
    load_safetensors(&path)
}

pub fn assert_allclose(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) {
    assert_eq!(
        a.shape().dims(),
        b.shape().dims(),
        "assert_allclose: несовпадение shape: {:?} vs {:?}",
        a.shape().dims(),
        b.shape().dims()
    );
    assert_eq!(
        a.dtype(),
        b.dtype(),
        "assert_allclose: несовпадение dtype: {:?} vs {:?}",
        a.dtype(),
        b.dtype()
    );
    let a_vals = tensor_to_f64_flat(a);
    let b_vals = tensor_to_f64_flat(b);
    let n = a_vals.len();
    let mut max_diff = 0.0_f64;
    let mut fail_count = 0usize;
    let mut first_fails: Vec<(usize, f64, f64, f64)> = Vec::new();
    for i in 0..n {
        let av = a_vals[i];
        let bv = b_vals[i];
        let diff = (av - bv).abs();
        let tol = atol + rtol * bv.abs();
        if diff > tol || av.is_nan() || bv.is_nan() {
            fail_count += 1;
            if first_fails.len() < 5 {
                first_fails.push((i, av, bv, diff));
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    if fail_count > 0 {
        let mean_diff: f64 = a_vals.iter().zip(b_vals.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / n as f64;
        let mut msg = format!(
            "assert_allclose FAIL: {} элементов из {} не совпали\n  shape={:?} dtype={:?}\n  atol={} rtol={}\n  max_abs_diff={:.3e}  mean_abs_diff={:.3e}\n  первые несовпадения:",
            fail_count, n, a.shape().dims(), a.dtype(), atol, rtol, max_diff, mean_diff
        );
        for (idx, av, bv, diff) in &first_fails {
            msg.push_str(&format!("\n    [{}]: a={:.6e}  b={:.6e}  diff={:.3e}", idx, av, bv, diff));
        }
        panic!("{}", msg);
    }
}

pub fn assert_exact_eq(a: &Tensor, b: &Tensor) {
    assert_eq!(
        a.shape().dims(),
        b.shape().dims(),
        "assert_exact_eq: несовпадение shape"
    );
    assert_eq!(
        a.dtype(),
        b.dtype(),
        "assert_exact_eq: несовпадение dtype"
    );
    let a_vals = tensor_to_i64_flat(a);
    let b_vals = tensor_to_i64_flat(b);
    let n = a_vals.len();
    let mut fail_count = 0usize;
    let mut first_fails: Vec<(usize, i64, i64)> = Vec::new();
    for i in 0..n {
        if a_vals[i] != b_vals[i] {
            fail_count += 1;
            if first_fails.len() < 5 {
                first_fails.push((i, a_vals[i], b_vals[i]));
            }
        }
    }
    if fail_count > 0 {
        let mut msg = format!(
            "assert_exact_eq FAIL: {} элементов из {} не совпали\n  shape={:?} dtype={:?}\n  первые несовпадения:",
            fail_count, n, a.shape().dims(), a.dtype()
        );
        for (idx, av, bv) in &first_fails {
            msg.push_str(&format!("\n    [{}]: a={}  b={}", idx, av, bv));
        }
        panic!("{}", msg);
    }
}

pub fn assert_no_nan(t: &Tensor) {
    let vals = tensor_to_f64_flat(t);
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "assert_no_nan: обнаружено {} NaN в тензоре shape={:?}", nan_count, t.shape().dims());
}

pub fn assert_no_inf(t: &Tensor) {
    let vals = tensor_to_f64_flat(t);
    let inf_count = vals.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(inf_count, 0, "assert_no_inf: обнаружено {} Inf в тензоре shape={:?}", inf_count, t.shape().dims());
}

fn st_dtype_to_synaptix(d: safetensors::Dtype) -> DType {
    match d {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::BOOL => DType::U8,
        _ => panic!("st_dtype_to_synaptix: неподдерживаемый dtype {:?}", d),
    }
}

fn tensor_to_f64_flat(t: &Tensor) -> Vec<f64> {
    let t = t.contiguous().expect("tensor_to_f64_flat: contiguous failed");
    let flat = t.flatten_all().expect("tensor_to_f64_flat: flatten_all failed");
    match flat.dtype() {
        DType::F64 => flat.to_vec1::<f64>().unwrap(),
        DType::F32 => flat.to_vec1::<f32>().unwrap().into_iter().map(|x| x as f64).collect(),
        DType::F16 => flat.to_vec1::<f16>().unwrap().into_iter().map(|x| x.to_f64()).collect(),
        DType::BF16 => flat.to_vec1::<bf16>().unwrap().into_iter().map(|x| x.to_f64()).collect(),
        DType::I64 => flat.to_vec1::<i64>().unwrap().into_iter().map(|x| x as f64).collect(),
        DType::I32 => flat.to_vec1::<i32>().unwrap().into_iter().map(|x| x as f64).collect(),
        DType::U32 => flat.to_vec1::<u32>().unwrap().into_iter().map(|x| x as f64).collect(),
        DType::U8 => flat.to_vec1::<u8>().unwrap().into_iter().map(|x| x as f64).collect(),
        d => panic!("tensor_to_f64_flat: неподдерживаемый dtype {:?}", d),
    }
}

fn tensor_to_i64_flat(t: &Tensor) -> Vec<i64> {
    let t = t.contiguous().expect("tensor_to_i64_flat: contiguous failed");
    let flat = t.flatten_all().expect("tensor_to_i64_flat: flatten_all failed");
    match flat.dtype() {
        DType::I64 => flat.to_vec1::<i64>().unwrap(),
        DType::I32 => flat.to_vec1::<i32>().unwrap().into_iter().map(|x| x as i64).collect(),
        DType::U32 => flat.to_vec1::<u32>().unwrap().into_iter().map(|x| x as i64).collect(),
        DType::U8 => flat.to_vec1::<u8>().unwrap().into_iter().map(|x| x as i64).collect(),
        DType::F32 => flat.to_vec1::<f32>().unwrap().into_iter().map(|x| x as i64).collect(),
        DType::F64 => flat.to_vec1::<f64>().unwrap().into_iter().map(|x| x as i64).collect(),
        d => panic!("tensor_to_i64_flat: неподдерживаемый dtype {:?}", d),
    }
}
