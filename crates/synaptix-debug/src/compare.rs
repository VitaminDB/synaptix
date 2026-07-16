use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_core::tensor::storage::Storage;

use crate::dump::TensorDump;
use crate::error::{DebugError, Result};

#[derive(Debug, Clone, Copy)]
pub struct CompareReport {
    pub cos_sim: f64,
    pub max_abs: f64,
    pub rel_err: f64,
    pub l1: f64,
    pub l2: f64,
    pub numel: usize,
}

pub fn cos_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

pub fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let eps = 1e-12f64;
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let scale = x.abs().max(y.abs()).max(eps);
            (x - y).abs() / scale
        })
        .fold(0.0, f64::max)
}

pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

pub fn compare_slices(a: &[f64], b: &[f64]) -> CompareReport {
    CompareReport {
        cos_sim: cos_sim(a, b),
        max_abs: max_abs(a, b),
        rel_err: rel_err(a, b),
        l1: l1_distance(a, b),
        l2: l2_distance(a, b),
        numel: a.len(),
    }
}

pub fn compare_tensors(a: &Tensor, b: &Tensor) -> Result<CompareReport> {
    if a.dims() != b.dims() {
        return Err(DebugError::ShapeMismatch {
            expected: a.dims().to_vec(),
            got: b.dims().to_vec(),
        });
    }
    let av = tensor_to_f64(a)?;
    let bv = tensor_to_f64(b)?;
    Ok(compare_slices(&av, &bv))
}

pub fn compare_dumps(a: &TensorDump, b: &TensorDump) -> Result<CompareReport> {
    if a.dims != b.dims {
        return Err(DebugError::ShapeMismatch { expected: a.dims.clone(), got: b.dims.clone() });
    }
    let av = dump_to_f64(a)?;
    let bv = dump_to_f64(b)?;
    Ok(compare_slices(&av, &bv))
}

pub fn tensor_to_f64(t: &Tensor) -> Result<Vec<f64>> {
    let contig = t.contiguous()?;
    let dtype = contig.dtype();
    let storage = contig.storage();
    let Storage::Cpu(buf) = storage else {
        return Err(DebugError::Other("compare: non-CPU storage".into()));
    };
    let off = contig.layout().byte_offset();
    let bytes = &buf.as_bytes()[off..];
    let numel = contig.numel();
    let expected_len = dtype.bytes_for_numel(numel);
    decode_to_f64(&bytes[..expected_len], dtype, numel)
}

pub fn dump_to_f64(d: &TensorDump) -> Result<Vec<f64>> {
    decode_to_f64(&d.data, d.dtype, d.numel())
}

fn decode_to_f64(bytes: &[u8], dtype: DType, numel: usize) -> Result<Vec<f64>> {
    match dtype {
        DType::F32 => Ok((0..numel)
            .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as f64)
            .collect()),
        DType::F64 => Ok((0..numel)
            .map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect()),
        DType::F16 => Ok((0..numel)
            .map(|i| half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f64())
            .collect()),
        DType::BF16 => Ok((0..numel)
            .map(|i| half::bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f64())
            .collect()),
        DType::U8 => Ok(bytes.iter().take(numel).map(|&b| b as f64).collect()),
        DType::U32 => Ok((0..numel)
            .map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as f64)
            .collect()),
        DType::I32 => Ok((0..numel)
            .map(|i| i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as f64)
            .collect()),
        DType::I64 => Ok((0..numel)
            .map(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()) as f64)
            .collect()),
        _ => Err(DebugError::Other(format!(
            "compare: dtype {dtype:?} not yet supported (quantized)"
        ))),
    }
}
