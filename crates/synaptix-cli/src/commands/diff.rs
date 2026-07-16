use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

pub struct DiffArgs {
    pub file_a: PathBuf,
    pub file_b: PathBuf,
    pub atol: f32,
    pub rtol: f32,
}

impl Default for DiffArgs {
    fn default() -> Self {
        Self { file_a: PathBuf::new(), file_b: PathBuf::new(), atol: 1e-4, rtol: 1e-3 }
    }
}

pub fn run(args: DiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.file_a.exists() {
        return Err(format!("file_a not found: {}", args.file_a.display()).into());
    }
    if !args.file_b.exists() {
        return Err(format!("file_b not found: {}", args.file_b.display()).into());
    }
    let bytes_a = std::fs::read(&args.file_a)?;
    let bytes_b = std::fs::read(&args.file_b)?;
    let a = SafeTensors::deserialize(&bytes_a).map_err(|e| format!("a: {e}"))?;
    let b = SafeTensors::deserialize(&bytes_b).map_err(|e| format!("b: {e}"))?;

    let names_a: BTreeSet<&str> = a.names().into_iter().collect();
    let names_b: BTreeSet<&str> = b.names().into_iter().collect();
    let only_a: Vec<&&str> = names_a.difference(&names_b).collect();
    let only_b: Vec<&&str> = names_b.difference(&names_a).collect();

    println!("diff {} vs {}", args.file_a.display(), args.file_b.display());
    if !only_a.is_empty() {
        println!("  only in A ({}): {:?}", only_a.len(), only_a.iter().take(5).collect::<Vec<_>>());
    }
    if !only_b.is_empty() {
        println!("  only in B ({}): {:?}", only_b.len(), only_b.iter().take(5).collect::<Vec<_>>());
    }

    let common: Vec<&str> = names_a.intersection(&names_b).copied().collect();
    println!("  common tensors: {}", common.len());

    let mut max_abs_global = 0.0_f32;
    let mut min_cos_global = 1.0_f32;
    let mut over_tol = Vec::new();
    for name in &common {
        let ta = a.tensor(name).map_err(|e| format!("a[{name}]: {e}"))?;
        let tb = b.tensor(name).map_err(|e| format!("b[{name}]: {e}"))?;
        if ta.shape() != tb.shape() {
            println!("  shape mismatch '{name}': {:?} vs {:?}", ta.shape(), tb.shape());
            continue;
        }
        let va = view_to_f32(&ta, &args.file_a)?;
        let vb = view_to_f32(&tb, &args.file_b)?;
        let (max_abs, cos) = compare_f32(&va, &vb);
        if max_abs > max_abs_global {
            max_abs_global = max_abs;
        }
        if cos < min_cos_global {
            min_cos_global = cos;
        }
        let tol = args.atol + args.rtol * va.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs > tol {
            over_tol.push((name.to_string(), max_abs, cos, tol));
        }
    }

    println!("  global max_abs: {:.6}", max_abs_global);
    println!("  global min_cos: {:.6}", min_cos_global);
    println!("  tensors over tolerance: {}", over_tol.len());
    for (n, ma, c, t) in over_tol.iter().take(10) {
        println!("    {n}: max_abs={ma:.4e} cos={c:.6} tol={t:.4e}");
    }
    Ok(())
}

fn view_to_f32(tv: &safetensors::tensor::TensorView<'_>, path: &Path) -> Result<Vec<f32>, String> {
    let dtype = match tv.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F64 => DType::F64,
        other => return Err(format!("{}: unsupported dtype {other:?}", path.display())),
    };
    let bytes = tv.data().to_vec();
    let t = Tensor::from_raw_bytes(bytes, tv.shape().to_vec(), dtype, synaptix_core::device::Device::Cpu)
        .map_err(|e| e.to_string())?;
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())
}

fn compare_f32(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    let cos = (dot / (na.sqrt() * nb.sqrt()).max(1.0e-30)) as f32;
    (max_abs, cos.clamp(-1.0, 1.0))
}
