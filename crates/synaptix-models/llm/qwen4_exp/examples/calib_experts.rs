//! Калибровка квантования экспертов (то же, что `synaptix calibrate <bundle> <текст.txt> -o <файл>`):
//!
//!   calib_experts <bundle.syn> <текст.txt> <calib.safetensors> [токенов, 32768]
//!
//! Окно и кусок — `CALIB_WINDOW` (4096), `CALIB_CHUNK` (1024), вес априорной оценки — `CALIB_PRIOR` (64).

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_llm_qwen4_exp::calibrate::{calibrate_experts, CalibOpts};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: calib_experts <bundle.syn> <текст> <calib.safetensors> [токенов, 32768]".into());
    }
    let text = String::from_utf8_lossy(&std::fs::read(&a[2])?).into_owned();
    let out = PathBuf::from(&a[3]);
    let opts = CalibOpts {
        tokens: a.get(4).and_then(|s| s.parse().ok()).unwrap_or(32768),
        window: env("CALIB_WINDOW", 4096),
        chunk: env("CALIB_CHUNK", 1024),
        prior: env("CALIB_PRIOR", 64),
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let r = calibrate_experts(&PathBuf::from(&a[1]), &text, &out, Device::Cuda(0), &opts, &|done, n| eprint!("\r  {done}/{n}"))?;
    eprintln!();
    println!(
        "калибровка: {} — {} матриц экспертов, из них {} видели меньше 64 строк; матриц связей входов {}; {} токенов за {:.0} с",
        out.display(), r.matrices, r.rare, r.grams, r.tokens, r.seconds
    );
    Ok(())
}
