//! `synaptix quantize` — квантование и перекодировка модели в `.syn`.
//!
//! Источник — `.syn` (плотный или уже квантованный NVFP4/MXFP8/SQ) или
//! `.gguf` (блоки ggml). Матрицы attn/mlp/экспертов/головы деквантуются на
//! карте в F16 и кодируются целевым форматом (`Tensor::quantize_to`),
//! остальные тензоры и файлы копируются как есть. Двойной квант из уже
//! квантованного источника — осознанный выбор (`--keep-quant` его отключает).

use std::path::PathBuf;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::parse_dtype;
use synaptix_io::weights::transcode::TranscodeSpec;

pub struct QuantizeArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    pub format: String,
    pub attn: Option<String>,
    pub lm_head: Option<String>,
    pub embed: Option<String>,
    pub keep_quant: bool,
    pub device: String,
}

fn fmt(s: &str, what: &str) -> Result<Option<DType>, String> {
    match s.to_ascii_lowercase().as_str() {
        "none" | "keep" | "-" => Ok(None),
        other => match parse_dtype(other) {
            Some(d) if d.is_quantized() && !matches!(d, DType::Ggml(_)) => Ok(Some(d)),
            _ => Err(format!("{what}: `{s}` — ожидается nvfp4 | mxfp8 | sq1…sq8 | none")),
        },
    }
}

pub fn run(args: QuantizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.input.is_dir() {
        return Err("каталог HF не поддержан: сперва `synaptix convert <dir> model.syn`, затем quantize".into());
    }
    let main = fmt(&args.format, "--format")?.ok_or("--format обязан быть форматом кванта")?;
    let attn = match &args.attn {
        Some(a) => fmt(a, "--attn")?,
        None => Some(main),
    };
    let lm_head = match &args.lm_head {
        Some(a) => fmt(a, "--lm-head")?,
        None => Some(main),
    };
    let embed = match &args.embed {
        Some(a) => {
            let d = fmt(a, "--embed")?;
            if matches!(d, Some(DType::NVFP4)) {
                return Err("--embed nvfp4: у NVFP4-таблицы нет gather — используйте mxfp8 или sqN".into());
            }
            d
        }
        None => None,
    };
    let spec = TranscodeSpec { attn, mlp: Some(main), experts: Some(main), lm_head, embed, requant: !args.keep_quant, quantize_dense: true };
    let device = if args.device.starts_with("cuda") {
        let ord = args.device.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        Device::Cuda(ord)
    } else {
        return Err("квантование считается на карте: --device cuda[:N]".into());
    };
    synaptix::init()?;
    let src_formats = synaptix::facade::arch::bundle_quant_formats(&args.input);
    println!("Quantize {} → {}", args.input.display(), args.output.display());
    println!("  spec:       {}", spec.key());
    if !src_formats.is_empty() {
        let s: Vec<String> = src_formats.iter().map(|(f, n)| format!("{f}×{n}")).collect();
        println!("  источник:   уже квантован ({}) — {}", s.join(", "), if args.keep_quant { "оставляем как есть" } else { "двойной квант" });
    }
    let progress: synaptix_io::weights::transcode::Progress = Arc::new(|done, total, name| {
        eprint!("\r  [{done}/{total}] {name:<70}");
    });
    let report = synaptix::facade::llm::transcode_bundle(&args.input, &args.output, &spec, device, Some(progress))?;
    eprintln!();
    println!("Done: {}", args.output.display());
    println!("  тензоров:   {} перекодировано", report.tensors);
    println!("  веса:       {:.2} → {:.2} ГБ", report.bytes_before as f64 / 1e9, report.bytes_after as f64 / 1e9);
    println!("  время:      {:.1} с", report.seconds);
    Ok(())
}
