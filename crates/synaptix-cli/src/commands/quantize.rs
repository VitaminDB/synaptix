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
    pub calib: Option<PathBuf>,
    pub include: Vec<String>,
    pub vision: Option<String>,
    pub rule: Vec<String>,
    pub policy: Option<String>,
}

/// Поле `config.json` бандла (в т. ч. внутри `text_config`).
fn config_u64(path: &std::path::Path, key: &str) -> Option<u64> {
    let bytes = synaptix_io::weights::syn_bundle::read_model_file(path, "config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let get = |v: &serde_json::Value| v.get(key).and_then(|n| n.as_u64());
    get(&v).or_else(|| v.get("text_config").and_then(get))
}

fn num_layers(path: &std::path::Path) -> Option<usize> {
    config_u64(path, "num_hidden_layers").map(|n| n as usize)
}


/// Пресет `llm`: правила крайних слоёв (первый и последний), общих экспертов
/// и таблиц позиций.
fn llm_rules(path: &std::path::Path) -> Result<Vec<(String, Option<DType>)>, String> {
    let n = num_layers(path).ok_or("--policy llm: нет num_hidden_layers в config.json")?;
    let m = Some(DType::MXFP8);
    Ok(vec![
        (".layers.0.".into(), m),
        (format!(".layers.{}.", n - 1), m),
        ("shared_expert".into(), m),
        // таблицы, из которых читают строки, — как есть
        ("pos_embed".into(), None),
    ])
}

fn fmt(s: &str, what: &str) -> Result<Option<DType>, String> {
    match s.to_ascii_lowercase().as_str() {
        "none" | "keep" | "-" => Ok(None),
        other => match parse_dtype(other) {
            Some(d) if d.is_quantized() && !matches!(d, DType::Ggml(_)) => Ok(Some(d)),
            _ => Err(format!("{what}: `{s}` — ожидается nvfp4 | mxfp8 | sq1…sq8 | q8row | none")),
        },
    }
}

pub fn run(args: QuantizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.input.is_dir() {
        return Err("каталог HF не поддержан: сперва `synaptix convert <dir> model.syn`, затем quantize".into());
    }
    let llm = match args.policy.as_deref() {
        None => false,
        Some("llm") => true,
        Some(p) => return Err(format!("--policy {p}: известен только `llm`").into()),
    };
    let main = if llm { DType::NVFP4 } else { fmt(&args.format, "--format")?.ok_or("--format обязан быть форматом кванта")? };
    let mx = || Some("mxfp8".to_string());
    let (attn_a, head_a, embed_a, vision_a) = if llm {
        // Качество важнее «влезет ли»: размещение — дело движка (оффлоад).
        (args.attn.clone().or_else(mx), args.lm_head.clone().or_else(mx), args.embed.clone().or_else(mx), args.vision.clone().or_else(mx))
    } else {
        (args.attn.clone(), args.lm_head.clone(), args.embed.clone(), args.vision.clone())
    };
    let args = QuantizeArgs { attn: attn_a, lm_head: head_a, embed: embed_a, vision: vision_a, ..args };
    let mut rules: Vec<(String, Option<DType>)> = Vec::new();
    for r in &args.rule {
        let (p, f) = r.rsplit_once('=').ok_or_else(|| format!("--rule `{r}`: ожидается ПОДСТРОКА=ФОРМАТ"))?;
        rules.push((p.to_string(), fmt(f, "--rule")?));
    }
    if llm {
        rules.extend(llm_rules(&args.input)?);
    }
    let vision = match &args.vision {
        Some(v) => fmt(v, "--vision")?,
        None => None,
    };
    let attn = match &args.attn {
        Some(a) => fmt(a, "--attn")?,
        None => Some(main),
    };
    let q8 = main == DType::Q8Row;
    let lm_head = match &args.lm_head {
        Some(a) => fmt(a, "--lm-head")?,
        None if q8 => None,
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
    let experts = if q8 { None } else { Some(main) };
    let spec = TranscodeSpec { attn, mlp: Some(main), experts, lm_head, embed, requant: !args.keep_quant, quantize_dense: true, include: args.include.clone(), vision, rules };
    let device = if args.device.starts_with("cuda") {
        let ord = args.device.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        Device::Cuda(ord)
    } else if q8 {
        Device::Cpu
    } else {
        return Err("квантование считается на карте: --device cuda[:N]".into());
    };
    synaptix::init()?;
    if let Some(c) = &args.calib {
        let t = synaptix_core::quant::calib::load(c)?;
        println!("  калибровка: {} ({} тензоров)", c.display(), t.len());
        synaptix_core::quant::calib::set_table(Some(t));
    }
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
