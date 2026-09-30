use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_embedding_bge_m3::{BgeM3, BgeReranker};

use crate::commands::device;

pub struct EmbedArgs {
    pub model: PathBuf,
    pub texts: Vec<String>,
    pub file: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub max_tokens: Option<usize>,
    pub batch_size: usize,
    pub device: String,
    pub compute_dtype: Option<String>,
}

pub struct RerankArgs {
    pub model: PathBuf,
    pub query: String,
    pub docs: Vec<String>,
    pub file: Option<PathBuf>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub json: bool,
    pub device: String,
    pub compute_dtype: Option<String>,
}

fn dtype_for(dev: Device, name: Option<&str>) -> Result<DType, String> {
    match name {
        Some("f16") => Ok(DType::F16),
        Some("bf16") => Ok(DType::BF16),
        Some("f32") => Ok(DType::F32),
        None if dev == Device::Cpu => Ok(DType::F32),
        None => Ok(DType::F16),
        Some(o) => Err(format!("unknown compute-dtype {o}")),
    }
}

fn gather(inline: Vec<String>, file: &Option<PathBuf>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut all = inline;
    if let Some(p) = file {
        let body = if p.as_os_str() == "-" { std::io::read_to_string(std::io::stdin())? } else { std::fs::read_to_string(p)? };
        all.extend(body.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
    }
    if all.is_empty() {
        return Err("нет текстов: передайте аргументами или --file (строка — текст; - — stdin)".into());
    }
    Ok(all)
}

pub fn run_embed(args: EmbedArgs) -> Result<(), Box<dyn std::error::Error>> {
    let texts = gather(args.texts, &args.file)?;
    let dev = device::resolve(&args.device);
    let dtype = dtype_for(dev, args.compute_dtype.as_deref())?;
    let t0 = std::time::Instant::now();
    let mut model = if args.model.is_dir() { BgeM3::from_unpacked(&args.model, &dev, dtype)? } else { BgeM3::from_syn(&args.model, &dev, dtype)? };
    if let Some(n) = args.max_tokens {
        model.set_max_tokens(n);
    }
    let mut vectors = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(args.batch_size.max(1)) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        vectors.extend(model.encode(&refs)?);
    }
    eprintln!(
        "synaptix embed: {} текстов × {} за {:.2}s",
        vectors.len(),
        model.dim(),
        t0.elapsed().as_secs_f32()
    );
    let rows: Vec<serde_json::Value> = texts
        .iter()
        .zip(&vectors)
        .map(|(t, v)| serde_json::json!({"text": t, "embedding": v}))
        .collect();
    let out = serde_json::to_string(&rows)? + "\n";
    match &args.output {
        Some(p) => {
            std::fs::write(p, out)?;
            eprintln!("synaptix embed: wrote {}", p.display());
        }
        None => print!("{out}"),
    }
    Ok(())
}

pub fn run_rerank(args: RerankArgs) -> Result<(), Box<dyn std::error::Error>> {
    let docs = gather(args.docs, &args.file)?;
    let dev = device::resolve(&args.device);
    let dtype = dtype_for(dev, args.compute_dtype.as_deref())?;
    let mut model = if args.model.is_dir() {
        BgeReranker::from_unpacked(&args.model, dev, dtype)?
    } else {
        BgeReranker::from_syn(&args.model, dev, dtype)?
    };
    if let Some(n) = args.max_tokens {
        model.set_max_tokens(n.max(64));
    }
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let ranked = model.rerank(&args.query, &refs, args.top_k.unwrap_or(docs.len()))?;
    if args.json {
        let rows: Vec<serde_json::Value> = ranked
            .iter()
            .map(|(i, s)| serde_json::json!({"index": i, "score": s, "text": docs[*i]}))
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        for (i, s) in &ranked {
            println!("{s:>8.4}  [{i}] {}", docs[*i]);
        }
    }
    Ok(())
}
