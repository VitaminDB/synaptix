use std::path::PathBuf;

use synaptix_bundle::BundleBuilder;

pub struct ConvertArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    pub format: String,
    pub arch: Option<String>,
    pub component: Option<String>,
    pub mmproj: Option<PathBuf>,
    pub dtype: String,
    pub tokenizer: Option<PathBuf>,
    pub id: Option<String>,
    pub sha256: bool,
    pub blake3: bool,
}

pub fn run(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    let from_ext = args.input.extension().and_then(|e| e.to_str()).unwrap_or("");
    let to_ext = args.output.extension().and_then(|e| e.to_str()).unwrap_or("");
    match (from_ext, to_ext) {
        ("safetensors", "syn") => convert_safetensors_to_syn(&args),
        ("gguf", "syn") => convert_gguf_to_syn(&args),
        _ => Err(format!(
            "unsupported conversion: {} → {} (supported: safetensors → syn, gguf → syn)",
            from_ext, to_ext
        ).into()),
    }
}

#[cfg(not(feature = "gguf"))]
fn convert_gguf_to_syn(_args: &ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    Err("поддержка GGUF выключена: пересоберите с `--features gguf`".into())
}

#[cfg(feature = "gguf")]
fn convert_gguf_to_syn(args: &ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use synaptix_gguf::{convert_to_syn, ConvertOptions, OutDtype};

    let dtype = match args.dtype.to_ascii_lowercase().as_str() {
        "auto" => OutDtype::Auto,
        "f16" | "fp16" | "half" => OutDtype::F16,
        "bf16" => OutDtype::BF16,
        "f32" | "fp32" => OutDtype::F32,
        other => return Err(format!("неизвестный --dtype `{other}` (auto|f16|bf16|f32)").into()),
    };

    let opts = ConvertOptions {
        dtype,
        bundle_id: args.id.clone(),
        mmproj: args.mmproj.clone(),
        tokenizer_json: args.tokenizer.clone(),
        extra_files: Vec::new(),
        sha256: args.sha256,
        blake3: args.blake3,
    };

    println!("Converting {} → {}", args.input.display(), args.output.display());
    if let Some(m) = &args.mmproj {
        println!("  mmproj:     {}", m.display());
    }
    println!("  dtype:      {}", args.dtype);

    let total = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicU64::new(0));
    let last = Arc::new(AtomicU64::new(0));
    let (t, d, l) = (total.clone(), done.clone(), last.clone());
    let cb: synaptix_bundle::ProgressCallback = Arc::new(move |ev| {
        use synaptix_bundle::ProgressEvent as E;
        match ev {
            E::Plan { total_bytes, total_items, payload_bytes } => {
                t.store(total_bytes.max(1), Ordering::Relaxed);
                println!(
                    "  план: {total_items} чанков, payload {:.2} ГБ",
                    payload_bytes as f64 / 1e9
                );
            }
            E::ItemStart { index, name, bytes } => {
                println!("  [{index}] {name} — {:.2} ГБ", bytes as f64 / 1e9);
            }
            E::Bytes { delta } => {
                let cur = d.fetch_add(delta, Ordering::Relaxed) + delta;
                let tot = t.load(Ordering::Relaxed);
                let pct = cur * 100 / tot.max(1);
                if pct > l.load(Ordering::Relaxed) {
                    l.store(pct, Ordering::Relaxed);
                    print!("\r      {pct}% ({:.1}/{:.1} ГБ)", cur as f64 / 1e9, tot as f64 / 1e9);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            E::ItemDone { .. } => println!(),
            E::Finalizing => println!("  финализация..."),
            E::Done => {}
        }
    });

    let report = convert_to_syn(&args.input, &args.output, &opts, Some(cb))?;
    println!("Done: {}", report.output.display());
    println!("  arch:       {}", report.arch);
    println!("  bundle_id:  {}", report.bundle_id);
    for (name, n) in &report.components {
        println!("  component {name}: {n} тензоров");
    }
    println!("  files:      {}", report.files.join(", "));
    println!("  payload:    {:.2} ГБ", report.payload_bytes as f64 / 1e9);
    Ok(())
}

fn convert_safetensors_to_syn(args: &ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    let id = args.id.clone().unwrap_or_else(|| {
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("converted")
            .to_string()
    });
    let arch = args.arch.clone().unwrap_or_else(|| "unknown".into());
    let component = args.component.clone().unwrap_or_else(|| "main".into());

    println!("Converting {} → {}", args.input.display(), args.output.display());
    println!("  bundle_id:  {}", id);
    println!("  arch:       {}", arch);
    println!("  component:  {}", component);

    let builder = BundleBuilder::new(id, "0.1.0")
        .arch(arch)
        .component(component.clone(), "")
        .add_safetensors_component(&component, vec![args.input.clone()], None);

    let count = builder.item_count();
    println!("  tensors to pack: {}", count);
    println!("Writing bundle...");
    builder.write(&args.output)?;
    println!("Done: {}", args.output.display());
    Ok(())
}
