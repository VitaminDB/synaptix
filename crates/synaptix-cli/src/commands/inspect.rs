use std::path::{Path, PathBuf};

use synaptix_bundle::Bundle;

pub struct InspectArgs {
    pub file: PathBuf,
    pub verbose: bool,
    pub filter: Option<String>,
}

pub fn run(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ext = args.file.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "syn" => inspect_syn(&args.file, args.verbose, args.filter.as_deref()),
        "safetensors" => inspect_safetensors(&args.file, args.verbose, args.filter.as_deref()),
        "gguf" => Err("GGUF не поддерживается synaptix; используйте `.syn` или конвертируйте через llama.cpp tools".into()),
        _ => Err(format!("unknown format: {ext}").into()),
    }
}

fn matches_filter(name: &str, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(f) => name.contains(f),
    }
}

fn inspect_syn(path: &Path, verbose: bool, filter: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::open(path)?;
    println!("=== Synaptix Bundle: {} ===", path.display());
    let (vmajor, vminor) = bundle.version();
    println!("  bundle_id: {}", bundle.id());
    println!("  version:   v{}.{}", vmajor, vminor);
    println!("  size:      {} bytes", bundle.size());
    let cdir = bundle.cdir();
    println!("  chunks:    {}", cdir.entries.len());
    if !cdir.bundle_meta.components.is_empty() {
        println!("  components:");
        for name in cdir.bundle_meta.components.keys() {
            println!("    - {}", name);
        }
    }
    println!();
    let mut shown = 0usize;
    let total = cdir.entries.len();
    for entry in cdir.entries.iter() {
        if !matches_filter(&entry.name, filter) {
            continue;
        }
        if verbose {
            println!("  [{:>3}] type={:?} name={} raw_len={} payload_len={} flags=0x{:04x}",
                entry.id, entry.kind_typed(), entry.name, entry.raw_len, entry.payload_len, entry.flags);
        } else {
            println!("  [{:>3}] {:?} {} ({} bytes)",
                entry.id, entry.kind_typed(), entry.name, entry.raw_len);
        }
        shown += 1;
    }
    println!();
    println!("  shown {} of {} chunks", shown, total);
    Ok(())
}

fn inspect_safetensors(path: &Path, verbose: bool, filter: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| format!("safetensors parse: {e}"))?;
    println!("=== SafeTensors: {} ===", path.display());
    let names = st.names();
    let total = names.len();
    println!("  total tensors: {}", total);
    println!();
    let mut shown = 0usize;
    let mut total_bytes = 0u64;
    for (name, view) in st.tensors() {
        total_bytes += view.data().len() as u64;
        if !matches_filter(name.as_str(), filter) {
            continue;
        }
        let shape: Vec<String> = view.shape().iter().map(|s| s.to_string()).collect();
        if verbose {
            println!("  {} dtype={:?} shape=[{}] bytes={}",
                name, view.dtype(), shape.join(","), view.data().len());
        } else {
            println!("  {:<60} {:?} [{}]", name, view.dtype(), shape.join(","));
        }
        shown += 1;
    }
    println!();
    println!("  shown {} of {} tensors, total {} bytes ({:.2} MB)",
        shown, total, total_bytes, total_bytes as f64 / 1024.0 / 1024.0);
    Ok(())
}

