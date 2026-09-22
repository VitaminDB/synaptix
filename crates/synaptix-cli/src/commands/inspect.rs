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
        "gguf" => inspect_gguf(&args.file, args.verbose, args.filter.as_deref()),
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


/// GGUF: метаданные, типы тензоров и что из этого движок исполняет напрямую.
fn inspect_gguf(path: &Path, verbose: bool, filter: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "gguf"))]
    {
        let _ = (path, verbose, filter);
        return Err("поддержка GGUF выключена: пересоберите с `--features gguf`".into());
    }
    #[cfg(feature = "gguf")]
    {
        use std::collections::BTreeMap;
        let f = synaptix_gguf::GgufFile::open(path)?;
        println!("=== GGUF: {} ===", path.display());
        println!("  version:      {}", f.version);
        println!("  architecture: {}", f.architecture().unwrap_or("?"));
        println!("  tensors:      {}", f.tensors().len());
        let mut by_type: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
        for t in f.tensors() {
            let e = by_type.entry(t.ty.name()).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.byte_len() as u64;
        }
        println!("  by type:");
        for (ty, (n, bytes)) in &by_type {
            println!("    {ty:<8} {n:>5} тензоров  {:.2} ГБ", *bytes as f64 / 1e9);
        }
        let mmproj = None::<&synaptix_gguf::GgufFile>;
        match synaptix_gguf::arch::build_plan(&f, mmproj, "inspect") {
            Ok(plan) => {
                let cfg: serde_json::Value = plan
                    .files
                    .iter()
                    .find(|x| x.path == "config.json")
                    .and_then(|x| serde_json::from_slice(&x.bytes).ok())
                    .unwrap_or_default();
                println!("  engine:       model_type={} (маппер {})", cfg["model_type"].as_str().unwrap_or("?"), plan.arch);
                println!("  hf tensors:   {}", plan.tensor_count());
                println!("  files:        {}", plan.files.iter().map(|x| x.path.as_str()).collect::<Vec<_>>().join(", "));
            }
            Err(e) => println!("  engine:       не исполняется напрямую — {e}"),
        }
        println!("  metadata:");
        let mut keys: Vec<&String> = f.metadata.keys().collect();
        keys.sort();
        for k in keys {
            if !matches_filter(k, filter) {
                continue;
            }
            let v = &f.metadata[k];
            let shown = match v {
                synaptix_gguf::Value::Array(a) if a.len() > 8 && !verbose => format!("<{} items>", a.len()),
                synaptix_gguf::Value::String(s) if s.len() > 120 && !verbose => format!("{:?}…", &s[..s.char_indices().nth(120).map(|(i, _)| i).unwrap_or(s.len())]),
                other => format!("{other:?}"),
            };
            println!("    {k} = {shown}");
        }
        if verbose {
            println!("  tensors:");
            for t in f.tensors() {
                if matches_filter(&t.name, filter) {
                    println!("    {:<48} {:<8} {:?}", t.name, t.ty.name(), t.hf_shape());
                }
            }
        }
        Ok(())
    }
}
