//! Извлечь .syn-бандл в нативные safetensors + встроенные файлы (config/tokenizer).
//! Использование: extract_syn <bundle.syn> <out_dir> [safetensors_basename]
//! tensors → <out_dir>/<basename>.safetensors (дефолт model). Файлы → как есть.

use std::fs;
use synaptix_bundle::Bundle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: extract_syn <bundle.syn> <out_dir> [safetensors_basename]");
        std::process::exit(1);
    }
    let path = &args[1];
    let outdir = &args[2];
    let base = args.get(3).cloned().unwrap_or_else(|| "model".into());
    fs::create_dir_all(outdir)?;

    let bundle = Bundle::open(path)?;
    let meta = bundle.meta();
    println!("bundle id={} ver={:?}", bundle.id(), bundle.version());
    println!("components: {:?}", meta.components.keys().collect::<Vec<_>>());

    // тензоры (встроенный safetensors-блоб)
    if meta.components.is_empty() {
        let slice = bundle.tensors_slice()?;
        let out = format!("{outdir}/{base}.safetensors");
        fs::write(&out, slice)?;
        println!("  tensors → {out} ({} bytes)", slice.len());
    } else {
        for comp in meta.components.keys() {
            let (slice, _) = bundle.tensors_slice_for(comp)?;
            let name = if meta.components.len() == 1 { base.clone() } else { comp.clone() };
            let out = format!("{outdir}/{name}.safetensors");
            fs::write(&out, slice)?;
            println!("  tensors[{comp}] → {out} ({} bytes)", slice.len());
        }
    }

    // встроенные файлы (config.json, tokenizer.json, ...)
    for e in bundle.list_files() {
        // пропускаем HF-download кэш-мусор (.lock/.metadata)
        if e.name.contains("/download/") || e.name.ends_with(".lock") {
            continue;
        }
        let bytes = bundle.read_file(&e.name)?;
        let out = format!("{outdir}/{}", e.name);
        if let Some(parent) = std::path::Path::new(&out).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, &bytes)?;
        println!("  file → {out} ({} bytes)", bytes.len());
    }
    println!("done.");
    Ok(())
}
