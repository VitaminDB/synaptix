//! syn-unpack — restore an in-bundle directory tree to disk.
//!
//! Inverse of syn-pack: writes every alive File chunk back to its in-bundle
//! path under the output directory, plus the embedded safetensors stream as
//! `model.safetensors`.

use std::path::PathBuf;

use synaptix_bundle::{Bundle, ChunkType};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: syn-unpack <bundle.syn> -o <dir>"))?
        .into();
    let mut out_dir: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--output" => out_dir = Some(PathBuf::from(args.next().unwrap())),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let out_dir = out_dir.ok_or_else(|| anyhow::anyhow!("missing -o <dir>"))?;
    std::fs::create_dir_all(&out_dir)?;

    let b = Bundle::open(&path)?;
    for e in b.cdir().entries.iter().filter(|e| e.is_alive()) {
        match e.kind_typed() {
            ChunkType::Tensors => {
                let comp = e.name.rsplit(':').next().unwrap_or("");
                let fname = if comp.is_empty() || comp == "main" {
                    "model.safetensors".to_string()
                } else {
                    format!("model-{comp}.safetensors")
                };
                let dst = out_dir.join(&fname);
                let bytes = b.read_raw_chunk(e)?;
                std::fs::write(&dst, &*bytes)?;
                eprintln!("  > {fname} ({} bytes)", bytes.len());
            }
            ChunkType::File => {
                let dst = out_dir.join(&e.name);
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let bytes = b.read_file(&e.name)?;
                std::fs::write(&dst, &*bytes)?;
                eprintln!("  > {} ({} bytes)", e.name, bytes.len());
            }
            _ => {}
        }
    }
    eprintln!("unpacked → {}", out_dir.display());
    Ok(())
}
