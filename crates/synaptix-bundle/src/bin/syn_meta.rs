//! syn-meta — правка метаданных .syn-бандла на месте (без переупаковки).
//!
//! Usage:
//!     syn-meta <bundle.syn> [--arch A] [--purpose P] [--id I] [--version V] \
//!         [--component <name>:<tensor_prefix>]...

use std::path::PathBuf;

use synaptix_bundle::BundleEditor;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "syn-meta <bundle.syn> [--arch A] [--purpose P] [--id I] [--version V] [--component name:prefix]..."
        );
        return Ok(());
    }
    let mut path: Option<PathBuf> = None;
    let mut arch: Option<String> = None;
    let mut purpose: Option<String> = None;
    let mut id: Option<String> = None;
    let mut version: Option<String> = None;
    let mut components: Vec<(String, String)> = Vec::new();

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--arch" => arch = Some(it.next().ok_or_else(|| anyhow::anyhow!("--arch requires value"))?),
            "--purpose" => purpose = Some(it.next().ok_or_else(|| anyhow::anyhow!("--purpose requires value"))?),
            "--id" => id = Some(it.next().ok_or_else(|| anyhow::anyhow!("--id requires value"))?),
            "--version" => version = Some(it.next().ok_or_else(|| anyhow::anyhow!("--version requires value"))?),
            "--component" => {
                let v = it.next().ok_or_else(|| anyhow::anyhow!("--component requires name:prefix"))?;
                let (name, prefix) = v
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("--component expects <name>:<tensor_prefix>, got {v:?}"))?;
                components.push((name.to_string(), prefix.to_string()));
            }
            _ if path.is_none() && !a.starts_with('-') => path = Some(PathBuf::from(a)),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let path = path.ok_or_else(|| anyhow::anyhow!("bundle path required"))?;

    let mut ed = BundleEditor::open(&path)?;
    let mut meta = ed.meta().clone();
    if let Some(v) = arch {
        meta.arch = v;
    }
    if let Some(v) = purpose {
        meta.purpose = v;
    }
    if let Some(v) = id {
        meta.id = v;
    }
    if let Some(v) = version {
        meta.version = v;
    }
    for (name, prefix) in components {
        meta.components.insert(name, prefix);
    }
    ed.set_meta(meta.clone());
    ed.commit()?;
    println!(
        "ok: id={} version={} arch={} purpose={} components={:?}",
        meta.id, meta.version, meta.arch, meta.purpose, meta.components
    );
    Ok(())
}
