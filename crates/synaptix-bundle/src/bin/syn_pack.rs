//! syn-pack — repack a HuggingFace-style model directory into one .syn bundle.
//!
//! Auto-detects:
//! - single `model.safetensors`;
//! - shards via `model.safetensors.index.json` (HF convention);
//! - glob `*.safetensors` (fallback when only shards live in the dir).
//!
//! For multi-component bundles (VoxCPM2, ACE-Step subnets) use repeatable
//! `--component <name>:<dir>[:<prefix>]` instead of a positional input dir.
//!
//! Usage:
//!     syn-pack <input_dir> -o <output.syn> [--id NAME] [--version VER] [--arch A] [--purpose P] [--prefix P]
//!     syn-pack -o <output.syn> [--id NAME] ... \
//!         --component minicpm4:/path/to/minicpm4_dir:minicpm4 \
//!         --component locdit:/path/to/locdit_dir:locdit \
//!         --files /path/to/aux_dir         # config.json + tokenizer.json + ...

use std::path::{Path, PathBuf};

use synaptix_bundle::{resolve_safetensors_in_dir, BundleBuilder, FileTag, Result};

#[derive(Default)]
struct Component {
    name: String,
    dir: PathBuf,
    prefix: Option<String>,
}

fn parse_component(s: &str) -> anyhow::Result<Component> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() < 2 {
        anyhow::bail!("--component expects <name>:<dir>[:<prefix>], got {s:?}");
    }
    let name = parts[0].to_string();
    let dir = PathBuf::from(parts[1]);
    // Empty prefix (`name:dir:`) → no tensor namespace, equivalent to None.
    let prefix = parts.get(2).and_then(|p| {
        if p.is_empty() { None } else { Some(p.to_string()) }
    });
    Ok(Component { name, dir, prefix })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return Ok(());
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut id: Option<String> = None;
    let mut version: String = "1.0.0".into();
    let mut arch: Option<String> = None;
    let mut purpose: Option<String> = None;
    let mut prefix: Option<String> = None;
    let mut components: Vec<Component> = Vec::new();
    let mut files_dir: Option<PathBuf> = None;
    let mut sha256 = false;
    let mut blake3 = false;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(it.next().ok_or_else(|| anyhow::anyhow!("-o requires value"))?)),
            "--id" => id = Some(it.next().ok_or_else(|| anyhow::anyhow!("--id requires value"))?),
            "--version" => version = it.next().ok_or_else(|| anyhow::anyhow!("--version requires value"))?,
            "--arch" => arch = Some(it.next().ok_or_else(|| anyhow::anyhow!("--arch requires value"))?),
            "--purpose" => purpose = Some(it.next().ok_or_else(|| anyhow::anyhow!("--purpose requires value"))?),
            "--prefix" => prefix = Some(it.next().ok_or_else(|| anyhow::anyhow!("--prefix requires value"))?),
            "--component" => {
                let v = it.next().ok_or_else(|| anyhow::anyhow!("--component requires <name>:<dir>[:<prefix>]"))?;
                components.push(parse_component(&v)?);
            }
            "--files" => files_dir = Some(PathBuf::from(it.next().ok_or_else(|| anyhow::anyhow!("--files requires value"))?)),
            "--sha256" => sha256 = true,
            "--blake3" => blake3 = true,
            _ if input.is_none() && !a.starts_with('-') => input = Some(PathBuf::from(a)),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let output = output.ok_or_else(|| anyhow::anyhow!("missing -o <output.syn>"))?;

    let id = id.unwrap_or_else(|| {
        if let Some(i) = &input {
            i.file_name().and_then(|s| s.to_str()).unwrap_or("model").to_string()
        } else if !components.is_empty() {
            components[0].name.clone()
        } else {
            "model".to_string()
        }
    });

    let mut b = BundleBuilder::new(&id, &version);
    if let Some(a) = arch { b = b.arch(a); }
    if let Some(p) = purpose { b = b.purpose(p); }
    if sha256 {
        #[cfg(feature = "sha256")]
        { b = b.with_sha256(true); }
        #[cfg(not(feature = "sha256"))]
        anyhow::bail!("--sha256 requires syn-pack built with `--features sha256`");
    }
    if blake3 {
        #[cfg(feature = "blake3")]
        { b = b.with_blake3(true); }
        #[cfg(not(feature = "blake3"))]
        anyhow::bail!("--blake3 requires syn-pack built with `--features blake3`");
    }

    // Tensor sources.
    if !components.is_empty() {
        if input.is_some() {
            anyhow::bail!("--component is incompatible with positional <input_dir>");
        }
        for c in &components {
            // Record components map only when an explicit prefix was given;
            // unprefixed components stay with empty prefix → loader uses raw
            // tensor names.
            b = b.component(&c.name, c.prefix.clone().unwrap_or_default());
            let paths = resolve_safetensors_in_dir(&c.dir).map_err(into_anyhow)?;
            eprintln!(
                "  component {} ({} shard{}, prefix={})",
                c.name,
                paths.len(),
                if paths.len() == 1 { "" } else { "s" },
                c.prefix.as_deref().unwrap_or("<none>")
            );
            for p in &paths {
                eprintln!("    + {} ({} bytes)", p.display(), std::fs::metadata(p)?.len());
            }
            b = b.add_safetensors_component(&c.name, paths, c.prefix.as_deref());
        }

        // Auxiliary files from --files dir (or from the first component's dir
        // if --files not given — useful for VoxCPM2 where tokenizer.json lives
        // alongside the first subnet). Walks recursively, skipping files that
        // live inside any registered component's directory (those .safetensors
        // are already packed as tensors-chunks; their non-safetensors siblings
        // like `audio_tokenizer/config.json` come back through this walk).
        let aux_dir = files_dir.as_ref().or_else(|| Some(&components[0].dir));
        if let Some(aux) = aux_dir {
            add_aux_files_recursive(&mut b, aux, &components)?;
        }
    } else {
        // Single-positional mode.
        let input = input.ok_or_else(|| anyhow::anyhow!("missing input directory or --component"))?;
        let paths = resolve_safetensors_in_dir(&input).map_err(into_anyhow)?;
        if paths.len() == 1 {
            eprintln!("  tensors: {} (single shard)", paths[0].display());
        } else {
            eprintln!("  tensors: {} shards", paths.len());
            for p in &paths {
                eprintln!("    {} ({} bytes)", p.display(), std::fs::metadata(p)?.len());
            }
        }
        b = b.add_safetensors_component("main", paths.clone(), prefix.as_deref());

        // Add every non-safetensors file in the dir.
        let safetensors_set: std::collections::BTreeSet<PathBuf> = paths.iter().cloned().collect();
        for entry in walk(&input)? {
            if safetensors_set.contains(&entry) {
                continue;
            }
            if entry.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json") {
                // Skip — we synthesised our own concatenated stream.
                continue;
            }
            let rel = entry.strip_prefix(&input).unwrap().to_string_lossy().to_string();
            let tag = tag_for(&rel);
            b = b.add_file_path(&rel, &entry, tag).map_err(into_anyhow)?;
            eprintln!("  + {rel} ({} bytes, {})", std::fs::metadata(&entry)?.len(), tag.as_str());
        }
    }

    eprintln!("packing → {}", output.display());
    b.write(&output).map_err(into_anyhow)?;
    let size = std::fs::metadata(&output)?.len();
    eprintln!("done: {} ({} bytes)", output.display(), size);
    Ok(())
}

fn add_aux_files_recursive(
    b: &mut BundleBuilder,
    dir: &Path,
    components: &[Component],
) -> anyhow::Result<()> {
    // Component-dirs already contributed their `.safetensors` shards to
    // dedicated Tensors chunks. We still want their *non*-safetensors siblings
    // (e.g. `audio_tokenizer/config.json`) packed as File chunks under their
    // in-bundle path so loaders can read codec/tokenizer configs from the
    // bundle. The `components` parameter is currently informational only —
    // the `.safetensors` skip below already prevents duplicate packing.
    let _ = components;

    for entry in walk(dir)? {
        let Some(name) = entry.file_name().and_then(|s| s.to_str()) else { continue };
        // Skip safetensors (already in Tensors chunks) and HF index files.
        if name.ends_with(".safetensors") || name == "model.safetensors.index.json" {
            continue;
        }
        if name == ".DS_Store" || name.ends_with(".swp") {
            continue;
        }
        let rel = match entry.strip_prefix(dir) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        let bytes = std::fs::read(&entry)?;
        let tag = tag_for(&rel);
        let new_b = std::mem::replace(b, BundleBuilder::new("", ""));
        *b = new_b.add_file_bytes(&rel, bytes, tag).map_err(into_anyhow)?;
        eprintln!("  + {rel} ({} bytes, {})", std::fs::metadata(&entry)?.len(), tag.as_str());
    }
    Ok(())
}

fn walk(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk_into(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_into(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        // Skip directories that are pure HuggingFace / OS noise. `.cache/`
        // can hold gigabytes of partially-downloaded shards (`*.incomplete`)
        // that have no business inside a model bundle.
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name == ".cache" || name == ".git" || name == "__pycache__" {
                continue;
            }
        }
        if p.is_dir() {
            walk_into(root, &p, out)?;
        } else if p.is_file() {
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name == ".DS_Store" || name.ends_with(".swp") {
                    continue;
                }
                // HF download leftovers.
                if name.ends_with(".incomplete")
                    || name.ends_with(".lock")
                    || name.ends_with(".metadata")
                {
                    continue;
                }
            }
            out.push(p);
        }
    }
    Ok(())
}

fn tag_for(rel: &str) -> FileTag {
    let lower = rel.to_ascii_lowercase();
    if lower.contains("readme") || lower.ends_with("license") || lower.ends_with("license.md") {
        FileTag::Doc
    } else if lower.starts_with("examples/") {
        FileTag::Example
    } else {
        FileTag::Inference
    }
}

fn usage() {
    eprintln!(
        "syn-pack — pack a HuggingFace-style model directory into a single .syn bundle\n\n\
         USAGE:\n\
         \x20 syn-pack <input_dir> -o <output.syn> [--id NAME] [--version VER] [--arch A] [--purpose P] [--prefix P]\n\
         \x20 syn-pack -o <output.syn> --component <name>:<dir>[:<prefix>] [--component ...] [--files <aux_dir>]\n\n\
         FLAGS:\n\
         \x20 -o, --output <path>          target .syn path\n\
         \x20 --id <name>                  bundle id (default: dir basename)\n\
         \x20 --version <v>                default 1.0.0\n\
         \x20 --arch <name>                e.g. xlm-roberta, qwen3, voxcpm2\n\
         \x20 --purpose <p>                e.g. embed, rerank, tts, music\n\
         \x20 --prefix <p>                 apply <p>. tensor name prefix (single-component mode)\n\
         \x20 --component <name>:<dir>[:<prefix>]   register one component (repeatable)\n\
         \x20 --files <aux_dir>            directory with config.json / tokenizer.json (multi-component mode)\n\
         \x20 --sha256                     compute SHA-256 for every chunk + manifest hash (needs feature `sha256`)\n\
         \x20 --blake3                     compute Blake3 — 5-10× faster than SHA-256 (needs feature `blake3`)\n\n\
         Auto-detects safetensors layout: model.safetensors → model.safetensors.index.json shards → *.safetensors glob."
    );
}

fn into_anyhow(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

#[allow(dead_code)]
fn _doc_check(_: Result<()>) {}
