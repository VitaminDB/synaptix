//! syn-add — add a file from disk into an existing bundle.
//!
//! Usage:
//!     syn-add <bundle.syn> <path-in-bundle> <local-file> [--tag inference|doc|example|asset]

use std::path::PathBuf;

use synaptix_bundle::{BundleEditor, FileTag};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return Ok(());
    }

    let mut bundle: Option<PathBuf> = None;
    let mut in_path: Option<String> = None;
    let mut local: Option<PathBuf> = None;
    let mut tag = FileTag::Inference;

    let mut positional = 0usize;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tag" => tag = parse_tag(&it.next().ok_or_else(|| anyhow::anyhow!("--tag requires value"))?)?,
            "-h" | "--help" => { usage(); return Ok(()); }
            _ => {
                match positional {
                    0 => bundle = Some(PathBuf::from(a)),
                    1 => in_path = Some(a),
                    2 => local = Some(PathBuf::from(a)),
                    _ => anyhow::bail!("extra positional arg: {a}"),
                }
                positional += 1;
            }
        }
    }
    let bundle = bundle.ok_or_else(|| anyhow::anyhow!("missing <bundle.syn>"))?;
    let in_path = in_path.ok_or_else(|| anyhow::anyhow!("missing <path-in-bundle>"))?;
    let local = local.ok_or_else(|| anyhow::anyhow!("missing <local-file>"))?;

    let bytes = std::fs::read(&local)?;
    let mut ed = BundleEditor::open(&bundle)?;
    ed.add_file(&in_path, bytes, tag)?;
    ed.commit()?;
    eprintln!("added {in_path} ({}) from {}", tag.as_str(), local.display());
    Ok(())
}

fn parse_tag(s: &str) -> anyhow::Result<FileTag> {
    Ok(match s {
        "inference" => FileTag::Inference,
        "doc" => FileTag::Doc,
        "example" => FileTag::Example,
        "asset" => FileTag::Asset,
        other => anyhow::bail!("unknown tag `{other}` (expected inference|doc|example|asset)"),
    })
}

fn usage() {
    eprintln!(
        "syn-add — add a file from disk into an existing .syn bundle\n\n\
         USAGE:\n  syn-add <bundle.syn> <path-in-bundle> <local-file> [--tag TAG]\n\n\
         tag: inference (default) | doc | example | asset"
    );
}
