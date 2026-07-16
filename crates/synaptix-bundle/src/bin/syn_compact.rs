//! syn-compact — rebuild a bundle dropping all tombstones (reclaims disk space).
//!
//! Usage:
//!     syn-compact <bundle.syn>              # in-place (writes to .tmp + rename)
//!     syn-compact <bundle.syn> -o <out.syn> # write fresh copy

use std::path::PathBuf;

use synaptix_bundle::compact;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: syn-compact <bundle.syn> [-o <out.syn>]");
        return Ok(());
    }
    let mut src: Option<PathBuf> = None;
    let mut dst: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => dst = Some(PathBuf::from(it.next().ok_or_else(|| anyhow::anyhow!("-o requires value"))?)),
            _ if src.is_none() => src = Some(PathBuf::from(a)),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let src = src.ok_or_else(|| anyhow::anyhow!("missing input bundle"))?;
    let dst = dst.unwrap_or_else(|| src.clone());

    let old_size = std::fs::metadata(&src)?.len();
    if src == dst {
        // In-place: compact() handles temp + rename internally because
        // BundleBuilder writes to .syn.tmp and then renames to the target.
        compact(&src, &src)?;
    } else {
        compact(&src, &dst)?;
    }
    let new_size = std::fs::metadata(&dst)?.len();
    let saved = old_size.saturating_sub(new_size);
    eprintln!(
        "compacted: {} -> {} ({} bytes reclaimed)",
        old_size, new_size, saved
    );
    Ok(())
}
