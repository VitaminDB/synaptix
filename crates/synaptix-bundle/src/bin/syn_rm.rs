//! syn-rm — tombstone files in an existing bundle.

use std::path::PathBuf;

use synaptix_bundle::BundleEditor;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: syn-rm <bundle.syn> <path> [<path>...]");
        return Ok(());
    }
    let bundle = PathBuf::from(&args[0]);
    let paths = &args[1..];
    if paths.is_empty() {
        anyhow::bail!("at least one path required");
    }
    let mut ed = BundleEditor::open(&bundle)?;
    for p in paths {
        ed.remove_file(p)?;
        eprintln!("removed {p}");
    }
    ed.commit()?;
    Ok(())
}
