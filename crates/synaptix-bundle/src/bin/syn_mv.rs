//! syn-mv — rename a file inside a bundle.

use std::path::PathBuf;

use synaptix_bundle::BundleEditor;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: syn-mv <bundle.syn> <old> <new>");
        return Ok(());
    }
    if args.len() != 3 {
        anyhow::bail!("expected exactly 3 args: <bundle.syn> <old> <new>");
    }
    let bundle = PathBuf::from(&args[0]);
    let old = &args[1];
    let new = &args[2];
    let mut ed = BundleEditor::open(&bundle)?;
    ed.rename(old, new)?;
    ed.commit()?;
    eprintln!("renamed {old} -> {new}");
    Ok(())
}
