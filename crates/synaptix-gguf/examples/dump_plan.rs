use std::path::PathBuf;

use synaptix_gguf::{convert::plan_for, GgufFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model: PathBuf = args.next().expect("usage: dump_plan <model.gguf> [mmproj.gguf] -o <dir>").into();
    let mut mmproj: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" => out = Some(args.next().unwrap().into()),
            other => mmproj = Some(other.into()),
        }
    }
    let m = GgufFile::open(&model)?;
    let p = match &mmproj {
        Some(p) => Some(GgufFile::open(p)?),
        None => None,
    };
    let plan = plan_for(&m, p.as_ref(), "dump")?;
    println!("arch={} components={}", plan.arch, plan.components.len());
    for c in &plan.components {
        println!("  {}: {} tensors", c.name, c.tensors.len());
        for t in c.tensors.iter().take(6) {
            println!("     {} <- {:?} shape={:?} tr={:?}", t.hf_name, t.producer, t.shape, t.transform);
        }
    }
    if let Some(dir) = out {
        std::fs::create_dir_all(&dir)?;
        for f in &plan.files {
            std::fs::write(dir.join(&f.path), &f.bytes)?;
            println!("  wrote {} ({} bytes)", f.path, f.bytes.len());
        }
    }
    Ok(())
}
