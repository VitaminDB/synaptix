use std::path::PathBuf;

use synaptix_bundle::BundleBuilder;

pub struct ConvertArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    pub format: String,
    pub arch: Option<String>,
    pub component: Option<String>,
}

pub fn run(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    let from_ext = args.input.extension().and_then(|e| e.to_str()).unwrap_or("");
    let to_ext = args.output.extension().and_then(|e| e.to_str()).unwrap_or("");
    match (from_ext, to_ext) {
        ("safetensors", "syn") => convert_safetensors_to_syn(&args),
        _ => Err(format!(
            "unsupported conversion: {} → {} (supported: safetensors → syn)",
            from_ext, to_ext
        ).into()),
    }
}

fn convert_safetensors_to_syn(args: &ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    let id = args.input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("converted")
        .to_string();
    let arch = args.arch.clone().unwrap_or_else(|| "unknown".into());
    let component = args.component.clone().unwrap_or_else(|| "main".into());

    println!("Converting {} → {}", args.input.display(), args.output.display());
    println!("  bundle_id:  {}", id);
    println!("  arch:       {}", arch);
    println!("  component:  {}", component);

    let builder = BundleBuilder::new(id, "0.1.0")
        .arch(arch)
        .component(component.clone(), "")
        .add_safetensors_component(&component, vec![args.input.clone()], None);

    let count = builder.item_count();
    println!("  tensors to pack: {}", count);
    println!("Writing bundle...");
    builder.write(&args.output)?;
    println!("Done: {}", args.output.display());
    Ok(())
}
