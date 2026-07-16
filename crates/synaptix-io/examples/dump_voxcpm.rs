use safetensors::SafeTensors;
use std::collections::BTreeMap;
use synaptix_bundle::Bundle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: dump_voxcpm <bundle.syn>");
    let bundle = Bundle::open(&path)?;
    println!("=== bundle {path} ===");
    println!("id={} ver={:?}", bundle.id(), bundle.version());
    let meta = bundle.meta();
    println!("arch={:?} purpose={:?}", meta.arch, meta.purpose);
    println!("components:");
    for (name, prefix) in meta.components.iter() {
        println!("  {name} -> prefix '{prefix}'");
    }
    println!("\n=== files ===");
    let mut files: Vec<(String, u64)> = Vec::new();
    for e in bundle.list_files() {
        files.push((e.name.clone(), e.raw_len));
    }
    files.sort();
    for (n, l) in &files {
        println!("  {n} ({l} bytes)");
    }

    for cfgname in ["config.json", "audiovae_config.json", "generation_config.json"] {
        if let Ok(bytes) = bundle.read_file(cfgname) {
            println!("\n=== {cfgname} ===\n{}", String::from_utf8_lossy(&bytes));
        }
    }

    let comps: Vec<String> = if meta.components.is_empty() {
        vec![String::new()]
    } else {
        meta.components.keys().cloned().collect()
    };

    let mut byprefix: BTreeMap<String, usize> = BTreeMap::new();
    for comp in &comps {
        let slice = match if comp.is_empty() {
            bundle.tensors_slice()
        } else {
            bundle.tensors_slice_for(comp).map(|(s, _)| s)
        } {
            Ok(s) => s,
            Err(e) => {
                println!("\n[component '{comp}'] no tensors slice: {e}");
                continue;
            }
        };
        let st = SafeTensors::deserialize(slice)?;
        println!("\n=== component '{comp}': {} tensors ===", st.len());
        let mut names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        names.sort();
        for name in &names {
            let v = st.tensor(name)?;
            println!("  {name}\t{:?}\t{:?}", v.dtype(), v.shape());
            let top = name.split('.').next().unwrap_or("").to_string();
            *byprefix.entry(top).or_default() += 1;
        }
    }
    println!("\n=== top-level prefix counts ===");
    for (p, c) in &byprefix {
        println!("  {p}: {c}");
    }
    Ok(())
}
