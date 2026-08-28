//! syn-info — print metadata, version, capability flags, and chunk listing.

use std::path::PathBuf;

use synaptix_bundle::{Bundle, DirEntry};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: syn-info <bundle.syn> [--tree | --layers | --json]"))?
        .into();
    let mode = args.next().unwrap_or_else(|| "--tree".into());

    let b = Bundle::open(&path)?;
    let (maj, min) = b.version();
    let meta = b.meta();

    println!("{}  ({} bytes)", path.display(), b.size());
    println!("  id:           {}", meta.id);
    println!("  version:      {}", meta.version);
    println!("  format:       {}.{}", maj, min);
    println!("  arch:         {}", meta.arch);
    println!("  purpose:      {}", meta.purpose);
    if !meta.components.is_empty() {
        println!("  components:");
        for (name, prefix) in &meta.components {
            println!("    {name} -> {prefix:?}");
        }
    }
    if !meta.required_caps.is_empty() {
        println!("  required_caps: {:?}", meta.required_caps);
    }
    if !meta.optional_caps.is_empty() {
        println!("  optional_caps: {:?}", meta.optional_caps);
    }
    if !meta.refs.is_empty() {
        println!("  refs:");
        for r in &meta.refs {
            println!("    - {} (purpose={}, prefix={})", r.id, r.purpose, r.tensor_prefix);
        }
    }

    let total: u64 = b.cdir().entries.iter().map(|e| e.payload_len).sum();
    let alive: u64 = b.cdir().entries.iter().filter(|e| e.is_alive()).map(|e| e.payload_len).sum();
    let tombs: u64 = total - alive;
    println!("  alive payload: {} bytes", alive);
    if tombs > 0 {
        let pct = (tombs as f64 / total as f64) * 100.0;
        println!("  tombstoned:    {} bytes ({:.1} % fragmented)", tombs, pct);
    }
    if let Some(m) = &meta.manifest_sha256 {
        let hex: String = m.iter().take(8).map(|b| format!("{b:02x}")).collect();
        println!("  manifest_sha256: {hex}…");
    }
    if let Some(m) = &meta.manifest_blake3 {
        let hex: String = m.iter().take(8).map(|b| format!("{b:02x}")).collect();
        println!("  manifest_blake3: {hex}…");
    }
    println!();

    if mode == "--json" {
        let cdir = b.cdir();
        serde_json::to_writer_pretty(std::io::stdout().lock(), cdir)?;
        println!();
    } else if mode == "--layers" {
        print_layers(&b)?;
    } else {
        println!("contents:");
        let shallow = b.list_dir_shallow("");
        print_dir(&b, "", &shallow, 1);
        if let Some(t) = b.cdir().entries.iter().find(|e| e.is_alive() && e.kind_typed() == synaptix_bundle::ChunkType::Tensors) {
            println!("  [{}]  {} bytes", t.name, t.payload_len);
        }
    }

    Ok(())
}

/// `--layers` — состав каждого `tensors:*`-чанка: группы имён со схлопнутыми
/// индексами слоёв, роль, dtype и вес. Читается только safetensors-заголовок
/// внутри mmap, поэтому работает мгновенно даже на 77-гигабайтном бандле.
fn print_layers(b: &Bundle) -> anyhow::Result<()> {
    use synaptix_bundle::inspect;

    let names: Vec<String> = b
        .cdir()
        .entries
        .iter()
        .filter(|e| e.is_alive() && e.kind_typed() == synaptix_bundle::ChunkType::Tensors)
        // `tensors_slice_named` сам приписывает `tensors:` — в cdir имя уже
        // полное, поэтому префикс здесь снимаем.
        .map(|e| e.name.trim_start_matches("tensors:").to_string())
        .collect();
    if names.is_empty() {
        println!("в бандле нет tensors-чанков");
        return Ok(());
    }
    // Подсказка для нераспознанных имён: у однокомпонентного `acestep_vae.syn`
    // чанк зовётся `main`, и роль читается только из id/purpose бандла.
    let meta = b.meta();
    for name in names {
        let hint = format!("{name} {} {}", meta.purpose, meta.id);
        let slice = b.tensors_slice_named(&name)?;
        let tensors = inspect::read_header_slice(slice)?;
        let total: u64 = tensors.iter().map(|t| t.bytes).sum();
        println!("[{name}]  {} тензоров, {}", tensors.len(), human(total));

        let by_role = inspect::bytes_by_role(&tensors, Some(&hint));
        for (role, est) in &by_role {
            let pct = if total == 0 { 0.0 } else { est.dense as f64 / total as f64 * 100.0 };
            // Заодно видно, во что превратился бы этот кусок под квантом —
            // тот же расчёт, что показывает мастер упаковки.
            println!(
                "  {:<10} {:>12}  {:>5.1} %   nvfp4 {:>10}  mxfp8 {:>10}",
                role.key(),
                human(est.dense),
                pct,
                human(est.nvfp4),
                human(est.mxfp8)
            );
        }
        println!();
        for g in inspect::group_tensors(&tensors, Some(&hint)) {
            let shape = if g.shape.is_empty() {
                "—".to_string()
            } else {
                g.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×")
            };
            println!(
                "  {:<10} ×{:<4} {:<6} {:<20} {:>12}  {}",
                g.role.key(),
                g.count,
                g.dtype,
                shape,
                human(g.bytes),
                g.pattern
            );
        }
        println!();
    }
    Ok(())
}

fn human(n: u64) -> String {
    const KB: f64 = 1024.0;
    let n = n as f64;
    if n >= KB * KB * KB {
        format!("{:.2} GB", n / (KB * KB * KB))
    } else if n >= KB * KB {
        format!("{:.1} MB", n / (KB * KB))
    } else if n >= KB {
        format!("{:.0} KB", n / KB)
    } else {
        format!("{n} B")
    }
}

fn print_dir(b: &Bundle, prefix: &str, entries: &[DirEntry<'_>], depth: usize) {
    let indent = "  ".repeat(depth);
    for e in entries {
        match e {
            DirEntry::File(f) => {
                let tag = f.tag.map(|t| format!(" [{}]", t.as_str())).unwrap_or_default();
                println!("{indent}{}  ({} bytes){tag}", f.name.trim_start_matches(&format!("{prefix}/")), f.payload_len);
            }
            DirEntry::Subdir(name) => {
                println!("{indent}{name}/");
                let new_prefix = if prefix.is_empty() { name.to_string() } else { format!("{prefix}/{name}") };
                let sub = b.list_dir_shallow(&new_prefix);
                print_dir(b, &new_prefix, &sub, depth + 1);
            }
        }
    }
}
