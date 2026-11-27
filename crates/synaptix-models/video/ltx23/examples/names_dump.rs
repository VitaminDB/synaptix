//! Имена, dtype и формы тензоров бандла/safetensors: `names_dump FILE [ПРЕФИКС]`.
use std::path::Path;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader as _;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let l = if a[0].ends_with(".syn") { SafetensorsLoader::open_bundle(Path::new(&a[0]), None).expect("bundle") } else { SafetensorsLoader::open(Path::new(&a[0])).expect("st") };
    let pre = a.get(1).cloned().unwrap_or_default();
    let mut n: Vec<&str> = l.names().into_iter().filter(|n| n.starts_with(&pre)).collect();
    n.sort();
    for k in n {
        if let Some((_, d, s)) = l.raw_bytes(k) { println!("{k} {d:?} {s:?}"); } else { println!("{k} ?"); }
    }
}
