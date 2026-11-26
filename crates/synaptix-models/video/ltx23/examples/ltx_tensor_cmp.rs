//! Побайтная сверка тензоров двух чекпойнтов (safetensors или .syn) с
//! переименованием префикса: `ltx_tensor_cmp A PREF_A B PREF_B` — для каждого
//! тензора A с префиксом PREF_A ищется B[PREF_B + остаток]. Печатает итог:
//! совпало, отличается, нет в B.
use std::path::Path;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader as _;

fn open(p: &str) -> SafetensorsLoader {
    if p.ends_with(".syn") {
        SafetensorsLoader::open_bundle(Path::new(p), None).expect("bundle")
    } else {
        SafetensorsLoader::open(Path::new(p)).expect("safetensors")
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (fa, pa, fb, pb) = (&a[0], &a[1], &a[2], &a[3]);
    let (la, lb) = (open(fa), open(fb));
    let (mut same, mut diff, mut missing) = (0, 0, 0);
    let mut names: Vec<&str> = la.names().into_iter().filter(|n| n.starts_with(pa.as_str())).collect();
    names.sort();
    for n in names {
        let nb = format!("{pb}{}", &n[pa.len()..]);
        let Some((ba, da, sa)) = la.raw_bytes(n) else { continue };
        match lb.raw_bytes(&nb) {
            None => {
                missing += 1;
                if missing <= 5 {
                    println!("нет в B: {nb}");
                }
            }
            Some((bb, db, sb)) if da == db && sa == sb && ba == bb => same += 1,
            Some((_, db, sb)) => {
                diff += 1;
                if diff <= 5 {
                    println!("отличается: {n} {da:?}{sa:?} vs {db:?}{sb:?}");
                }
            }
        }
    }
    println!("совпало {same}, отличается {diff}, нет в B {missing}");
}
