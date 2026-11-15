//! Сравнить два `.syn` по содержимому: тензоры (включая блобы квантов
//! `.qpacked`/`.qscales`) байт в байт и файлы. Раскладка внутри файла
//! (порядок, выравнивание, заголовки) не важна.
//!
//! `cargo run --profile fast-release -p synaptix-io --example bundle_cmp -- a.syn b.syn`

use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [pa, pb] = a.as_slice() else {
        eprintln!("использование: bundle_cmp a.syn b.syn");
        std::process::exit(2);
    };
    let la = SynBundleLoader::open(pa).expect("a");
    let lb = SynBundleLoader::open(pb).expect("b");
    let na: std::collections::BTreeSet<&str> = la.names().into_iter().collect();
    let nb: std::collections::BTreeSet<&str> = lb.names().into_iter().collect();
    let mut bad = 0usize;
    for n in na.symmetric_difference(&nb) {
        println!("только в {}: {n}", if na.contains(n) { "a" } else { "b" });
        bad += 1;
    }
    let (mut same, mut bytes) = (0usize, 0u64);
    for n in na.intersection(&nb) {
        if la.quant_kind(n).is_some() || lb.quant_kind(n).is_some() {
            let (ka, kb) = (la.quant_kind(n), lb.quant_kind(n));
            let (da, db) = (la.quant_dims(n), lb.quant_dims(n));
            if ka != kb || da != db {
                println!("{n}: квант {ka:?}{da:?} ≠ {kb:?}{db:?}");
                bad += 1;
                continue;
            }
            match (la.quant_blob_slices(n), lb.quant_blob_slices(n)) {
                (Some(Ok((pa, sa))), Some(Ok((pb, sb)))) => {
                    if pa != pb || sa != sb {
                        let dp = pa.iter().zip(pb).filter(|(p, q)| p != q).count();
                        let ds = sa.iter().zip(sb).filter(|(p, q)| p != q).count();
                        println!("{n} ({ka:?}): отличаются {dp} из {} байт весов, {ds} из {} байт масштабов", pa.len(), sa.len());
                        bad += 1;
                    } else {
                        same += 1;
                        bytes += (pa.len() + sa.len()) as u64;
                    }
                }
                _ => {
                    println!("{n}: блоб кванта не читается");
                    bad += 1;
                }
            }
            continue;
        }
        match (la.mapped_bytes(n), lb.mapped_bytes(n)) {
            (Some((x, dx, sx)), Some((y, dy, sy))) => {
                if dx != dy || sx != sy {
                    println!("{n}: {dx:?}{sx:?} ≠ {dy:?}{sy:?}");
                    bad += 1;
                } else if x != y {
                    let diff = x.iter().zip(y).filter(|(p, q)| p != q).count();
                    println!("{n}: {diff} из {} байт отличаются", x.len());
                    bad += 1;
                } else {
                    same += 1;
                    bytes += x.len() as u64;
                }
            }
            _ => {
                println!("{n}: не читается как тензор");
                bad += 1;
            }
        }
    }
    let fa: std::collections::BTreeSet<String> = la.file_names().into_iter().collect();
    let fb: std::collections::BTreeSet<String> = lb.file_names().into_iter().collect();
    for f in fa.symmetric_difference(&fb) {
        println!("файл только в {}: {f}", if fa.contains(f) { "a" } else { "b" });
        bad += 1;
    }
    for f in fa.intersection(&fb) {
        if la.read_file(f) != lb.read_file(f) {
            println!("файл {f}: содержимое отличается");
            bad += 1;
        }
    }
    println!("совпало тензоров: {same} ({:.2} ГБ), файлов: {}, расхождений: {bad}", bytes as f64 / 1e9, fa.intersection(&fb).count());
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
