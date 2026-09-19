//! `.syn`-бандл FLUX против исходного каталога diffusers: те же тензоры байт
//! в байт, те же конфиги и токенайзеры. Без GPU.
//!
//! ```sh
//! FLUX_DIR=…/FLUX.1-dev FLUX_SYN=…/flux.1-dev.syn \
//!   cargo test --release -p synaptix-image-flux --test bundle_source -- --ignored --nocapture
//! ```

use synaptix_core::device::Device;
use synaptix_image_flux::{FluxComponent, FluxSource};

fn paths() -> Option<(String, String)> {
    Some((std::env::var("FLUX_DIR").ok()?, std::env::var("FLUX_SYN").ok()?))
}

#[test]
#[ignore]
fn bundle_matches_directory_byte_for_byte() {
    let Some((dir, syn)) = paths() else {
        eprintln!("SKIP: задайте FLUX_DIR и FLUX_SYN");
        return;
    };
    let a = FluxSource::open(&dir).unwrap();
    let b = FluxSource::open(&syn).unwrap();
    assert!(!a.is_bundle() && b.is_bundle());

    for c in FluxComponent::ALL {
        let la = a.loader(c, Device::Cpu).unwrap();
        let lb = b.loader(c, Device::Cpu).unwrap();
        let mut names_a: Vec<String> = la.infos().map(|(n, _, _)| n.to_string()).collect();
        let mut names_b: Vec<String> = lb.infos().map(|(n, _, _)| n.to_string()).collect();
        names_a.sort();
        names_b.sort();
        assert_eq!(names_a, names_b, "{}: разный набор тензоров", c.name());
        let mut bytes = 0u64;
        for n in &names_a {
            let (ra, da, sa) = la.raw_bytes(n).unwrap();
            let (rb, db, sb) = lb.raw_bytes(n).unwrap();
            assert_eq!((da, sa), (db, sb), "{}: {n}: dtype/форма", c.name());
            assert!(ra == rb, "{}: {n}: байты расходятся", c.name());
            bytes += ra.len() as u64;
        }
        eprintln!("{:<15} {:>5} тензоров, {:.2} ГБ — совпало", c.name(), names_a.len(), bytes as f64 / 1e9);
    }

    for rel in [
        "model_index.json",
        "transformer/config.json",
        "scheduler/scheduler_config.json",
        "vae/config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer.json",
    ] {
        assert_eq!(a.read(rel).unwrap(), b.read(rel).unwrap(), "{rel}");
    }
}

#[test]
#[ignore]
fn model_opens_from_bundle() {
    let Some((_, syn)) = paths() else {
        eprintln!("SKIP: задайте FLUX_SYN");
        return;
    };
    let m = synaptix_image_flux::FluxModel::open(
        &syn,
        Device::Cpu,
        synaptix_core::dtype::DType::F32,
        synaptix_core::dtype::DType::F32,
    )
    .unwrap();
    assert!(m.guidance_distilled(), "FLUX.1-dev — guidance-distilled");
    assert_eq!(m.default_max_seq_len(), 512);
    let bytes = m.source().component_bytes(FluxComponent::Transformer).unwrap();
    assert!(bytes > 20_000_000_000, "трансформер {bytes} байт");
}
