//! `.syn`-бандл FLUX.2 против исходного каталога diffusers: все тензоры
//! компонентов байт в байт, конфиги и токенайзер совпадают, модель
//! открывается из бандла.
//!
//! ```sh
//! FLUX2_DIR=…/FLUX.2-klein-4B FLUX2_SYN=…/flux.2-klein-4b.syn \
//!   cargo test --release -p synaptix-image-flux2 --test bundle_source -- --ignored --nocapture
//! ```

use synaptix_core::{device::Device, dtype::DType};
use synaptix_image_flux2::source::COMPONENTS;
use synaptix_image_flux2::{Flux2Model, Flux2Source};

fn pair() -> Option<(Flux2Source, Flux2Source)> {
    let dir = std::env::var("FLUX2_DIR").ok()?;
    let syn = std::env::var("FLUX2_SYN").ok()?;
    Some((Flux2Source::open(dir).unwrap(), Flux2Source::open(syn).unwrap()))
}

const AUX: &[&str] = &[
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

#[test]
#[ignore]
fn bundle_matches_directory() {
    let Some((dir, syn)) = pair() else { return };
    assert!(syn.is_bundle());
    let mut total = 0usize;
    let mut bytes = 0u64;
    for c in COMPONENTS {
        let (a, b) = (dir.weights(c).unwrap(), syn.weights(c).unwrap());
        let names = a.names();
        assert_eq!(names.len(), b.names().len(), "{c}: число тензоров");
        for n in &names {
            let (ra, da, sa) = a.raw(n).unwrap();
            let (rb, db, sb) = b.raw(n).unwrap_or_else(|| panic!("{c}: в бандле нет {n}"));
            assert_eq!((da, sa), (db, sb), "{c}/{n}: dtype/shape");
            assert!(ra == rb, "{c}/{n}: байты расходятся");
            bytes += ra.len() as u64;
        }
        total += names.len();
    }
    for f in AUX {
        if let Some(x) = dir.read_opt(f) {
            assert_eq!(Some(x), syn.read_opt(f), "{f}");
        }
    }
    eprintln!("совпало {total} тензоров, {:.2} ГБ, и {} вспомогательных файлов", bytes as f64 / 1e9, AUX.len());
}

#[test]
#[ignore]
fn model_opens_from_bundle() {
    let Some((_, syn)) = pair() else { return };
    let m = Flux2Model::open(syn.path(), Device::Cpu, DType::F32, DType::F32).unwrap();
    eprintln!("{:?}: {:?}", m.variant(), m.config());
}
