//! `.syn`-бандл SDXL против исходного каталога diffusers: все тензоры
//! компонентов байт в байт (в каталоге берётся тот же набор, что упаковщик:
//! основной, а при одних вариантах точности — fp16), конфиги и токенайзеры
//! совпадают, чекпойнт открывается из бандла.
//!
//! ```sh
//! SDXL_DIR=~/models/stabilityai/stable-diffusion-xl-base-1.0 \
//! SDXL_SYN=~/Storage/syn_models/sdxl-base-1.0.syn \
//!   cargo test --release -p synaptix-image-sdxl --test bundle_source -- --ignored --nocapture
//! ```

use synaptix_core::device::Device;
use synaptix_image_sdxl::source::COMPONENTS;
use synaptix_image_sdxl::{SdxlCheckpoint, SdxlSource};

fn pair() -> Option<(SdxlSource, SdxlSource)> {
    let dir = std::env::var("SDXL_DIR").ok()?;
    let syn = std::env::var("SDXL_SYN").ok()?;
    Some((SdxlSource::open(dir).unwrap(), SdxlSource::open(syn).unwrap()))
}

const AUX: &[&str] = &[
    "model_index.json",
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/tokenizer_config.json",
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
    SdxlCheckpoint::open(syn.path(), Device::Cpu).expect("чекпойнт из бандла");
    eprintln!("совпало {total} тензоров, {:.2} ГБ, и вспомогательные файлы", bytes as f64 / 1e9);
}
