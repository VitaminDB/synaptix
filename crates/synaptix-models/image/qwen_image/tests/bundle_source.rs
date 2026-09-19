//! `.syn`-бандл Qwen-Image против исходного каталога diffusers: все тензоры
//! компонентов байт в байт, конфиги и токенайзер совпадают, модель
//! открывается из бандла.
//!
//! ```sh
//! QWEN_DIR=…/Qwen-Image-Edit-2511 QWEN_SYN=…/qwen-image-edit-2511.syn \
//!   cargo test --release -p synaptix-image-qwen --test bundle_source -- --ignored --nocapture
//! ```

use synaptix_core::{device::Device, dtype::DType};
use synaptix_image_qwen::source::COMPONENTS;
use synaptix_image_qwen::{QwenImageModel, QwenImageSource};

fn pair() -> Option<(QwenImageSource, QwenImageSource)> {
    let dir = std::env::var("QWEN_DIR").ok()?;
    let syn = std::env::var("QWEN_SYN").ok()?;
    Some((QwenImageSource::open(dir).unwrap(), QwenImageSource::open(syn).unwrap()))
}

const AUX: &[&str] = &[
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "processor/tokenizer.json",
    "processor/preprocessor_config.json",
    "processor/tokenizer_config.json",
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
    let m = QwenImageModel::open(syn.path(), Device::Cpu, DType::F32, DType::F32).unwrap();
    eprintln!("{:?}: {:?}", m.variant(), m.config());
    assert!(m.variant().is_edit());
}
