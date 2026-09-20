//! KB-модели из `.syn`-бандлов: эмбеддер BGE-M3 и реранкер BGE-reranker-v2-m3
//! через фасады `facade::embedding` / `facade::rerank` — тем же путём, каким
//! их грузит synthos.
//!
//! Бандлы ищутся в `KB_MODELS_DIR` (по умолчанию `~/Storage/syn_models`); нет
//! файла — тест пропускается. `KB_DEVICE=cuda` гоняет на GPU (F16), иначе CPU F32.
//!
//! ```text
//! cargo test -p synaptix --release --test kb_bundles -- --nocapture --test-threads=1
//! ```

use std::path::PathBuf;

use synaptix::facade::embedding::{load_embedder, DType, Device, EmbedderConfig};
use synaptix::facade::rerank::{load_reranker, RerankerConfig};

fn bundle(name: &str) -> Option<PathBuf> {
    let dir = std::env::var("KB_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Storage/syn_models")
    });
    let path = dir.join(name);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: нет {}", path.display());
        None
    }
}

fn target() -> (Device, DType) {
    match std::env::var("KB_DEVICE").as_deref() {
        Ok("cuda") => (Device::Cuda(0), DType::F16),
        _ => (Device::Cpu, DType::F32),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const QUERY: &str = "С какого возраста можно вступать в брак?";
const DOCS: [&str; 3] = [
    "Брачный возраст устанавливается в восемнадцать лет для мужчин и женщин.",
    "Для приготовления борща понадобятся свёкла, капуста, картофель и мясо.",
    "The marriageable age is eighteen years for both men and women.",
];

#[test]
fn embedder_from_syn_bundle() {
    let Some(path) = bundle("bge-m3.syn") else { return };
    let (device, dtype) = target();
    let t0 = std::time::Instant::now();
    let embedder =
        load_embedder(EmbedderConfig::new(path).with_device(device).with_dtype(dtype)).unwrap();
    eprintln!("embedder {device:?}: загрузка {:.1} с", t0.elapsed().as_secs_f32());
    assert_eq!(embedder.dim(), 1024);

    let q = embedder.encode_query(QUERY).unwrap();
    let t1 = std::time::Instant::now();
    let docs = embedder.encode(&DOCS).unwrap();
    eprintln!("encode 3 docs: {:.0} мс", t1.elapsed().as_secs_f32() * 1e3);
    assert_eq!(docs.len(), 3);
    for v in docs.iter().chain([&q]) {
        assert_eq!(v.len(), 1024);
        let norm = dot(v, v).sqrt();
        assert!((norm - 1.0).abs() < 1e-2, "L2-норма {norm}");
    }
    let sims: Vec<f32> = docs.iter().map(|d| dot(&q, d)).collect();
    eprintln!("cos: {sims:?}");
    // По теме (и на другом языке) ближе, чем рецепт.
    assert!(sims[0] > sims[1] + 0.15, "{sims:?}");
    assert!(sims[2] > sims[1] + 0.10, "{sims:?}");
}

#[test]
fn reranker_from_syn_bundle() {
    let Some(path) = bundle("bge-reranker-v2-m3.syn") else { return };
    let (device, dtype) = target();
    let t0 = std::time::Instant::now();
    let reranker =
        load_reranker(RerankerConfig::new(path).with_device(device).with_dtype(dtype)).unwrap();
    eprintln!("reranker {device:?}: загрузка {:.1} с", t0.elapsed().as_secs_f32());

    let ranked = reranker.rerank(QUERY, &DOCS, 3).unwrap();
    eprintln!("rerank: {ranked:?}");
    assert_eq!(ranked.len(), 3);
    assert_ne!(ranked[0].0, 1, "рецепт не может быть первым: {ranked:?}");
    assert_eq!(ranked[2].0, 1, "рецепт — последним: {ranked:?}");
    assert!(ranked[0].1 > ranked[2].1);
}
