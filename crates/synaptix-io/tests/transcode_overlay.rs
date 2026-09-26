//! Перекодировка при загрузке (`weights::transcode`): оверлей в памяти и в
//! дисковом кэше поверх `.gguf` (Qwen3-0.6B Q8_0), полный бандл через
//! `write_bundle`. Пропуск без эталонного файла или CUDA.

use std::path::PathBuf;
use std::sync::Arc;

use synaptix_bundle::inspect::QuantKind;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::transcode::{self, Placement, Request, TranscodeSpec};
use synaptix_io::weights::WeightLoader;
use synaptix_io::SynBundleLoader;

fn source() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    Some(home.join("Storage/syn_models/gguf-tests/Qwen3-0.6B-Q8_0.gguf")).filter(|p| p.is_file())
}

fn setup() -> Option<(PathBuf, Device)> {
    let src = source()?;
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return None;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    Some((src, Device::Cuda(0)))
}

fn to_vec(t: &synaptix_core::tensor::Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Перекодированный вес близок к исходному: MSE против дисперсии.
fn check_close(a: &SynBundleLoader, b: &SynBundleLoader, name: &str, dev: Device, tol: f32) {
    let qa = a.load_quant(name, dev).unwrap().unwrap();
    let qb = b.load_quant(name, dev).unwrap().unwrap();
    let va = to_vec(&qa.dequantize(DType::F16).unwrap());
    let vb = to_vec(&qb.dequantize(DType::F16).unwrap());
    let var: f32 = va.iter().map(|x| x * x).sum::<f32>() / va.len() as f32;
    let mse: f32 = va.iter().zip(&vb).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / va.len() as f32;
    assert!(mse < var * tol, "{name}: mse {mse} против дисперсии {var}");
}

#[test]
fn overlay_in_memory_and_on_disk() {
    let Some((src, dev)) = setup() else { return };
    let base = SynBundleLoader::open(&src).unwrap();
    let q = "model.layers.0.mlp.down_proj.weight";
    assert_eq!(base.quant_kind(q), Some(QuantKind::Ggml(synaptix_core::quant::GgmlType::Q8_0)));

    // Память: attn/mlp → SQ4, голова/эмбеддинг как есть.
    let mut mem = SynBundleLoader::open(&src).unwrap();
    let spec = TranscodeSpec { lm_head: None, embed: None, ..TranscodeSpec::uniform(DType::Sq { bits: 4 }) };
    let report = mem.transcode(&src, spec.clone(), dev, Placement::Memory, None).unwrap();
    assert_eq!(report.placement, "memory");
    assert!(report.tensors >= 28 * 7, "перекодировано {} тензоров", report.tensors);
    assert!(report.bytes_after < report.bytes_before, "SQ4 должен быть компактнее Q8_0");
    assert_eq!(mem.quant_kind(q), Some(QuantKind::Sq(4)));
    assert_eq!(mem.quant_dims(q), base.quant_dims(q));
    // Эмбеддинг не перекодирован.
    assert_eq!(mem.quant_kind("model.embed_tokens.weight"), base.quant_kind("model.embed_tokens.weight"));
    check_close(&base, &mem, q, dev, 0.02);
    // Блоб оверлея — ровно SQ4-размер.
    let (packed, scales) = mem.quant_blob_slices(q).unwrap().unwrap();
    let (_, n, k) = base.quant_dims(q).unwrap();
    assert_eq!(packed.len(), n * synaptix_core::quant::sq::row_bytes(4, k));
    assert!(scales.is_empty());
    // Плотное чтение перекодированного веса — деквант SQ на хосте.
    let dense = mem.load_to(q, Device::Cpu, DType::F32).unwrap();
    assert_eq!(dense.dims(), &[n, k]);

    // Диск: кэш пишется один раз, второе открытие его находит.
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("SYN_TRANSCODE_CACHE", tmp.path());
    let mut disk = SynBundleLoader::open(&src).unwrap();
    let r1 = disk.transcode(&src, spec.clone(), dev, Placement::Disk, None).unwrap();
    assert_eq!(r1.placement, "disk");
    let cache = r1.cache_path.clone().unwrap();
    assert!(cache.is_file(), "кэш {} не записан", cache.display());
    assert_eq!(disk.quant_kind(q), Some(QuantKind::Sq(4)));
    check_close(&mem, &disk, q, dev, 1e-9);
    // Второй раз — из кэша (реестр оверлеев чист: заявки нет).
    let mut disk2 = SynBundleLoader::open(&src).unwrap();
    let r2 = disk2.transcode(&src, spec.clone(), dev, Placement::Disk, None).unwrap();
    assert_eq!(r2.placement, "disk-cached");
    assert_eq!(r2.cache_path.as_deref(), Some(cache.as_path()));
    check_close(&disk, &disk2, q, dev, 1e-9);
    std::env::remove_var("SYN_TRANSCODE_CACHE");
}

#[test]
fn request_applies_to_every_open() {
    let Some((src, dev)) = setup() else { return };
    let spec = TranscodeSpec { lm_head: None, embed: None, ..TranscodeSpec::uniform(DType::Sq { bits: 3 }) };
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let s2 = seen.clone();
    let progress: transcode::Progress = Arc::new(move |_, _, _| {
        s2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    {
        let _g = transcode::request(&src, Request { spec: spec.clone(), device: dev, placement: Placement::Memory, progress: Some(progress) });
        let a = SynBundleLoader::open(&src).unwrap();
        let b = SynBundleLoader::open(&src).unwrap();
        assert_eq!(a.quant_kind("model.layers.1.self_attn.q_proj.weight"), Some(QuantKind::Sq(3)));
        assert_eq!(b.quant_kind("model.layers.1.self_attn.q_proj.weight"), Some(QuantKind::Sq(3)));
        assert!(Arc::ptr_eq(a.overlay().unwrap(), b.overlay().unwrap()), "оверлей должен разделяться");
        let n = seen.load(std::sync::atomic::Ordering::Relaxed);
        assert!(n > 0 && n == a.overlay().unwrap().report.tensors, "прогресс {n}");
    }
    // Заявка снята: обычное открытие — без оверлея.
    let c = SynBundleLoader::open(&src).unwrap();
    assert!(c.overlay().is_none());
    assert_eq!(c.quant_kind("model.layers.1.self_attn.q_proj.weight"), Some(QuantKind::Ggml(synaptix_core::quant::GgmlType::Q8_0)));
}

#[test]
fn full_bundle_is_standalone() {
    let Some((src, dev)) = setup() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3-sq4.syn");
    let spec = TranscodeSpec { requant: true, quantize_dense: true, embed: Some(DType::Sq { bits: 8 }), ..TranscodeSpec::uniform(DType::Sq { bits: 4 }) };
    let report = transcode::write_bundle(&src, &out, &spec, dev, None).unwrap();
    assert!(out.is_file());
    let l = SynBundleLoader::open(&out).unwrap();
    assert!(l.overlay().is_none());
    assert_eq!(l.quant_kind("model.layers.0.mlp.gate_proj.weight"), Some(QuantKind::Sq(4)));
    assert_eq!(l.quant_kind("model.embed_tokens.weight"), Some(QuantKind::Sq(8)));
    // Нормы скопированы плотно, файлы модели — на месте.
    let norm = l.load_to("model.layers.0.input_layernorm.weight", Device::Cpu, DType::F32).unwrap();
    assert_eq!(norm.dims().len(), 1);
    assert!(l.read_file("config.json").is_some() && l.read_file("tokenizer.json").is_some());
    assert!(synaptix_bundle::Bundle::open(&out).unwrap().meta().required_caps.iter().any(|c| c == synaptix_bundle::CAP_QUANT_WEIGHTS));
    let base = SynBundleLoader::open(&src).unwrap();
    check_close(&base, &l, "model.layers.0.mlp.gate_proj.weight", dev, 0.02);
    eprintln!("{} тензоров, {:.2} → {:.2} ГБ за {:.1} с", report.tensors, report.bytes_before as f64 / 1e9, report.bytes_after as f64 / 1e9, report.seconds);
}

/// Плотный бандл из одной MLP-матрицы 64×256.
fn tiny_dense_bundle(dir: &std::path::Path) -> PathBuf {
    use safetensors::tensor::{Dtype, TensorView};
    let vals: Vec<u8> = (0..64 * 256)
        .flat_map(|i| (((i * 37) % 101) as f32 / 50.0 - 1.0).to_le_bytes())
        .collect();
    let mut t = std::collections::HashMap::new();
    t.insert("model.layers.0.mlp.down_proj.weight", TensorView::new(Dtype::F32, vec![64, 256], &vals).unwrap());
    let st = dir.join("model.safetensors");
    std::fs::write(&st, safetensors::serialize(&t, None).unwrap()).unwrap();
    let out = dir.join("tiny.syn");
    synaptix_bundle::BundleBuilder::new("tiny", "1.0.0").add_tensors_from_safetensors(&st).write(&out).unwrap();
    out
}

/// Обрубок под именем кэша (прерванная запись старой версии) не валит
/// загрузку: кэш пересобирается. Матрица крошечная — на карте только
/// CUDA-контекст (CPU-бэкенд квантовать не умеет).
#[test]
fn broken_disk_cache_is_rebuilt() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let work = tempfile::tempdir().unwrap();
    let src = tiny_dense_bundle(work.path());
    let dev = Device::Cuda(0);
    let spec = TranscodeSpec { quantize_dense: true, ..TranscodeSpec::uniform(DType::Sq { bits: 4 }) };
    std::env::set_var("SYN_TRANSCODE_CACHE", work.path().join("cache"));
    let q = "model.layers.0.mlp.down_proj.weight";
    let mut first = SynBundleLoader::open(&src).unwrap();
    let r1 = first.transcode(&src, spec.clone(), dev, Placement::Disk, None).unwrap();
    assert_eq!((r1.placement, r1.tensors), ("disk", 1));
    let cache = r1.cache_path.clone().unwrap();
    assert!(!cache.with_extension("syn.part").exists(), "временный файл остался");
    let len = std::fs::metadata(&cache).unwrap().len();
    std::fs::OpenOptions::new().write(true).open(&cache).unwrap().set_len(len / 3).unwrap();
    let mut again = SynBundleLoader::open(&src).unwrap();
    let r2 = again.transcode(&src, spec, dev, Placement::Disk, None).unwrap();
    assert_eq!(r2.placement, "disk", "битый кэш должен пересобраться");
    assert_eq!(std::fs::metadata(&cache).unwrap().len(), len);
    assert_eq!(again.quant_kind(q), Some(QuantKind::Sq(4)));
    std::env::remove_var("SYN_TRANSCODE_CACHE");
}
