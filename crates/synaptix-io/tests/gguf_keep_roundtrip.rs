//! GGUF → `.syn` в режиме `OutDtype::Keep` → `SynBundleLoader`: кванты
//! остаются блоками ggml (`.qpacked` + `quant_manifest.json`,
//! `required_caps` = `syn-quant-v1`), а читатель бандла отдаёт те же байты и
//! ту же плотную матрицу, что и прямое чтение `.gguf`. Пропуск, если
//! эталонного файла нет: `~/Storage/syn_models/gguf-tests/Qwen3-0.6B-Q8_0.gguf`
//! (или `SYN_GGUF_KEEP_SRC`). Плотный деквант сверяется на CPU.

use std::path::PathBuf;

use synaptix_bundle::inspect::QuantKind;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::GgmlType;
use synaptix_gguf::convert::{convert_to_syn, ConvertOptions};
use synaptix_gguf::OutDtype;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

fn source() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SYN_GGUF_KEEP_SRC") {
        return Some(PathBuf::from(p)).filter(|p| p.is_file());
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    Some(home.join("Storage/syn_models/gguf-tests/Qwen3-0.6B-Q8_0.gguf")).filter(|p| p.is_file())
}

#[test]
fn keep_conversion_roundtrips_through_bundle_loader() {
    let Some(src) = source() else {
        eprintln!("нет эталонного GGUF — пропуск");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("keep.syn");
    let opts = ConvertOptions { dtype: OutDtype::Keep, ..Default::default() };
    let report = convert_to_syn(&src, &out, &opts, None).expect("convert keep");
    assert!(report.kept_quant > 0, "ни один тензор не остался блоками");
    assert!(report.files.iter().any(|f| f == "quant_manifest.json"));

    let gguf = SynBundleLoader::open(&src).expect("open gguf");
    let g = gguf.gguf().expect("gguf backend").clone();
    let syn = SynBundleLoader::open(&out).expect("open keep.syn");
    assert!(syn.gguf().is_none(), "keep.syn должен читаться как бандл, не как gguf");
    assert!(
        synaptix_bundle::Bundle::open(&out).unwrap().meta().required_caps.iter().any(|c| c == synaptix_bundle::CAP_QUANT_WEIGHTS),
        "required_caps без syn-quant-v1"
    );

    // Квант-веса: тот же формат, форма и байты.
    let mut checked = 0usize;
    for name in g.names() {
        let Some((slices, n, k)) = g.quant_dims(name) else {
            // Плотный тензор (нормы): в бандле тоже плотный и равный.
            let a = g.load_to(name, Device::Cpu, DType::F32).unwrap();
            let b = syn.load_to(name, Device::Cpu, DType::F32).unwrap();
            assert_eq!(a.dims(), b.dims(), "{name}: форма плотного тензора");
            let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(va, vb, "{name}: плотный тензор");
            continue;
        };
        let ty = g.quant_kind(name).unwrap();
        assert_eq!(syn.quant_kind(name), Some(QuantKind::Ggml(ty)), "{name}: формат в манифесте");
        assert_eq!(syn.quant_dims(name), Some((slices, n, k)), "{name}: размеры в манифесте");
        if slices == 1 {
            let a = g.load_quant(name, Device::Cpu).unwrap().unwrap();
            let b = syn.load_quant(name, Device::Cpu).unwrap().unwrap();
            assert_eq!(b.dtype(), DType::Ggml(ty));
            let pa = a.packed_arc().unwrap();
            let pb = b.packed_arc().unwrap();
            assert_eq!(pa.as_cpu().unwrap().as_bytes(), pb.as_cpu().unwrap().as_bytes(), "{name}: байты блоба");
            checked += 1;
        }
    }
    assert!(checked >= 10, "проверено слишком мало квант-весов: {checked}");
    // Плотное чтение кванта из бандла на CPU совпадает с GGUF.
    let q = "model.layers.0.self_attn.q_proj.weight";
    let a = g.load_to(q, Device::Cpu, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = syn.load_to(q, Device::Cpu, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "{q}: плотный деквант из бандла");
    let _ = GgmlType::Q8_0;
    eprintln!("keep.syn: {} тензоров блоками, {checked} сверено байт в байт", report.kept_quant);
}
