//! Чтение квантованных стопок `[E, N, K]` — так в бандле лежат веса
//! экспертов MoE: один блоб `.qpacked` и один `.qscales` на всю стопку,
//! срезы внутри идут подряд по ведущей оси.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::quant_layout::{QuantEntry, QuantManifest, MANIFEST_NAME};
use synaptix_bundle::{BundleBuilder, FileTag, CAP_QUANT_WEIGHTS};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::syn_bundle::SynBundleLoader;

const E: usize = 3;
const N: usize = 4;
const K: usize = 32;
/// MXFP8: байт на вес плюс один масштаб на каждые 32 веса.
const PACKED_PER_EXPERT: usize = N * K;
const SCALES_PER_EXPERT: usize = N * (K / 32);

fn tempdir(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    d.push(format!("{name}_{stamp}"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Маркеры вместо настоящего кванта: у эксперта `i` все упакованные байты
/// равны `i + 1`, а масштабы — `0x80 + i`. Так видно, что срез взят свой, а
/// не соседний и не начало блоба.
fn marked(per_slice: usize, mark: impl Fn(usize) -> u8) -> Vec<u8> {
    (0..E).flat_map(|i| std::iter::repeat_n(mark(i), per_slice)).collect()
}

fn make_bundle(work: &Path) -> PathBuf {
    let stack_packed = marked(PACKED_PER_EXPERT, |i| i as u8 + 1);
    let stack_scales = marked(SCALES_PER_EXPERT, |i| 0x80 + i as u8);
    // Одиночная матрица рядом со стопкой: читатель обязан работать с обеими
    // формами через один и тот же вызов.
    let flat_packed = vec![0x11u8; PACKED_PER_EXPERT];
    let flat_scales = vec![0x22u8; SCALES_PER_EXPERT];

    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "experts.down_proj.qpacked",
        TensorView::new(Dtype::U8, vec![stack_packed.len()], &stack_packed).unwrap(),
    );
    tensors.insert(
        "experts.down_proj.qscales",
        TensorView::new(Dtype::U8, vec![stack_scales.len()], &stack_scales).unwrap(),
    );
    tensors.insert(
        "mlp.up_proj.weight.qpacked",
        TensorView::new(Dtype::U8, vec![flat_packed.len()], &flat_packed).unwrap(),
    );
    tensors.insert(
        "mlp.up_proj.weight.qscales",
        TensorView::new(Dtype::U8, vec![flat_scales.len()], &flat_scales).unwrap(),
    );
    let blob = safetensors::serialize(&tensors, None).unwrap();
    let st_path = work.join("model.safetensors");
    std::fs::write(&st_path, blob).unwrap();

    let mut manifest = QuantManifest::new();
    manifest.tensors.insert(
        "experts.down_proj".into(),
        QuantEntry { format: "mxfp8".into(), shape: vec![E, N, K] },
    );
    manifest.tensors.insert(
        "mlp.up_proj.weight".into(),
        QuantEntry { format: "mxfp8".into(), shape: vec![N, K] },
    );

    let bundle_path = work.join("moe.syn");
    BundleBuilder::new("quant-stack-test", "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes(MANIFEST_NAME, serde_json::to_vec(&manifest).unwrap(), FileTag::Inference)
        .unwrap()
        .require_capability(CAP_QUANT_WEIGHTS)
        .write(&bundle_path)
        .unwrap();
    bundle_path
}

/// Байты квант-веса с хоста — то, что реально уедет в ядро.
fn bytes_of(w: &synaptix_core::tensor::quant::QuantWeight) -> (Vec<u8>, Vec<u8>) {
    let packed = w.packed_arc().expect("packed на месте");
    let packed = packed.as_cpu().expect("packed на хосте").as_bytes().to_vec();
    let scales = w.scales().as_cpu().expect("scales на хосте").as_bytes().to_vec();
    (packed, scales)
}

#[test]
fn stack_splits_into_one_weight_per_expert() {
    synaptix_kernels_cpu::ensure_registered();
    let work = tempdir("syn_quant_stack");
    let path = make_bundle(&work);
    let loader = SynBundleLoader::open(&path).unwrap();

    let stack = loader
        .load_quant_stack("experts.down_proj", Device::Cpu)
        .expect("вес квантован")
        .expect("стопка читается");
    assert_eq!(stack.len(), E);

    for (i, w) in stack.iter().enumerate() {
        assert_eq!(w.dtype(), DType::MXFP8);
        // Форма — одной матрицы, ведущая ось стопки в неё не входит.
        assert_eq!((w.n(), w.k()), (N, K));
        let (packed, scales) = bytes_of(w);
        assert_eq!(packed.len(), PACKED_PER_EXPERT);
        assert_eq!(scales.len(), SCALES_PER_EXPERT);
        assert!(packed.iter().all(|b| *b == i as u8 + 1), "эксперт {i}: чужие веса");
        assert!(scales.iter().all(|b| *b == 0x80 + i as u8), "эксперт {i}: чужие масштабы");
    }

    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn single_expert_reads_without_materialising_the_stack() {
    synaptix_kernels_cpu::ensure_registered();
    let work = tempdir("syn_quant_expert");
    let path = make_bundle(&work);
    let loader = SynBundleLoader::open(&path).unwrap();

    let w = loader
        .load_quant_expert("experts.down_proj", 2, Device::Cpu)
        .expect("вес квантован")
        .expect("эксперт читается");
    let (packed, scales) = bytes_of(&w);
    assert!(packed.iter().all(|b| *b == 3));
    assert!(scales.iter().all(|b| *b == 0x82));

    // Выход за границу стопки — ошибка, а не молчаливое чтение соседних байт.
    let out_of_range = loader
        .load_quant_expert("experts.down_proj", E, Device::Cpu)
        .expect("вес квантован");
    assert!(out_of_range.is_err());

    assert_eq!(loader.quant_dims("experts.down_proj"), Some((E, N, K)));

    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn plain_matrix_still_loads_as_one_weight() {
    synaptix_kernels_cpu::ensure_registered();
    let work = tempdir("syn_quant_flat");
    let path = make_bundle(&work);
    let loader = SynBundleLoader::open(&path).unwrap();

    // Обычная матрица — стопка из одного элемента: вызывающему не нужно
    // различать MoE и не-MoE.
    let stack = loader
        .load_quant_stack("mlp.up_proj.weight", Device::Cpu)
        .expect("вес квантован")
        .expect("матрица читается");
    assert_eq!(stack.len(), 1);
    assert_eq!((stack[0].n(), stack[0].k()), (N, K));

    let one = loader
        .load_quant("mlp.up_proj.weight", Device::Cpu)
        .expect("вес квантован")
        .expect("матрица читается");
    assert_eq!(bytes_of(&one).0, bytes_of(&stack[0]).0);

    // А вот стопку через одиночный вызов брать нельзя: вернуть первую
    // матрицу молча значило бы подсунуть модели веса одного эксперта вместо
    // всех.
    let err = match loader.load_quant("experts.down_proj", Device::Cpu).expect("вес квантован") {
        Ok(_) => panic!("стопку нельзя отдавать как одну матрицу"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("load_quant_stack"), "ошибка должна указывать путь: {err}");

    // Неквантованное имя — `None`, обычный путь загрузки.
    assert!(loader.load_quant_stack("нет.такого", Device::Cpu).is_none());

    std::fs::remove_dir_all(&work).ok();
}
