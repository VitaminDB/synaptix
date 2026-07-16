use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{BundleBuilder, FileTag};
use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

fn tempdir(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    d.push(format!("{}_{}", name, stamp));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn make_bundle(work: &Path) -> PathBuf {
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_bytes(&[10.0, 20.0]);
    let c = f32_bytes(&[-1.0, 0.5, 7.0, 8.0, 9.0, 11.0]);
    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert("model.a.weight", TensorView::new(Dtype::F32, vec![2, 2], &a).unwrap());
    tensors.insert("model.b.weight", TensorView::new(Dtype::F32, vec![2], &b).unwrap());
    tensors.insert("model.c.weight", TensorView::new(Dtype::F32, vec![3, 2], &c).unwrap());
    let blob = safetensors::serialize(&tensors, None).unwrap();
    let st_path = work.join("model.safetensors");
    std::fs::write(&st_path, blob).unwrap();

    let bundle_path = work.join("model.syn");
    BundleBuilder::new("loader-test", "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .write(&bundle_path)
        .unwrap();
    bundle_path
}

#[test]
fn loads_bit_exact_and_caches_index() {
    synaptix_kernels_cpu::ensure_registered();
    let work = tempdir("syn_loader");
    let path = make_bundle(&work);

    let loader = SynBundleLoader::open(&path).unwrap();

    // names() exposes every tensor regardless of read order.
    let mut names = loader.names();
    names.sort();
    assert_eq!(names, vec!["model.a.weight", "model.b.weight", "model.c.weight"]);

    // Repeated loads return identical bytes — the cached index must not corrupt
    // offsets across calls (the bug this guards: re-deriving the slice per call).
    for _ in 0..3 {
        let a = loader.load("model.a.weight").unwrap();
        assert_eq!(a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                   vec![1.0, 2.0, 3.0, 4.0]);
        let c = loader.load("model.c.weight").unwrap();
        assert_eq!(c.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                   vec![-1.0, 0.5, 7.0, 8.0, 9.0, 11.0]);
        let b = loader.load("model.b.weight").unwrap();
        assert_eq!(b.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![10.0, 20.0]);
    }

    // load_to with a dtype conversion (F32 -> F16 -> back) stays bit-exact for
    // these exactly-representable values.
    let a16 = loader.load_to("model.a.weight", Device::Cpu, DType::F16).unwrap();
    assert_eq!(a16.dtype(), DType::F16);
    assert_eq!(a16.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
               vec![1.0, 2.0, 3.0, 4.0]);

    assert!(loader.load("model.missing.weight").is_err());
}
