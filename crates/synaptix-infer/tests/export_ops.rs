//! F8: export safetensors / .syn → round-trip read → bit-exact.

use std::collections::HashMap;

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_infer::export::{export_safetensors, export_syn};
use synaptix_kernels_cpu::ensure_registered;

fn setup() { ensure_registered(); }

fn sample_tensors() -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    m.insert(
        "encoder.weight".to_string(),
        Tensor::from_vec::<_, f32>((0..12).map(|i| i as f32 * 0.5).collect(), vec![3, 4], Device::Cpu).unwrap(),
    );
    m.insert(
        "encoder.bias".to_string(),
        Tensor::from_vec::<_, f32>(vec![-1.0, 0.0, 1.0], vec![3], Device::Cpu).unwrap(),
    );
    m.insert(
        "token_ids".to_string(),
        Tensor::from_vec::<_, i64>(vec![5, 10, 15, 20], vec![2, 2], Device::Cpu).unwrap(),
    );
    m
}

fn assert_bit_exact(st_bytes: &[u8], original: &HashMap<String, Tensor>) {
    let st = SafeTensors::deserialize(st_bytes).expect("valid safetensors");
    assert_eq!(st.names().len(), original.len(), "tensor count");
    for (name, t) in original {
        let tv = st.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}"));
        assert_eq!(tv.shape(), t.dims(), "shape of {name}");
        assert_eq!(tv.data(), t.to_bytes().unwrap().as_slice(), "bytes of {name}");
    }
}

fn tmp_path(stem: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("synaptix_infer_{}_{}.{}", stem, std::process::id(), ext))
}

#[test]
fn t45_1_safetensors_roundtrip() {
    setup();
    let tensors = sample_tensors();
    let path = tmp_path("st", "safetensors");
    export_safetensors(&tensors, &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_bit_exact(&bytes, &tensors);

    std::fs::remove_file(&path).ok();
}

#[test]
fn t45_2_syn_bundle_roundtrip() {
    setup();
    let tensors = sample_tensors();
    let path = tmp_path("syn", "syn");
    export_syn(&tensors, &path).unwrap();

    let bundle = Bundle::open(&path).unwrap();
    let slice = bundle.tensors_slice().unwrap();
    assert_bit_exact(slice, &tensors);
    drop(bundle);

    std::fs::remove_file(&path).ok();
    // Промежуточный staged-файл должен быть убран.
    assert!(!path.with_extension("export_stage.safetensors").exists());
}
