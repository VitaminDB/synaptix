use std::collections::HashMap;
use std::path::PathBuf;

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{compact, Bundle, BundleBuilder, BundleEditor, FileTag};

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

fn write_safetensors(path: &std::path::Path) {
    let w_bytes: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|v: f32| v.to_le_bytes().to_vec())
        .collect();
    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "layer.weight",
        TensorView::new(Dtype::F32, vec![2, 2], &w_bytes).unwrap(),
    );
    let blob = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(path, blob).unwrap();
}

fn make_bundle(work: &PathBuf, name: &str) -> PathBuf {
    let st_path = work.join(format!("{name}.safetensors"));
    write_safetensors(&st_path);

    let bundle_path = work.join(format!("{name}.syn"));
    BundleBuilder::new(name, "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes("config.json", br#"{"hidden":4}"#.to_vec(), FileTag::Inference)
        .unwrap()
        .add_file_bytes("README.md", b"hello".to_vec(), FileTag::Doc)
        .unwrap()
        .add_file_bytes(
            "examples/sample.txt",
            b"example".to_vec(),
            FileTag::Example,
        )
        .unwrap()
        .write(&bundle_path)
        .unwrap();
    bundle_path
}

fn read_layer_weight(b: &Bundle) -> Vec<f32> {
    let slice = b.tensors_slice().unwrap();
    let st = safetensors::SafeTensors::deserialize(slice).unwrap();
    let v = st.tensor("layer.weight").unwrap();
    v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn remove_then_reopen_hides_file() {
    let work = tempdir("synaptix_edit_rm");
    let path = make_bundle(&work, "model");
    let size_before = std::fs::metadata(&path).unwrap().len();

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.remove_file("README.md").unwrap();
    ed.commit().unwrap();

    let b = Bundle::open(&path).unwrap();
    assert!(b.read_file("README.md").is_err());
    assert_eq!(&*b.read_file("config.json").unwrap(), br#"{"hidden":4}"#);
    assert_eq!(&*b.read_file("examples/sample.txt").unwrap(), b"example");
    b.verify_full().unwrap();

    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(size_after < size_before + 4096);

    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn add_then_reopen_makes_file_readable() {
    let work = tempdir("synaptix_edit_add");
    let path = make_bundle(&work, "model");

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_file(
        "notes/extra.txt",
        b"hello from edit".to_vec(),
        FileTag::Inference,
    )
    .unwrap();
    ed.commit().unwrap();

    let b = Bundle::open(&path).unwrap();
    assert_eq!(&*b.read_file("notes/extra.txt").unwrap(), b"hello from edit");
    assert_eq!(read_layer_weight(&b), vec![1.0, 2.0, 3.0, 4.0]);
    b.verify_full().unwrap();
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn rename_changes_lookup_path() {
    let work = tempdir("synaptix_edit_mv");
    let path = make_bundle(&work, "model");

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.rename("README.md", "docs/intro.md").unwrap();
    ed.commit().unwrap();

    let b = Bundle::open(&path).unwrap();
    assert!(b.read_file("README.md").is_err());
    assert_eq!(&*b.read_file("docs/intro.md").unwrap(), b"hello");
    b.verify_full().unwrap();
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn replace_swaps_payload() {
    let work = tempdir("synaptix_edit_replace");
    let path = make_bundle(&work, "model");

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.replace_file(
        "config.json",
        br#"{"hidden":99}"#.to_vec(),
        FileTag::Inference,
    )
    .unwrap();
    ed.commit().unwrap();

    let b = Bundle::open(&path).unwrap();
    assert_eq!(&*b.read_file("config.json").unwrap(), br#"{"hidden":99}"#);
    assert_eq!(&*b.read_file("README.md").unwrap(), b"hello");
    b.verify_full().unwrap();
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn compact_reclaims_tombstones() {
    let work = tempdir("synaptix_edit_compact");
    let path = make_bundle(&work, "model");

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_file("filler.bin", vec![0u8; 4096], FileTag::Asset).unwrap();
    ed.commit().unwrap();

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.remove_file("filler.bin").unwrap();
    ed.commit().unwrap();

    let size_fragmented = std::fs::metadata(&path).unwrap().len();

    compact(&path, &path).unwrap();
    let size_compacted = std::fs::metadata(&path).unwrap().len();
    assert!(size_compacted < size_fragmented);

    let b = Bundle::open(&path).unwrap();
    assert!(b.read_file("filler.bin").is_err());
    assert_eq!(&*b.read_file("config.json").unwrap(), br#"{"hidden":4}"#);
    assert_eq!(&*b.read_file("README.md").unwrap(), b"hello");
    assert_eq!(read_layer_weight(&b), vec![1.0, 2.0, 3.0, 4.0]);
    b.verify_full().unwrap();
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn cannot_add_duplicate_paths() {
    let work = tempdir("synaptix_edit_dup");
    let path = make_bundle(&work, "model");
    let mut ed = BundleEditor::open(&path).unwrap();
    let err = ed
        .add_file("config.json", b"x".to_vec(), FileTag::Inference)
        .unwrap_err();
    assert!(matches!(err, synaptix_bundle::Error::InvalidPath { .. }));
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn remove_pending_add_drops_it() {
    let work = tempdir("synaptix_edit_drop");
    let path = make_bundle(&work, "model");
    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_file("scratch.txt", b"temp".to_vec(), FileTag::Asset).unwrap();
    ed.remove_file("scratch.txt").unwrap();
    ed.commit().unwrap();
    let b = Bundle::open(&path).unwrap();
    assert!(b.read_file("scratch.txt").is_err());
    let _ = std::fs::remove_dir_all(&work);
}

fn tensors_payload(name: &str, values: &[f32]) -> Vec<u8> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert(name, TensorView::new(Dtype::F32, vec![values.len()], &bytes).unwrap());
    safetensors::serialize(&tensors, None).unwrap()
}

fn read_named(b: &Bundle, component: &str, tensor: &str) -> Vec<f32> {
    let st = safetensors::SafeTensors::deserialize(b.tensors_slice_named(component).unwrap()).unwrap();
    st.tensor(tensor)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Вторая модель докладывается в готовый бандл своим чанком: основные веса не
/// трогаются, компонент виден в метаданных, повтор заменяет прежний чанк.
#[test]
fn add_tensors_component_keeps_main_weights() {
    let work = tempdir("synaptix_edit_component");
    let path = make_bundle(&work, "model");

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_tensors_component("extra", tensors_payload("x", &[5.0, 6.0])).unwrap();
    assert!(ed.add_tensors_component("main", Vec::new()).is_err(), "main — имя основного чанка");
    ed.commit().unwrap();

    let b = Bundle::open(&path).unwrap();
    assert_eq!(read_layer_weight(&b), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_named(&b, "extra", "x"), vec![5.0, 6.0]);
    assert_eq!(b.meta().components.get("extra").map(String::as_str), Some(""));
    b.verify_full().unwrap();

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_tensors_component("extra", tensors_payload("x", &[7.0])).unwrap();
    ed.commit().unwrap();
    let b = Bundle::open(&path).unwrap();
    assert_eq!(read_named(&b, "extra", "x"), vec![7.0]);
    assert_eq!(read_layer_weight(&b), vec![1.0, 2.0, 3.0, 4.0]);
}

/// `compact` многокомпонентного бандла: каждый тензорный чанк остаётся под
/// своим именем (раньше все шли через один временный файл, и выживал только
/// последний — под именем main).
#[test]
fn compact_keeps_every_tensors_component() {
    let work = tempdir("synaptix_edit_compact_components");
    let path = make_bundle(&work, "model");
    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_tensors_component("extra", tensors_payload("x", &[5.0, 6.0])).unwrap();
    ed.commit().unwrap();

    let out = work.join("compacted.syn");
    compact(&path, &out).unwrap();
    let b = Bundle::open(&out).unwrap();
    assert_eq!(read_layer_weight(&b), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_named(&b, "extra", "x"), vec![5.0, 6.0]);
    assert!(b.meta().components.contains_key("extra"));
    b.verify_full().unwrap();
}
