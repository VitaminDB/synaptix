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
