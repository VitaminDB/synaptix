use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{Bundle, BundleBuilder, BundleEditor, FileTag};

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

fn write_safetensors(path: &Path) {
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
        .add_file_bytes("README.md", b"hello".to_vec(), FileTag::Doc)
        .unwrap()
        .write(&bundle_path)
        .unwrap();
    bundle_path
}

#[test]
fn editor_takes_exclusive_lock() {
    let work = tempdir("synaptix_flock");
    let path = make_bundle(&work, "model");

    let editor1 = BundleEditor::open(&path).unwrap();

    let start = std::time::Instant::now();
    let err = BundleEditor::open(&path);
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 1);
    assert!(matches!(err, Err(synaptix_bundle::Error::BundleBusy(_))));

    let _bundle = Bundle::open(&path).expect("reader still works while editor holds lock");

    drop(editor1);
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn lock_released_on_drop() {
    let work = tempdir("synaptix_flock_drop");
    let path = make_bundle(&work, "model");

    {
        let _editor = BundleEditor::open(&path).unwrap();
    }
    let _editor2 = BundleEditor::open(&path).expect("lock released on drop");
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn editor_commit_is_append_only() {
    let work = tempdir("synaptix_append_only");
    let path = make_bundle(&work, "model");
    let before = std::fs::read(&path).unwrap();

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_file("notes.txt", b"hi".to_vec(), FileTag::Inference).unwrap();
    ed.commit().unwrap();

    let after = std::fs::read(&path).unwrap();
    assert!(after.len() > before.len());
    assert_eq!(&after[..before.len()], &before[..]);

    let bundle = Bundle::open(&path).unwrap();
    assert_eq!(&*bundle.read_file("notes.txt").unwrap(), b"hi");
    bundle.verify_full().unwrap();
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn scan_back_recovers_from_torn_tail() {
    let work = tempdir("synaptix_torn_tail");
    let path = make_bundle(&work, "model");
    let pre_content = std::fs::read(&path).unwrap();

    let mut ed = BundleEditor::open(&path).unwrap();
    ed.add_file(
        "late_addition.txt",
        b"this edit gets torn".to_vec(),
        FileTag::Inference,
    )
    .unwrap();
    ed.commit().unwrap();
    let size_after_commit = std::fs::metadata(&path).unwrap().len();
    assert!(size_after_commit > pre_content.len() as u64);

    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(size_after_commit - 100).unwrap();
    drop(f);

    let bundle = Bundle::open(&path).expect("scan-back recovery should succeed");
    assert!(bundle.read_file("late_addition.txt").is_err());
    assert_eq!(&*bundle.read_file("README.md").unwrap(), b"hello");
    bundle.verify_full().unwrap();
    drop(bundle);

    let post_content = std::fs::read(&path).unwrap();
    let common = pre_content.len().min(post_content.len());
    assert_eq!(&pre_content[..common], &post_content[..common]);
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn scan_back_refuses_completely_corrupt_file() {
    let work = tempdir("synaptix_corrupt");
    let path = make_bundle(&work, "model");
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(32).unwrap();
    drop(f);
    let err = Bundle::open(&path);
    assert!(err.is_err());
    let _ = std::fs::remove_dir_all(&work);
}
