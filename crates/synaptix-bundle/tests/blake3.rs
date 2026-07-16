#![cfg(feature = "blake3")]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{Bundle, BundleBuilder, FileTag, CAP_BLAKE3_MANIFEST};

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

fn make_bundle(work: &PathBuf, with_blake3: bool) -> PathBuf {
    let st = work.join("model.safetensors");
    write_safetensors(&st);

    let path = work.join("model.syn");
    let mut b = BundleBuilder::new("blake3-test", "1.0.0")
        .add_tensors_from_safetensors(&st)
        .add_file_bytes("config.json", br#"{"k":1}"#.to_vec(), FileTag::Inference)
        .unwrap()
        .add_file_bytes("README.md", b"hello".to_vec(), FileTag::Doc)
        .unwrap();
    if with_blake3 {
        b = b.with_blake3(true);
    }
    b.write(&path).unwrap();
    path
}

#[test]
fn blake3_present_when_built() {
    let work = tempdir("blake3_present");
    let path = make_bundle(&work, true);
    let b = Bundle::open(&path).unwrap();
    assert!(b.meta().optional_caps.contains(&CAP_BLAKE3_MANIFEST.to_string()));
    assert!(b.meta().manifest_blake3.is_some());
    for e in b.cdir().entries.iter() {
        assert!(e.blake3.is_some());
        assert_eq!(e.blake3.as_ref().unwrap().len(), 32);
    }
    assert!(b.verify_blake3().unwrap());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn blake3_absent_when_not_requested() {
    let work = tempdir("blake3_absent");
    let path = make_bundle(&work, false);
    let b = Bundle::open(&path).unwrap();
    assert!(b.meta().manifest_blake3.is_none());
    for e in b.cdir().entries.iter() {
        assert!(e.blake3.is_none());
    }
    assert!(!b.verify_blake3().unwrap());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn blake3_detects_payload_bitflip() {
    let work = tempdir("blake3_bitflip");
    let path = make_bundle(&work, true);

    let b = Bundle::open(&path).unwrap();
    let readme = b.cdir().entries.iter().find(|e| e.name == "README.md").unwrap().clone();
    drop(b);

    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(readme.payload_off)).unwrap();
    let mut byte = [0u8; 1];
    f.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xff;
    f.seek(SeekFrom::Start(readme.payload_off)).unwrap();
    f.write_all(&byte).unwrap();
    f.sync_all().unwrap();
    drop(f);

    let b = Bundle::open(&path).unwrap();
    assert!(b.verify_full().is_err());
    assert!(b.verify_blake3().is_err());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
#[cfg(feature = "sha256")]
fn blake3_and_sha256_coexist() {
    let work = tempdir("blake3_sha256_coexist");
    let st = work.join("model.safetensors");
    write_safetensors(&st);

    let path = work.join("both.syn");
    BundleBuilder::new("both", "1.0.0")
        .add_tensors_from_safetensors(&st)
        .add_file_bytes("config.json", b"{}".to_vec(), FileTag::Inference)
        .unwrap()
        .with_sha256(true)
        .with_blake3(true)
        .write(&path)
        .unwrap();

    let b = Bundle::open(&path).unwrap();
    assert!(b.meta().manifest_sha256.is_some());
    assert!(b.meta().manifest_blake3.is_some());
    assert!(b.meta().optional_caps.contains(&"sha256-manifest".to_string()));
    assert!(b.meta().optional_caps.contains(&"blake3-chunks".to_string()));
    for e in b.cdir().entries.iter() {
        assert!(e.sha256.is_some());
        assert!(e.blake3.is_some());
    }
    assert!(b.verify_sha256().unwrap());
    assert!(b.verify_blake3().unwrap());
    let _ = std::fs::remove_dir_all(&work);
}
