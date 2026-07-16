#![cfg(feature = "sha256")]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{Bundle, BundleBuilder, FileTag, CAP_SHA256_MANIFEST};

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

fn make_bundle(work: &PathBuf, with_sha256: bool) -> PathBuf {
    let st_path = work.join("model.safetensors");
    write_safetensors(&st_path);

    let bundle_path = work.join("model.syn");
    let mut b = BundleBuilder::new("sha-model", "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes("config.json", br#"{"hidden":4}"#.to_vec(), FileTag::Inference)
        .unwrap()
        .add_file_bytes("README.md", b"hello".to_vec(), FileTag::Doc)
        .unwrap();
    if with_sha256 {
        b = b.with_sha256(true);
    }
    b.write(&bundle_path).unwrap();
    bundle_path
}

#[test]
fn sha256_present_in_cdir_when_built_with_feature() {
    let work = tempdir("sha256_present");
    let path = make_bundle(&work, true);
    let b = Bundle::open(&path).unwrap();
    assert!(b.meta().optional_caps.contains(&CAP_SHA256_MANIFEST.to_string()));
    assert!(b.meta().manifest_sha256.is_some());
    for e in b.cdir().entries.iter() {
        assert!(e.sha256.is_some());
        assert_eq!(e.sha256.as_ref().unwrap().len(), 32);
    }
    assert!(b.verify_sha256().unwrap());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn sha256_absent_when_built_without() {
    let work = tempdir("sha256_absent");
    let path = make_bundle(&work, false);
    let b = Bundle::open(&path).unwrap();
    assert!(b.meta().manifest_sha256.is_none());
    for e in b.cdir().entries.iter() {
        assert!(e.sha256.is_none());
    }
    assert!(!b.verify_sha256().unwrap());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn payload_bitflip_detected_by_sha256() {
    let work = tempdir("sha256_bitflip");
    let path = make_bundle(&work, true);

    let b = Bundle::open(&path).unwrap();
    let readme = b.cdir().entries.iter().find(|e| e.name == "README.md").unwrap().clone();
    drop(b);

    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    f.seek(SeekFrom::Start(readme.payload_off)).unwrap();
    let mut byte = [0u8; 1];
    f.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xff;
    f.seek(SeekFrom::Start(readme.payload_off)).unwrap();
    f.write_all(&byte).unwrap();
    f.sync_all().unwrap();
    drop(f);

    let b = Bundle::open(&path).unwrap();
    let crc_err = b.verify_full().unwrap_err();
    assert!(format!("{crc_err}").contains("crc mismatch"));
    let sha_err = b.verify_sha256().unwrap_err();
    let msg = format!("{sha_err}");
    assert!(msg.contains("crc mismatch") || msg.contains("manifest"));
    let _ = std::fs::remove_dir_all(&work);
}
