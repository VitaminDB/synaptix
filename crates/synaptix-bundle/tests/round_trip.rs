use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::{Bundle, BundleBuilder, ChunkType, DirEntry, FileTag};

fn make_safetensors_blob() -> Vec<u8> {
    let w_bytes: Vec<u8> = (0..4)
        .flat_map(|i| (i as f32 + 1.0).to_le_bytes().to_vec())
        .collect();
    let b_bytes: Vec<u8> = vec![10.0f32, 20.0]
        .iter()
        .flat_map(|v| v.to_le_bytes().to_vec())
        .collect();
    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "layer.weight",
        TensorView::new(Dtype::F32, vec![2, 2], &w_bytes).unwrap(),
    );
    tensors.insert(
        "layer.bias",
        TensorView::new(Dtype::F32, vec![2], &b_bytes).unwrap(),
    );
    safetensors::serialize(&tensors, None).unwrap()
}

fn tempdir_path(name: &str) -> std::path::PathBuf {
    let mut d = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    d.push(format!("{}_{}", name, stamp));
    d
}

#[test]
fn pack_open_read_round_trip() {
    let work = tempdir_path("synaptix_bundle_rt");
    std::fs::create_dir_all(&work).unwrap();

    let st_bytes = make_safetensors_blob();
    let st_path = work.join("model.safetensors");
    std::fs::write(&st_path, &st_bytes).unwrap();
    let config = br#"{"hidden_size": 128, "num_layers": 4}"#;
    std::fs::write(work.join("config.json"), config).unwrap();
    let readme = b"# Test model\nNot a real model.";
    std::fs::write(work.join("README.md"), readme).unwrap();

    let bundle_path = work.join("model.syn");
    BundleBuilder::new("test-model", "1.0.0")
        .arch("test-arch")
        .purpose("embed")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes("config.json", config.to_vec(), FileTag::Inference)
        .unwrap()
        .add_file_bytes("README.md", readme.to_vec(), FileTag::Doc)
        .unwrap()
        .write(&bundle_path)
        .unwrap();

    let b = Bundle::open(&bundle_path).unwrap();
    assert_eq!(b.id(), "test-model");
    assert_eq!(b.version(), (1, 0));
    assert_eq!(b.meta().arch, "test-arch");
    assert_eq!(b.meta().purpose, "embed");

    let cfg = b.read_file("config.json").unwrap();
    assert_eq!(&*cfg, config);
    let rd = b.read_file("README.md").unwrap();
    assert_eq!(&*rd, readme);

    let files: Vec<&str> = b.list_files().map(|e| e.name.as_str()).collect();
    assert!(files.contains(&"config.json"));
    assert!(files.contains(&"README.md"));

    let slice = b.tensors_slice().unwrap();
    let st = safetensors::SafeTensors::deserialize(slice).unwrap();
    let w = st.tensor("layer.weight").unwrap();
    let bias = st.tensor("layer.bias").unwrap();
    assert_eq!(w.shape(), &[2, 2]);
    assert_eq!(bias.shape(), &[2]);
    let w_v: Vec<f32> = bytemuck_cast_to_f32(w.data());
    let bias_v: Vec<f32> = bytemuck_cast_to_f32(bias.data());
    assert_eq!(w_v, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(bias_v, vec![10.0, 20.0]);

    b.verify_full().unwrap();

    let tensors_entry = b
        .cdir()
        .entries
        .iter()
        .find(|e| e.is_alive() && e.kind_typed() == ChunkType::Tensors)
        .expect("tensors chunk must exist");
    assert_eq!(tensors_entry.name, "tensors:main");

    let _ = std::fs::remove_dir_all(&work);
}

fn bytemuck_cast_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn invalid_paths_rejected() {
    use synaptix_bundle::path::normalize;
    assert!(normalize("../escape").is_err());
    assert!(normalize("/abs").is_err());
    assert!(normalize("a//b").is_err());
}

#[test]
fn nested_paths_round_trip() {
    let work = tempdir_path("synaptix_bundle_nested");
    std::fs::create_dir_all(&work).unwrap();
    let st_bytes = make_safetensors_blob();
    let st_path = work.join("model.safetensors");
    std::fs::write(&st_path, &st_bytes).unwrap();

    let bundle_path = work.join("nested.syn");
    BundleBuilder::new("nested-model", "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes("config.json", b"{}".to_vec(), FileTag::Inference)
        .unwrap()
        .add_file_bytes(
            "examples/voice/sample.wav",
            b"FAKEWAV".to_vec(),
            FileTag::Example,
        )
        .unwrap()
        .add_file_bytes(
            "examples/voice/info.json",
            b"{}".to_vec(),
            FileTag::Example,
        )
        .unwrap()
        .add_file_bytes("docs/architecture.md", b"# notes".to_vec(), FileTag::Doc)
        .unwrap()
        .write(&bundle_path)
        .unwrap();

    let b = Bundle::open(&bundle_path).unwrap();
    let root = b.list_dir_shallow("");
    let mut names: Vec<String> = root
        .iter()
        .map(|e| match e {
            DirEntry::File(f) => f.name.clone(),
            DirEntry::Subdir(s) => format!("{s}/"),
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["config.json", "docs/", "examples/"]);

    let voice = b.list_dir_shallow("examples/voice");
    let voice_names: std::collections::BTreeSet<String> = voice
        .iter()
        .map(|e| match e {
            DirEntry::File(f) => f.name.clone(),
            DirEntry::Subdir(s) => s.to_string(),
        })
        .collect();
    assert!(voice_names.contains("examples/voice/sample.wav"));
    assert!(voice_names.contains("examples/voice/info.json"));

    let wav = b.read_file("examples/voice/sample.wav").unwrap();
    assert_eq!(&*wav, b"FAKEWAV");

    let _ = std::fs::remove_dir_all(&work);
}
