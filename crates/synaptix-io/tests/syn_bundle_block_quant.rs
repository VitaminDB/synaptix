//! Одноблобные форматы в бандле: `.qpacked` без `.qscales`, манифест
//! `format = "sq4"` / `"ggml:q4_0"`. Читатель отдаёт `QuantWeight` без
//! масштабов; на карте вес деквантуется бит в бит с CPU-эталоном.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::f16;
use safetensors::tensor::{Dtype, TensorView};
use synaptix_bundle::quant_layout::{QuantEntry, QuantManifest, MANIFEST_NAME};
use synaptix_bundle::{BundleBuilder, FileTag, CAP_QUANT_WEIGHTS};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::{dequant_row_f32, sq, GgmlType};
use synaptix_io::weights::syn_bundle::SynBundleLoader;

const N: usize = 8;
const K: usize = 288; // SQ: неполный хвостовой супер-блок; Q4_0: 9 блоков.
const E: usize = 2;

fn tempdir(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    d.push(format!("{name}_{stamp}"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn det(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Q4_0-блоки по маркеру: d = 0.5, ниббл = (индекс блока + эксперт) & 0xF.
fn q4_0_blob(expert: usize) -> Vec<u8> {
    let bb = GgmlType::Q4_0.block_bytes();
    let blocks = N * K / 32;
    let mut v = vec![0u8; blocks * bb];
    for b in 0..blocks {
        let blk = &mut v[b * bb..(b + 1) * bb];
        blk[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        let nib = ((b + expert) & 0xF) as u8;
        for q in &mut blk[2..18] {
            *q = nib | (nib << 4);
        }
    }
    v
}

fn make_bundle(work: &Path) -> (PathBuf, Vec<u8>, Vec<Vec<u8>>) {
    let x = det(11, N * K);
    let sq_blob = sq::quantize_matrix(4, &x, N, K).unwrap();
    let experts: Vec<Vec<u8>> = (0..E).map(q4_0_blob).collect();
    let stack: Vec<u8> = experts.concat();

    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert("mlp.w.qpacked", TensorView::new(Dtype::U8, vec![sq_blob.len()], &sq_blob).unwrap());
    tensors.insert("experts.down.qpacked", TensorView::new(Dtype::U8, vec![stack.len()], &stack).unwrap());
    let blob = safetensors::serialize(&tensors, None).unwrap();
    let st_path = work.join("model.safetensors");
    std::fs::write(&st_path, blob).unwrap();

    let mut manifest = QuantManifest::new();
    manifest.tensors.insert("mlp.w".into(), QuantEntry { format: "sq4".into(), shape: vec![N, K] });
    manifest
        .tensors
        .insert("experts.down".into(), QuantEntry { format: "ggml:q4_0".into(), shape: vec![E, N, K] });

    let bundle_path = work.join("block.syn");
    BundleBuilder::new("block-quant-test", "1.0.0")
        .add_tensors_from_safetensors(&st_path)
        .add_file_bytes(MANIFEST_NAME, serde_json::to_vec(&manifest).unwrap(), FileTag::Inference)
        .unwrap()
        .require_capability(CAP_QUANT_WEIGHTS)
        .write(&bundle_path)
        .unwrap();
    (bundle_path, sq_blob, experts)
}

#[test]
fn block_formats_load_without_scales() {
    synaptix_kernels_cpu::ensure_registered();
    let work = tempdir("syn_block_quant");
    let (path, sq_blob, experts) = make_bundle(&work);
    let loader = SynBundleLoader::open(&path).unwrap();

    let w = loader.load_quant("mlp.w", Device::Cpu).expect("квантован").expect("читается");
    assert_eq!(w.dtype(), DType::Sq { bits: 4 });
    assert_eq!((w.n(), w.k()), (N, K));
    assert!(w.scales_opt().is_none(), "у SQ нет отдельных масштабов");
    let packed = w.packed_arc().unwrap();
    assert_eq!(packed.as_cpu().unwrap().as_bytes(), &sq_blob[..]);

    let stack = loader.load_quant_stack("experts.down", Device::Cpu).expect("квантован").expect("стопка");
    assert_eq!(stack.len(), E);
    for (i, e) in stack.iter().enumerate() {
        assert_eq!(e.dtype(), DType::Ggml(GgmlType::Q4_0));
        assert_eq!(e.packed_arc().unwrap().as_cpu().unwrap().as_bytes(), &experts[i][..]);
    }
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn block_formats_dequantize_on_gpu_like_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let work = tempdir("syn_block_quant_gpu");
    let (path, _, _) = make_bundle(&work);
    let loader = SynBundleLoader::open(&path).unwrap();

    for (name, expert) in [("mlp.w", None), ("experts.down", Some(1usize))] {
        let w = match expert {
            None => loader.load_quant(name, Device::Cuda(0)).unwrap().unwrap(),
            Some(i) => loader.load_quant_expert(name, i, Device::Cuda(0)).unwrap().unwrap(),
        };
        let host = w.to_device(Device::Cpu).unwrap();
        let blob = host.packed_arc().unwrap();
        let blob = blob.as_cpu().unwrap().as_bytes().to_vec();
        let rb = w.block_row_bytes().unwrap();
        let mut want = vec![0f32; N * K];
        for r in 0..N {
            dequant_row_f32(w.dtype(), &blob[r * rb..(r + 1) * rb], K, &mut want[r * K..(r + 1) * K]).unwrap();
        }
        for out_dt in [DType::F16, DType::BF16] {
            let got = w.dequantize(out_dt).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for i in 0..N * K {
                let w32 = match out_dt {
                    DType::F16 => f16::from_f32(want[i]).to_f32(),
                    _ => half::bf16::from_f32(want[i]).to_f32(),
                };
                assert_eq!(got[i].to_bits(), w32.to_bits(), "{name} {out_dt:?} [{i}]: {} vs {}", got[i], w32);
            }
        }
    }
    std::fs::remove_dir_all(&work).ok();
}
