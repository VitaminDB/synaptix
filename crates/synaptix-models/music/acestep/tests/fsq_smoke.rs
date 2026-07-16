use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_acestep::fsq::Fsq;
use synaptix_music_acestep::loader::CompLoader;

fn dit_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_v15_xl_base.syn");
    p.exists().then_some(p)
}

#[test]
fn fsq_codebook_and_output() {
    let Some(path) = dit_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = CompLoader::open(&path, None, Device::Cpu).expect("open dit bundle");
    let fsq = Fsq::load(&ck, "tokenizer.quantizer").expect("load fsq");
    assert_eq!(fsq.dim(), 2048);

    // index 0 → all digits 0 → (0 - half)/half = -1 per dim
    assert_eq!(fsq.code_vec(0), [-1.0; 6]);
    // index 1 → digit 1, L=8 → 1·2/7 − 1 = -0.7142857 (preserve_symmetry), rest -1
    let c1 = fsq.code_vec(1);
    assert!((c1[0] - (2.0 / 7.0 - 1.0)).abs() < 1e-5, "c1[0]={}", c1[0]);
    assert!((c1[1] - (-1.0)).abs() < 1e-6);
    // last-dim carry: basis[5]=12800, L=5 → index 12800 → digit5=1 → 1·2/4 − 1 = -0.5
    let c = fsq.code_vec(12800);
    assert!((c[5] - (-0.5)).abs() < 1e-6);

    let out = fsq.get_output_from_indices(&[0u32, 1, 100, 63999]).expect("output");
    assert_eq!(out.dims(), &[1, 4, 2048]);
    let v: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
    eprintln!("[acestep-fsq] code_vec(1)={c1:?} out dims OK");
}
