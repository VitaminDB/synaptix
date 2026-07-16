use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::config::DitConfig;
use synaptix_music_acestep::dit::Dit;
use synaptix_music_acestep::loader::CompLoader;

fn dit_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_v15_xl_base.syn");
    p.exists().then_some(p)
}

#[test]
fn dit_forward_shape_and_finite() {
    if std::env::var("ACESTEP_DIT").is_err() {
        return;
    }
    let Some(path) = dit_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = CompLoader::open(&path, None, Device::Cpu).expect("open dit bundle");
    let cfg = DitConfig::xl_base();
    let dit = Dit::load(&ck, &cfg, DType::F32, DType::F32).expect("load dit");

    let t = 8usize;
    let hidden = Tensor::zeros(vec![1usize, t, 64], DType::F32, Device::Cpu).unwrap();
    let context = Tensor::zeros(vec![1usize, t, 128], DType::F32, Device::Cpu).unwrap();
    let enc = Tensor::zeros(vec![1usize, 12usize, 2048], DType::F32, Device::Cpu).unwrap();

    let v = dit.forward(&hidden, 1.0, 0.0, &context, &enc).expect("dit forward");
    let dims = v.dims().to_vec();
    eprintln!("[acestep-dit] forward -> velocity {dims:?}");
    assert_eq!(dims, vec![1, t, 64]);
    let vv: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
    assert!(vv.iter().all(|x| x.is_finite()), "velocity must be finite");
}
