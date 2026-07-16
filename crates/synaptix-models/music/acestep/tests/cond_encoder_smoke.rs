use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::cond_encoder::ConditionEncoder;
use synaptix_music_acestep::config::DitConfig;
use synaptix_music_acestep::loader::CompLoader;

fn dit_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_v15_xl_base.syn");
    p.exists().then_some(p)
}

#[test]
fn condition_encoder_fuses_text_lyric() {
    let Some(path) = dit_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = CompLoader::open(&path, None, Device::Cpu).expect("open dit bundle");
    let cfg = DitConfig::xl_base();
    let enc = ConditionEncoder::load(&ck, &cfg).expect("cond encoder");

    let lt = 4usize;
    let ll = 8usize;
    let text = Tensor::zeros(vec![1usize, lt, cfg.text_hidden_dim], DType::F32, Device::Cpu).unwrap();
    let lyric = Tensor::zeros(vec![1usize, ll, cfg.text_hidden_dim], DType::F32, Device::Cpu).unwrap();
    let out = enc.forward(&text, &lyric).expect("forward");
    let dims = out.dims().to_vec();
    eprintln!("[acestep-cond] text[1,{lt},1024]+lyric[1,{ll},1024] -> {dims:?}");
    assert_eq!(dims, vec![1, ll + lt, cfg.encoder_hidden_size]);
    let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));

    // timbre: ref_latent [1,T,64] → [1,1,2048]
    let refl = Tensor::zeros(vec![1usize, 30usize, 64], DType::F32, Device::Cpu).unwrap();
    let timbre = enc.timbre_emb(&refl).expect("timbre");
    assert_eq!(timbre.dims(), &[1, 1, cfg.encoder_hidden_size]);
    let tv: Vec<f32> = timbre.flatten_all().unwrap().to_vec1().unwrap();
    assert!(tv.iter().all(|x| x.is_finite()));
}
