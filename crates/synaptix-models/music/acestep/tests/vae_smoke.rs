use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::AceStepVae;

fn vae_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_vae.syn");
    p.exists().then_some(p)
}

#[test]
fn vae_decode_shape_and_finite() {
    let Some(path) = vae_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let vae = AceStepVae::open(&path, Device::Cpu).expect("open vae");
    let hop = vae.config().hop_length();
    assert_eq!(hop, 1920);

    let t = 6usize;
    let lat = vae.config().decoder_input_channels;
    let z = Tensor::zeros(vec![1usize, lat, t], DType::F32, Device::Cpu).unwrap();
    let out = vae.decode(&z).expect("decode");
    let dims = out.dims().to_vec();
    eprintln!("[acestep-vae] decode [1,{lat},{t}] -> {dims:?}");
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], vae.config().audio_channels);
    assert_eq!(dims[2], t * hop);
    let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "decode output has non-finite values");

    // tiled decode: T>chunk → halo-tiling, длина = T·hop, конечно
    let z2 = Tensor::zeros(vec![1usize, lat, 6usize], DType::F32, Device::Cpu).unwrap();
    let tiled = vae.decode_tiled(&z2, 2, 1).expect("decode_tiled");
    assert_eq!(tiled.dims(), &[1, vae.config().audio_channels, 6 * hop]);
}

#[test]
fn vae_encode_decode_roundtrip_shapes() {
    let Some(path) = vae_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let vae = AceStepVae::open(&path, Device::Cpu).expect("open vae");
    let hop = vae.config().hop_length();
    let frames = 4usize;
    let samples = frames * hop;
    let audio = Tensor::zeros(vec![1usize, vae.config().audio_channels, samples], DType::F32, Device::Cpu).unwrap();
    let mean = vae.encode_mean(&audio).expect("encode");
    let md = mean.dims().to_vec();
    eprintln!("[acestep-vae] encode [1,2,{samples}] -> {md:?}");
    assert_eq!(md[0], 1);
    assert_eq!(md[1], vae.config().decoder_input_channels);
    assert_eq!(md[2], frames);
}
