use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tts_voxcpm::loader::VoxCheckpoint;
use synaptix_tts_voxcpm::model::VoxCpmModel;
use synaptix_tts_voxcpm::pipeline::{GenerateOptions, VoxCpmPipeline};
use synaptix_tts_voxcpm::tokenizer::TextTokenizer;

fn bundle() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/voxcpm2.syn");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn loads_model_and_validates_tensors() {
    let Some(path) = bundle() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = VoxCheckpoint::open(&path, Device::Cpu, DType::F32).expect("open bundle");
    assert_eq!(ck.config.architecture, "voxcpm2");
    assert_eq!(ck.config.lm_config.num_hidden_layers, 28);
    assert_eq!(ck.config.patch_size, 4);
    let model = VoxCpmModel::load(&ck).expect("load model");
    eprintln!("[voxcpm] model loaded, layers={}", model.config.lm_config.num_hidden_layers);
}

#[test]
fn tokenizer_roundtrip() {
    let Some(path) = bundle() else { return };
    let ck = VoxCheckpoint::open(&path, Device::Cpu, DType::F32).expect("open");
    let bytes = ck.read_file("tokenizer.json").expect("tokenizer.json");
    let tok = TextTokenizer::from_bytes(&bytes).expect("tokenizer");
    let ids = tok.encode("Hello world.").expect("encode");
    assert!(!ids.is_empty());
    eprintln!("[voxcpm] 'Hello world.' -> {ids:?}");
}

#[test]
fn audiovae_decode_shape() {
    let Some(path) = bundle() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = VoxCheckpoint::open(&path, Device::Cpu, DType::F32).expect("open");
    let model = VoxCpmModel::load(&ck).expect("load");
    let t = 4usize;
    let latent = Tensor::zeros(vec![1usize, ck.config.feat_dim, t], DType::F32, Device::Cpu).unwrap();
    let out = model.audio_vae.decode(&latent).expect("decode");
    let dims = out.dims().to_vec();
    eprintln!("[voxcpm] vae decode {t} latent -> {dims:?}");
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 1);
    let expect = t * ck.config.audio_vae_config.decode_chunk_size();
    assert_eq!(dims[2], expect);
}

#[test]
fn synthesize_tiny() {
    if std::env::var("VOXCPM_GENERATE").is_err() {
        return;
    }
    let Some(path) = bundle() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let pipe = VoxCpmPipeline::from_bundle(&path, Device::Cpu, DType::F32).expect("pipeline");
    let opts = GenerateOptions { min_len: 2, max_len: 4, n_timesteps: 2, ..GenerateOptions::default() };
    let wav = pipe.synthesize("Hello world.", &opts).expect("synthesize");
    eprintln!("[voxcpm] pcm len={} sr={}", wav.pcm.len(), wav.sample_rate);
    assert!(!wav.pcm.is_empty());
    assert!(wav.pcm.iter().all(|x| x.is_finite()));
    assert_eq!(wav.sample_rate, 48000);
}
