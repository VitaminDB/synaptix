use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_acestep::config::DitConfig;
use synaptix_music_acestep::detokenizer::Detokenizer;
use synaptix_music_acestep::fsq::Fsq;
use synaptix_music_acestep::loader::CompLoader;

fn dit_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_v15_xl_base.syn");
    p.exists().then_some(p)
}

#[test]
fn fsq_detokenizer_to_lm_hints() {
    let Some(path) = dit_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let ck = CompLoader::open(&path, None, Device::Cpu).expect("open dit bundle");
    let cfg = DitConfig::xl_base();
    let fsq = Fsq::load(&ck, "tokenizer.quantizer").expect("fsq");
    let detok = Detokenizer::load(&ck, &cfg).expect("detok");

    let t = 4usize;
    let indices: Vec<u32> = (0..t as u32).map(|i| i * 7919 % 64000).collect();
    let codes = fsq.get_output_from_indices(&indices).expect("fsq out"); // [1,T,2048]
    assert_eq!(codes.dims(), &[1, t, 2048]);

    let hints = detok.forward(&codes).expect("detok"); // [1, T*pool, 64]
    let dims = hints.dims().to_vec();
    eprintln!("[acestep-detok] codes[1,{t},2048] -> lm_hints {dims:?}");
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], t * cfg.pool_window_size);
    assert_eq!(dims[2], cfg.audio_acoustic_hidden_dim);
    let v: Vec<f32> = hints.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "lm_hints must be finite");
}
