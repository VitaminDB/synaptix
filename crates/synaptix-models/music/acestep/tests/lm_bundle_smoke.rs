//! Smoke загрузки 5Hz-LM из произвольного .syn-бандла: конфиг читается из
//! `config.json` бандла, имена тензоров — HF (`model.…`) или плоские.
//!
//! Запуск (пропускается без переменной):
//!   ACESTEP_LM=/path/acestep_5hz_lm_4b.syn cargo test --release \
//!       -p synaptix-music-acestep --test lm_bundle_smoke -- --nocapture
//! `ACESTEP_LM_CPU=1` — форсировать CPU (по умолчанию CUDA:0, если доступна).

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::lm::AceStepLm;
use synaptix_music_acestep::text_encoder::TextEncoder;

fn device() -> Device {
    if std::env::var("ACESTEP_LM_CPU").is_ok() {
        return Device::Cpu;
    }
    synaptix_kernels_cuda::ensure_registered();
    Device::Cuda(0)
}

#[test]
fn lm_bundle_loads_and_forwards() {
    let Ok(p) = std::env::var("ACESTEP_LM") else { return };
    let path = PathBuf::from(p);
    synaptix_kernels_cpu::ensure_registered();
    let dev = device();
    let t = std::time::Instant::now();
    let lm = AceStepLm::open(&path, dev, DType::BF16, DType::BF16, 64).expect("open lm");
    eprintln!(
        "[lm-bundle] {} за {:.1}s: layers={} hidden={} heads={}/{} vocab={}",
        path.display(),
        t.elapsed().as_secs_f32(),
        lm.config.num_hidden_layers,
        lm.config.hidden_size,
        lm.config.num_attention_heads,
        lm.config.num_key_value_heads,
        lm.config.vocab_size
    );
    let mut kv = lm.make_kv(1, 16).expect("kv");
    let ids = Tensor::from_vec(vec![lm.config.bos_token_id, 100u32, 200u32], vec![1usize, 3], dev).unwrap();
    let logits = lm.forward(&ids, &mut kv).expect("forward");
    assert_eq!(logits.dims(), &[1, lm.config.vocab_size]);
    let v: Vec<f32> = logits
        .to_device(Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    let (argmax, _) = v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(am, mv), (i, &x)| {
        if x > mv { (i, x) } else { (am, mv) }
    });
    eprintln!("[lm-bundle] argmax={argmax}");
}

#[test]
fn text_encoder_bundle_loads() {
    let Ok(p) = std::env::var("ACESTEP_TE") else { return };
    synaptix_kernels_cpu::ensure_registered();
    let dev = device();
    let te = TextEncoder::open(&p, dev, DType::BF16, DType::BF16, 64).expect("open text encoder");
    let ids = Tensor::from_vec(vec![151643u32, 100, 200], vec![1usize, 3], dev).unwrap();
    let h = te.caption_hidden(&ids).expect("caption_hidden");
    eprintln!("[te-bundle] {p}: hidden dims={:?}", h.dims());
    assert_eq!(h.dims()[1], 3);
}
