use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::lm::AceStepLm;

fn lm_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_5hz_lm_1.7b.syn");
    p.exists().then_some(p)
}

#[test]
fn lm_loads_and_forwards() {
    let Some(path) = lm_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let lm = AceStepLm::open(&path, Device::Cpu, DType::BF16, DType::BF16, 64).expect("open lm");
    assert_eq!(lm.config.num_hidden_layers, 28);
    assert_eq!(lm.config.vocab_size, 217204);

    let mut kv = lm.make_kv(1, 16).expect("kv");
    let ids = Tensor::from_vec(
        vec![lm.config.bos_token_id, 100u32, 200u32],
        vec![1usize, 3],
        Device::Cpu,
    )
    .unwrap();
    let logits = lm.forward(&ids, &mut kv).expect("forward");
    assert_eq!(logits.dims(), &[1, lm.config.vocab_size]);
    let v: Vec<f32> = logits.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    let (argmax, _) = v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(am, mv), (i, &x)| {
        if x > mv { (i, x) } else { (am, mv) }
    });
    eprintln!("[acestep-lm] argmax={argmax} vocab={}", lm.config.vocab_size);
}
