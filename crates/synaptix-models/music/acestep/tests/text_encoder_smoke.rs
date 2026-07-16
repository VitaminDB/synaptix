use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::text_encoder::TextEncoder;

fn te_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/qwen3-embedding-0.6b.syn");
    p.exists().then_some(p)
}

#[test]
fn text_encoder_caption_and_lyric() {
    let Some(path) = te_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let te = TextEncoder::open(&path, Device::Cpu, DType::BF16, DType::BF16, 64).expect("open text encoder");

    let ids = Tensor::from_vec(vec![151643u32, 100, 200, 300], vec![1usize, 4], Device::Cpu).unwrap();
    let cap = te.caption_hidden(&ids).expect("caption");
    assert_eq!(cap.dims(), &[1, 4, 1024]);
    let cv: Vec<f32> = cap.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert!(cv.iter().all(|x| x.is_finite()));

    let lyr = te.lyric_embed(&ids).expect("lyric");
    assert_eq!(lyr.dims(), &[1, 4, 1024]);
    eprintln!("[acestep-te] caption {:?} lyric {:?}", cap.dims(), lyr.dims());
}
