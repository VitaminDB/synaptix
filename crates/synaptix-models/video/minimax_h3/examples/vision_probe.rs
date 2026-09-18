//! Башня зрения энкодера H3 отдельно от текста: грузит только vision-веса из
//! каталога или `.syn` и гоняет один кадр. `vision_probe <encoder> [H W]`.
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_video_minimax_h3::H3EncoderSource;
use synaptix_vlm_qwen3::{
    pipeline::DirWeights,
    preprocess::{prepare_tensor, PreprocessLimits},
    VisionConfig, VisionTower, VisionWeights,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(384);
    let w: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(640);
    let device = Device::Cuda(0);
    let src = H3EncoderSource::open(&args[1])?;
    let cfg = VisionConfig::from_hf_bytes(&src.read("config.json")?)?;
    let weights = DirWeights::from_loader(src.loader(device)?);
    for key in [
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.norm1.weight",
    ] {
        match weights.tensor(key, device, DType::BF16) {
            Ok(t) => eprintln!("{key}: {:?} {:?}", t.dtype(), t.dims()),
            Err(e) => eprintln!("{key}: {e}"),
        }
    }
    let tower = VisionTower::build(cfg.clone(), &weights, device, DType::BF16)?;
    let img = Tensor::zeros(vec![3, h, w], DType::F32, Device::Cpu)?.add_scalar(0.5)?;
    let p = prepare_tensor(&img, &cfg, PreprocessLimits::default(), device)?;
    eprintln!("patches {:?} {:?}, grid {:?}", p.patches.dtype(), p.patches.dims(), p.grid);
    let (out, taps) = tower.forward_deepstack(&p.patches, p.grid)?;
    eprintln!("ok: {:?} {:?}, taps {}", out.dtype(), out.dims(), taps.len());
    Ok(())
}
