use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::vision::{
    clip_vision::ClipVision, dino_v2::DinoV2, efficient_net::EfficientNet,
    navit::NaViT, resnet::ResNet, sam_vit::SamVit, siglip::SigLip, swin::Swin,
};

const D: Device = Device::Cpu;

fn rand_image(b: usize, c: usize, h: usize, w: usize) -> Tensor {
    let data: Vec<f32> = (0..b * c * h * w).map(|i| (i as f32) * 0.001 - 0.05).collect();
    Tensor::from_slice(&data, &[b, c, h, w], D).unwrap()
}

// ── CLIP-ViT: forward выдаёт embedding [B, embed_dim] ──
#[test]
fn clip_vision_shape() {
    ensure_registered();
    let m = ClipVision::new(3, 4, 16, 32, 4, 0, 64, 16, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 16]);
}

// ── SigLIP: те же shape semantics ──
#[test]
fn siglip_shape() {
    ensure_registered();
    let m = SigLip::new(3, 4, 16, 32, 4, 0, 64, 8, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 8]);
}

// ── DINOv2: с регистровыми токенами и без ──
#[test]
fn dino_v2_with_registers() {
    ensure_registered();
    let m = DinoV2::new(3, 4, 16, 32, 4, 0, 64, 4, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 32]);
    assert!(m.register_tokens.is_some());
    assert_eq!(m.num_registers, 4);
}

#[test]
fn dino_v2_no_registers() {
    ensure_registered();
    let m = DinoV2::new(3, 4, 16, 32, 4, 0, 64, 0, D, DType::F32).unwrap();
    let img = rand_image(1, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[1, 32]);
    assert!(m.register_tokens.is_none());
}

// ── SAM-ViT: [B, num_patches, neck_dim] ──
#[test]
fn sam_vit_neck_shape() {
    ensure_registered();
    let m = SamVit::new(3, 4, 16, 32, 4, 0, 64, 256, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    // 16x16 → 4x4 patches = 16 tokens
    assert_eq!(y.dims(), &[2, 16, 256]);
}

// ── NaViT: variable H/W ──
#[test]
fn navit_variable_aspect_ratio() {
    ensure_registered();
    let m = NaViT::new(3, 4, 16, 32, 4, 0, 64, D, DType::F32).unwrap();
    // Не-квадратное: 16×8 (оба кратны patch=4)
    let img = rand_image(2, 3, 16, 8);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 32]);
}

// ── Swin: patch_embed + final norm + mean-pool без stage'ов ──
#[test]
fn swin_no_stages_shape() {
    ensure_registered();
    let m = Swin::new(3, 4, 32, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 32]);
}

// ── ResNet: stem-only (без stage'ов) → GAP → linear ──
#[test]
fn resnet_stem_only_shape() {
    ensure_registered();
    // stem_out=8, head_in=8 (=stem_out, поскольку stages пустые), 10 classes
    let m = ResNet::new(3, 10, 8, 8, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 10]);
}

// ── EfficientNet: stem + head без MBConv блоков ──
#[test]
fn efficient_net_no_blocks_shape() {
    ensure_registered();
    let m = EfficientNet::new(3, 10, 8, 8, 16, D, DType::F32).unwrap();
    let img = rand_image(2, 3, 16, 16);
    let y = m.forward(&img).unwrap();
    assert_eq!(y.dims(), &[2, 10]);
}
