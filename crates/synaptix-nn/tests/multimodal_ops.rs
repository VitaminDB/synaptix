use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::multimodal::{
    AnyToAnyProjector, CrossModalAttention, InstantIdProjector, MlpProjector,
    PerceiverResampler, QFormer, VlmBlock,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

// ── MlpProjector: fc1(identity) → gelu_exact → fc2(identity) ──
#[test]
fn mlp_projector_identity() {
    ensure_registered();
    let fc1_w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let fc1_b = t1(&[0.0, 0.0], &[2]);
    let fc2_w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let fc2_b = t1(&[0.0, 0.0], &[2]);
    let proj = MlpProjector::from_weights(fc1_w, Some(fc1_b), fc2_w, Some(fc2_b)).unwrap();
    let x = t1(&[1.0, -1.0], &[1, 2]);
    let y = proj.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    // gelu_exact(1.0) ≈ 0.8413; gelu_exact(-1.0) ≈ -0.1587
    assert!((v[0][0] - 0.8413).abs() < 1e-3);
    assert!((v[0][1] - (-0.1587)).abs() < 1e-3);
}

// ── InstantIdProjector: Linear ──
#[test]
fn instant_id_projects() {
    ensure_registered();
    let w = t1(&[1.0, 2.0,   3.0, 4.0,   5.0, 6.0], &[3, 2]);
    let b = t1(&[0.0, 0.0, 0.0], &[3]);
    let proj = InstantIdProjector::from_weights(w, Some(b)).unwrap();
    // id = [1, 2] → out = [1*1+2*2, 3+8, 5+12] = [5, 11, 17]
    let id = t1(&[1.0, 2.0], &[1, 2]);
    let y = proj.forward(&id).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], vec![5.0, 11.0, 17.0]);
}

// ── AnyToAnyProjector: с normom и без ──
#[test]
fn any_to_any_without_norm() {
    ensure_registered();
    let w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let proj = AnyToAnyProjector::from_weights(w, Some(b), None, 1e-6).unwrap();
    let x = t1(&[3.0, 4.0], &[1, 2]);
    let y = proj.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], vec![3.0, 4.0]);
}

#[test]
fn any_to_any_with_norm_changes_scale() {
    ensure_registered();
    let w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let norm = t1(&[1.0, 1.0], &[2]);
    let proj = AnyToAnyProjector::from_weights(w, Some(b), Some(norm), 1e-6).unwrap();
    let x = t1(&[3.0, 4.0], &[1, 2]);
    let y = proj.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    // RMS of (3,4) = sqrt(mean(9, 16)) = sqrt(12.5) ≈ 3.5355
    // normed = (3/3.5355, 4/3.5355) ≈ (0.8485, 1.1314)
    assert!((v[0][0] - 3.0_f32 / (12.5_f32).sqrt()).abs() < 1e-4);
    assert!((v[0][1] - 4.0_f32 / (12.5_f32).sqrt()).abs() < 1e-4);
}

// ── CrossModalAttention: shape preservation ──
#[test]
fn cross_modal_shape_preserves() {
    ensure_registered();
    let attn = CrossModalAttention::new(8, 16, 2, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let ctx_data: Vec<f32> = (0..2 * 6 * 16).map(|i| (i as f32) * 0.005).collect();
    let x = t1(&x_data, &[2, 4, 8]);
    let ctx = t1(&ctx_data, &[2, 6, 16]);
    let y = attn.forward(&x, &ctx, None).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

// ── QFormer: image_features [B, Sk, C] → [B, num_q, hidden] ──
#[test]
fn q_former_compresses() {
    ensure_registered();
    let q = QFormer::new(4, 8, 16, 2, D, DType::F32).unwrap();
    let img: Vec<f32> = (0..2 * 12 * 16).map(|i| (i as f32) * 0.005).collect();
    let image = t1(&img, &[2, 12, 16]);
    let y = q.forward(&image).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

// ── PerceiverResampler: latents независимы от input length ──
#[test]
fn perceiver_resampler_fixed_output_length() {
    ensure_registered();
    let r = PerceiverResampler::new(4, 8, 16, 2, D, DType::F32).unwrap();
    let ctx_data: Vec<f32> = (0..1 * 20 * 16).map(|i| (i as f32) * 0.005).collect();
    let ctx = t1(&ctx_data, &[1, 20, 16]);
    let y = r.forward(&ctx).unwrap();
    assert_eq!(y.dims(), &[1, 4, 8]);
}

// ── VlmBlock: cross-attn + residual, preserves shape ──
#[test]
fn vlm_block_shape_preserves() {
    ensure_registered();
    let block = VlmBlock::new(8, 16, 2, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.01).collect();
    let ctx_data: Vec<f32> = (0..2 * 6 * 16).map(|i| (i as f32) * 0.005).collect();
    let x = t1(&x_data, &[2, 4, 8]);
    let ctx = t1(&ctx_data, &[2, 6, 16]);
    let y = block.forward(&x, &ctx).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}
