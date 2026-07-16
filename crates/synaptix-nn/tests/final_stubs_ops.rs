use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::dit::dit_block::DitBlock;
use synaptix_nn::dit::dit_joint::DitJoint;
use synaptix_nn::squeezeformer::Squeezeformer;

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

// ── DitJoint: shape preservation `[B, T_img, hidden]` (img-stream возвращается). ──
#[test]
fn dit_joint_shape_preserves_img_stream() {
    ensure_registered();
    let hidden = 8;
    let img_blocks = vec![
        DitBlock::new(hidden, 2, 16, hidden, D, DType::F32).unwrap(),
        DitBlock::new(hidden, 2, 16, hidden, D, DType::F32).unwrap(),
    ];
    let txt_blocks = vec![
        DitBlock::new(hidden, 2, 16, hidden, D, DType::F32).unwrap(),
        DitBlock::new(hidden, 2, 16, hidden, D, DType::F32).unwrap(),
    ];
    let dj = DitJoint::from_blocks(img_blocks, txt_blocks).unwrap();

    let img_data: Vec<f32> = (0..2 * 4 * hidden).map(|i| (i as f32) * 0.01).collect();
    let img = t1(&img_data, &[2, 4, hidden]);
    let txt_data: Vec<f32> = (0..2 * 6 * hidden).map(|i| (i as f32) * 0.01).collect();
    let txt = t1(&txt_data, &[2, 6, hidden]);
    let cond_data: Vec<f32> = (0..2 * hidden).map(|i| (i as f32) * 0.01).collect();
    let cond = t1(&cond_data, &[2, hidden]);

    let y = dj.forward(&img, &txt, &cond).unwrap();
    assert_eq!(y.dims(), &[2, 4, hidden]);
}

// ── Squeezeformer: чётная длина T=6 → T_after=6 (без padding). ──
#[test]
fn squeezeformer_even_t_no_padding() {
    ensure_registered();
    let s = Squeezeformer::new(4, 8, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 6 * 4).map(|i| (i as f32) * 0.01).collect();
    let x = t1(&x_data, &[2, 6, 4]);
    let y = s.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 6, 8]);
}

// ── Squeezeformer: нечётная длина T=5 → padding до T=5. ──
#[test]
fn squeezeformer_odd_t_padding_preserves_length() {
    ensure_registered();
    let s = Squeezeformer::new(3, 6, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..1 * 5 * 3).map(|i| (i as f32) * 0.01).collect();
    let x = t1(&x_data, &[1, 5, 3]);
    let y = s.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 5, 6]);
}

// ── Squeezeformer: T<2 → pure projection (без resampling). ──
#[test]
fn squeezeformer_short_seq_passthrough_projection() {
    ensure_registered();
    let s = Squeezeformer::new(2, 4, D, DType::F32).unwrap();
    let x = t1(&[1.0, 2.0], &[1, 1, 2]);
    let y = s.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
}
