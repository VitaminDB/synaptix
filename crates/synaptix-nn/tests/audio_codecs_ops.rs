use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::audio::{
    dac::Dac, encodec::EnCodec, fsq::FiniteScalarQuantizer, higgs_audio::HiggsAudio,
    lfq::Lfq, mimi::Mimi, parakeet_enc::ParakeetEnc, rvq::ResidualVQ, snac::Snac,
    speaker_encoder::SpeakerEncoder,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

// ── FSQ: round-trip levels=[3,3], dim=2 ──
#[test]
fn fsq_levels_3_3_roundtrip() {
    ensure_registered();
    let q = FiniteScalarQuantizer::new(vec![3, 3]);
    assert_eq!(q.codebook_size, 9);
    // z=(0.0, 1.0) → tanh→(0, 0.7616) → *half=(0, 0.7616) → round→(0, 1).
    // levels=3 odd → offset=0 → shifted index = (0+1, 1+1) = (1, 2). idx = 1 + 2*3 = 7.
    let z = t1(&[0.0, 1.0,   -1.0, -0.5], &[2, 2]);
    let (codes, indices) = q.quantize(&z).unwrap();
    let cv = codes.to_vec2::<f32>().unwrap();
    let iv = indices.to_vec1::<i64>().unwrap();
    assert!((cv[0][0] - 0.0).abs() < 1e-5);
    assert!((cv[0][1] - 1.0).abs() < 1e-5);
    // dequant round-trip
    let dq = q.dequantize(&indices, DType::F32).unwrap();
    let dv = dq.to_vec2::<f32>().unwrap();
    assert_eq!(dv, cv);
    assert!(iv[0] >= 0 && iv[0] < 9);
}

// ── LFQ: sign-based binarization, dim=3, codebook_size=8 ──
#[test]
fn lfq_sign_quantization() {
    ensure_registered();
    let q = Lfq::new(8, 3);
    let z = t1(&[1.0, -2.0, 0.5,   -0.1, 0.1, -3.0], &[2, 3]);
    let (codes, indices) = q.quantize(&z).unwrap();
    let cv = codes.to_vec2::<f32>().unwrap();
    let iv = indices.to_vec1::<i64>().unwrap();
    // row 0: (+, -, +) → codes (1, -1, 1); idx bits {bit0=1, bit1=0, bit2=1} = 0b101 = 5
    assert_eq!(cv[0], vec![1.0, -1.0, 1.0]);
    assert_eq!(iv[0], 5);
    // row 1: (-, +, -) → codes (-1, 1, -1); idx = 0b010 = 2
    assert_eq!(cv[1], vec![-1.0, 1.0, -1.0]);
    assert_eq!(iv[1], 2);
    // dequant: idx → codes ±1
    let dq = q.dequantize(&indices, DType::F32).unwrap();
    let dv = dq.to_vec2::<f32>().unwrap();
    assert_eq!(dv, cv);
}

// ── RVQ: 2 codebooks, residual encoding ──
#[test]
fn rvq_2_codebooks_residual() {
    ensure_registered();
    // CB1 = {(1, 0), (0, 1)};  CB2 = {(0.5, 0.5), (-0.5, -0.5)}.
    let cb1 = t1(&[1.0, 0.0,  0.0, 1.0], &[2, 2]);
    let cb2 = t1(&[0.5, 0.5,  -0.5, -0.5], &[2, 2]);
    let rvq = ResidualVQ::from_codebooks(vec![cb1, cb2]).unwrap();
    // x = (1.5, 0.5):
    //   CB1: nearest = (1, 0) (dist=0.5) [vs (0,1) dist=2.5]
    //     residual = (0.5, 0.5)
    //   CB2: nearest = (0.5, 0.5) (dist=0)
    //     residual = (0, 0)
    //   indices = (0, 0). decoded = (1+0.5, 0+0.5) = (1.5, 0.5) exact.
    let x = t1(&[1.5, 0.5], &[1, 2]);
    let indices = rvq.encode(&x).unwrap();
    let iv = indices.to_vec2::<i64>().unwrap();
    assert_eq!(iv[0], vec![0, 0]);
    let recon = rvq.decode(&indices, DType::F32).unwrap();
    let rv = recon.to_vec2::<f32>().unwrap();
    assert!((rv[0][0] - 1.5).abs() < 1e-5);
    assert!((rv[0][1] - 0.5).abs() < 1e-5);
}

// ── DAC: shape preservation + forward не падает ──
#[test]
fn dac_shape_preserves() {
    ensure_registered();
    let dac = Dac::new(8, 16, 4, 32, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 5 * 8).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let x = t1(&x_data, &[2, 5, 8]);
    let y = dac.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 5, 8]);
    let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(yv.iter().all(|v| v.is_finite()));
}

// ── EnCodec: shape preservation ──
#[test]
fn encodec_shape_preserves() {
    ensure_registered();
    let codec = EnCodec::new(8, 16, 4, 32, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 5 * 8).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let x = t1(&x_data, &[2, 5, 8]);
    let y = codec.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 5, 8]);
}

// ── SNAC: multi-scale (3 scale), shape preservation ──
#[test]
fn snac_multi_scale_shape() {
    ensure_registered();
    let snac = Snac::new(4, 8, vec![2, 2, 2], 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[1, 3, 4]);
    let y = snac.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 4]);
}

// ── Mimi: semantic+acoustic split, shape preservation ──
#[test]
fn mimi_semantic_acoustic_shape() {
    ensure_registered();
    let mimi = Mimi::new(4, 8, 1, 3, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[1, 3, 4]);
    let y = mimi.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 4]);
}

// ── HiggsAudio: shape preservation ──
#[test]
fn higgs_audio_shape_preserves() {
    ensure_registered();
    let higgs = HiggsAudio::new(4, 8, 4, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[1, 3, 4]);
    let y = higgs.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 4]);
}

// ── ParakeetEnc: in_proj → blocks=[] → final_ln → out_proj ──
#[test]
fn parakeet_enc_empty_blocks_shape() {
    ensure_registered();
    let enc = ParakeetEnc::new(80, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 5 * 80).map(|i| (i as f32) * 0.001).collect();
    let x = t1(&x_data, &[2, 5, 80]);
    let y = enc.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 5, 16]);
}

// ── SpeakerEncoder: [B, T, in] → [B, embedding_size] ──
#[test]
fn speaker_encoder_pooling() {
    ensure_registered();
    let enc = SpeakerEncoder::new(40, 16, 8, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 10 * 40).map(|i| (i as f32) * 0.005).collect();
    let x = t1(&x_data, &[2, 10, 40]);
    let y = enc.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8]);
}
