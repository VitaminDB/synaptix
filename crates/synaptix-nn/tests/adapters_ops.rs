use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::adapters::{
    BoftLinear, GaloreLinear, OftLinear, PTuningV2, PromptTuning, QLoraLinear, ReftAdapter,
    VeraLinear,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── QLoRA: zero-LoRA → output = base.forward(x). ──
#[test]
fn qlora_zero_lora_equals_base() {
    ensure_registered();
    // base = identity 2x2; lora_a = zeros [r=2, in=2]; lora_b = zeros [out=2, r=2].
    let base = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let lora_a = t1(&[0.0; 4], &[2, 2]);
    let lora_b = t1(&[0.0; 4], &[2, 2]);
    let q = QLoraLinear::from_weights(base, lora_a, lora_b, 1.0).unwrap();
    let x = t1(&[3.0, -4.0], &[1, 2]);
    let y = q.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], vec![3.0, -4.0]);
}

// ── VeRA: lambda_b = 0 → output = base(x); lambda_b ≠ 0 → delta = (B·diag(λ_d)·A·x) ⊙ λ_b. ──
#[test]
fn vera_zero_lambda_b_equals_base() {
    ensure_registered();
    let base = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let a = t1(&[1.0, 1.0,   1.0, -1.0], &[2, 2]);
    let b = t1(&[1.0, 1.0,   1.0, -1.0], &[2, 2]);
    let ld = t1(&[1.0, 1.0], &[2]);
    let lb = t1(&[0.0, 0.0], &[2]);
    let v = VeraLinear::from_weights(base, a, b, ld, lb).unwrap();
    let x = t1(&[2.5, -0.5], &[1, 2]);
    let y = v.forward(&x).unwrap();
    let vec = y.to_vec2::<f32>().unwrap();
    approx_eq(vec[0][0], 2.5, 1e-6);
    approx_eq(vec[0][1], -0.5, 1e-6);
}

#[test]
fn vera_diag_lambda_d_scales_through() {
    ensure_registered();
    // base = 0, A=I, B=I, λ_d=(2, 3), λ_b=(1, 1) → out = (2·x0, 3·x1).
    let base = t1(&[0.0; 4], &[2, 2]);
    let a = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let ld = t1(&[2.0, 3.0], &[2]);
    let lb = t1(&[1.0, 1.0], &[2]);
    let v = VeraLinear::from_weights(base, a, b, ld, lb).unwrap();
    let x = t1(&[5.0, -1.0], &[1, 2]);
    let y = v.forward(&x).unwrap();
    let vec = y.to_vec2::<f32>().unwrap();
    approx_eq(vec[0][0], 10.0, 1e-6);
    approx_eq(vec[0][1], -3.0, 1e-6);
}

// ── OFT: R=I → output = base.forward(x). ──
#[test]
fn oft_identity_r_equals_base() {
    ensure_registered();
    let base = t1(&[1.0, 2.0,   3.0, 4.0], &[2, 2]);
    let r = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let oft = OftLinear::from_weights(base, r).unwrap();
    let x = t1(&[1.0, 1.0], &[1, 2]);
    // base.forward: y = x @ W^T = (1·1 + 1·2, 1·3 + 1·4) = (3, 7)
    let y = oft.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    approx_eq(v[0][0], 3.0, 1e-6);
    approx_eq(v[0][1], 7.0, 1e-6);
}

// ── OFT Cayley: q_raw = 0 → R = I. ──
#[test]
fn oft_cayley_zero_q_is_identity() {
    ensure_registered();
    let q = t1(&[0.0; 9], &[3, 3]);
    let r = synaptix_nn::adapters::oft::cayley_orthogonal_cpu(&q).unwrap();
    let v = r.to_vec2::<f32>().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx_eq(v[i][j], expected, 1e-6);
        }
    }
}

// ── OFT Cayley: R должна быть ортогональной (R · Rᵀ = I). ──
#[test]
fn oft_cayley_produces_orthogonal_matrix() {
    ensure_registered();
    let q = t1(&[
        0.0, 0.2, -0.1,
        0.0, 0.0, 0.3,
        0.0, 0.0, 0.0,
    ], &[3, 3]);
    let r = synaptix_nn::adapters::oft::cayley_orthogonal_cpu(&q).unwrap();
    let r_t = r.transpose(0, 1).unwrap().contiguous().unwrap();
    let rrt = r.matmul(&r_t).unwrap();
    let v = rrt.to_vec2::<f32>().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx_eq(v[i][j], expected, 1e-5);
        }
    }
}

// ── BOFT: identity block-diag → output = base.forward(x). ──
#[test]
fn boft_identity_r_equals_base() {
    ensure_registered();
    let base = t1(&[1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 0.0, 1.0, 2.0,
                    3.0, 4.0, 5.0, 6.0], &[4, 4]);
    let mut id = vec![0.0f32; 16];
    for i in 0..4 { id[i * 4 + i] = 1.0; }
    let r = t1(&id, &[4, 4]);
    let boft = BoftLinear::from_weights(base, r, 2).unwrap();
    let x = t1(&[1.0, 0.0, 0.0, 0.0], &[1, 4]);
    let y = boft.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    // y = x @ Wᵀ — берём первую строку Wᵀ = первый столбец W = (1, 5, 9, 3).
    approx_eq(v[0][0], 1.0, 1e-6);
    approx_eq(v[0][1], 5.0, 1e-6);
    approx_eq(v[0][2], 9.0, 1e-6);
    approx_eq(v[0][3], 3.0, 1e-6);
}

#[test]
fn boft_assemble_block_diag_zero_q_is_identity() {
    ensure_registered();
    let q0 = t1(&[0.0; 4], &[2, 2]);
    let q1 = t1(&[0.0; 4], &[2, 2]);
    let r = synaptix_nn::adapters::boft::assemble_block_diag_r_cpu(&[q0, q1]).unwrap();
    let v = r.to_vec2::<f32>().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx_eq(v[i][j], expected, 1e-6);
        }
    }
}

// ── GaLore: forward — pass-through через base. ──
#[test]
fn galore_forward_equals_base() {
    ensure_registered();
    let base = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let g = GaloreLinear::from_weights(base, 4, 0.25).unwrap();
    let x = t1(&[3.0, -2.0], &[1, 2]);
    let y = g.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    approx_eq(v[0][0], 3.0, 1e-6);
    approx_eq(v[0][1], -2.0, 1e-6);
}

// ── LoReFT: интервенция со zero (W=R, b=0) → h' = h. ──
#[test]
fn reft_identity_intervention_no_change() {
    ensure_registered();
    let r_proj = t1(&[1.0, 0.0, 0.0,   0.0, 1.0, 0.0], &[2, 3]);
    let w = t1(&[1.0, 0.0, 0.0,   0.0, 1.0, 0.0], &[2, 3]);
    let b = t1(&[0.0, 0.0], &[2]);
    let reft = ReftAdapter::from_weights(r_proj, w, b).unwrap();
    let h = t1(&[0.5, -0.5, 0.3,   1.0, 2.0, 3.0], &[1, 2, 3]);
    let y = reft.forward(&h).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    approx_eq(v[0][0][0], 0.5, 1e-6);
    approx_eq(v[0][0][1], -0.5, 1e-6);
    approx_eq(v[0][0][2], 0.3, 1e-6);
    approx_eq(v[0][1][0], 1.0, 1e-6);
    approx_eq(v[0][1][1], 2.0, 1e-6);
    approx_eq(v[0][1][2], 3.0, 1e-6);
}

// ── LoReFT shape preservation: out.dims == h.dims. ──
#[test]
fn reft_shape_preserves() {
    ensure_registered();
    let r = ReftAdapter::new(8, 3, D, DType::F32).unwrap();
    let h_data: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.01).collect();
    let h = t1(&h_data, &[2, 4, 8]);
    let y = r.forward(&h).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

// ── PromptTuning: prepend увеличивает T на num_tokens. ──
#[test]
fn prompt_tuning_prepend_grows_seq_dim() {
    ensure_registered();
    let soft = t1(&[1.0, 2.0,   3.0, 4.0,   5.0, 6.0], &[3, 2]);
    let pt = PromptTuning::from_weights(soft).unwrap();
    let x = t1(&[10.0, 20.0,   30.0, 40.0], &[1, 2, 2]);
    let y = pt.prepend(&x).unwrap();
    assert_eq!(y.dims(), &[1, 5, 2]);
    let v = y.to_vec3::<f32>().unwrap();
    // первые 3 строки — soft, дальше — x.
    approx_eq(v[0][0][0], 1.0, 1e-6);
    approx_eq(v[0][0][1], 2.0, 1e-6);
    approx_eq(v[0][1][0], 3.0, 1e-6);
    approx_eq(v[0][1][1], 4.0, 1e-6);
    approx_eq(v[0][2][0], 5.0, 1e-6);
    approx_eq(v[0][2][1], 6.0, 1e-6);
    approx_eq(v[0][3][0], 10.0, 1e-6);
    approx_eq(v[0][3][1], 20.0, 1e-6);
    approx_eq(v[0][4][0], 30.0, 1e-6);
    approx_eq(v[0][4][1], 40.0, 1e-6);
}

#[test]
fn prompt_tuning_prepend_batches_independently() {
    ensure_registered();
    let soft = t1(&[7.0, 8.0], &[1, 2]);
    let pt = PromptTuning::from_weights(soft).unwrap();
    let x = t1(&[1.0, 2.0,   3.0, 4.0], &[2, 1, 2]);
    let y = pt.prepend(&x).unwrap();
    assert_eq!(y.dims(), &[2, 2, 2]);
    let v = y.to_vec3::<f32>().unwrap();
    approx_eq(v[0][0][0], 7.0, 1e-6);
    approx_eq(v[1][0][0], 7.0, 1e-6);
    approx_eq(v[0][1][0], 1.0, 1e-6);
    approx_eq(v[1][1][0], 3.0, 1e-6);
}

// ── P-TuningV2: forward выдаёт [L, 2, P, H]. ──
#[test]
fn p_tuning_v2_shape() {
    ensure_registered();
    let p = PTuningV2::new(4, 3, 6, D, DType::F32).unwrap();
    let full = p.forward().unwrap();
    assert_eq!(full.dims(), &[3, 2, 4, 6]);
    let (k, v) = p.layer_kv(&full, 0).unwrap();
    assert_eq!(k.dims(), &[4, 6]);
    assert_eq!(v.dims(), &[4, 6]);
}
