use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::heads::{
    BboxHead, CtcHead, KeypointHead, MlmHead, QaHead, RegressionActivation, RegressionHead,
    RnnTHead, SegmentationHead,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── CTC: forward = log_softmax(proj(x)) ──
#[test]
fn ctc_head_log_softmax() {
    ensure_registered();
    // 1×2×3 input, identity proj (vocab=3, hidden=3)
    let w = t1(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let b = t1(&[0.0, 0.0, 0.0], &[3]);
    let head = CtcHead::from_weights(w, Some(b)).unwrap();
    let x = t1(&[1.0, 2.0, 3.0,   0.0, 0.0, 0.0], &[1, 2, 3]);
    let y = head.forward(&x).unwrap();
    let out = y.to_vec3::<f32>().unwrap();
    // row 0: logits=(1,2,3); log_softmax(3-max=0) = log(exp(-2)+exp(-1)+1) = log(1.5032) = 0.4076
    // log_softmax = shifted - log_sum_exp = (1-3, 2-3, 3-3) - log_sum = (-2,-1,0) - 0.4076
    let lse_row0 = ((1.0_f32 - 3.0).exp() + (2.0_f32 - 3.0).exp() + 1.0).ln();
    approx_eq(out[0][0][0], -2.0 - lse_row0, 1e-5);
    approx_eq(out[0][0][1], -1.0 - lse_row0, 1e-5);
    approx_eq(out[0][0][2],  0.0 - lse_row0, 1e-5);
    // row 1: logits=(0,0,0); log_softmax = -ln 3
    let want = -3.0_f32.ln();
    approx_eq(out[0][1][0], want, 1e-5);
    approx_eq(out[0][1][1], want, 1e-5);
    approx_eq(out[0][1][2], want, 1e-5);
}

// ── MLM: forward = out(LN(GELU(dense(x)))) ──
#[test]
fn mlm_head_dense_gelu_ln_out() {
    ensure_registered();
    // hidden=2, vocab=2. dense=identity, GELU(0)=0, LN(0,0)=(0,0) → out=identity → all zeros
    let dense_w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let dense_b = t1(&[0.0, 0.0], &[2]);
    let ln_w = Some(t1(&[1.0, 1.0], &[2]));
    let ln_b = Some(t1(&[0.0, 0.0], &[2]));
    let out_w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let out_b = t1(&[0.0, 0.0], &[2]);
    let head = MlmHead::from_weights(dense_w, Some(dense_b), ln_w, ln_b, 1e-12, out_w, Some(out_b)).unwrap();
    // Non-zero input: dense=id, GELU(x), LN over last dim, out=id
    let x = t1(&[1.0, -1.0], &[1, 2]);
    let y = head.forward(&x).unwrap();
    let v = y.to_vec2::<f32>().unwrap();
    // GELU(1)=0.841192, GELU(-1)=-0.158808
    // LN: mean=(0.841192-0.158808)/2=0.341192; var=((0.5+ε)*((0.5)^2+(0.5)^2)) ≈ 0.25; std≈0.5
    // normed = (0.841192-0.341192)/0.5, (-0.158808-0.341192)/0.5 = (1.0, -1.0)
    approx_eq(v[0][0],  1.0, 5e-3);
    approx_eq(v[0][1], -1.0, 5e-3);
}

// ── QA: forward returns last-dim=2, forward_split returns (start, end) without last dim ──
#[test]
fn qa_head_start_end() {
    ensure_registered();
    // hidden=3 → 2; W rows: row0=start (sums), row1=end (last elem)
    let w = t1(&[1.0, 1.0, 1.0,   0.0, 0.0, 1.0], &[2, 3]);
    let b = t1(&[0.0, 0.0], &[2]);
    let head = QaHead::from_weights(w, Some(b)).unwrap();
    let x = t1(&[1.0, 2.0, 3.0,   4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = head.forward(&x).unwrap();
    let yv = y.to_vec3::<f32>().unwrap();
    approx_eq(yv[0][0][0], 6.0, 1e-5);
    approx_eq(yv[0][0][1], 3.0, 1e-5);
    approx_eq(yv[0][1][0], 15.0, 1e-5);
    approx_eq(yv[0][1][1], 6.0, 1e-5);
    let (start, end) = head.forward_split(&x).unwrap();
    let sv = start.contiguous().unwrap().to_vec2::<f32>().unwrap();
    let ev = end.contiguous().unwrap().to_vec2::<f32>().unwrap();
    approx_eq(sv[0][0], 6.0, 1e-5);
    approx_eq(sv[0][1], 15.0, 1e-5);
    approx_eq(ev[0][0], 3.0, 1e-5);
    approx_eq(ev[0][1], 6.0, 1e-5);
}

// ── Segmentation: BHWC permute + flatten + Linear + reshape back ──
#[test]
fn segmentation_head_bchw_path() {
    ensure_registered();
    // hidden=2 → num_classes=2. W=identity → output = input по каналам
    let w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let head = SegmentationHead::from_weights(w, Some(b)).unwrap();
    let x = t1(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ], &[1, 2, 2, 2]); // B=1, C=2, H=2, W=2
    let y = head.forward_bchw(&x).unwrap();
    let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(yv, xv);
}

// ── BBox: sigmoid + reshape last → (num_classes, 4) ──
#[test]
fn bbox_head_reshape_sigmoid() {
    ensure_registered();
    // hidden=2, num_classes=1 (4 outputs). W=identity → 4 outputs = first 4 of x
    let w = t1(&[
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, -1.0,
    ], &[4, 2]);
    let b = t1(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let head = BboxHead::from_weights(w, Some(b), 1, true).unwrap();
    let x = t1(&[0.0, 0.0], &[1, 2]);
    let y = head.forward(&x).unwrap();
    let yv = y.to_vec3::<f32>().unwrap();
    // logits = (0,0,0,0) → sigmoid = 0.5 каждый. Shape [1, 1, 4]
    assert_eq!(y.dims(), &[1, 1, 4]);
    for c in 0..1 {
        for k in 0..4 {
            approx_eq(yv[0][c][k], 0.5, 1e-5);
        }
    }
    // sigmoid_output=false → raw logits
    let w2 = t1(&[
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, -1.0,
    ], &[4, 2]);
    let b2 = t1(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let head_raw = BboxHead::from_weights(w2, Some(b2), 1, false).unwrap();
    let x2 = t1(&[1.0, 2.0], &[1, 2]);
    let y2 = head_raw.forward(&x2).unwrap();
    let yv2 = y2.to_vec3::<f32>().unwrap();
    approx_eq(yv2[0][0][0], 1.0, 1e-5);
    approx_eq(yv2[0][0][1], 2.0, 1e-5);
    approx_eq(yv2[0][0][2], 3.0, 1e-5);
    approx_eq(yv2[0][0][3], -1.0, 1e-5);
}

// ── Regression: dense → activation → out, аналитика на Tanh ──
#[test]
fn regression_head_dense_tanh_out() {
    ensure_registered();
    // hidden=2, output_dim=1. dense=identity, out=sum
    let dense_w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let dense_b = t1(&[0.0, 0.0], &[2]);
    let out_w = t1(&[1.0, 1.0], &[1, 2]);
    let out_b = t1(&[0.0], &[1]);
    let head = RegressionHead::from_weights(
        dense_w, Some(dense_b), out_w, Some(out_b), RegressionActivation::Tanh,
    ).unwrap();
    // dense(x)=x, tanh(0,0)=(0,0), out=0+0=0
    let x = t1(&[0.0, 0.0], &[1, 2]);
    let y = head.forward(&x).unwrap();
    approx_eq(y.to_vec2::<f32>().unwrap()[0][0], 0.0, 1e-5);
    // x=(1, -1): tanh(1)+tanh(-1) = 0
    let x2 = t1(&[1.0, -1.0], &[1, 2]);
    let y2 = head.forward(&x2).unwrap();
    approx_eq(y2.to_vec2::<f32>().unwrap()[0][0], 0.0, 1e-5);
    // x=(1, 1): 2*tanh(1) = 1.523188...
    let x3 = t1(&[1.0, 1.0], &[1, 2]);
    let y3 = head.forward(&x3).unwrap();
    approx_eq(y3.to_vec2::<f32>().unwrap()[0][0], 2.0 * 1.0_f32.tanh(), 1e-5);
}

// ── Keypoint: reshape last (3*num_kp) → (num_kp, 3) + sigmoid на vis ──
#[test]
fn keypoint_head_reshape_visibility() {
    ensure_registered();
    // hidden=3, num_kp=1 (3 outputs). W = одинаковые единицы 3 раза
    let w = t1(&[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], &[3, 3]);
    let b = t1(&[0.0, 0.0, 0.0], &[3]);
    let head = KeypointHead::from_weights(w, Some(b), 1, true).unwrap();
    let x = t1(&[2.0, 3.0, 0.0], &[1, 3]);
    let y = head.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let yv = y.to_vec3::<f32>().unwrap();
    approx_eq(yv[0][0][0], 2.0, 1e-5);
    approx_eq(yv[0][0][1], 3.0, 1e-5);
    approx_eq(yv[0][0][2], 0.5, 1e-5);
    // sigmoid_visibility=false → raw третий компонент
    let w2 = t1(&[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], &[3, 3]);
    let b2 = t1(&[0.0, 0.0, 0.0], &[3]);
    let head_raw = KeypointHead::from_weights(w2, Some(b2), 1, false).unwrap();
    let x2 = t1(&[1.0, 2.0, 4.0], &[1, 3]);
    let y2 = head_raw.forward(&x2).unwrap();
    let yv2 = y2.to_vec3::<f32>().unwrap();
    approx_eq(yv2[0][0][2], 4.0, 1e-5);
}

// ── RNN-T joint: [B,T,F]+[B,U,F] → [B,T,U,V] ──
#[test]
fn rnn_t_head_joint_broadcast() {
    ensure_registered();
    // enc_dim=2, pred_dim=2, joint_dim=2, vocab=1. Все веса единичные, нулевые bias.
    let enc_w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let enc_b = t1(&[0.0, 0.0], &[2]);
    let pred_w = t1(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let pred_b = t1(&[0.0, 0.0], &[2]);
    let out_w = t1(&[1.0, 1.0], &[1, 2]);
    let out_b = t1(&[0.0], &[1]);
    let head = RnnTHead::from_weights(
        enc_w, Some(enc_b), pred_w, Some(pred_b), out_w, Some(out_b),
    ).unwrap();
    // B=1, T=2, U=1, F=2
    let enc = t1(&[1.0, 1.0,   0.0, 0.0], &[1, 2, 2]);
    let pred = t1(&[2.0, 2.0], &[1, 1, 2]);
    let y = head.forward(&enc, &pred).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1, 1]);
    let yv: Vec<f32> = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // [t=0]: f=(1,1), g=(2,2); joint=tanh(3,3); out=2*tanh(3)
    // [t=1]: f=(0,0), g=(2,2); joint=tanh(2,2); out=2*tanh(2)
    approx_eq(yv[0], 2.0 * 3.0_f32.tanh(), 1e-5);
    approx_eq(yv[1], 2.0 * 2.0_f32.tanh(), 1e-5);
}
