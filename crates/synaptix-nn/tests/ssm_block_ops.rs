use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::ssm_block::mamba2_block::Mamba2Block;
use synaptix_nn::ssm_block::xlstm_block::{XLstmBlock, XLstmKind};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

// ── Mamba2Block: проверка shape preservation + sanity по seq ──
//
// Полная Mamba2 SSD-математика проверяется в ref-тесте против Python-портa
// той же step-loop логики. Здесь — что forward не падает на типичных формах
// и сохраняет [B, L, hidden].
#[test]
fn mamba2_block_shape_preserves() {
    ensure_registered();
    let hidden = 8;
    let d_state = 4;
    let num_heads = 2;
    let head_dim = 4; // d_inner = num_heads*head_dim = 8 = hidden
    let d_conv = 3;
    let block = Mamba2Block::new(hidden, d_state, num_heads, head_dim, d_conv, D, synaptix_core::dtype::DType::F32).unwrap();
    let b_sz = 2;
    let seq = 5;
    let x_data: Vec<f32> = (0..b_sz * seq * hidden).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let x = t1(&x_data, &[b_sz, seq, hidden]);
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[b_sz, seq, hidden]);
    let y_flat = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(y_flat.iter().all(|v| v.is_finite()),
        "Mamba2Block output contains NaN/Inf");
}

// ── Mamba2Block: with zero out_proj → output ≈ 0 (sanity) ──
#[test]
fn mamba2_block_zero_out_proj_yields_zero() {
    ensure_registered();
    let hidden = 4;
    let d_state = 4;
    let num_heads = 2;
    let head_dim = 2;
    let d_conv = 3;
    let in_dim = 2 * (num_heads * head_dim) + 2 * d_state + num_heads;
    let in_proj_w = t1(&vec![0.1_f32; in_dim * hidden], &[in_dim, hidden]);
    let conv_w = t1(&vec![1.0_f32; (num_heads * head_dim) * d_conv], &[num_heads * head_dim, 1, d_conv]);
    let out_proj_w = t1(&vec![0.0_f32; hidden * (num_heads * head_dim)], &[hidden, num_heads * head_dim]);
    let a_log = t1(&vec![0.0_f32; num_heads], &[num_heads]);
    let d_param = t1(&vec![1.0_f32; num_heads], &[num_heads]);
    let dt_bias = t1(&vec![0.0_f32; num_heads], &[num_heads]);
    let norm_w = t1(&vec![1.0_f32; num_heads * head_dim], &[num_heads * head_dim]);
    let block = Mamba2Block::from_weights(
        in_proj_w, conv_w, None, out_proj_w, a_log, d_param, dt_bias, norm_w,
        hidden, d_state, num_heads, head_dim, d_conv, 1e-5,
    ).unwrap();
    let x = t1(&[0.1_f32; 1 * 3 * 4], &[1, 3, 4]);
    let y = block.forward(&x).unwrap();
    let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for x in v { assert!(x.abs() < 1e-6, "output not ≈ 0: {x}"); }
}

// ── XLstmBlock sLSTM: T=1, проверка скалярного gate-вывода ──
#[test]
fn xlstm_block_slstm_t1_manual() {
    ensure_registered();
    let h = 2;
    // gate_proj выдаёт [z, i, f, o] = 4*h каналов
    // Положим gate_proj = identity-ish: input x=[1, 0] → gate = [1, 0, 0, 0, ..., 0, 1, 0, 0]
    // Чтобы упростить: gate_w = zeros; gate_b = [1.0 для z[0], 0 для остальных, 1.0 для o[0]]
    let gate_w = t1(&vec![0.0_f32; (4 * h) * h], &[4 * h, h]);
    // bias: 8 элементов: z(2), i(2), f(2), o(2)
    // z=[1,1], i=[10,10] (sigmoid≈1), f=[-10,-10] (sigmoid≈0), o=[10,10] (sigmoid≈1)
    let gate_b = t1(&[1.0, 1.0, 10.0, 10.0, -10.0, -10.0, 10.0, 10.0], &[4 * h]);
    let out_w = t1(&[1.0, 0.0, 0.0, 1.0], &[h, h]);
    let block = XLstmBlock::from_weights(
        gate_w, Some(gate_b), out_w, None, XLstmKind::SLstm,
    ).unwrap();
    // T=1: c_init=0, c_new = σ(f)*0 + σ(i)*tanh(z) ≈ 1 * tanh(1) = 0.7616
    // h_new = σ(o)*tanh(c_new) ≈ 1 * tanh(0.7616) = 0.6418
    let x = t1(&[0.0, 0.0], &[1, 1, h]);
    let y = block.forward(&x).unwrap();
    let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = (1.0_f32.tanh()).tanh();
    let atol = 1e-3;
    for &val in v.iter() {
        assert!((val - expected).abs() < atol, "expected ≈ {expected}, got {val}");
    }
}

// ── XLstmBlock mLSTM: T=1, проверка матричной памяти ──
#[test]
fn xlstm_block_mlstm_t1() {
    ensure_registered();
    let h = 2;
    // gate=[q, k, v] = 3*h каналов
    let gate_w = t1(&vec![0.0_f32; (3 * h) * h], &[3 * h, h]);
    // gate_b = q=[1,2], k=[1,1], v=[2,3]
    let gate_b = t1(&[1.0, 2.0,  1.0, 1.0,  2.0, 3.0], &[3 * h]);
    let out_w = t1(&[1.0, 0.0, 0.0, 1.0], &[h, h]);
    let block = XLstmBlock::from_weights(
        gate_w, Some(gate_b), out_w, None, XLstmKind::MLstm,
    ).unwrap();
    // C_new[i,j] = 0 + v[i]·k[j]; n_new = 0 + k = (1,1).
    // C_new = [[2*1, 2*1], [3*1, 3*1]] = [[2,2],[3,3]]
    // num[i] = Σ_j C_new[i,j]·q[j] = (2*1+2*2, 3*1+3*2) = (6, 9)
    // den = Σ n[j]·q[j] = 1*1+1*2 = 3 → max(|3|,1) = 3
    // out = (6/3, 9/3) = (2, 3)
    // out_proj = identity → (2, 3)
    let x = t1(&[0.0, 0.0], &[1, 1, h]);
    let y = block.forward(&x).unwrap();
    let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((v[0] - 2.0).abs() < 1e-4, "expected 2.0, got {}", v[0]);
    assert!((v[1] - 3.0).abs() < 1e-4, "expected 3.0, got {}", v[1]);
}

// ── XLstmBlock sLSTM: shape preservation на [B,L,H] ──
#[test]
fn xlstm_block_slstm_shape_preserves() {
    ensure_registered();
    let block = XLstmBlock::new(4, XLstmKind::SLstm, D, synaptix_core::dtype::DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.05).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

// ── XLstmBlock mLSTM: shape preservation на [B,L,H] ──
#[test]
fn xlstm_block_mlstm_shape_preserves() {
    ensure_registered();
    let block = XLstmBlock::new(4, XLstmKind::MLstm, D, synaptix_core::dtype::DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.05).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}
