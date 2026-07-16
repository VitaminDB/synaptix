use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::conformer::{
    attention_module::AttentionModule, conv_module::ConvModule, ff_module::FeedForwardModule,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── ff_module: macaron half-step ──
//
// Контролируемый случай: norm gain=1, bias=0; fc1=identity(2x2),
// fc1_b=0; fc2=identity, fc2_b=0. LN над 2D-вектором даёт центр-норм.
// Цель — проверить, что residual прибавляется с коэффициентом 0.5.
#[test]
fn ff_module_half_step_residual() {
    ensure_registered();
    let h = 2;
    let norm_w = t1(&[1.0, 1.0], &[h]);
    let norm_b = t1(&[0.0, 0.0], &[h]);
    let fc1_w = t1(&[1.0, 0.0,   0.0, 1.0], &[h, h]);
    let fc1_b = t1(&[0.0, 0.0], &[h]);
    let fc2_w = t1(&[1.0, 0.0,   0.0, 1.0], &[h, h]);
    let fc2_b = t1(&[0.0, 0.0], &[h]);
    let ff = FeedForwardModule::from_weights(
        norm_w, norm_b,
        fc1_w, Some(fc1_b),
        fc2_w, Some(fc2_b),
        1e-5,
    ).unwrap();
    // x[0,0]=( 1.0, -1.0) → mean=0, var=1 → LN=(1,-1).
    // fc1=I → silu(1)=0.7311, silu(-1)=-0.2689; fc2=I → same.
    // output = x + 0.5*silu(LN(x)) = (1 + 0.5·0.7311, -1 + 0.5·(-0.2689))
    let x = t1(&[1.0, -1.0], &[1, 1, h]);
    let y = ff.forward(&x).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    let silu_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp()); // sigmoid(1)
    let silu_m1 = -1.0_f32 / (1.0 + (1.0_f32).exp()); // -sigmoid(-1) (silu(-1) = -sigmoid(-1·sign)=, -1*sigmoid(-1))
    approx_eq(v[0][0][0], 1.0 + 0.5 * silu_1, 1e-5);
    approx_eq(v[0][0][1], -1.0 + 0.5 * silu_m1, 1e-5);
}

#[test]
fn ff_module_shape_preserves() {
    ensure_registered();
    let ff = FeedForwardModule::new(8, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let x = t1(&x_data, &[2, 4, 8]);
    let y = ff.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

// ── conv_module: residual + shape preservation ──
//
// При нулевых pw1/dw/pw2 веса = 0 → conv_module возвращает x + 0 = x.
// Это проверяет, что residual ветвь подключена правильно.
#[test]
fn conv_module_zero_weights_is_identity() {
    ensure_registered();
    let h = 4;
    let k = 3;
    let norm_w = t1(&[1.0, 1.0, 1.0, 1.0], &[h]);
    let norm_b = t1(&[0.0, 0.0, 0.0, 0.0], &[h]);
    // pw1 [2h, h, 1] = zeros
    let pw1 = Tensor::zeros(vec![2 * h, h, 1], DType::F32, D).unwrap();
    let dw = Tensor::zeros(vec![h, 1, k], DType::F32, D).unwrap();
    let bn_mean = Tensor::zeros(vec![h], DType::F32, D).unwrap();
    let bn_var = Tensor::ones(vec![h], DType::F32, D).unwrap();
    let bn_w = Tensor::ones(vec![h], DType::F32, D).unwrap();
    let bn_b = Tensor::zeros(vec![h], DType::F32, D).unwrap();
    let pw2 = Tensor::zeros(vec![h, h, 1], DType::F32, D).unwrap();
    let conv = ConvModule::from_weights(
        norm_w, norm_b,
        pw1, None,
        dw, None,
        bn_mean, bn_var, Some(bn_w), Some(bn_b),
        pw2, None,
        1e-5, 1e-5,
    ).unwrap();
    let x_data: Vec<f32> = (0..1 * 5 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[1, 5, 4]);
    let y = conv.forward(&x).unwrap();
    let xv = x.to_vec3::<f32>().unwrap();
    let yv = y.to_vec3::<f32>().unwrap();
    for s in 0..5 {
        for c in 0..4 {
            approx_eq(yv[0][s][c], xv[0][s][c], 1e-5);
        }
    }
}

#[test]
fn conv_module_shape_preserves() {
    ensure_registered();
    let conv = ConvModule::new(8, 5, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 6 * 8).map(|i| (i as f32) * 0.01).collect();
    let x = t1(&x_data, &[2, 6, 8]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 6, 8]);
}

// ── attention_module: residual when out_proj=0 ──
//
// out_proj = 0 → output = x + 0 = x. Проверяет residual цепочку.
#[test]
fn attention_module_zero_out_is_identity() {
    ensure_registered();
    let h = 4;
    let nh = 2;
    let norm_w = t1(&[1.0; 4], &[h]);
    let norm_b = t1(&[0.0; 4], &[h]);
    // Q,K,V identity, O=zeros
    let id = t1(&[1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0], &[h, h]);
    let zero = t1(&[0.0; 16], &[h, h]);
    let zb = t1(&[0.0; 4], &[h]);
    let attn = AttentionModule::from_weights(
        norm_w, norm_b,
        id.clone(), Some(zb.clone()),
        id.clone(), Some(zb.clone()),
        id, Some(zb.clone()),
        zero, Some(zb),
        nh, 1e-5,
    ).unwrap();
    let x_data: Vec<f32> = (0..1 * 3 * 4).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let x = t1(&x_data, &[1, 3, h]);
    let y = attn.forward(&x, None, None).unwrap();
    let yv = y.to_vec3::<f32>().unwrap();
    let xv = x.to_vec3::<f32>().unwrap();
    for s in 0..3 {
        for c in 0..h {
            approx_eq(yv[0][s][c], xv[0][s][c], 1e-5);
        }
    }
}

#[test]
fn attention_module_shape_preserves() {
    ensure_registered();
    let attn = AttentionModule::new(8, 2, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 4 * 8).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let x = t1(&x_data, &[2, 4, 8]);
    let y = attn.forward(&x, None, None).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

#[test]
fn rel_pos_indices_centered_table() {
    // S=3, max_distance=2 → 2*M+1 = 5 entries (0..4). Центр (i=j) → index 2.
    let idx = AttentionModule::relative_position_indices(3, 2);
    assert_eq!(idx.len(), 9);
    // (0,0): 0-0+2=2; (0,1): -1+2=1; (0,2): -2+2=0
    assert_eq!(idx[0], 2); assert_eq!(idx[1], 1); assert_eq!(idx[2], 0);
    // (1,0): 1-0+2=3; (1,1): 2; (1,2): 1
    assert_eq!(idx[3], 3); assert_eq!(idx[4], 2); assert_eq!(idx[5], 1);
    // (2,0): 4 (clamp); (2,1): 3; (2,2): 2
    assert_eq!(idx[6], 4); assert_eq!(idx[7], 3); assert_eq!(idx[8], 2);
}
