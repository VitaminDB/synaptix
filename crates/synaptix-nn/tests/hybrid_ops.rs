use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::hybrid::{
    FalconMamba, GriffinBlock, Hymba, Jamba, LayerKind, MixPolicy, Samba, Zamba,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── Каждая модель: zero-init out projection → output = x. ──
#[test]
fn falcon_mamba_zero_out_equals_x() {
    ensure_registered();
    let f = FalconMamba::new(4, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = f.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 {
        for t in 0..3 {
            for h in 0..4 {
                approx_eq(v_y[b][t][h], v_x[b][t][h], 1e-5);
            }
        }
    }
}

#[test]
fn griffin_block_zero_out_equals_x() {
    ensure_registered();
    let g = GriffinBlock::new(4, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = g.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 { for t in 0..3 { for h in 0..4 {
        approx_eq(v_y[b][t][h], v_x[b][t][h], 1e-5);
    }}}
}

#[test]
fn hymba_zero_fuse_equals_x() {
    ensure_registered();
    let h_mod = Hymba::new(4, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = h_mod.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 { for t in 0..3 { for k in 0..4 {
        approx_eq(v_y[b][t][k], v_x[b][t][k], 1e-5);
    }}}
}

#[test]
fn jamba_zero_experts_equals_x() {
    ensure_registered();
    // `new()` стартует с KaimingUniform-experts, потому используем from_weights
    // с zero-experts: gate любой → softmax-blend(0, 0) = 0 → out = x.
    let nw = Tensor::ones(vec![4], DType::F32, D).unwrap();
    let nb = Tensor::zeros(vec![4], DType::F32, D).unwrap();
    let gw = Tensor::zeros(vec![2, 4], DType::F32, D).unwrap();
    let e0w = Tensor::zeros(vec![4, 4], DType::F32, D).unwrap();
    let e0b = Tensor::zeros(vec![4], DType::F32, D).unwrap();
    let e1w = Tensor::zeros(vec![4, 4], DType::F32, D).unwrap();
    let e1b = Tensor::zeros(vec![4], DType::F32, D).unwrap();
    let j = Jamba::from_weights(nw, nb, gw, e0w, Some(e0b), e1w, Some(e1b), 1e-5).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = j.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 { for t in 0..3 { for k in 0..4 {
        approx_eq(v_y[b][t][k], v_x[b][t][k], 1e-5);
    }}}
}

#[test]
fn samba_zero_out_equals_x() {
    ensure_registered();
    let s = Samba::new(4, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = s.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 { for t in 0..3 { for k in 0..4 {
        approx_eq(v_y[b][t][k], v_x[b][t][k], 1e-5);
    }}}
}

#[test]
fn zamba_zero_out_equals_x() {
    ensure_registered();
    let z = Zamba::new(4, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let y = z.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 { for t in 0..3 { for k in 0..4 {
        approx_eq(v_y[b][t][k], v_x[b][t][k], 1e-5);
    }}}
}

// ── MixPolicy: kind_at + counts + functional dispatch. ──
#[test]
fn mix_policy_counts_and_kind() {
    let schedule = vec![
        LayerKind::Attention, LayerKind::Ssm, LayerKind::Ssm,
        LayerKind::Attention, LayerKind::MoE,
    ];
    let p = MixPolicy::new(schedule);
    assert_eq!(p.len(), 5);
    assert!(!p.is_empty());
    assert_eq!(*p.kind_at(0).unwrap(), LayerKind::Attention);
    assert_eq!(*p.kind_at(1).unwrap(), LayerKind::Ssm);
    assert_eq!(*p.kind_at(4).unwrap(), LayerKind::MoE);
    assert!(p.kind_at(5).is_none());
    assert_eq!(p.counts(), (2, 2, 1));
}

#[test]
fn mix_policy_dispatches_to_correct_fn() {
    ensure_registered();
    let p = MixPolicy::new(vec![
        LayerKind::Attention,
        LayerKind::Ssm,
        LayerKind::MoE,
    ]);
    let x = t1(&[1.0, 2.0], &[2]);

    let y0 = p.apply(&x, 0,
        |t| t.affine(10.0, 0.0),
        |t| t.affine(100.0, 0.0),
        |t| t.affine(1000.0, 0.0),
    ).unwrap();
    let y1 = p.apply(&x, 1,
        |t| t.affine(10.0, 0.0),
        |t| t.affine(100.0, 0.0),
        |t| t.affine(1000.0, 0.0),
    ).unwrap();
    let y2 = p.apply(&x, 2,
        |t| t.affine(10.0, 0.0),
        |t| t.affine(100.0, 0.0),
        |t| t.affine(1000.0, 0.0),
    ).unwrap();
    let err = p.apply(&x, 3,
        |t| t.affine(10.0, 0.0),
        |t| t.affine(100.0, 0.0),
        |t| t.affine(1000.0, 0.0),
    );
    approx_eq(y0.to_vec1::<f32>().unwrap()[0], 10.0, 1e-6);
    approx_eq(y1.to_vec1::<f32>().unwrap()[0], 100.0, 1e-6);
    approx_eq(y2.to_vec1::<f32>().unwrap()[0], 1000.0, 1e-6);
    assert!(err.is_err());
}
