use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::diffusion::{Apg, Cfg, ControlNet, Gligen, IpAdapter, Pag, T2iAdapter};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── CFG: scale=1 → output=cond; scale=0 → output=uncond. ──
#[test]
fn cfg_scale_one_equals_cond() {
    ensure_registered();
    let cond = t1(&[1.0, 2.0, 3.0], &[3]);
    let uncond = t1(&[10.0, 20.0, 30.0], &[3]);
    let cfg = Cfg::new(1.0);
    let y = cfg.apply(&cond, &uncond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 1.0, 1e-6);
    approx_eq(v[1], 2.0, 1e-6);
    approx_eq(v[2], 3.0, 1e-6);
}

#[test]
fn cfg_scale_zero_equals_uncond() {
    ensure_registered();
    let cond = t1(&[1.0, 2.0, 3.0], &[3]);
    let uncond = t1(&[10.0, 20.0, 30.0], &[3]);
    let cfg = Cfg::new(0.0);
    let y = cfg.apply(&cond, &uncond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 10.0, 1e-6);
    approx_eq(v[1], 20.0, 1e-6);
    approx_eq(v[2], 30.0, 1e-6);
}

// ── PAG: cond + scale*(cond - perturbed). При perturbed=cond → output=cond. ──
#[test]
fn pag_perturbed_equals_cond_no_change() {
    ensure_registered();
    let cond = t1(&[2.0, 3.0], &[2]);
    let pag = Pag::new(5.0);
    let y = pag.apply(&cond, &cond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 2.0, 1e-6);
    approx_eq(v[1], 3.0, 1e-6);
}

#[test]
fn pag_amplifies_difference() {
    ensure_registered();
    let cond = t1(&[3.0], &[1]);
    let perturbed = t1(&[1.0], &[1]);
    let pag = Pag::new(2.0);
    // 3 + 2*(3-1) = 7
    let y = pag.apply(&cond, &perturbed).unwrap();
    approx_eq(y.to_vec1::<f32>().unwrap()[0], 7.0, 1e-6);
}

// ── APG: cond==uncond → diff=0 → output=cond. ──
#[test]
fn apg_zero_diff_equals_cond() {
    ensure_registered();
    let cond = t1(&[1.0, 2.0, 3.0], &[3]);
    let apg = Apg::new(7.5, 0.0);
    let y = apg.apply(&cond, &cond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 1.0, 1e-6);
    approx_eq(v[1], 2.0, 1e-6);
    approx_eq(v[2], 3.0, 1e-6);
}

// ── APG: orthogonal-проекция убирает компоненту вдоль cond. ──
//
// cond = (1, 0), uncond = (0, 0) → diff = (1, 0) — целиком вдоль cond → ortho = 0
//  → output = cond.
#[test]
fn apg_diff_parallel_to_cond_removed() {
    ensure_registered();
    let cond = t1(&[1.0, 0.0], &[2]);
    let uncond = t1(&[0.0, 0.0], &[2]);
    let apg = Apg::new(10.0, 0.0).with_norm_threshold(1000.0);
    let y = apg.apply(&cond, &uncond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 1.0, 1e-5);
    approx_eq(v[1], 0.0, 1e-5);
}

// cond = (1, 0), uncond = (1, -1) → diff = (0, 1) — целиком ортогональна cond.
#[test]
fn apg_diff_orthogonal_passes_through() {
    ensure_registered();
    let cond = t1(&[1.0, 0.0], &[2]);
    let uncond = t1(&[1.0, -1.0], &[2]);
    let apg = Apg::new(2.0, 0.0).with_norm_threshold(1000.0);
    // diff = (0, 1), ortho = (0, 1), out = cond + 2*(0,1) = (1, 2).
    let y = apg.apply(&cond, &uncond).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    approx_eq(v[0], 1.0, 1e-5);
    approx_eq(v[1], 2.0, 1e-5);
}

// ── ControlNet: zero proj → output = x (zero-init свойство). ──
#[test]
fn controlnet_zero_proj_passes_through() {
    ensure_registered();
    let cn = ControlNet::new(4, 8, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 8]);
    let ctrl_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.05).collect();
    let ctrl = t1(&ctrl_data, &[2, 3, 4]);
    let y = cn.forward(&x, &ctrl).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 {
        for t in 0..3 {
            for h in 0..8 {
                approx_eq(v_y[b][t][h], v_x[b][t][h], 1e-6);
            }
        }
    }
}

#[test]
fn controlnet_identity_proj_adds_control() {
    ensure_registered();
    // proj = identity 2×2, bias=0, scale=1 → y = x + control.
    let w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let cn = ControlNet::from_weights(w, Some(b), 1.0).unwrap();
    let x = t1(&[1.0, 2.0,   3.0, 4.0], &[1, 2, 2]);
    let ctrl = t1(&[10.0, 20.0,   30.0, 40.0], &[1, 2, 2]);
    let y = cn.forward(&x, &ctrl).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    approx_eq(v[0][0][0], 11.0, 1e-6);
    approx_eq(v[0][1][1], 44.0, 1e-6);
}

// ── T2I-Adapter: аналогично ControlNet. ──
#[test]
fn t2i_identity_adds_condition() {
    ensure_registered();
    let w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let ad = T2iAdapter::from_weights(w, Some(b), 0.5).unwrap();
    let x = t1(&[2.0, 4.0], &[1, 1, 2]);
    let cnd = t1(&[10.0, 20.0], &[1, 1, 2]);
    let y = ad.forward(&x, &cnd).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    // y = (2,4) + 0.5*(10,20) = (7, 14)
    approx_eq(v[0][0][0], 7.0, 1e-6);
    approx_eq(v[0][0][1], 14.0, 1e-6);
}

// ── IP-Adapter: бродкаст по seq. ──
#[test]
fn ip_adapter_broadcasts_image_embedding() {
    ensure_registered();
    let w = t1(&[1.0, 0.0,   0.0, 1.0], &[2, 2]);
    let b = t1(&[0.0, 0.0], &[2]);
    let ip = IpAdapter::from_weights(w, Some(b), 1.0).unwrap();
    let x = t1(&[0.0; 8], &[2, 2, 2]);
    let img = t1(&[1.0, 2.0,   3.0, 4.0], &[2, 2]);
    let y = ip.forward(&x, &img).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    // Бродкаст по T=2: для batch 0 → (1,2) на обоих t; batch 1 → (3,4).
    approx_eq(v[0][0][0], 1.0, 1e-6);
    approx_eq(v[0][1][0], 1.0, 1e-6);
    approx_eq(v[1][0][1], 4.0, 1e-6);
    approx_eq(v[1][1][1], 4.0, 1e-6);
}

// ── GLIGEN: gate=0 → tanh(0)=0 → output = x. ──
#[test]
fn gligen_zero_gate_no_change() {
    ensure_registered();
    let g = Gligen::new(4, 8, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, 8]);
    let boxes_data: Vec<f32> = (0..2 * 2 * 4).map(|i| (i as f32) * 0.05).collect();
    let boxes = t1(&boxes_data, &[2, 2, 4]);
    let emb_data: Vec<f32> = (0..2 * 2 * 4).map(|i| (i as f32) * 0.05 + 0.1).collect();
    let emb = t1(&emb_data, &[2, 2, 4]);
    let y = g.forward(&x, &boxes, &emb).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 {
        for t in 0..3 {
            for h in 0..8 {
                approx_eq(v_y[b][t][h], v_x[b][t][h], 1e-6);
            }
        }
    }
}
