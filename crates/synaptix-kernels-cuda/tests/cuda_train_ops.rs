//! Ядра обучения (`TrainOp`) против эталона на хосте.

use synaptix_core::backend::train::TrainOp;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn dev() -> Option<Device> {
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).ok()?;
    Some(Device::Cuda(0))
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn rms_norm_bwd_matches_host() {
    let Some(d) = dev() else { return };
    let (r, h, eps) = (7usize, 333usize, 1e-6f32);
    let x = det(1, r * h, 2.0);
    let w = det(2, h, 1.0);
    let dy = det(3, r * h, 1.0);
    let xt = Tensor::from_vec(x.clone(), vec![r, h], d).unwrap();
    let wt = Tensor::from_vec(w.clone(), vec![h], d).unwrap();
    let dyt = Tensor::from_vec(dy.clone(), vec![r, h], d).unwrap();
    let out = Tensor::train_op(TrainOp::RmsNormBwd { eps }, &[&xt, &wt, &dyt], &[(vec![r, h], DType::F32)]).unwrap();
    // Эталон — численная производная скалярного Σ dy·y по x.
    let f = |x: &[f32]| -> f64 {
        let mut s = 0f64;
        for i in 0..r {
            let row = &x[i * h..(i + 1) * h];
            let ms: f64 = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h as f64;
            let rr = 1.0 / (ms + eps as f64).sqrt();
            for j in 0..h {
                s += dy[i * h + j] as f64 * row[j] as f64 * rr * w[j] as f64;
            }
        }
        s
    };
    let got = host(&out[0]);
    let mut xp = x.clone();
    for idx in [0usize, 5, 333, 1000, r * h - 1] {
        let e = 1e-2f32;
        xp[idx] = x[idx] + e;
        let fp = f(&xp);
        xp[idx] = x[idx] - e;
        let fm = f(&xp);
        xp[idx] = x[idx];
        let num = ((fp - fm) / (2.0 * e as f64)) as f32;
        assert!((num - got[idx]).abs() < 2e-3 * (1.0 + num.abs()), "idx {idx}: {num} vs {}", got[idx]);
    }
    // bf16-вход.
    let out16 = Tensor::train_op(
        TrainOp::RmsNormBwd { eps },
        &[&xt.to_dtype(DType::BF16).unwrap(), &wt.to_dtype(DType::BF16).unwrap(), &dyt],
        &[(vec![r, h], DType::F32)],
    )
    .unwrap();
    assert!(max_err(&host(&out16[0]), &got) < 0.05);
}

#[test]
fn silu_mul_bwd_matches_host() {
    let Some(d) = dev() else { return };
    let n = 5000;
    let g = det(4, n, 4.0);
    let u = det(5, n, 2.0);
    let dy = det(6, n, 1.0);
    let t = |v: &Vec<f32>| Tensor::from_vec(v.clone(), vec![n], d).unwrap();
    let out = Tensor::train_op(TrainOp::SiluMulBwd, &[&t(&g), &t(&u), &t(&dy)], &[(vec![n], DType::F32), (vec![n], DType::F32)]).unwrap();
    let (dg, du) = (host(&out[0]), host(&out[1]));
    for i in 0..n {
        let s = 1.0 / (1.0 + (-g[i]).exp());
        let eg = dy[i] * u[i] * s * (1.0 + g[i] * (1.0 - s));
        let eu = dy[i] * g[i] * s;
        assert!((eg - dg[i]).abs() < 1e-4 && (eu - du[i]).abs() < 1e-4, "i {i}");
    }
}

#[test]
fn cross_entropy_matches_host() {
    let Some(d) = dev() else { return };
    let (r, v) = (5usize, 3001usize);
    let x = det(7, r * v, 8.0);
    let labels: Vec<i32> = vec![3, -100, 3000, 0, 1234];
    let xt = Tensor::from_vec(x.clone(), vec![r, v], d).unwrap();
    let lt = Tensor::from_vec(labels.clone(), vec![r], d).unwrap();
    let scale = 0.25f32;
    let out = Tensor::train_op(TrainOp::CrossEntropy { scale }, &[&xt, &lt], &[(vec![r], DType::F32), (vec![r, v], DType::F32)]).unwrap();
    let (loss, grad) = (host(&out[0]), host(&out[1]));
    for i in 0..r {
        let row = &x[i * v..(i + 1) * v];
        if labels[i] < 0 {
            assert_eq!(loss[i], 0.0);
            assert!(grad[i * v..(i + 1) * v].iter().all(|g| *g == 0.0));
            continue;
        }
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let s: f64 = row.iter().map(|z| ((z - m) as f64).exp()).sum();
        let lse = m as f64 + s.ln();
        let l = labels[i] as usize;
        assert!(((lse - row[l] as f64) as f32 - loss[i]).abs() < 1e-3, "loss row {i}");
        for j in [0usize, l, v - 1] {
            let p = ((row[j] as f64 - lse).exp() - if j == l { 1.0 } else { 0.0 }) as f32 * scale;
            assert!((p - grad[i * v + j]).abs() < 1e-5, "grad {i},{j}");
        }
    }
}

#[test]
fn causal_softmax_and_bwd_match_host() {
    let Some(d) = dev() else { return };
    let (b, t) = (3usize, 37usize);
    let s = det(8, b * t * t, 3.0);
    let seg: Vec<i32> = (0..t).map(|i| if i < 10 { 0 } else if i < 30 { 1 } else { 2 }).collect();
    let st = Tensor::from_vec(s.clone(), vec![b, t, t], d).unwrap();
    let sg = Tensor::from_vec(seg.clone(), vec![t], d).unwrap();
    let scale = 0.7f32;
    let p = Tensor::train_op(TrainOp::CausalSoftmax { scale, has_seg: true }, &[&st, &sg], &[(vec![b, t, t], DType::F32)]).unwrap();
    let ph = host(&p[0]);
    let mut pref = vec![0f32; b * t * t];
    for bb in 0..b {
        for i in 0..t {
            let row = &s[(bb * t + i) * t..(bb * t + i + 1) * t];
            let ok = |j: usize| j <= i && seg[j] == seg[i];
            let m = (0..t).filter(|&j| ok(j)).map(|j| row[j] * scale).fold(f32::MIN, f32::max);
            let z: f32 = (0..t).filter(|&j| ok(j)).map(|j| (row[j] * scale - m).exp()).sum();
            for j in 0..t {
                pref[(bb * t + i) * t + j] = if ok(j) { (row[j] * scale - m).exp() / z } else { 0.0 };
            }
        }
    }
    assert!(max_err(&ph, &pref) < 1e-5);
    let dp = det(9, b * t * t, 1.0);
    let dpt = Tensor::from_vec(dp.clone(), vec![b, t, t], d).unwrap();
    let ds = Tensor::train_op(TrainOp::SoftmaxBwd { scale }, &[&p[0], &dpt], &[(vec![b, t, t], DType::F32)]).unwrap();
    let dsh = host(&ds[0]);
    for row in 0..b * t {
        let pr = &pref[row * t..(row + 1) * t];
        let dr = &dp[row * t..(row + 1) * t];
        let acc: f32 = pr.iter().zip(dr).map(|(a, b)| a * b).sum();
        for j in 0..t {
            let e = scale * pr[j] * (dr[j] - acc);
            assert!((e - dsh[row * t + j]).abs() < 1e-5);
        }
    }
}

#[test]
fn adamw_matches_host() {
    let Some(d) = dev() else { return };
    let n = 1000;
    let (p, g, m, v) = (det(10, n, 1.0), det(11, n, 1.0), det(12, n, 0.1), det(13, n, 0.1).iter().map(|x| x.abs()).collect::<Vec<_>>());
    let t = |x: &Vec<f32>| Tensor::from_vec(x.clone(), vec![n], d).unwrap();
    let (lr, b1, b2, eps, wd, bc1, bc2) = (1e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32, 0.19f32, 0.002f32);
    let out = Tensor::train_op(
        TrainOp::AdamW { lr, beta1: b1, beta2: b2, eps, weight_decay: wd, bc1, bc2 },
        &[&t(&p), &t(&g), &t(&m), &t(&v)],
        &[(vec![n], DType::F32), (vec![n], DType::F32), (vec![n], DType::F32)],
    )
    .unwrap();
    let (po, mo, vo) = (host(&out[0]), host(&out[1]), host(&out[2]));
    for i in 0..n {
        let mi = b1 * m[i] + (1.0 - b1) * g[i];
        let vi = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let pi = p[i] - lr * ((mi / bc1) / ((vi / bc2).sqrt() + eps) + wd * p[i]);
        assert!((mi - mo[i]).abs() < 1e-6 && (vi - vo[i]).abs() < 1e-6 && (pi - po[i]).abs() < 1e-5);
    }
}

#[test]
fn nvfp4_dequant_from_shuffled_matches_linear() {
    let Some(d) = dev() else { return };
    let (n, k) = (192usize, 320usize);
    let w = Tensor::from_vec(det(14, n * k, 1.0), vec![n, k], d).unwrap().to_dtype(DType::F16).unwrap();
    let q = w.quantize_to_nvfp4().unwrap();
    let linear = host(&q.dequantize(DType::BF16).unwrap());
    q.ensure_shuffled().unwrap();
    if q.packed_arc().is_some() {
        eprintln!("карта без FP4-MMA: перемешанной копии нет, проверка пропущена");
        return;
    }
    let shuf = host(&q.dequantize_bf16().unwrap());
    assert_eq!(max_err(&linear, &shuf), 0.0);
}

/// Пакетный matmul F32 и BF16 на карте — на нём стоит внимание обучения.
#[test]
fn batched_matmul_f32_and_transposed_views() {
    let Some(d) = dev() else { return };
    let (b, m, k, n) = (4usize, 33usize, 64usize, 17usize);
    let a = det(15, b * m * k, 1.0);
    let bm = det(16, b * k * n, 1.0);
    let at = Tensor::from_vec(a.clone(), vec![b, m, k], d).unwrap();
    let bt = Tensor::from_vec(bm.clone(), vec![b, k, n], d).unwrap();
    let c = host(&at.matmul(&bt).unwrap());
    let mut cref = vec![0f32; b * m * n];
    for bb in 0..b {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[(bb * m + i) * k + kk] * bm[(bb * k + kk) * n + j];
                }
                cref[(bb * m + i) * n + j] = s;
            }
        }
    }
    assert!(max_err(&c, &cref) < 1e-3, "f32 {}", max_err(&c, &cref));
    // A · Bᵀ через транспонированный вид.
    let btt = bt.transpose(1, 2).unwrap().contiguous().unwrap();
    let c2 = host(&at.matmul(&btt.transpose(1, 2).unwrap()).unwrap());
    assert!(max_err(&c2, &cref) < 1e-3);
    let c3 = host(&at.to_dtype(DType::BF16).unwrap().matmul(&bt.to_dtype(DType::BF16).unwrap()).unwrap());
    assert!(max_err(&c3, &cref) < 0.3, "bf16 {}", max_err(&c3, &cref));
}

#[test]
fn lora_kernels_match_host() {
    let Some(d) = dev() else { return };
    let (t, k, n, r) = (300usize, 520usize, 333usize, 24usize);
    let x = det(20, t * k, 1.0);
    let w = det(21, r * k, 0.5);
    let base = det(22, t * n, 1.0);
    let m = det(23, n * r, 0.3);
    let xt = Tensor::from_vec(x.clone(), vec![t, k], d).unwrap();
    let wt = Tensor::from_vec(w.clone(), vec![r, k], d).unwrap();
    let z = Tensor::train_op(TrainOp::LoraDown, &[&xt, &wt], &[(vec![t, r], DType::F32)]).unwrap().remove(0);
    let zh = host(&z);
    let mut zr = vec![0f32; t * r];
    for i in 0..t {
        for j in 0..r {
            zr[i * r + j] = (0..k).map(|q| x[i * k + q] * w[j * k + q]).sum();
        }
    }
    assert!(max_err(&zh, &zr) < 1e-3, "down {}", max_err(&zh, &zr));
    // bf16-вход
    let z16 = Tensor::train_op(TrainOp::LoraDown, &[&xt.to_dtype(DType::BF16).unwrap(), &wt], &[(vec![t, r], DType::F32)]).unwrap().remove(0);
    assert!(max_err(&host(&z16), &zr) < 0.2);

    let bt = Tensor::from_vec(base.clone(), vec![t, n], d).unwrap();
    let mt = Tensor::from_vec(m.clone(), vec![n, r], d).unwrap();
    let s = 0.7f32;
    let up = Tensor::train_op(TrainOp::LoraUpAdd { scale: s }, &[&bt, &z, &mt], &[(vec![t, n], DType::F32)]).unwrap().remove(0);
    let uh = host(&up);
    let mut ur = base.clone();
    for i in 0..t {
        for q in 0..n {
            ur[i * n + q] += s * (0..r).map(|j| zr[i * r + j] * m[q * r + j]).sum::<f32>();
        }
    }
    assert!(max_err(&uh, &ur) < 1e-3, "up {}", max_err(&uh, &ur));

    let parts = t.div_ceil(128);
    let o = Tensor::train_op(TrainOp::LoraOuter, &[&z, &xt], &[(vec![parts, r, k], DType::F32)]).unwrap().remove(0);
    let oh = host(&o.sum_keepdim(0).unwrap());
    let mut or = vec![0f32; r * k];
    for j in 0..r {
        for q in 0..k {
            or[j * k + q] = (0..t).map(|i| zr[i * r + j] * x[i * k + q]).sum();
        }
    }
    assert!(max_err(&oh, &or) < 2e-2, "outer {}", max_err(&oh, &or));
}

fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect()
}

fn flash_case(h: usize, hkv: usize, t: usize, d: usize, seg: Option<Vec<i32>>) {
    let Some(dv_) = dev() else { return };
    let scale = 1.0 / (d as f32).sqrt();
    let q = bf16_round(&det(30, h * t * d, 1.5));
    let k = bf16_round(&det(31, hkv * t * d, 1.5));
    let v = bf16_round(&det(32, hkv * t * d, 1.0));
    let dout = bf16_round(&det(33, h * t * d, 1.0));
    let tb = |x: &Vec<f32>, dims: Vec<usize>| Tensor::from_vec(x.clone(), dims, dv_).unwrap().to_dtype(DType::BF16).unwrap();
    let (qt, kt, vt) = (tb(&q, vec![h, t, d]), tb(&k, vec![hkv, t, d]), tb(&v, vec![hkv, t, d]));
    let segt = seg.as_ref().map(|s| Tensor::from_vec(s.clone(), vec![t], dv_).unwrap());
    let mut ins = vec![&qt, &kt, &vt];
    if let Some(s) = &segt {
        ins.push(s);
    }
    let has_seg = seg.is_some();
    let out = Tensor::train_op(TrainOp::FlashFwd { scale, has_seg }, &ins, &[(vec![h, t, d], DType::BF16), (vec![h, t], DType::F32)]).unwrap();
    let (oh, lh) = (host(&out[0]), host(&out[1]));
    // Эталон
    let g = h / hkv;
    let vis = |i: usize, j: usize| j <= i && seg.as_ref().is_none_or(|s| s[i] == s[j]);
    let mut o_ref = vec![0f32; h * t * d];
    let mut lse_ref = vec![0f32; h * t];
    let mut p_all = vec![0f64; h * t * t];
    for hh in 0..h {
        let hk = hh / g;
        for i in 0..t {
            let mut s = vec![f64::NEG_INFINITY; t];
            for j in 0..t {
                if vis(i, j) {
                    s[j] = (0..d).map(|c| q[(hh * t + i) * d + c] as f64 * k[(hk * t + j) * d + c] as f64).sum::<f64>() * scale as f64;
                }
            }
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            lse_ref[hh * t + i] = (m + z.ln()) as f32;
            for j in 0..t {
                let p = (s[j] - m).exp() / z;
                p_all[(hh * t + i) * t + j] = p;
                for c in 0..d {
                    o_ref[(hh * t + i) * d + c] += (p * v[(hk * t + j) * d + c] as f64) as f32;
                }
            }
        }
    }
    assert!(max_err(&lh, &lse_ref) < 2e-2, "lse d{d}: {}", max_err(&lh, &lse_ref));
    assert!(max_err(&oh, &o_ref) < 3e-2, "o d{d}: {}", max_err(&oh, &o_ref));

    // Обратный проход: delta = Σ dout∘o (с BF16-выходом ядра, как в обучении).
    let o16 = bf16_round(&oh);
    let delta: Vec<f32> = (0..h * t).map(|r| (0..d).map(|c| dout[r * d + c] * o16[r * d + c]).sum()).collect();
    let dot = tb(&dout, vec![h, t, d]);
    let dlt = Tensor::from_vec(delta.clone(), vec![h, t], dv_).unwrap();
    let mut bins = vec![&qt, &kt, &vt, &dot, &out[1], &dlt];
    if let Some(s) = &segt {
        bins.push(s);
    }
    let tp = t.div_ceil(32) * 32;
    let g_out = Tensor::train_op(
        TrainOp::FlashBwd { scale, has_seg },
        &bins,
        &[(vec![h, t, d], DType::F32), (vec![hkv, tp, d], DType::F32), (vec![hkv, tp, d], DType::F32)],
    )
    .unwrap();
    let dq = host(&g_out[0]);
    let dk = host(&g_out[1].narrow(1, 0, t).unwrap().contiguous().unwrap());
    let dvv = host(&g_out[2].narrow(1, 0, t).unwrap().contiguous().unwrap());
    let mut dq_r = vec![0f64; h * t * d];
    let mut dk_r = vec![0f64; hkv * t * d];
    let mut dv_r = vec![0f64; hkv * t * d];
    for hh in 0..h {
        let hk = hh / g;
        for i in 0..t {
            let dl = delta[hh * t + i] as f64;
            for j in 0..t {
                let p = p_all[(hh * t + i) * t + j];
                if p == 0.0 {
                    continue;
                }
                let dp: f64 = (0..d).map(|c| dout[(hh * t + i) * d + c] as f64 * v[(hk * t + j) * d + c] as f64).sum();
                let ds = p * (dp - dl) * scale as f64;
                for c in 0..d {
                    dv_r[(hk * t + j) * d + c] += p * dout[(hh * t + i) * d + c] as f64;
                    dk_r[(hk * t + j) * d + c] += ds * q[(hh * t + i) * d + c] as f64;
                    dq_r[(hh * t + i) * d + c] += ds * k[(hk * t + j) * d + c] as f64;
                }
            }
        }
    }
    let f = |x: Vec<f64>| x.into_iter().map(|v| v as f32).collect::<Vec<f32>>();
    let rel = |a: &[f32], b: &[f32]| {
        let n: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / n.max(1e-6)
    };
    let (dq_r, dk_r, dv_r) = (f(dq_r), f(dk_r), f(dv_r));
    assert!(rel(&dvv, &dv_r) < 2e-2, "dv d{d}: {}", rel(&dvv, &dv_r));
    assert!(rel(&dk, &dk_r) < 3e-2, "dk d{d}: {}", rel(&dk, &dk_r));
    assert!(rel(&dq, &dq_r) < 3e-2, "dq d{d}: {}", rel(&dq, &dq_r));
}

#[test]
fn flash_train_d128_gqa_tail() {
    flash_case(4, 2, 100, 128, None);
}

#[test]
fn flash_train_d128_packed() {
    let t = 150;
    let seg: Vec<i32> = (0..t).map(|i| if i < 37 { 0 } else if i < 110 { 1 } else { 2 }).collect();
    flash_case(4, 1, t, 128, Some(seg));
}

#[test]
fn flash_train_d64_and_d256() {
    flash_case(2, 2, 70, 64, None);
    let seg: Vec<i32> = (0..90).map(|i| (i / 40) as i32).collect();
    flash_case(4, 2, 90, 256, Some(seg));
}
