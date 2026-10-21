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
