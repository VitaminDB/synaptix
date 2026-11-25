//! Ядра `hy4.cu` против прямого счёта на CPU (по формулам эталона transformers/vLLM).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_hy_v4_flash::kern::{self, Keys};

fn rnd(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn dev() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).ok().map(|_| Device::Cuda(0))
}

fn up(v: &[f32], shape: Vec<usize>, d: Device) -> Tensor {
    Tensor::from_vec(v.to_vec(), shape, d).unwrap()
}

fn down(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: длины");
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs() / (1.0 + y.abs());
        worst = worst.max(d);
    }
    assert!(worst < tol, "{what}: расхождение {worst}");
}

fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn hc_pre_post_head() {
    let Some(d) = dev() else { return };
    let (t, m, h) = (3usize, 4usize, 96usize);
    let x = rnd(1, t * m * h, 2.0);
    let fnw = rnd(2, 2 * m * m * h, 0.05);
    let base = rnd(3, 2 * m, 1.0);
    let scale = vec![0.7f32, 1.3];
    let (eps, hce, mag) = (1e-5f32, 1e-6f32, 2.0f32);
    let xt = up(&x, vec![t, m, h], d);
    let mixes = xt.reshape(vec![t, m * h]).unwrap().linear(&up(&fnw, vec![2 * m, m * h], d)).unwrap().contiguous().unwrap();
    let (hh, post) = kern::hc_pre(&xt, &mixes, &up(&base, vec![2 * m], d), &up(&scale, vec![2], d), false, eps, hce, mag).unwrap();
    let (mut want_h, mut want_p) = (vec![0f32; t * h], vec![0f32; t * m]);
    for ti in 0..t {
        let row = &x[ti * m * h..(ti + 1) * m * h];
        let r = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / (m * h) as f32 + eps).sqrt();
        let mix: Vec<f32> = (0..2 * m).map(|j| (0..m * h).map(|k| fnw[j * m * h + k] * row[k]).sum::<f32>() * r).collect();
        let pre: Vec<f32> = (0..m).map(|i| sigm(mix[i] * scale[0] + base[i]) + hce).collect();
        for i in 0..m {
            want_p[ti * m + i] = mag * sigm(mix[m + i] * scale[1] + base[m + i]) + hce;
        }
        for dd in 0..h {
            want_h[ti * h + dd] = (0..m).map(|i| pre[i] * row[i * h + dd]).sum();
        }
    }
    close(&down(&hh), &want_h, 1e-4, "hc_pre h");
    let post = post.unwrap();
    close(&down(&post), &want_p, 1e-4, "hc_pre post");
    // Возврат в потоки.
    let y = rnd(4, t * h, 1.0);
    kern::hc_post(&xt, &post, &up(&y, vec![t, h], d)).unwrap();
    let mut want_x = x.clone();
    for ti in 0..t {
        for i in 0..m {
            for dd in 0..h {
                want_x[(ti * m + i) * h + dd] += want_p[ti * m + i] * y[ti * h + dd];
            }
        }
    }
    close(&down(&xt), &want_x, 1e-4, "hc_post");
}

#[test]
fn rope_interleaved_tail() {
    let Some(d) = dev() else { return };
    let (t, heads, hs, off, w) = (3usize, 2usize, 12usize, 4usize, 8usize);
    let x = rnd(5, t * heads * hs, 1.0);
    let xt = up(&x, vec![t, heads * hs], d);
    let pos = Tensor::from_vec(vec![0i32, 7, 1000], vec![t], d).unwrap();
    let theta = 1e7f64;
    kern::rope(&xt, &pos, heads * hs, heads, hs, off, w, theta).unwrap();
    let mut want = x.clone();
    for (ti, p) in [0f32, 7.0, 1000.0].iter().enumerate() {
        for hh in 0..heads {
            let b = ti * heads * hs + hh * hs + off;
            for j in 0..w / 2 {
                let inv = 1.0 / (theta as f32).powf(2.0 * j as f32 / w as f32);
                let (s, c) = (p * inv).sin_cos();
                let (a, bb) = (x[b + 2 * j], x[b + 2 * j + 1]);
                want[b + 2 * j] = a * c - bb * s;
                want[b + 2 * j + 1] = bb * c + a * s;
            }
        }
    }
    close(&down(&xt), &want, 1e-4, "rope");
}

/// Внимание по кэшу: окно и группы против прямого softmax с синком.
#[test]
fn mla_attn_window_and_groups() {
    let Some(d) = dev() else { return };
    let (t, nh, l, r) = (3usize, 2usize, 64usize, 16usize);
    let dd = l + r;
    let cap = 40usize;
    let cache = rnd(6, cap * dd, 1.0);
    let qa = rnd(7, t * nh * dd, 0.3);
    let sinks = vec![0.5f32, -1.0];
    let posv = vec![20i32, 31, 39];
    let pos = Tensor::from_vec(posv.clone(), vec![t], d).unwrap();
    let scale = 0.125f32;
    let ct = up(&cache, vec![cap, dd], d);
    let qt = up(&qa, vec![t, nh, dd], d);
    let st = up(&sinks, vec![nh], d);
    let reference = |keys: &dyn Fn(usize) -> Vec<usize>| -> Vec<f32> {
        let mut out = vec![0f32; t * nh * l];
        for ti in 0..t {
            let ks = keys(ti);
            for h in 0..nh {
                let q = &qa[(ti * nh + h) * dd..(ti * nh + h + 1) * dd];
                let sc: Vec<f32> = ks.iter().map(|&k| (0..dd).map(|i| q[i] * cache[k * dd + i]).sum::<f32>() * scale).collect();
                let mx = sc.iter().cloned().fold(sinks[h], f32::max);
                let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                let den: f32 = e.iter().sum::<f32>() + (sinks[h] - mx).exp();
                for c in 0..l {
                    out[(ti * nh + h) * l + c] = ks.iter().zip(&e).map(|(&k, w)| w * cache[k * dd + c]).sum::<f32>() / den;
                }
            }
        }
        out
    };
    let o = kern::mla_attn(&qt, &ct, &st, &pos, l, Keys::Window(8), scale).unwrap();
    let want = reference(&|ti| ((posv[ti] - 7).max(0) as usize..=posv[ti] as usize).collect());
    close(&down(&o), &want, 1e-4, "окно");
    // Группы по 4: у каждого запроса две чужие группы (одна — пустая) плюс своя.
    let groups = vec![0u32, u32::MAX, 2, 4, 1, 3];
    let gt = Tensor::from_vec(groups.clone(), vec![t, 2], d).unwrap();
    let o = kern::mla_attn(&qt, &ct, &st, &pos, l, Keys::Groups(&gt, 4), scale).unwrap();
    let want = reference(&|ti| {
        let p = posv[ti] as usize;
        let mut ks = Vec::new();
        for &g in &groups[ti * 2..ti * 2 + 2] {
            if g != u32::MAX && g as usize != p / 4 {
                ks.extend((g as usize * 4..g as usize * 4 + 4).filter(|&k| k <= p));
            }
        }
        ks.extend((p / 4 * 4..p / 4 * 4 + 4).filter(|&k| k <= p));
        ks
    });
    close(&down(&o), &want, 1e-4, "группы");
    // Блок драфтера: общая строка групп, группы целиком ниже lo и отрезок lo..pos.
    let dpos = vec![20i32, 21, 22];
    let dpt = Tensor::from_vec(dpos.clone(), vec![t], d).unwrap();
    let drow = vec![0u32, 2, 4, u32::MAX];
    let drt = Tensor::from_vec(drow.clone(), vec![1, 4], d).unwrap();
    let o = kern::mla_attn(&qt, &ct, &st, &dpt, l, Keys::Draft { groups: &drt, size: 4, lo: 16 }, scale).unwrap();
    let want = reference(&|ti| {
        let mut ks: Vec<usize> = (0..4).chain(8..12).collect();
        ks.extend(16..=dpos[ti] as usize);
        ks
    });
    close(&down(&o), &want, 1e-4, "блок драфтера");
    // BF16-кэш читается так же.
    let cb = ct.to_dtype(DType::BF16).unwrap();
    let o = kern::mla_attn(&qt, &cb, &st, &pos, l, Keys::Window(8), scale).unwrap();
    let want = reference(&|ti| ((posv[ti] - 7).max(0) as usize..=posv[ti] as usize).collect());
    close(&down(&o), &want, 2e-2, "окно BF16");
}

#[test]
fn indexer_pool_and_score() {
    let Some(d) = dev() else { return };
    let (c, g, ng) = (128usize, 4usize, 3usize);
    let toks = rnd(8, ng * g * c, 1.0);
    let wv = rnd(9, c * c, 0.1);
    let wg = rnd(10, c * c, 0.1);
    let ape = rnd(11, g * c, 0.5);
    let kn = rnd(12, c, 1.0);
    let theta = 1e7f64;
    let tok = up(&toks, vec![ng * g, c], d);
    let out = kern::zeros(vec![ng, c], DType::F32, d).unwrap();
    kern::idx_pool(&tok, &up(&wv, vec![c, c], d), &up(&wg, vec![c, c], d), &up(&ape, vec![g, c], d), &up(&kn, vec![c], d), &out, 0, ng, g, 64, 1e-5, theta).unwrap();
    let mut want = vec![0f32; ng * c];
    for gi in 0..ng {
        let mut pooled = vec![0f32; c];
        for ch in 0..c {
            let mut v = [0f32; 4];
            let mut gt = [0f32; 4];
            for i in 0..g {
                let k = &toks[(gi * g + i) * c..(gi * g + i + 1) * c];
                v[i] = (0..c).map(|j| wv[ch * c + j] * k[j]).sum();
                gt[i] = (0..c).map(|j| wg[ch * c + j] * k[j]).sum::<f32>() + ape[i * c + ch];
            }
            let mx = gt.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = gt.iter().map(|x| (x - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            pooled[ch] = (0..g).map(|i| v[i] * e[i] / s).sum();
        }
        let r = 1.0 / (pooled.iter().map(|x| x * x).sum::<f32>() / c as f32 + 1e-5).sqrt();
        let p: Vec<f32> = pooled.iter().zip(&kn).map(|(x, w)| x * r * w).collect();
        let pos = (gi * g) as f32;
        for ch in 0..c {
            want[gi * c + ch] = if ch < 64 {
                p[ch]
            } else {
                let j = (ch - 64) / 2;
                let inv = 1.0 / (theta as f32).powf(2.0 * j as f32 / 64.0);
                let (s, co) = (pos * inv).sin_cos();
                let (a, b) = (p[64 + 2 * j], p[64 + 2 * j + 1]);
                if (ch - 64) % 2 == 1 { b * co + a * s } else { a * co - b * s }
            };
        }
    }
    close(&down(&out), &want, 1e-3, "пулинг групп");
    // Оценки: запрос на позиции 13 видит группы 0..3 (своя — 3).
    let nhi = 2usize;
    let q = rnd(13, nhi * c, 1.0);
    let w = vec![0.3f32, -0.2];
    let pos = Tensor::from_vec(vec![13i32], vec![1], d).unwrap();
    let sc = kern::idx_score(&up(&q, vec![1, nhi, c], d), &up(&w, vec![1, nhi], d), &out, &pos, ng, g).unwrap();
    let got = down(&sc);
    for gi in 0..ng {
        let k = &want[gi * c..(gi + 1) * c];
        let s: f32 = (0..nhi).map(|h| w[h] * (0..c).map(|j| q[h * c + j] * k[j]).sum::<f32>().max(0.0)).sum();
        assert!((got[gi] - s).abs() < 1e-3 * (1.0 + s.abs()), "оценка группы {gi}: {} против {s}", got[gi]);
    }
}

#[test]
fn engram_gate_update() {
    let Some(d) = dev() else { return };
    let (t, m, h) = (2usize, 4usize, 64usize);
    let x = rnd(14, t * m * h, 3.0);
    let proj = rnd(15, t * (m + 1) * h, 1.0);
    let kn = rnd(16, m * h, 1.0);
    let qn = rnd(17, m * h, 1.0);
    let pg = rnd(18, h, 1.0);
    let hs = rnd(19, m * h, 0.01);
    let xt = up(&x, vec![t, m, h], d);
    let live = Tensor::from_vec(vec![1u8, 0], vec![t], d).unwrap();
    let eps = f32::EPSILON;
    kern::engram_gate(&xt, &up(&proj, vec![t, (m + 1) * h], d), &up(&kn, vec![m, h], d), &up(&qn, vec![m, h], d), &up(&pg, vec![h], d), &up(&hs, vec![m, h], d), &live, eps).unwrap();
    let mut want = x.clone();
    let rms = |v: &[f32]| 1.0 / (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + eps).sqrt();
    let ti = 0;
    let val = &proj[(ti * (m + 1) + m) * h..(ti * (m + 1) + m + 1) * h];
    let rv = rms(val);
    for c in 0..m {
        let key = &proj[(ti * (m + 1) + c) * h..(ti * (m + 1) + c + 1) * h];
        let xr = &x[(ti * m + c) * h..(ti * m + c + 1) * h];
        let (rk, rx) = (rms(key), rms(xr));
        let dot: f32 = (0..h).map(|i| key[i] * rk * kn[c * h + i] * xr[i] * rx * qn[c * h + i]).sum::<f32>() / (h as f32).sqrt();
        let gate = sigm(dot.signum() * dot.abs().max(1e-6).sqrt());
        for i in 0..h {
            want[(ti * m + c) * h + i] += (gate + hs[c * h + i]) * pg[i] * val[i] * rv;
        }
    }
    close(&down(&xt), &want, 1e-4, "гейт engram (вторая строка не тронута)");
}

/// Поглощённые проекции декода: x — срез строки q (шаги как у q [T, NH·qk]),
/// W BF16 [NH, Din, Dout]; T от 1 до 8.
#[test]
fn head_gemv_matches_cpu() {
    let Some(d) = dev() else { return };
    for &(t, nh, qk, din, dout) in &[(1usize, 4usize, 256usize, 192usize, 512usize), (3, 2, 512, 512, 256), (8, 3, 192, 192, 128)] {
        let x = rnd(11, t * nh * qk, 1.0);
        let w = rnd(12, nh * din * dout, 0.1);
        // Вес в BF16 и обратно — эталон считает с теми же округлёнными весами.
        let wb: Vec<f32> = w.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect();
        let wt = up(&wb, vec![nh, din, dout], d).to_dtype(DType::BF16).unwrap();
        let got = down(&kern::head_gemv(&up(&x, vec![t, nh * qk], d), nh * qk, qk, &wt, t).unwrap());
        let mut want = vec![0f32; t * nh * dout];
        for ti in 0..t {
            for h in 0..nh {
                for c in 0..dout {
                    let mut s = 0f32;
                    for k in 0..din {
                        s += x[ti * nh * qk + h * qk + k] * wb[(h * din + k) * dout + c];
                    }
                    want[(ti * nh + h) * dout + c] = s;
                }
            }
        }
        close(&got, &want, 1e-4, &format!("head_gemv t={t} din={din} dout={dout}"));
    }
}
