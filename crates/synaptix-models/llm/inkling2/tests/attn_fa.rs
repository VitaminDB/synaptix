//! Flash-ядро префилла `ink2_attn_fa` против прежнего `ink2_attn` (скалярный F32-путь) на
//! случайных входах: SWA с кольцом KV и без, полные слои, T = 1…512 в одном прогоне,
//! позиции до 200K+, dist около границ extent и окна, одна часть (синк и гейт в ядре) и
//! несколько частей (сборка `ink2_attn_merge`). По строкам (токен, голова): макс. абсолютная
//! ошибка, относительная (к макс. |эталона| строки) и минимальный косинус.
//!
//! `cargo test --profile fast-release -p synaptix-llm-inkling2 --test attn_fa -- --nocapture`

use synaptix_core::backend::{registry, Ink2Launch};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const D: usize = 128;
const DR: usize = 16;

struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    /// Приближённо N(0, 1): сумма 4 равномерных.
    fn n(&mut self) -> f32 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
    fn vec(&mut self, len: usize, s: f32) -> Vec<f32> {
        (0..len).map(|_| self.n() * s).collect()
    }
}

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn dev_f32(v: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec::<_, f32>(v, shape, Device::Cuda(0)).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).unwrap()
}

fn launch(name: &str, a: &Ink2Launch, grid: (u32, u32, u32)) {
    let dev = Device::Cuda(0);
    registry::backend_for(dev).unwrap().inkling2_launch(dev, name, a, grid, 128).unwrap();
}

fn ad(t: &Tensor) -> u64 {
    t.device_address().unwrap()
}

struct Case {
    name: &'static str,
    nh: usize,
    nkv: usize,
    past: usize,
    t: usize,
    window: usize,
    extent: usize,
    ring: usize,
    cap: usize,
    splits: usize,
}

struct Stat {
    abs: f32,
    rel: f32,
    cos: f32,
    /// Ошибка (макс. отн. к макс. |f64| строки) нового и прежнего ядра против f64 на CPU
    /// для выборочных строк — только на длинном контексте.
    f64: Option<(f64, f64)>,
}

fn run(c: &Case, seed: u64) -> Stat {
    let mut rng = Rng(seed);
    let (nh, nkv, t) = (c.nh, c.nkv, c.t);
    // q уже L2-клипнуты и ×tau: размах логитов q·k·scale ~ σ = 3.
    let q = dev_f32(rng.vec(t * nh * D, 3.0), vec![t, nh * D]);
    let r = dev_f32(rng.vec(t * nh * DR, 1.0), vec![t, nh * DR]);
    let pscale = if std::env::var_os("FA_TEST_NOBIAS").is_some() { 0.0 } else { 0.4 };
    let proj = dev_f32(rng.vec(DR * c.extent, pscale), vec![DR, c.extent]);
    let sink = dev_f32(rng.vec(t * nh, 2.0), vec![t, nh]);
    let gate = dev_f32(rng.vec(t * nh * D, 1.0), vec![t, nh * D]);
    let kc = dev_f32(rng.vec(nkv * c.cap * D, 1.0), vec![nkv, c.cap, D]).to_dtype(DType::BF16).unwrap();
    let vc = dev_f32(rng.vec(nkv * c.cap * D, 1.0), vec![nkv, c.cap, D]).to_dtype(DType::BF16).unwrap();
    let pos = Tensor::from_vec::<_, u32>(vec![c.past as u32], vec![1], Device::Cuda(0)).unwrap();
    let mut a = Ink2Launch::default();
    a.a[0] = ad(&q);
    a.a[1] = ad(&r);
    a.a[2] = ad(&proj);
    a.a[3] = ad(&sink);
    a.a[4] = ad(&kc);
    a.a[5] = ad(&vc);
    a.a[6] = ad(&gate);
    a.a[10] = ad(&pos);
    a.u[0] = t as u32;
    a.u[1] = nh as u32;
    a.u[2] = nkv as u32;
    a.u[3] = D as u32;
    a.u[4] = DR as u32;
    a.u[5] = c.cap as u32;
    a.u[6] = 0xdead_beef; // позиция — только с карты
    a.u[7] = c.window as u32;
    a.u[8] = c.extent as u32;
    a.u[10] = c.ring as u32;
    a.f[0] = 1.0 / (D as f32).sqrt();

    // Эталон: прежнее ядро, на длинном контексте — части по ключам + merge (как на декоде).
    let out_ref = Tensor::zeros(vec![t, nh * D], DType::F32, Device::Cuda(0)).unwrap();
    let sref = if c.past > 4096 { 32 } else { 1 };
    let pref = Tensor::zeros(vec![t * nh * sref * 130], DType::F32, Device::Cuda(0)).unwrap();
    let mut ar = a;
    ar.a[7] = ad(&out_ref);
    ar.a[8] = ad(&pref);
    ar.u[9] = sref as u32;
    launch("ink2_attn", &ar, (t as u32, nh as u32, sref as u32));
    if sref > 1 {
        launch("ink2_attn_merge", &ar, (t as u32, nh as u32, 1));
    }

    let out = Tensor::zeros(vec![t, nh * D], DType::F32, Device::Cuda(0)).unwrap();
    let pn = Tensor::zeros(vec![t * nh * c.splits.max(1) * 130], DType::F32, Device::Cuda(0)).unwrap();
    let mut an = a;
    an.a[7] = ad(&out);
    an.a[8] = ad(&pn);
    an.u[9] = c.splits as u32;
    launch("ink2_attn_fa", &an, (t.div_ceil(64) as u32, nh as u32, c.splits.max(1) as u32));
    if c.splits > 1 {
        launch("ink2_attn_merge", &an, (t as u32, nh as u32, 1));
    }

    let (x, y) = (host(&out), host(&out_ref));
    let mut st = Stat { abs: 0.0, rel: 0.0, cos: 1.0, f64: None };
    if c.past > 100_000 || c.t == 512 {
        let (qh, rh, ph, sh, gh) = (host(&q), host(&r), host(&proj), host(&sink), host(&gate));
        let (kh, vh) = (host(&kc.to_dtype(DType::F32).unwrap()), host(&vc.to_dtype(DType::F32).unwrap()));
        let (mut en, mut eo) = (0f64, 0f64);
        for &tt in &[0, t / 2, t - 1] {
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let pos = c.past + tt;
                let qo = (tt * nh + h) * D;
                let lo = if c.window > 0 && pos + 1 > c.window { pos + 1 - c.window } else { 0 };
                let row = |j: usize| kvh * c.cap + if c.ring > 0 { j % c.ring } else { j };
                let mut lg = Vec::with_capacity(pos + 1);
                for j in lo..=pos {
                    let kr = &kh[row(j) * D..(row(j) + 1) * D];
                    let mut s = 0f64;
                    for i in 0..D {
                        s += qh[qo + i] as f64 * (1.0 / (D as f64).sqrt()) * kr[i] as f64;
                    }
                    let dist = pos - j;
                    if dist < c.extent {
                        for ch in 0..DR {
                            s += rh[(tt * nh + h) * DR + ch] as f64 * ph[ch * c.extent + dist] as f64;
                        }
                    }
                    lg.push(s);
                }
                let sk = sh[tt * nh + h] as f64;
                let m = lg.iter().cloned().fold(sk, f64::max);
                let mut l = (sk - m).exp();
                let mut o = [0f64; D];
                for (j, &s) in lg.iter().enumerate() {
                    let p = (s - m).exp();
                    l += p;
                    let vr = &vh[row(lo + j) * D..(row(lo + j) + 1) * D];
                    for i in 0..D {
                        o[i] += p * vr[i] as f64;
                    }
                }
                let (mut mx, mut dn, mut dold) = (0f64, 0f64, 0f64);
                for i in 0..D {
                    let g = gh[qo + i] as f64;
                    let e = o[i] / l / (1.0 + (-g).exp());
                    mx = mx.max(e.abs());
                    dn = dn.max((x[qo + i] as f64 - e).abs());
                    dold = dold.max((y[qo + i] as f64 - e).abs());
                }
                en = en.max(dn / mx);
                eo = eo.max(dold / mx);
            }
        }
        st.f64 = Some((en, eo));
    }
    for row in 0..t * nh {
        let (xs, ys) = (&x[row * D..(row + 1) * D], &y[row * D..(row + 1) * D]);
        let (mut e, mut m, mut xy, mut xx, mut yy) = (0f32, 0f32, 0f64, 0f64, 0f64);
        for i in 0..D {
            assert!(xs[i].is_finite(), "{}: не число в строке {row}", c.name);
            e = e.max((xs[i] - ys[i]).abs());
            m = m.max(ys[i].abs());
            xy += xs[i] as f64 * ys[i] as f64;
            xx += xs[i] as f64 * xs[i] as f64;
            yy += ys[i] as f64 * ys[i] as f64;
        }
        st.abs = st.abs.max(e);
        st.rel = st.rel.max(e / m.max(1e-20));
        st.cos = st.cos.min((xy / (xx.sqrt() * yy.sqrt()).max(1e-30)) as f32);
    }
    st
}

#[test]
fn attn_fa_matches_scalar() {
    if !setup() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    let full = |name, past, t, splits| Case { name, nh: 24, nkv: 8, past, t, window: 0, extent: 1024, ring: 0, cap: past + t + 16, splits };
    let swa = |name, past, t, ring, splits| Case {
        name,
        nh: 24,
        nkv: 8,
        past,
        t,
        window: 1024,
        extent: 1024,
        ring,
        cap: if ring > 0 { ring } else { past + t + 16 },
        splits,
    };
    let long = |name, past, t, splits| Case { name, nh: 6, nkv: 2, past, t, window: 0, extent: 1024, ring: 0, cap: past + t + 16, splits };
    let cases = [
        full("полный, past 0, T 512", 0, 512, 1),
        full("полный, past 0, T 1", 0, 1, 1),
        full("полный, past 0, T 7", 0, 7, 1),
        full("полный, past 900, T 200 (extent)", 900, 200, 1),
        full("полный, past 961, T 65 (extent)", 961, 65, 3),
        full("полный, past 1023, T 64", 1023, 64, 1),
        full("полный, past 3000, T 333", 3000, 333, 1),
        full("полный, past 3000, T 333, S 4", 3000, 333, 4),
        full("полный, past 8000, T 512, S 8", 8000, 512, 8),
        swa("SWA лин., past 0, T 512", 0, 512, 0, 1),
        swa("SWA лин., past 1000, T 100 (окно)", 1000, 100, 0, 1),
        swa("SWA лин., past 3000, T 300", 3000, 300, 0, 2),
        swa("SWA кольцо, past 600, T 512", 600, 512, 1536, 1),
        swa("SWA кольцо, past 5000, T 512", 5000, 512, 1536, 1),
        swa("SWA кольцо, past 5000, T 512, S 2", 5000, 512, 1536, 2),
        swa("SWA кольцо, past 7777, T 33", 7777, 33, 1536, 1),
        swa("SWA кольцо, past 1535, T 2", 1535, 2, 1536, 1),
        long("полный, past 131072, T 512, S 16", 131072, 512, 16),
        long("полный, past 200000, T 512, S 16", 200000, 512, 16),
        long("полный, past 200000, T 129, S 64", 200000, 129, 64),
        long("полный, past 200000, T 512, S 1", 200000, 512, 1),
    ];
    let (mut wa, mut wr, mut wc) = (0f32, 0f32, 1f32);
    for (i, c) in cases.iter().enumerate() {
        let s = run(c, 1000 + i as u64);
        let f = s.f64.map(|(n, o)| format!("  | к f64: новое {n:.2e}, прежнее {o:.2e}")).unwrap_or_default();
        println!("{:<40} abs {:.2e}  rel {:.2e}  cos_min {:.8}{f}", c.name, s.abs, s.rel, s.cos);
        wa = wa.max(s.abs);
        wr = wr.max(s.rel);
        wc = wc.min(s.cos);
    }
    println!("итого: abs {wa:.2e}  rel {wr:.2e}  cos_min {wc:.8}");
    assert!(wr < 1e-3 && wc > 0.999999, "расхождение с ink2_attn");
}

/// Скорость: прежнее ink2_attn (префилл, одна часть) и ink2_attn_fa (части по
/// `kern::attn_fa_splits`) на T = 512, nh 24 / nkv 8 — время одного слоя.
/// `cargo test --profile fast-release -p synaptix-llm-inkling2 --test attn_fa bench -- --ignored --nocapture`
#[test]
#[ignore]
fn attn_fa_bench() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    let (nh, nkv, t) = (24usize, 8usize, 512usize);
    let sync = || synaptix_core::device::cuda::synchronize(0).unwrap();
    for &(past, swa) in &[(8192usize, true), (8192, false), (32768, false), (131072, false), (200000, false)] {
        let mut rng = Rng(7);
        let cap = if swa { 1536 } else { past + t };
        let q = dev_f32(rng.vec(t * nh * D, 3.0), vec![t, nh * D]);
        let r = dev_f32(rng.vec(t * nh * DR, 1.0), vec![t, nh * DR]);
        let proj = dev_f32(rng.vec(DR * 1024, 0.4), vec![DR, 1024]);
        let sink = dev_f32(rng.vec(t * nh, 2.0), vec![t, nh]);
        let gate = dev_f32(rng.vec(t * nh * D, 1.0), vec![t, nh * D]);
        let kc = Tensor::zeros(vec![nkv, cap, D], DType::BF16, dev).unwrap();
        let vc = Tensor::zeros(vec![nkv, cap, D], DType::BF16, dev).unwrap();
        let pos = Tensor::from_vec::<_, u32>(vec![past as u32], vec![1], dev).unwrap();
        let out = Tensor::zeros(vec![t, nh * D], DType::F32, dev).unwrap();
        let keys = if swa { 1024 + 64 } else { past + t };
        let splits = synaptix_llm_inkling2::kern::attn_fa_splits(t, nh, keys);
        let parts = Tensor::zeros(vec![t * nh * splits * 130], DType::F32, dev).unwrap();
        let mut a = Ink2Launch::default();
        for (i, x) in [&q, &r, &proj, &sink, &kc, &vc, &gate, &out, &parts].iter().enumerate() {
            a.a[i] = ad(x);
        }
        a.a[10] = ad(&pos);
        a.u = [t as u32, nh as u32, nkv as u32, D as u32, DR as u32, cap as u32, 0, if swa { 1024 } else { 0 }, 1024, 1, if swa { 1536 } else { 0 }, 0];
        a.f[0] = 1.0 / (D as f32).sqrt();
        let time = |f: &dyn Fn(), reps: usize| {
            f();
            sync();
            let s = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            sync();
            s.elapsed().as_secs_f64() * 1e3 / reps as f64
        };
        let old = time(&|| launch("ink2_attn", &a, (t as u32, nh as u32, 1)), if past > 50_000 { 1 } else { 3 });
        let mut an = a;
        an.u[9] = splits as u32;
        let new = time(
            &|| {
                launch("ink2_attn_fa", &an, (t.div_ceil(64) as u32, nh as u32, splits as u32));
                if splits > 1 {
                    launch("ink2_attn_merge", &an, (t as u32, nh as u32, 1));
                }
            },
            10,
        );
        let flop = 4.0 * (t * nh * D) as f64 * if swa { 1024.0 } else { past as f64 + t as f64 / 2.0 };
        println!(
            "{} past {past:>6}: прежнее {old:8.2} мс, flash {new:7.3} мс (S {splits}), ×{:.0}, {:.1} TFLOPS полезных",
            if swa { "SWA   " } else { "полный" },
            old / new,
            flop / new / 1e9
        );
    }
}
