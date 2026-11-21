//! Квантованный KV полных слотов (MXFP8 и INT8 с масштабом E8M0 на 32 канала).
//!
//! 1. Запись: `ink2_qkv` в квант-формате против того же ядра в BF16 — ошибка квантования
//!    k и v (отн. к норме строки), масштаб без насыщения.
//! 2. Чтение: деквант на хосте (свой декодер) → BF16-кэш; `ink2_attn` и `ink2_attn_fa` на
//!    квант-кэше обязаны совпасть бит в бит с собой же на этом BF16-кэше (деквант точен в BF16).
//! 3. Ошибка внимания из-за квантования: квант-кэш против исходного BF16 (тот же `ink2_attn`).
//!
//! `cargo test --profile fast-release -p synaptix-llm-inkling2 --test kv_quant -- --nocapture`

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

fn dev<T: synaptix_core::dtype::SynaptixScalar>(v: Vec<T>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec::<_, T>(v, shape, Device::Cuda(0)).unwrap()
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn host_u8(t: &Tensor) -> Vec<u8> {
    t.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<u8>().unwrap()
}

fn launch(name: &str, a: &Ink2Launch, grid: (u32, u32, u32), block: u32) {
    let d = Device::Cuda(0);
    registry::backend_for(d).unwrap().inkling2_launch(d, name, a, grid, block).unwrap();
}

fn ad(t: &Tensor) -> u64 {
    t.device_address().unwrap()
}

fn e4m3(b: u8) -> f32 {
    let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f32);
    let v = if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if b & 0x80 != 0 { -v } else { v }
}

/// Деквант на хосте: байты `[nkv, cap, D]` и масштабы `[nkv, cap, D/32]` (одна половина a11).
fn dequant(bytes: &[u8], scales: &[u8], fmt: u32) -> Vec<f32> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let sc = 2f32.powi(scales[i / 32] as i32 - 127);
            if fmt == 1 { e4m3(b) * sc } else { (b as i8) as f32 * sc }
        })
        .collect()
}

struct Kv {
    /// Квант-кэш (байты k, v; масштабы) и BF16-кэш того же ядра без квантования.
    k8: Tensor,
    v8: Tensor,
    sc: Tensor,
    kb: Tensor,
    vb: Tensor,
}

/// Записать `n` позиций ядром `ink2_qkv` в оба кэша.
fn write_kv(rng: &mut Rng, nkv: usize, cap: usize, n: usize, fmt: u32) -> Kv {
    let nh = nkv; // q не важны
    let q = dev(rng.vec(n * nh * D, 1.0), vec![n, nh * D]);
    // k — как у модели после L2-клипа: норма строки ~ 10, отдельные выбросы по каналам.
    let mut kh = rng.vec(n * nkv * D, 1.0);
    for (i, x) in kh.iter_mut().enumerate() {
        if i % 37 == 5 {
            *x *= 6.0;
        }
    }
    let k = dev(kh, vec![n, nkv * D]);
    let v = dev(rng.vec(n * nkv * D, 1.0), vec![n, nkv * D]);
    let r = dev(rng.vec(n * nh * DR, 1.0), vec![n, nh * DR]);
    let qs = dev(vec![1f32; nh * D], vec![nh * D]);
    let rs = dev(vec![1f32; nh * DR], vec![nh * DR]);
    let qo = Tensor::zeros(vec![n, nh * D], DType::F32, Device::Cuda(0)).unwrap();
    let ro = Tensor::zeros(vec![n, nh * DR], DType::F32, Device::Cuda(0)).unwrap();
    let k8 = Tensor::zeros(vec![nkv, cap, D], DType::U8, Device::Cuda(0)).unwrap();
    let v8 = Tensor::zeros(vec![nkv, cap, D], DType::U8, Device::Cuda(0)).unwrap();
    let sc = Tensor::zeros(vec![2, nkv, cap, D / 32], DType::U8, Device::Cuda(0)).unwrap();
    let kb = Tensor::zeros(vec![nkv, cap, D], DType::BF16, Device::Cuda(0)).unwrap();
    let vb = Tensor::zeros(vec![nkv, cap, D], DType::BF16, Device::Cuda(0)).unwrap();
    let mut a = Ink2Launch::default();
    for (i, t) in [&q, &k, &v, &r, &qs, &rs, &qo, &ro].iter().enumerate() {
        a.a[i] = ad(t);
    }
    a.u = [n as u32, nh as u32, nkv as u32, D as u32, DR as u32, cap as u32, 0, 1, 8192, 0, 0, 0];
    a.f = [1e30, 1e30, 1e-6, 0.1];
    let mut ab = a;
    ab.a[8] = ad(&kb);
    ab.a[9] = ad(&vb);
    launch("ink2_qkv", &ab, (n as u32, (nh + 2 * nkv) as u32, 1), D as u32);
    let mut aq = a;
    aq.a[8] = ad(&k8);
    aq.a[9] = ad(&v8);
    aq.a[11] = ad(&sc);
    aq.u[11] = fmt;
    launch("ink2_qkv", &aq, (n as u32, (nh + 2 * nkv) as u32, 1), D as u32);
    Kv { k8, v8, sc, kb, vb }
}

/// Внимание одним ядром на кэше: (k, v, масштабы|нет, формат) → выход `[T, nh·D]`.
#[allow(clippy::too_many_arguments)]
fn attn(kernel: &str, rng_seed: u64, nh: usize, nkv: usize, cap: usize, past: usize, t: usize, k: &Tensor, v: &Tensor, sc: Option<&Tensor>, fmt: u32, splits: usize) -> Vec<f32> {
    let mut rng = Rng(rng_seed);
    let q = dev(rng.vec(t * nh * D, 3.0), vec![t, nh * D]);
    let r = dev(rng.vec(t * nh * DR, 1.0), vec![t, nh * DR]);
    let proj = dev(rng.vec(DR * 1024, 0.4), vec![DR, 1024]);
    let sink = dev(rng.vec(t * nh, 2.0), vec![t, nh]);
    let gate = dev(rng.vec(t * nh * D, 1.0), vec![t, nh * D]);
    let pos = dev(vec![past as u32], vec![1]);
    let out = Tensor::zeros(vec![t, nh * D], DType::F32, Device::Cuda(0)).unwrap();
    let parts = Tensor::zeros(vec![t * nh * splits * 130], DType::F32, Device::Cuda(0)).unwrap();
    let mut a = Ink2Launch::default();
    for (i, x) in [&q, &r, &proj, &sink, k, v, &gate, &out, &parts].iter().enumerate() {
        a.a[i] = ad(x);
    }
    a.a[10] = ad(&pos);
    if let Some(s) = sc {
        a.a[11] = ad(s);
    }
    a.u = [t as u32, nh as u32, nkv as u32, D as u32, DR as u32, cap as u32, 0, 0, 1024, splits as u32, 0, fmt];
    a.f[0] = 1.0 / (D as f32).sqrt();
    if kernel == "ink2_attn_fa" {
        launch(kernel, &a, (t.div_ceil(64) as u32, nh as u32, splits as u32), 128);
    } else {
        launch(kernel, &a, (t as u32, nh as u32, splits as u32), 128);
    }
    if splits > 1 {
        launch("ink2_attn_merge", &a, (t as u32, nh as u32, 1), 128);
    }
    host_f32(&out)
}

/// Макс. отн. ошибка строки (к норме строки эталона) и мин. косинус.
fn cmp(x: &[f32], y: &[f32], row: usize) -> (f64, f64) {
    let (mut worst, mut cos) = (0f64, 1f64);
    for (xs, ys) in x.chunks(row).zip(y.chunks(row)) {
        let (mut e, mut n, mut xy, mut xx) = (0f64, 0f64, 0f64, 0f64);
        for (&a, &b) in xs.iter().zip(ys) {
            e += (a as f64 - b as f64).powi(2);
            n += (b as f64).powi(2);
            xy += a as f64 * b as f64;
            xx += (a as f64).powi(2);
        }
        worst = worst.max((e / n.max(1e-30)).sqrt());
        cos = cos.min(xy / (xx.sqrt() * n.sqrt()).max(1e-30));
    }
    (worst, cos)
}

#[test]
fn kv_quant_write_and_read() {
    if !setup() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    let (nh, nkv) = (24usize, 8usize);
    for fmt in [1u32, 2] {
        let name = if fmt == 1 { "MXFP8" } else { "INT8 " };
        for &(past, t, splits) in &[(0usize, 512usize, 1usize), (3000, 333, 4), (20000, 1, 32), (20000, 512, 8)] {
            let cap = past + t + 8;
            let mut rng = Rng(77 + past as u64);
            let kv = write_kv(&mut rng, nkv, cap, past + t, fmt);
            // 1. Запись.
            let (k8, v8, sc) = (host_u8(&kv.k8), host_u8(&kv.v8), host_u8(&kv.sc));
            let half = nkv * cap * D / 32;
            let (kd, vd) = (dequant(&k8, &sc[..half], fmt), dequant(&v8, &sc[half..], fmt));
            let (kb, vb) = (host_f32(&kv.kb), host_f32(&kv.vb));
            let (ek, _) = cmp(&kd, &kb, D);
            let (ev, _) = cmp(&vd, &vb, D);
            // Масштаб без насыщения: |x| ≤ sc·top у каждого блока, и sc минимален (amax > sc·top/2).
            let top = if fmt == 1 { 448f32 } else { 127.0 };
            for (blk, chunk) in kb.chunks(32).enumerate().take(nkv * (past + t) * D / 32) {
                let amax = chunk.iter().fold(0f32, |m, x| m.max(x.abs()));
                if amax == 0.0 {
                    continue;
                }
                let s = 2f32.powi(sc[blk] as i32 - 127);
                // amax здесь — по BF16-копии (округление 2⁻⁹), ядро видит F32 до округления.
                assert!(amax <= s * top * 1.004 && amax > s * top / 2.0 * 0.996, "{name}: масштаб блока {blk}: amax {amax}, sc {s}");
            }
            // 2. Чтение: квант-кэш == BF16-кэш из декванта, бит в бит.
            let kq = dev(kd.clone(), vec![nkv, cap, D]).to_dtype(DType::BF16).unwrap();
            let vq = dev(vd.clone(), vec![nkv, cap, D]).to_dtype(DType::BF16).unwrap();
            let s_attn = if t == 1 { splits } else { 1 };
            let a_q = attn("ink2_attn", 5, nh, nkv, cap, past, t, &kv.k8, &kv.v8, Some(&kv.sc), fmt, s_attn);
            let a_d = attn("ink2_attn", 5, nh, nkv, cap, past, t, &kq, &vq, None, 0, s_attn);
            assert!(a_q == a_d, "{name}: ink2_attn на квант-кэше ≠ на его декванте");
            if t > 1 {
                let f_q = attn("ink2_attn_fa", 5, nh, nkv, cap, past, t, &kv.k8, &kv.v8, Some(&kv.sc), fmt, splits);
                let f_d = attn("ink2_attn_fa", 5, nh, nkv, cap, past, t, &kq, &vq, None, 0, splits);
                assert!(f_q == f_d, "{name}: ink2_attn_fa на квант-кэше ≠ на его декванте");
            }
            // 3. Ошибка внимания от квантования (против исходного BF16-кэша).
            let a_b = attn("ink2_attn", 5, nh, nkv, cap, past, t, &kv.kb, &kv.vb, None, 0, s_attn);
            let (ea, ca) = cmp(&a_q, &a_b, D);
            println!(
                "{name} past {past:>5} T {t:>3}: k {ek:.2e}, v {ev:.2e} отн. (макс. по строкам); выход внимания {ea:.2e}, cos_min {ca:.6}"
            );
            assert!(ek < if fmt == 1 { 0.06 } else { 0.03 } && ev < if fmt == 1 { 0.06 } else { 0.03 });
        }
    }
}
