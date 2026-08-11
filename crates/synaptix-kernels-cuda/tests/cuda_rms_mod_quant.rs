//! Гейт fused «adaLN-модуляция + NVFP4-квант»: БИТ-эквивалентность decomposed-
//! цепочке rms_norm_fused(ones) → add_scalar(1) → broadcast_mul → broadcast_add
//! → to(F16) → nvfp4_quantize_act. Полное побайтовое сравнение y/packed/scales.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn case(m: usize, k: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let mk = m * k;
    let x = Tensor::from_vec(det(1, mk, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(2, mk, 0.5), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(3, mk, 0.3), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let ones = Tensor::ones(vec![k], dt, dev).unwrap();

    let y_ref = x
        .rms_norm_fused(&ones, 1e-6, false).unwrap()
        .broadcast_mul(&sc.add_scalar(1.0).unwrap()).unwrap()
        .broadcast_add(&sh).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().nvfp4_quantize_act().unwrap();

    let (y, pk, scl) = x.rms_mod_quant_nvfp4(&sc, &sh, 1e-6).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, b) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert_eq!(worst, 0.0, "y не бит-в-бит (m={m} k={k} {dt:?}): max|Δ|={worst}");

    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "packed расходится (m={m} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "scales расходятся (m={m} k={k} {dt:?})");
}

fn case_ln(b: usize, t: usize, k: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let m = b * t;
    let mk = m * k;
    let x = Tensor::from_vec(det(11, mk, 2.0), vec![b, t, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(12, b * k, 0.5), vec![b, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(13, b * k, 0.3), vec![b, k], dev).unwrap().to_dtype(dt).unwrap();
    let ones = Tensor::ones(vec![k], dt, dev).unwrap();

    // decomposed: FLUX ada_ln = LN(gamma=1, round→dtype) → modulate (bcast [B,1,K])
    let n = x.layer_norm_fused(&ones, None, 1e-6).unwrap();
    let scp = sc.add_scalar(1.0).unwrap().reshape(vec![b, 1, k]).unwrap();
    let shp = sh.reshape(vec![b, 1, k]).unwrap();
    let y_ref = n.broadcast_mul(&scp).unwrap().broadcast_add(&shp).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().nvfp4_quantize_act().unwrap();

    let (y, pk, scl) = x.ln_mod_quant_nvfp4(&sc, &sh, 1e-6).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, bb) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - bb).abs());
    }
    assert_eq!(worst, 0.0, "ln y не бит-в-бит (b={b} t={t} k={k} {dt:?})");
    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "ln packed расходится (b={b} t={t} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "ln scales расходятся (b={b} t={t} k={k} {dt:?})");
}

#[test]
fn ln_mod_quant_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(b, t, k) in &[(1usize, 128usize, 3072usize), (2, 99, 3072), (1, 512, 2048)] {
        case_ln(b, t, k, DType::BF16);
        case_ln(b, t, k, DType::F16);
    }
}

fn case_w(m: usize, k: usize, dt: DType, qwen: bool) {
    let dev = Device::Cuda(0);
    let mk = m * k;
    let x = Tensor::from_vec(det(21, mk, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let w = Tensor::from_vec(det(22, k, 1.0), vec![k], dev).unwrap().to_dtype(dt).unwrap();

    let y_ref = x.rms_norm_fused(&w, 1e-6, qwen).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().nvfp4_quantize_act().unwrap();
    let (y, pk, scl) = x.rms_quant_nvfp4(&w, 1e-6, qwen).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, b) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert_eq!(worst, 0.0, "rms_w y не бит-в-бит (m={m} k={k} {dt:?} qwen={qwen})");
    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "rms_w packed расходится (m={m} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "rms_w scales расходятся (m={m} k={k} {dt:?})");
}

#[test]
fn rms_w_quant_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(m, k) in &[(16usize, 5120usize), (350, 5120), (1, 4096)] {
        case_w(m, k, DType::F16, false);
        case_w(m, k, DType::BF16, false);
        case_w(m, k, DType::F16, true);
    }
}

#[test]
fn rms_mod_quant_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(m, k) in &[(16usize, 4096usize), (128, 4096), (197, 4096), (512, 2048)] {
        case(m, k, DType::BF16);
        case(m, k, DType::F16);
    }
}

// ── MXFP8-варианты: тот же контракт, эпилог = mxfp8_quantize_act бит-в-бит
// (packed [m·k] e4m3, scales natural [m·k/32] E8M0). ──
fn case_mx(m: usize, k: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let mk = m * k;
    let x = Tensor::from_vec(det(1, mk, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(2, mk, 0.5), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(3, mk, 0.3), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let ones = Tensor::ones(vec![k], dt, dev).unwrap();

    let y_ref = x
        .rms_norm_fused(&ones, 1e-6, false).unwrap()
        .broadcast_mul(&sc.add_scalar(1.0).unwrap()).unwrap()
        .broadcast_add(&sh).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().mxfp8_quantize_act().unwrap();

    let (y, pk, scl) = x.rms_mod_quant_mxfp8(&sc, &sh, 1e-6).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, b) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert_eq!(worst, 0.0, "mx y не бит-в-бит (m={m} k={k} {dt:?}): max|Δ|={worst}");

    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "mx packed расходится (m={m} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "mx scales расходятся (m={m} k={k} {dt:?})");
}

fn case_ln_mx(b: usize, t: usize, k: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let m = b * t;
    let mk = m * k;
    let x = Tensor::from_vec(det(11, mk, 2.0), vec![b, t, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(12, b * k, 0.5), vec![b, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(13, b * k, 0.3), vec![b, k], dev).unwrap().to_dtype(dt).unwrap();
    let ones = Tensor::ones(vec![k], dt, dev).unwrap();

    let n = x.layer_norm_fused(&ones, None, 1e-6).unwrap();
    let scp = sc.add_scalar(1.0).unwrap().reshape(vec![b, 1, k]).unwrap();
    let shp = sh.reshape(vec![b, 1, k]).unwrap();
    let y_ref = n.broadcast_mul(&scp).unwrap().broadcast_add(&shp).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().mxfp8_quantize_act().unwrap();

    let (y, pk, scl) = x.ln_mod_quant_mxfp8(&sc, &sh, 1e-6).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, bb) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - bb).abs());
    }
    assert_eq!(worst, 0.0, "ln_mx y не бит-в-бит (b={b} t={t} k={k} {dt:?})");
    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "ln_mx packed расходится (b={b} t={t} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "ln_mx scales расходятся (b={b} t={t} k={k} {dt:?})");
}

fn case_w_mx(m: usize, k: usize, dt: DType, qwen: bool) {
    let dev = Device::Cuda(0);
    let mk = m * k;
    let x = Tensor::from_vec(det(21, mk, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let w = Tensor::from_vec(det(22, k, 1.0), vec![k], dev).unwrap().to_dtype(dt).unwrap();

    let y_ref = x.rms_norm_fused(&w, 1e-6, qwen).unwrap();
    let (pk_ref, sc_ref) = y_ref.to_dtype(DType::F16).unwrap().mxfp8_quantize_act().unwrap();
    let (y, pk, scl) = x.rms_quant_mxfp8(&w, 1e-6, qwen).unwrap();

    let yb: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let yrb: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![mk]).unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (a, b) in yb.iter().zip(yrb.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert_eq!(worst, 0.0, "rms_w_mx y не бит-в-бит (m={m} k={k} {dt:?} qwen={qwen})");
    let pkb: Vec<u8> = pk.to_vec1().unwrap();
    let pkr: Vec<u8> = pk_ref.to_vec1().unwrap();
    assert_eq!(pkb, pkr, "rms_w_mx packed расходится (m={m} k={k} {dt:?})");
    let scb: Vec<u8> = scl.to_vec1().unwrap();
    let scr: Vec<u8> = sc_ref.to_vec1().unwrap();
    assert_eq!(scb, scr, "rms_w_mx scales расходятся (m={m} k={k} {dt:?})");
}

#[test]
fn rms_mod_quant_mxfp8_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(m, k) in &[(16usize, 4096usize), (128, 4096), (197, 4096), (512, 2048)] {
        case_mx(m, k, DType::BF16);
        case_mx(m, k, DType::F16);
    }
}

#[test]
fn ln_mod_quant_mxfp8_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(b, t, k) in &[(1usize, 128usize, 3072usize), (2, 99, 3072), (1, 512, 2048)] {
        case_ln_mx(b, t, k, DType::BF16);
        case_ln_mx(b, t, k, DType::F16);
    }
}

#[test]
fn rms_w_quant_mxfp8_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    for &(m, k) in &[(16usize, 5120usize), (350, 5120), (1, 4096)] {
        case_w_mx(m, k, DType::F16, false);
        case_w_mx(m, k, DType::BF16, false);
        case_w_mx(m, k, DType::F16, true);
    }
}

// prequant-путь: linear_quant(x f16) == linear_quant_prequant(mxfp8_quantize_act(x))
// бит-в-бит (тот же квант + тот же rot-GEMM).
#[test]
fn linear_quant_prequant_mxfp8_bitexact() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for &(m, n, k) in &[(128usize, 512usize, 2048usize), (197, 512, 2048), (16, 256, 1024)] {
        let x = Tensor::from_vec(det(31, m * k, 1.0), vec![m, k], dev).unwrap().to_dtype(DType::F16).unwrap();
        let w = Tensor::from_vec(det(32, n * k, 0.5), vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
        let qw = w.quantize_to_mxfp8().unwrap();
        let y_ref = x.linear_quant(&qw).unwrap();
        let (p, s) = x.mxfp8_quantize_act().unwrap();
        let y = p.linear_quant_prequant(&s, &qw, m).unwrap();
        let a: Vec<f32> = y_ref.to_dtype(DType::F32).unwrap().reshape(vec![m * n]).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = y.to_dtype(DType::F32).unwrap().reshape(vec![m * n]).unwrap().to_vec1().unwrap();
        let mut worst = 0f32;
        for (u, v) in a.iter().zip(b.iter()) {
            worst = worst.max((u - v).abs());
        }
        assert_eq!(worst, 0.0, "prequant mxfp8 расходится (m={m} n={n} k={k}): max|Δ|={worst}");
    }
}

#[test]
fn rms_mod_quant_bench() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (m, k) = (4992usize, 4096usize);
    let dt = DType::BF16;
    let mk = m * k;
    let x = Tensor::from_vec(det(1, mk, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(2, mk, 0.5), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(3, mk, 0.3), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let ones = Tensor::ones(vec![k], dt, dev).unwrap();
    let sync = || synaptix_core::device::cuda::synchronize(0).unwrap();

    let old_path = || {
        let y = x
            .rms_norm_fused(&ones, 1e-6, false).unwrap()
            .broadcast_mul(&sc.add_scalar(1.0).unwrap()).unwrap()
            .broadcast_add(&sh).unwrap();
        let (p, s) = y.to_dtype(DType::F16).unwrap().nvfp4_quantize_act().unwrap();
        std::hint::black_box((y, p, s));
    };
    let new_path = || {
        let r = x.rms_mod_quant_nvfp4(&sc, &sh, 1e-6).unwrap();
        std::hint::black_box(r);
    };
    for _ in 0..20 { old_path(); new_path(); }
    sync();
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n { old_path(); }
    sync();
    let t_old = t0.elapsed().as_secs_f64() / n as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..n { new_path(); }
    sync();
    let t_new = t1.elapsed().as_secs_f64() / n as f64;
    println!(
        "rms_mod_quant m={m} k={k}: decomposed {:.1}µs → fused {:.1}µs (×{:.2})",
        t_old * 1e6, t_new * 1e6, t_old / t_new
    );
}

/// Доля нибблов, отличающихся между пороговым e2m1 (GPU, при env=1) и
/// div-эталоном (хост, IEEE div = div.rn GPU). Информационная метрика
/// (запускается только с SYN_NVFP4_E2M1_THRESHOLD=1).
#[test]
fn e2m1_threshold_divergence() {
    if synaptix_core::device::cuda::get(0).is_err()
        || std::env::var("SYN_NVFP4_E2M1_THRESHOLD").as_deref() != Ok("1")
    {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (m, k) = (2048usize, 4096usize);
    let host = det(99, m * k, 2.0);
    let x = Tensor::from_vec(host.clone(), vec![m, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let (pk, _sc) = x.nvfp4_quantize_act().unwrap();
    let gpu: Vec<u8> = pk.to_vec1().unwrap();

    // хост-эталон div-версии (та же арифметика, f32)
    let xf: Vec<f32> = x.to_dtype(DType::F32).unwrap().reshape(vec![m * k]).unwrap().to_vec1().unwrap();
    let enc_e4m3 = |v: f32| -> u8 {
        let v = v.clamp(-448.0, 448.0);
        let a = v.abs();
        if a == 0.0 { return 0; }
        let e = a.log2().floor() as i32;
        let eb = e + 7;
        if eb < 1 {
            let mm = (a * 512.0).round_ties_even() as i32;
            return mm.clamp(0, 7) as u8;
        }
        if eb > 15 { return 0x7E; }
        let p2 = (e as f32).exp2();
        let mut mm = ((a / p2 - 1.0) * 8.0).round_ties_even() as i32;
        let mut eb = eb;
        if mm == 8 { mm = 0; eb += 1; if eb > 15 { return 0x7E; } }
        if eb == 15 && mm == 7 { mm = 6; }
        ((eb as u8) << 3) | (mm as u8)
    };
    let dec_e4m3 = |b: u8| -> f32 {
        let eb = (b >> 3) & 0xF; let mm = (b & 7) as f32;
        if eb == 0 { mm * 0.001953125 } else { (1.0 + mm * 0.125) * ((eb as i32 - 7) as f32).exp2() }
    };
    let enc_e2m1 = |v: f32| -> u8 {
        let s = if v.is_sign_negative() { 8u8 } else { 0 };
        let a = v.abs();
        let idx = if a >= 5.0 { 7 } else if a >= 3.5 { 6 } else if a >= 2.5 { 5 }
            else if a >= 1.75 { 4 } else if a >= 1.25 { 3 } else if a >= 0.75 { 2 }
            else if a >= 0.25 { 1 } else { 0 };
        s | idx
    };
    let mut diff = 0usize;
    let total = m * k;
    for g in 0..(total / 16) {
        let grp = &xf[g * 16..g * 16 + 16];
        let amax = grp.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        let sraw = if amax > 0.0 { amax / 6.0 } else { 1e-9 };
        let sb = enc_e4m3(sraw);
        let mut sq = dec_e4m3(sb);
        if sq == 0.0 { sq = 1e-9; }
        for i in 0..8 {
            let lo = enc_e2m1(grp[2 * i] / sq);
            let hi = enc_e2m1(grp[2 * i + 1] / sq);
            let refb = (lo & 0x0F) | ((hi & 0x0F) << 4);
            let gb = gpu[g * 8 + i];
            if refb != gb {
                let dl = (refb & 0xF) != (gb & 0xF);
                let dh = (refb >> 4) != (gb >> 4);
                diff += dl as usize + dh as usize;
            }
        }
    }
    println!(
        "e2m1 threshold-vs-div: {} расхождений из {} нибблов ({:.2e})",
        diff, total, diff as f64 / total as f64
    );
}
