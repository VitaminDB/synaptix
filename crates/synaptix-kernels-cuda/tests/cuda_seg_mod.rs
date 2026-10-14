//! Гейт слитых ядер сегментной модуляции и qk-нормы (MiniMax-H3): побайтовое
//! совпадение с прежними цепочками.
//! - rms_segmod_quant (NVFP4/MXFP8) ≡ rms_mod_quant_* от развёрнутых по-токенных
//!   scale/shift (строка таблицы по ridx);
//! - gate_residual_seg ≡ fused_gate_residual по строкам;
//! - qk_norm_rope ≡ to(F32) → rms_norm_fused → rope_split_partial_fused → to(T).

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

/// Сегменты: токены группами по строкам таблицы (как text/video/audio у H3).
fn ridx(m: usize, rows: usize) -> Vec<i32> {
    let bounds = [0, m / 7, m / 3, m / 2, m - m / 5, m];
    let mut r = vec![0i32; m];
    for (k, w) in bounds.windows(2).enumerate() {
        for v in &mut r[w[0]..w[1]] {
            *v = ((k * 5 + 1) % rows) as i32;
        }
    }
    r
}

fn bytes(t: &Tensor) -> Vec<u8> {
    t.to_device(Device::Cpu).unwrap().contiguous().unwrap().to_bytes().unwrap()
}

/// Развернуть таблицу [R, k] по ridx → [m, k].
fn expand(table: &Tensor, idx: &[i32]) -> Tensor {
    let parts: Vec<Tensor> = idx.iter().map(|&r| table.narrow(0, r as usize, 1).unwrap().contiguous().unwrap()).collect();
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0).unwrap().contiguous().unwrap()
}

fn segmod_case(m: usize, k: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let rows = 6;
    let x = Tensor::from_vec(det(1, m * k, 2.0), vec![m, k], dev).unwrap().to_dtype(dt).unwrap();
    let sc = Tensor::from_vec(det(2, rows * k, 0.5), vec![rows, k], dev).unwrap().to_dtype(dt).unwrap();
    let sh = Tensor::from_vec(det(3, rows * k, 0.3), vec![rows, k], dev).unwrap().to_dtype(dt).unwrap();
    let idx = ridx(m, rows);
    let ri = Tensor::from_vec(idx.clone(), vec![m], dev).unwrap();
    let (sc_e, sh_e) = (expand(&sc, &idx), expand(&sh, &idx));

    let (_, pk_ref, s_ref) = x.rms_mod_quant_nvfp4(&sc_e, &sh_e, 1e-6).unwrap();
    let (pk, s) = x.rms_segmod_quant(&sc, &sh, &ri, 1e-6, DType::NVFP4).unwrap();
    assert_eq!(bytes(&pk), bytes(&pk_ref), "nvfp4 packed (m={m} k={k} {dt:?})");
    assert_eq!(bytes(&s), bytes(&s_ref), "nvfp4 scales (m={m} k={k} {dt:?})");

    let (_, pk_ref, s_ref) = x.rms_mod_quant_mxfp8(&sc_e, &sh_e, 1e-6).unwrap();
    let (pk, s) = x.rms_segmod_quant(&sc, &sh, &ri, 1e-6, DType::MXFP8).unwrap();
    assert_eq!(bytes(&pk), bytes(&pk_ref), "mxfp8 packed (m={m} k={k} {dt:?})");
    assert_eq!(bytes(&s), bytes(&s_ref), "mxfp8 scales (m={m} k={k} {dt:?})");
}

fn gate_case(m: usize, d: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let rows = 6;
    let x = Tensor::from_vec(det(4, m * d, 3.0), vec![m, d], dev).unwrap().to_dtype(dt).unwrap();
    let y = Tensor::from_vec(det(5, m * d, 2.0), vec![m, d], dev).unwrap().to_dtype(dt).unwrap();
    let g = Tensor::from_vec(det(6, rows * d, 1.5), vec![rows, d], dev).unwrap().to_dtype(dt).unwrap();
    let idx = ridx(m, rows);
    let ri = Tensor::from_vec(idx.clone(), vec![m], dev).unwrap();
    let out = x.gate_residual_seg(&y, &g, &ri).unwrap();
    let reference = x.fused_gate_residual(&y, &expand(&g, &idx)).unwrap();
    assert_eq!(bytes(&out), bytes(&reference), "gate_residual_seg (m={m} d={d} {dt:?})");
}

fn qk_case(n: usize, heads: usize, d: usize, rot: usize, off: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let s_total = off + n + 3;
    let half = rot / 2;
    let x = Tensor::from_vec(det(7, n * heads * d, 4.0), vec![1, n, heads, d], dev).unwrap().to_dtype(dt).unwrap();
    let w = Tensor::from_vec(det(8, d, 1.0).iter().map(|v| 1.0 + v * 0.3).collect(), vec![d], dev).unwrap();
    let ang: Vec<f32> = (0..s_total * half).map(|i| (i as f32) * 0.013).collect();
    let cos = Tensor::from_vec(ang.iter().map(|a| a.cos()).collect(), vec![s_total, half], dev).unwrap();
    let sin = Tensor::from_vec(ang.iter().map(|a| a.sin()).collect(), vec![s_total, half], dev).unwrap();

    let out = x.qk_norm_rope(&w, &cos, &sin, off, rot, 1e-6).unwrap();
    let reference = x
        .to_dtype(DType::F32).unwrap()
        .rms_norm_fused(&w, 1e-6, false).unwrap()
        .rope_split_partial_fused(&cos.narrow(0, off, n).unwrap(), &sin.narrow(0, off, n).unwrap(), rot).unwrap()
        .to_dtype(dt).unwrap();
    assert_eq!(bytes(&out), bytes(&reference), "qk_norm_rope (n={n} h={heads} d={d} rot={rot} off={off} {dt:?})");
}

#[test]
fn segmod_quant_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    for dt in [DType::BF16, DType::F16] {
        segmod_case(300, 5376, dt);
        segmod_case(17, 256, dt);
    }
}

#[test]
fn gate_residual_seg_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    for dt in [DType::BF16, DType::F16] {
        gate_case(300, 5376, dt);
        gate_case(9, 64, dt);
    }
}

#[test]
fn qk_norm_rope_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    for dt in [DType::BF16, DType::F16] {
        qk_case(37, 42, 128, 128, 0, dt);
        qk_case(37, 42, 128, 96, 100, dt);
        qk_case(5, 3, 64, 32, 7, dt);
    }
}

#[test]
fn silu_mul_quant_mxfp8_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for dt in [DType::BF16, DType::F16] {
        for (m, f) in [(300usize, 1024usize), (7, 64)] {
            let h = Tensor::from_vec(det(9, m * 2 * f, 3.0), vec![m, 2 * f], dev).unwrap().to_dtype(dt).unwrap();
            let (pk, sc) = h.silu_mul_quant_mxfp8().unwrap();
            let gate = h.narrow(1, 0, f).unwrap().contiguous().unwrap();
            let up = h.narrow(1, f, f).unwrap().contiguous().unwrap();
            let act = gate.silu_and_mul(&up).unwrap();
            let (pk_ref, sc_ref) = act.mxfp8_quantize_act().unwrap();
            assert_eq!(bytes(&pk), bytes(&pk_ref), "packed (m={m} f={f} {dt:?})");
            assert_eq!(bytes(&sc), bytes(&sc_ref), "scales (m={m} f={f} {dt:?})");
        }
    }
}

#[test]
fn layer_norm_plain_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    for dt in [DType::BF16, DType::F16] {
        for (m, n) in [(300usize, 3072usize), (4106, 4096), (5, 512), (3, 9000)] {
            // смещение и выбросы — как остаточный поток Qwen-Image
            let v: Vec<f32> = det(31, m * n, 3.0).iter().enumerate().map(|(i, x)| if i % 101 == 0 { x * 900.0 + 50.0 } else { x + 50.0 }).collect();
            let x = Tensor::from_vec(v, vec![1, m, n], dev).unwrap().to_dtype(dt).unwrap();
            let y = x.layer_norm_plain(1e-6).unwrap();
            let xf = x.to_dtype(DType::F32).unwrap();
            let mean = xf.mean_keepdim(2).unwrap();
            let c = xf.broadcast_sub(&mean).unwrap();
            let var = c.sqr().unwrap().mean_keepdim(2).unwrap();
            let inv = var.add_scalar(1e-6).unwrap().sqrt().unwrap().recip().unwrap();
            let r = c.broadcast_mul(&inv).unwrap().to_dtype(dt).unwrap();
            assert_eq!(bytes(&y), bytes(&r), "layer_norm_plain (m={m} n={n} {dt:?})");
        }
    }
}

/// Режим FLUX: RoPE парами, таблицы полной ширины `[S, D]`, позиция со сдвигом.
fn qk_il_case(n: usize, heads: usize, off: usize, dt: DType) {
    let dev = Device::Cuda(0);
    let d = 128;
    let s_total = off + n + 3;
    let x = Tensor::from_vec(det(9, n * heads * d, 4.0), vec![1, n, heads, d], dev).unwrap().to_dtype(dt).unwrap();
    let w = Tensor::from_vec(det(10, d, 1.0).iter().map(|v| 1.0 + v * 0.3).collect(), vec![d], dev).unwrap();
    // Таблица как у FLUX: угол пары повторён в обоих элементах.
    let ang: Vec<f32> = (0..s_total * d).map(|i| ((i / 2) as f32) * 0.011).collect();
    let cos = Tensor::from_vec(ang.iter().map(|a| a.cos()).collect(), vec![1, s_total, 1, d], dev).unwrap();
    let sin = Tensor::from_vec(ang.iter().map(|a| a.sin()).collect(), vec![1, s_total, 1, d], dev).unwrap();

    let out = x.qk_norm_rope_interleaved(&w, &cos, &sin, off, d, 1e-6).unwrap();
    let reference = x
        .to_dtype(DType::F32).unwrap()
        .rms_norm_fused(&w, 1e-6, false).unwrap()
        .rope_interleaved_fused(&cos.narrow(1, off, n).unwrap().contiguous().unwrap(), &sin.narrow(1, off, n).unwrap().contiguous().unwrap()).unwrap()
        .to_dtype(dt).unwrap();
    assert_eq!(bytes(&out), bytes(&reference), "qk_norm_rope_interleaved (n={n} h={heads} off={off} {dt:?})");
}

#[test]
fn qk_norm_rope_interleaved_bitexact() {
    synaptix_kernels_cuda::ensure_registered();
    for dt in [DType::BF16, DType::F16] {
        qk_il_case(37, 24, 0, dt);
        qk_il_case(300, 24, 512, dt);
    }
}
