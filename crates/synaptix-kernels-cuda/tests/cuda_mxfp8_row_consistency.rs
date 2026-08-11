
//! СТРОГИЙ per-row аудит MXFP8 GEMM (gn_mxfp8_*): проверяет КАЖДУЮ строку (не 8),
//! и M-независимость (та же строка при разных M → идентична). Опровергает/подтверждает
//! аудит-гипотезу «b_data n-map ≠ b_scale n-map».

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_mxfp8::gemm_mxfp8_linear;
use synaptix_kernels_cuda::elementwise::quant::Mxfp8QuantKernels;

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32((((x >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

// ВСЕ строки vs f32-dense эталон. Ловит per-row scale-index mismatch (если есть,
// ровно одна group-of-rows будет иметь низкий cos, а не все).
#[test]
fn mxfp8_all_rows_vs_dense() {
    synaptix_kernels_cuda::ensure_registered();
    let Some(ctx) = synaptix_core::device::cuda::get(0).ok() else {
        return;
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let qk = Mxfp8QuantKernels::for_context(&ctx).expect("mxfp8 quant compile");

    // v1 = 128×128×128-кратные (M<128 паддится в dispatch, не здесь).
    for &(m, n, k) in &[(128u32, 256u32, 256u32), (256, 256, 512), (384, 256, 256), (512, 256, 256)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let x = det_f16(0xC0DE_BA5E, mu * ku, 0.5);
        let w = det_f16(0xA110_C8E1, nu * ku, 0.5);
        let xd: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
        let wd: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
        let mut y: CudaSlice<f16> = stream.alloc_zeros(mu * nu).unwrap();
        gemm_mxfp8_linear(&qk, &stream, &xd, &wd, &mut y.slice_mut(0..), m, n, k).unwrap();
        stream.synchronize().unwrap();
        let y_h: Vec<f16> = stream.clone_dtoh(&y).unwrap();

        // f32-dense эталон ВСЕХ строк
        let mut worst_row = (0usize, 1.0f32);
        for i in 0..mu {
            let mut yref = vec![0.0f32; nu];
            for c in 0..nu {
                let mut acc = 0.0f64;
                for kk in 0..ku {
                    acc += x[i * ku + kk].to_f32() as f64 * w[c * ku + kk].to_f32() as f64;
                }
                yref[c] = acc as f32;
            }
            let got: Vec<f32> = y_h[i * nu..(i + 1) * nu].iter().map(|v| v.to_f32()).collect();
            let rc = cos(&got, &yref);
            if rc < worst_row.1 {
                worst_row = (i, rc);
            }
        }
        eprintln!(
            "[mxfp8 ALL-rows {m}x{k}x{n}] worst_row={} cos={:.5}",
            worst_row.0, worst_row.1
        );
        assert!(
            worst_row.1 >= 0.97,
            "mxfp8 {m}x{k}x{n} row {} cos={} < 0.97 (per-row scale defect!)",
            worst_row.0,
            worst_row.1
        );
    }
}

// M-независимость: одна и та же строка row=10 при M={64,128,256,512} обязана совпасть
// (с точностью block-квантования X, которое для фикс-строки одинаково при любом M, т.к.
// квант поблочный по K внутри строки). Расхождение → M-зависимый scale-индекс.
#[test]
fn mxfp8_row_independent_of_m() {
    synaptix_kernels_cuda::ensure_registered();
    let Some(ctx) = synaptix_core::device::cuda::get(0).ok() else {
        return;
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let qk = Mxfp8QuantKernels::for_context(&ctx).expect("mxfp8 quant compile");

    let (n, k) = (256usize, 256usize);
    let row = 10usize;
    let ms = [128u32, 256, 384, 512];
    let maxm = *ms.iter().max().unwrap() as usize;
    let w = det_f16(0xA110_C8E1, n * k, 0.5);
    let wd: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let x_all = det_f16(0x5151_5151, maxm * k, 0.4);

    let mut refrow: Option<Vec<f32>> = None;
    let mut maxd = 0.0f32;
    for &m in &ms {
        let mu = m as usize;
        let x = x_all[..mu * k].to_vec();
        let xd: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
        let mut y: CudaSlice<f16> = stream.alloc_zeros(mu * n).unwrap();
        gemm_mxfp8_linear(&qk, &stream, &xd, &wd, &mut y.slice_mut(0..), m, n as u32, k as u32).unwrap();
        stream.synchronize().unwrap();
        let y_h: Vec<f16> = stream.clone_dtoh(&y).unwrap();
        let r: Vec<f32> = y_h[row * n..(row + 1) * n].iter().map(|v| v.to_f32()).collect();
        match &refrow {
            None => refrow = Some(r),
            Some(rr) => {
                let d = rr.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                maxd = maxd.max(d);
                eprintln!("[mxfp8 row-indep] M={m} row={row} max_abs_vs_ref={d:.4}");
            }
        }
    }
    eprintln!("[mxfp8 row-indep] row={row} max_abs across M={ms:?} = {maxd:.4}");
    assert!(maxd < 0.05, "mxfp8 row {row} M-dependent: max_abs={maxd} (defect!)");
}
