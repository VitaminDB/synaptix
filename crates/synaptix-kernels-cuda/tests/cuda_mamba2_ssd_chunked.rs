#![cfg(feature = "cuda")]

//! Bit-exact (с допуском, см. ниже) тесты Mamba2 chunked-SSD против
//! recurrent baseline ([`Mamba2SsdKernels::ssd`]) на тех же inputs.
//!
//! Замечание про precision:
//!  - Chunked форма всегда cast'ит промежуточные операнды (`C_QN`, `B_QN`,
//!    `dt_x`) в BF16 для bmm (bmm требует BF16 in / F32 acc — пока F32-path
//!    chunked нет).
//!  - Recurrent на F32 input делает всё в F32 — поэтому против F32-recurrent
//!    chunked даёт отклонение в пределах BF16 (порядка 1e-2 на K=128).
//!  - Recurrent на BF16 input — bf16-cast тоже на load_f, как у нас → отклонение
//!    только от порядка reduction (sequential vs bmm) и составляет ≲ 5e-3.

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::ssm::mamba2_ssd::Mamba2SsdKernels;
use synaptix_kernels_cuda::ssm::mamba2_ssd_chunked::Mamba2SsdChunkedKernels;

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale + offset
        })
        .collect()
}

fn upload<T: DeviceRepr + Copy>(stream: &Arc<CudaStream>, host: &[T]) -> CudaSlice<T> {
    let mut d = unsafe { stream.alloc::<T>(host.len()).expect("alloc") };
    stream.memcpy_htod(host, &mut d).expect("memcpy");
    d
}

fn alloc_zeros<T: DeviceRepr + Copy + Default>(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<T> {
    let h = vec![T::default(); n];
    let mut d = unsafe { stream.alloc::<T>(n).expect("alloc") };
    stream.memcpy_htod(&h, &mut d).expect("memcpy");
    d
}

fn download<T: DeviceRepr + Copy + Default>(stream: &Arc<CudaStream>, d: &CudaSlice<T>) -> Vec<T> {
    let mut h = vec![T::default(); d.len()];
    stream.memcpy_dtoh(d, &mut h).expect("memcpy dtoh");
    h
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}
fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}
fn bf16_to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}
fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: u32,
    h: u32,
    l: u32,
    p: u32,
    n: u32,
    q: u32,
}

fn run_f32(shape: Shape, has_d: bool, tol: f32, label: &str) {
    let Some((ctx, stream)) = setup() else { return };
    let recur = Mamba2SsdKernels::for_context(&ctx).expect("compile recurrent");
    let chunked = Mamba2SsdChunkedKernels::for_context(&ctx).expect("compile chunked");

    let Shape { b, h, l, p, n, q } = shape;
    let x_h = det_f32(0x100, (b * l * h * p) as usize, 0.5, 0.0);
    let dt_h = det_f32(0x101, (b * l * h) as usize, 0.2, 0.5); // > 0
    let a_h = det_f32(0x102, h as usize, 0.5, -1.5); // < 0
    let b_h_ = det_f32(0x103, (b * l * h * n) as usize, 0.5, 0.0);
    let c_h = det_f32(0x104, (b * l * h * n) as usize, 0.5, 0.0);
    let d_h = if has_d {
        det_f32(0x105, h as usize, 0.5, 0.0)
    } else {
        vec![]
    };

    let x = upload(&stream, &x_h);
    let dt = upload(&stream, &dt_h);
    let a = upload(&stream, &a_h);
    let bb = upload(&stream, &b_h_);
    let cc = upload(&stream, &c_h);
    let d_opt = if has_d {
        Some(upload(&stream, &d_h))
    } else {
        None
    };

    let mut y_recur = alloc_zeros::<f32>(&stream, (b * l * h * p) as usize);
    let mut y_chunk = alloc_zeros::<f32>(&stream, (b * l * h * p) as usize);

    recur
        .ssd_f32(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_recur,
            b,
            l,
            h,
            p,
            n,
        )
        .expect("recurrent");
    chunked
        .ssd(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_chunk,
            b,
            l,
            h,
            p,
            n,
            q,
            DType::F32,
        )
        .expect("chunked");
    stream.synchronize().expect("sync");

    let yr = download::<f32>(&stream, &y_recur);
    let yc = download::<f32>(&stream, &y_chunk);
    let err = max_abs(&yr, &yc);
    assert!(
        err <= tol,
        "[{label} F32 has_d={has_d}] max_abs={err:.3e} > tol={tol:.1e}"
    );
}

fn run_bf16(shape: Shape, has_d: bool, tol: f32, label: &str) {
    let Some((ctx, stream)) = setup() else { return };
    let recur = Mamba2SsdKernels::for_context(&ctx).unwrap();
    let chunked = Mamba2SsdChunkedKernels::for_context(&ctx).unwrap();

    let Shape { b, h, l, p, n, q } = shape;
    let x_h = to_bf16(&det_f32(0x200, (b * l * h * p) as usize, 0.5, 0.0));
    let dt_h = to_bf16(&det_f32(0x201, (b * l * h) as usize, 0.2, 0.5));
    let a_h = to_bf16(&det_f32(0x202, h as usize, 0.5, -1.5));
    let b_h_ = to_bf16(&det_f32(0x203, (b * l * h * n) as usize, 0.5, 0.0));
    let c_h = to_bf16(&det_f32(0x204, (b * l * h * n) as usize, 0.5, 0.0));
    let d_h = if has_d {
        to_bf16(&det_f32(0x205, h as usize, 0.5, 0.0))
    } else {
        vec![]
    };

    let x = upload(&stream, &x_h);
    let dt = upload(&stream, &dt_h);
    let a = upload(&stream, &a_h);
    let bb = upload(&stream, &b_h_);
    let cc = upload(&stream, &c_h);
    let d_opt = if has_d {
        Some(upload(&stream, &d_h))
    } else {
        None
    };

    let mut y_recur = alloc_zeros::<bf16>(&stream, (b * l * h * p) as usize);
    let mut y_chunk = alloc_zeros::<bf16>(&stream, (b * l * h * p) as usize);
    recur
        .ssd_bf16(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_recur,
            b,
            l,
            h,
            p,
            n,
        )
        .unwrap();
    chunked
        .ssd(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_chunk,
            b,
            l,
            h,
            p,
            n,
            q,
            DType::BF16,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let yr = bf16_to_f32(&download::<bf16>(&stream, &y_recur));
    let yc = bf16_to_f32(&download::<bf16>(&stream, &y_chunk));
    let err = max_abs(&yr, &yc);
    assert!(
        err <= tol,
        "[{label} BF16 has_d={has_d}] max_abs={err:.3e} > tol={tol:.1e}"
    );
}

fn run_f16(shape: Shape, has_d: bool, tol: f32, label: &str) {
    let Some((ctx, stream)) = setup() else { return };
    let recur = Mamba2SsdKernels::for_context(&ctx).unwrap();
    let chunked = Mamba2SsdChunkedKernels::for_context(&ctx).unwrap();

    let Shape { b, h, l, p, n, q } = shape;
    let x_h = to_f16(&det_f32(0x300, (b * l * h * p) as usize, 0.5, 0.0));
    let dt_h = to_f16(&det_f32(0x301, (b * l * h) as usize, 0.2, 0.5));
    let a_h = to_f16(&det_f32(0x302, h as usize, 0.5, -1.5));
    let b_h_ = to_f16(&det_f32(0x303, (b * l * h * n) as usize, 0.5, 0.0));
    let c_h = to_f16(&det_f32(0x304, (b * l * h * n) as usize, 0.5, 0.0));
    let d_h = if has_d {
        to_f16(&det_f32(0x305, h as usize, 0.5, 0.0))
    } else {
        vec![]
    };

    let x = upload(&stream, &x_h);
    let dt = upload(&stream, &dt_h);
    let a = upload(&stream, &a_h);
    let bb = upload(&stream, &b_h_);
    let cc = upload(&stream, &c_h);
    let d_opt = if has_d {
        Some(upload(&stream, &d_h))
    } else {
        None
    };

    let mut y_recur = alloc_zeros::<f16>(&stream, (b * l * h * p) as usize);
    let mut y_chunk = alloc_zeros::<f16>(&stream, (b * l * h * p) as usize);
    recur
        .ssd_f16(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_recur,
            b,
            l,
            h,
            p,
            n,
        )
        .unwrap();
    chunked
        .ssd(
            &stream,
            &x,
            &dt,
            &a,
            &bb,
            &cc,
            d_opt.as_ref(),
            &mut y_chunk,
            b,
            l,
            h,
            p,
            n,
            q,
            DType::F16,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let yr = f16_to_f32(&download::<f16>(&stream, &y_recur));
    let yc = f16_to_f32(&download::<f16>(&stream, &y_chunk));
    let err = max_abs(&yr, &yc);
    assert!(
        err <= tol,
        "[{label} F16 has_d={has_d}] max_abs={err:.3e} > tol={tol:.1e}"
    );
}

// ── F32 input: tolerance ≲ BF16 precision (chunked cast'ит в bf16). ─────────
// `Y` амплитуда ≈ O(L/Q · √N) = умеренно (state накапливает decay), bf16
// reduction даёт абс ошибку до ~5e-2 на N=128.

#[test]
fn chunked_small_f32_no_skip() {
    run_f32(
        Shape {
            b: 1,
            h: 2,
            l: 32,
            p: 16,
            n: 16,
            q: 16,
        },
        false,
        5e-2,
        "small_no_skip",
    );
}

#[test]
fn chunked_small_f32_with_skip() {
    run_f32(
        Shape {
            b: 1,
            h: 2,
            l: 32,
            p: 16,
            n: 16,
            q: 16,
        },
        true,
        5e-2,
        "small_skip",
    );
}

#[test]
fn chunked_med_f32() {
    run_f32(
        Shape {
            b: 1,
            h: 4,
            l: 64,
            p: 32,
            n: 32,
            q: 16,
        },
        false,
        5e-2,
        "med",
    );
}

#[test]
fn chunked_mamba2_27b_f32() {
    // B=1, H=8 (для скорости теста, не полные 64), Q=64, P=64, N=128. T=4.
    run_f32(
        Shape {
            b: 1,
            h: 8,
            l: 256,
            p: 64,
            n: 128,
            q: 64,
        },
        true,
        1e-1,
        "27b_like",
    );
}

// ── BF16 input: tolerance ≲ порядка reduction. ──────────────────────────────

#[test]
fn chunked_small_bf16() {
    run_bf16(
        Shape {
            b: 1,
            h: 2,
            l: 32,
            p: 16,
            n: 16,
            q: 16,
        },
        true,
        5e-2,
        "small_bf16",
    );
}

#[test]
fn chunked_med_bf16() {
    run_bf16(
        Shape {
            b: 1,
            h: 4,
            l: 64,
            p: 32,
            n: 32,
            q: 16,
        },
        false,
        5e-2,
        "med_bf16",
    );
}

// ── F16 input: f16 cast precision ≈ 0.5 ULP, recurrent тоже f16 → cast. ─────

#[test]
fn chunked_small_f16() {
    run_f16(
        Shape {
            b: 1,
            h: 2,
            l: 32,
            p: 16,
            n: 16,
            q: 16,
        },
        true,
        5e-2,
        "small_f16",
    );
}

// ── GQA chunk sizes Q=16 / 32 / 64 ──────────────────────────────────────────

#[test]
fn chunked_gqa_q32_f32() {
    run_f32(
        Shape {
            b: 1,
            h: 2,
            l: 64,
            p: 32,
            n: 32,
            q: 32,
        },
        false,
        5e-2,
        "gqa_q32",
    );
}

#[test]
fn chunked_gqa_q64_f32() {
    run_f32(
        Shape {
            b: 1,
            h: 2,
            l: 128,
            p: 64,
            n: 64,
            q: 64,
        },
        false,
        1e-1,
        "gqa_q64",
    );
}
