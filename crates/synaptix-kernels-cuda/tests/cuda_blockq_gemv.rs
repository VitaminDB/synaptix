//! Портируемый GEMV по квантованным весам против CPU-эталона: для каждого
//! формата (SQ1…8, все веса ggml, NVFP4/MXFP8 движка) `out = x · Wᵀ` на M = 1,
//! 5 и 8 строк активации, F16 и BF16; батчевый вариант — против одиночных.
//! Эталон: деквант CPU (f32) и скалярное произведение в f64.

use cudarc::driver::CudaSlice;
use half::{bf16, f16};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::{block_row_bytes, dequant_row_f32, sq, GgmlType};
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cuda::elementwise::blockq::{blockq_gemv, blockq_gemv_batched, BlockqGemvKernels};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn sane_f16(&mut self) -> [u8; 2] {
        let sign = ((self.next() & 1) as u16) << 15;
        let exp = ((5 + self.next() % 6) as u16) << 10; // 2^-10 … 2^-5
        let man = (self.next() % 1024) as u16;
        (sign | exp | man).to_le_bytes()
    }
}

fn f16_fields(t: GgmlType) -> &'static [usize] {
    use GgmlType::*;
    match t {
        Q3K => &[108],
        Q4_0 | Q5_0 | Q8_0 | Iq2Xxs | Iq2Xs | Iq2S | Iq3Xxs | Iq3S | Iq1S | Iq4Nl | Iq4Xs | Q1_0 | Q2_0 => &[0],
        Q4_1 | Q5_1 | Q4K | Q5K => &[0, 2],
        Q2K => &[80, 82],
        Q6K => &[208],
        Tq1_0 => &[52],
        Tq2_0 => &[64],
        _ => &[],
    }
}

fn random_blob(t: GgmlType, nblk: usize, rng: &mut Lcg) -> Vec<u8> {
    let bb = t.block_bytes();
    let mut v: Vec<u8> = (0..nblk * bb).map(|_| rng.byte()).collect();
    for b in 0..nblk {
        let blk = &mut v[b * bb..(b + 1) * bb];
        for &off in f16_fields(t) {
            blk[off..off + 2].copy_from_slice(&rng.sane_f16());
        }
        match t {
            GgmlType::Iq1M => blk[55] = (blk[55] & 0x0F) | 0x30,
            GgmlType::Mxfp4 => blk[0] = 120 + (rng.next() % 8) as u8,
            _ => {}
        }
    }
    v
}

struct Case {
    dtype: DType,
    /// Блоб веса (одноблобный) или packed NVFP4/MXFP8.
    w: Vec<u8>,
    /// Масштабы NVFP4/MXFP8 (пусто у одноблобных).
    sw: Vec<u8>,
    /// Деквантованный вес [n, k] f32 — эталон.
    wf: Vec<f32>,
}

fn blob_case(dtype: DType, n: usize, k: usize, rng: &mut Lcg) -> Case {
    let w = match dtype {
        DType::Sq { bits } => {
            let x: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
            sq::quantize_matrix(bits, &x, n, k).unwrap()
        }
        DType::Ggml(t) => random_blob(t, n * (k / t.block_elems()), rng),
        _ => unreachable!(),
    };
    let rb = block_row_bytes(dtype, k).unwrap();
    let mut wf = vec![0f32; n * k];
    for r in 0..n {
        dequant_row_f32(dtype, &w[r * rb..(r + 1) * rb], k, &mut wf[r * k..(r + 1) * k]).unwrap();
    }
    Case { dtype, w, sw: Vec::new(), wf }
}

/// NVFP4/MXFP8 движка: квантуем на карте существующим ядром, эталон — деквант
/// тем же движком (`nvfp4_dequant_f16` / `mxfp8_dequant`) через `QuantWeight`.
fn syn_case(dtype: DType, n: usize, k: usize, rng: &mut Lcg) -> Case {
    let x: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
    let t = Tensor::from_vec(x, (n, k), Device::Cuda(0)).unwrap().to_dtype(DType::F16).unwrap();
    let qw = match dtype {
        DType::NVFP4 => t.quantize_to_nvfp4().unwrap(),
        DType::MXFP8 => t.quantize_to_mxfp8().unwrap(),
        _ => unreachable!(),
    };
    let host = qw.to_device(Device::Cpu).unwrap();
    let w = host.packed_arc().unwrap().as_cpu().unwrap().as_bytes().to_vec();
    let sw = host.scales().as_cpu().unwrap().as_bytes().to_vec();
    let wf = match dtype {
        DType::MXFP8 => qw.dequantize(DType::F16).unwrap(),
        _ => {
            // NVFP4: деквант через linear_quant с единичной матрицей активаций
            // не нужен — считаем эталон на CPU из packed+scales тем же
            // порядком, что ядро GEMV: E2M1 × E4M3.
            let mut out = vec![0f32; n * k];
            let sf_inner = k.div_ceil(64) * 4;
            for r in 0..n {
                for c in 0..k {
                    let bc = c / 16;
                    let tile_row = r >> 7;
                    let tile_col = bc >> 2;
                    let lo = r & 127;
                    let li = bc & 3;
                    let off = (tile_col * 4 + tile_row * sf_inner) * 128 + (lo & 31) * 16 + (lo >> 5) * 4 + li;
                    let sc = e4m3(sw[off]);
                    let b = w[(r * k + c) / 2];
                    let nib = if c % 2 == 0 { b & 0xF } else { b >> 4 };
                    out[r * k + c] = e2m1(nib) * sc;
                }
            }
            return Case { dtype, w, sw, wf: out };
        }
    };
    let wf = wf.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    Case { dtype, w, sw, wf }
}

fn e4m3(b: u8) -> f32 {
    let sign = b & 0x80 != 0;
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 7) as f32;
    let v = if exp == 0 { man * 0.001953125 } else { (1.0 + man * 0.125) * 2f32.powi(exp - 7) };
    if sign { -v } else { v }
}
fn e2m1(nib: u8) -> f32 {
    let mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = mags[(nib & 7) as usize];
    if nib & 8 != 0 { -m } else { m }
}

fn check_case(case: &Case, n: usize, k: usize, m: usize, bf16_act: bool, rng: &mut Lcg) {
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let x: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
    let (x_bytes, xq): (Vec<u8>, Vec<f32>) = if bf16_act {
        let q: Vec<bf16> = x.iter().map(|v| bf16::from_f32(*v)).collect();
        (q.iter().flat_map(|h| h.to_le_bytes()).collect(), q.iter().map(|h| h.to_f32()).collect())
    } else {
        let q: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
        (q.iter().flat_map(|h| h.to_le_bytes()).collect(), q.iter().map(|h| h.to_f32()).collect())
    };
    // Эталон f64.
    let mut want = vec![0f64; m * n];
    for mm in 0..m {
        for r in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                acc += case.wf[r * k + c] as f64 * xq[mm * k + c] as f64;
            }
            want[mm * n + r] = acc;
        }
    }
    let dw: CudaSlice<u8> = stream.clone_htod(&case.w).unwrap();
    let dsw: CudaSlice<u8> = stream.clone_htod(if case.sw.is_empty() { &case.w[..1] } else { &case.sw }).unwrap();
    let dx: CudaSlice<u8> = stream.clone_htod(&x_bytes).unwrap();
    let mut dout: CudaSlice<u8> = stream.alloc_zeros(m * n * 2).unwrap();
    let kernels = if bf16_act { BlockqGemvKernels::for_context_bf16(&ctx) } else { BlockqGemvKernels::for_context(&ctx) }.unwrap();
    blockq_gemv(&kernels, &stream, case.dtype, &dw.as_view(), &dsw.as_view(), &dx.as_view(), &mut dout.as_view_mut(), n as u32, k as u32, m as u32, k as u32, n as u32).unwrap();
    let got: Vec<u8> = stream.clone_dtoh(&dout).unwrap();
    let scale = want.iter().map(|v| v.abs()).fold(0f64, f64::max).max(1e-3);
    let tol = if bf16_act { 2e-2 } else { 4e-3 } * scale + 1e-4;
    let mut worst = 0f64;
    for i in 0..m * n {
        let g = if bf16_act {
            bf16::from_le_bytes([got[2 * i], got[2 * i + 1]]).to_f32() as f64
        } else {
            f16::from_le_bytes([got[2 * i], got[2 * i + 1]]).to_f32() as f64
        };
        let d = (g - want[i]).abs();
        worst = worst.max(d);
        assert!(d <= tol, "{:?} m={m} bf16={bf16_act} [{i}]: gpu {g} vs {} (tol {tol})", case.dtype, want[i]);
    }
    let _ = worst;
}

fn all_formats() -> Vec<DType> {
    let mut v: Vec<DType> = (1..=8u8).map(|bits| DType::Sq { bits }).collect();
    v.extend(GgmlType::ALL.iter().filter(|t| t.is_weight_format()).map(|t| DType::Ggml(*t)));
    v
}

#[test]
fn gemv_every_format_matches_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    let mut rng = Lcg(0xA11CE);
    let (n, k) = (96usize, 768usize);
    for dtype in all_formats() {
        let case = blob_case(dtype, n, k, &mut rng);
        for m in [1usize, 5, 8] {
            check_case(&case, n, k, m, false, &mut rng);
        }
        check_case(&case, n, k, 3, true, &mut rng);
    }
}

#[test]
fn gemv_engine_nvfp4_mxfp8_match_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut rng = Lcg(0xBEEF);
    let (n, k) = (256usize, 1024usize);
    for dtype in [DType::NVFP4, DType::MXFP8] {
        let case = syn_case(dtype, n, k, &mut rng);
        for m in [1usize, 4, 8] {
            check_case(&case, n, k, m, false, &mut rng);
        }
        check_case(&case, n, k, 2, true, &mut rng);
    }
}

#[test]
fn gemv_batched_matches_single() {
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let mut rng = Lcg(77);
    let (n, k, e) = (64usize, 512usize, 3usize);
    for dtype in [DType::Sq { bits: 4 }, DType::Ggml(GgmlType::Q4K), DType::Ggml(GgmlType::Q8_0)] {
        let cases: Vec<Case> = (0..e).map(|_| blob_case(dtype, n, k, &mut rng)).collect();
        // Активация: e+1 строк, эксперт i читает строку (i + 1).
        let x: Vec<f16> = (0..(e + 1) * k).map(|_| f16::from_f32(rng.f32())).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|h| h.to_le_bytes()).collect();
        let dx: CudaSlice<u8> = stream.clone_htod(&x_bytes).unwrap();
        let dws: Vec<CudaSlice<u8>> = cases.iter().map(|c| stream.clone_htod(&c.w).unwrap()).collect();
        let mut douts: Vec<CudaSlice<u8>> = (0..e).map(|_| stream.alloc_zeros::<u8>(n * 2).unwrap()).collect();
        use cudarc::driver::DevicePtr;
        let ptr = |s: &CudaSlice<u8>| s.device_ptr(&stream).0;
        let w_ptrs: Vec<u64> = dws.iter().map(ptr).collect();
        let x_ptrs: Vec<u64> = (0..e).map(|i| ptr(&dx) + ((i + 1) * k * 2) as u64).collect();
        let out_ptrs: Vec<u64> = douts.iter().map(ptr).collect();
        let dwp: CudaSlice<u64> = stream.clone_htod(&w_ptrs).unwrap();
        let dxp: CudaSlice<u64> = stream.clone_htod(&x_ptrs).unwrap();
        let dop: CudaSlice<u64> = stream.clone_htod(&out_ptrs).unwrap();
        let kernels = BlockqGemvKernels::for_context(&ctx).unwrap();
        blockq_gemv_batched(&kernels, &stream, dtype, &dwp, &dwp, &dxp, &dop, e as u32, n as u32, k as u32).unwrap();
        for i in 0..e {
            let got: Vec<u8> = stream.clone_dtoh(&douts[i]).unwrap();
            let mut single: CudaSlice<u8> = stream.alloc_zeros(n * 2).unwrap();
            let xs = dx.slice((i + 1) * k * 2..(i + 2) * k * 2);
            blockq_gemv(&kernels, &stream, dtype, &dws[i].as_view(), &dws[i].as_view(), &xs, &mut single.as_view_mut(), n as u32, k as u32, 1, k as u32, n as u32).unwrap();
            let want: Vec<u8> = stream.clone_dtoh(&single).unwrap();
            assert_eq!(got, want, "{dtype:?}: эксперт {i} батчем и по одному разошлись");
        }
        douts.clear();
    }
}
