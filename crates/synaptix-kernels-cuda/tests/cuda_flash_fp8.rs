//! FP8-внимание против bf16 flash_splitq (BSHD, hd128): относительная ошибка
//! и косинус на данных с канальным смещением K (проверка сглаживания).

use cudarc::driver::CudaSlice;
use half::bf16;
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::attention::flash_fp8::{flash_fp8_bshd_mode, Fa8Mode, FlashFp8Kernels};
use synaptix_kernels_cuda::attention::flash_splitq::{flash_splitq_u8, FlashSplitQKernels};

fn det(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect()
}

fn mode(full: bool) -> Fa8Mode {
    if full { Fa8Mode::Full } else { Fa8Mode::Qk }
}

fn run(h: usize, sq: usize, skv: usize, full: bool) -> (f32, f32) {
    run_dt(h, sq, skv, mode(full), DType::BF16)
}

fn run_dt(h: usize, sq: usize, skv: usize, m: Fa8Mode, dt: DType) -> (f32, f32) {
    synaptix_kernels_cuda::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let d = 128;
    let mk = |seed: u64, s: usize, bias: f32, amp: f32| -> CudaSlice<u8> {
        let r = det(seed, s * h * d);
        let v: Vec<u8> = r
            .iter()
            .enumerate()
            .flat_map(|(i, x)| {
                let c = i % d;
                let b = if c % 7 == 0 { bias } else { 0.0 };
                let y = x * amp + b;
                if dt == DType::F16 { half::f16::from_f32(y).to_le_bytes() } else { bf16::from_f32(y).to_le_bytes() }
            })
            .collect();
        stream.clone_htod(&v).unwrap()
    };
    let (q, k, v) = (mk(1, sq, 0.0, 2.0), mk(2, skv, 6.0, 2.0), mk(3, skv, 0.0, 1.0));
    let n = sq * h * d;
    let scale = 1.0 / (d as f32).sqrt();
    let mut o_ref: CudaSlice<u8> = stream.alloc_zeros(n * 2).unwrap();
    let mut o8: CudaSlice<u8> = stream.alloc_zeros(n * 2).unwrap();
    let ks = FlashSplitQKernels::for_context(&ctx).unwrap();
    flash_splitq_u8(&ks, &stream, &q, 0, &k, 0, &v, 0, &mut o_ref, 0, 1, h as u32, h as u32, sq as u32, skv as u32, d as u32, scale, false, 0, dt, true).unwrap();
    let k8 = FlashFp8Kernels::for_dtype(&ctx, dt).unwrap();
    flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o8, 1, h, sq, skv, scale, m).unwrap();
    stream.synchronize().unwrap();
    let to_f = |s: &CudaSlice<u8>| -> Vec<f32> { stream.clone_dtoh(s).unwrap().chunks(2).map(|c| if dt == DType::F16 { half::f16::from_le_bytes([c[0], c[1]]).to_f32() } else { bf16::from_le_bytes([c[0], c[1]]).to_f32() }).collect() };
    let (a, b) = (to_f(&o8), to_f(&o_ref));
    let (mut dot, mut na, mut nb, mut err, mut refn) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        err += (x - y).powi(2);
        refn += y * y;
    }
    ((dot / (na.sqrt() * nb.sqrt())) as f32, (err / refn).sqrt() as f32)
}

#[test]
fn fp8_attention_close_to_bf16() {
    for full in [false, true] {
        for (h, sq, skv) in [(2usize, 300usize, 300usize), (3, 1100, 777), (2, 64, 2050)] {
            let (cos, rel) = run(h, sq, skv, full);
            eprintln!("[fp8 attn full={full} h={h} sq={sq} skv={skv}] cos={cos:.6} rel={rel:.4}");
            assert!(cos > 0.995, "cos {cos}");
            assert!(rel < if full { 0.08 } else { 0.05 }, "rel {rel}");
        }
    }
}

#[test]
fn fp8_attention_f16_close() {
    for full in [false, true] {
        for (h, sq, skv) in [(2usize, 300usize, 300usize), (3, 1100, 777), (2, 64, 2050)] {
            let (cos, rel) = run_dt(h, sq, skv, mode(full), DType::F16);
            eprintln!("[fp8 attn f16 full={full} h={h} sq={sq} skv={skv}] cos={cos:.6} rel={rel:.4}");
            assert!(cos > 0.995, "cos {cos}");
            assert!(rel < if full { 0.08 } else { 0.05 }, "rel {rel}");
        }
    }
}

#[test]
fn fp8_attention_acc16_close() {
    // f16-накопление против f32 того же FP8: длинные ключи — суммы по тысячам плиток.
    for dt in [DType::BF16, DType::F16] {
        for (h, sq, skv) in [(2usize, 300usize, 300usize), (3, 1100, 777), (2, 64, 2050), (2, 512, 20000)] {
            let (c32, r32) = run_dt(h, sq, skv, Fa8Mode::Full, dt);
            let (c16, r16) = run_dt(h, sq, skv, Fa8Mode::Acc16, dt);
            eprintln!("[fp8 attn {dt:?} h={h} sq={sq} skv={skv}] f32-acc cos={c32:.6} rel={r32:.4} | f16-acc cos={c16:.6} rel={r16:.4}");
            assert!(c16 > 0.995, "cos {c16}");
            assert!(r16 < 0.08, "rel {r16}");
        }
    }
}

#[test]
fn fp8_attention_kv_cache_and_sparse() {
    synaptix_kernels_cuda::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let (h, sq, skv, d) = (4usize, 512usize, 2048usize, 128usize);
    let mk = |seed: u64, s: usize, amp: f32| -> CudaSlice<u8> {
        let r = det(seed, s * h * d);
        let v: Vec<u8> = r.iter().flat_map(|x| bf16::from_f32(x * amp).to_le_bytes()).collect();
        stream.clone_htod(&v).unwrap()
    };
    let (q, k, v) = (mk(11, sq, 1.5), mk(12, skv, 1.5), mk(13, skv, 1.0));
    let scale = 1.0 / (d as f32).sqrt();
    let k8 = FlashFp8Kernels::for_dtype(&ctx, DType::BF16).unwrap();
    let bytes = sq * h * d * 2;
    let mut o_dense: CudaSlice<u8> = stream.alloc_zeros(bytes).unwrap();
    let mut o_cached: CudaSlice<u8> = stream.alloc_zeros(bytes).unwrap();
    let mut o_sp_full: CudaSlice<u8> = stream.alloc_zeros(bytes).unwrap();
    let mut o_sp_prot: CudaSlice<u8> = stream.alloc_zeros(bytes).unwrap();

    // 1. Dense uncached
    flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o_dense, 1, h, sq, skv, scale, Fa8Mode::Acc16).unwrap();

    // 2. Cached K/V across two calls must be bit-exact with uncached
    {
        let _kv = synaptix_core::precision::FlashFp8KvGuard::new();
        flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o_cached, 1, h, sq, skv, scale, Fa8Mode::Acc16).unwrap();
        flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o_cached, 1, h, sq, skv, scale, Fa8Mode::Acc16).unwrap();
    }

    // 3. Sparse with q_offset=0 and protected_prefix=sq -> ALL queries protected -> bit-exact with dense!
    {
        let _sp = synaptix_core::precision::AttentionSparseGuard::new(Some(synaptix_core::precision::AttentionSparseConfig {
            protected_prefix: sq,
            q_offset: 0,
            keep_ratio: 0.35,
            frame_tokens: 256,
        }));
        flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o_sp_prot, 1, h, sq, skv, scale, Fa8Mode::Acc16).unwrap();
    }

    // 4. Sparse on video region (q_offset=256, prot=128, keep=0.5) must stay finite & close
    {
        let _sp = synaptix_core::precision::AttentionSparseGuard::new(Some(synaptix_core::precision::AttentionSparseConfig {
            protected_prefix: 128,
            q_offset: 256,
            keep_ratio: 0.50,
            frame_tokens: 256,
        }));
        flash_fp8_bshd_mode(&k8, &stream, (&q, 0), (&k, 0), (&v, 0), &mut o_sp_full, 1, h, sq, skv, scale, Fa8Mode::Acc16).unwrap();
    }
    stream.synchronize().unwrap();

    let b_dense = stream.clone_dtoh(&o_dense).unwrap();
    let b_cached = stream.clone_dtoh(&o_cached).unwrap();
    let b_prot = stream.clone_dtoh(&o_sp_prot).unwrap();
    let b_sp = stream.clone_dtoh(&o_sp_full).unwrap();
    assert_eq!(b_dense, b_cached, "KV cache must be bit-exact with uncached");
    assert_eq!(b_dense, b_prot, "Protected prefix queries must be bit-exact with dense");

    let to_f = |buf: &[u8]| -> Vec<f32> { buf.chunks(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect() };
    let (fd, fs) = (to_f(&b_dense), to_f(&b_sp));
    assert!(fs.iter().all(|x| x.is_finite()), "sparse output contains NaN/Inf");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in fs.iter().zip(&fd) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y; na += x * x; nb += y * y;
    }
    let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
    eprintln!("[fp8 sparse keep=0.50] cos vs dense = {cos:.6}");
    assert!(cos > 0.85, "sparse cos too low: {cos}");
}
