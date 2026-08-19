//! Бенч MXFP8-KV attention на формах Qwen3.8-27B (24Q/4KV×256) при длинном
//! контексте. Запуск: cargo test -p synaptix-kernels-cuda --release
//!   --test zz_mxfp8_kv_bench -- --ignored --nocapture
//! Tq=1 — decode-шаг, Tq=2 — MTP-verify (сейчас уходит в WMMA-«префилл»).

use half::f16;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            f16::from_f32(((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn bench(t_q: usize, t_kv: usize, label: &str) {
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    let (b, nh, nkv, hd) = (1usize, 24usize, 4usize, 256usize);
    let nb = hd / 32;
    let max_seq = 57344usize; // реальный KV-ринг из панели synthos
    let scale = 1.0 / (hd as f32).sqrt();

    let q = Tensor::from_vec(det(11, b * nh * t_q * hd, 1.0), vec![b, nh, t_q, hd], dev).unwrap();
    let k = Tensor::from_vec(det(22, b * nkv * t_kv * hd, 1.0), vec![b, nkv, t_kv, hd], dev).unwrap();
    let v = Tensor::from_vec(det(33, b * nkv * t_kv * hd, 1.0), vec![b, nkv, t_kv, hd], dev).unwrap();

    let mut k_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut v_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut k_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();
    let mut v_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();
    k_buf.kv_append_quant_mxfp8_inplace(&mut k_sc, &k, 0).unwrap();
    v_buf.kv_append_quant_mxfp8_inplace(&mut v_sc, &v, 0).unwrap();

    let k_q = k_buf.narrow(2, 0, t_kv).unwrap();
    let v_q = v_buf.narrow(2, 0, t_kv).unwrap();
    let ks = k_sc.narrow(2, 0, t_kv).unwrap();
    let vs = v_sc.narrow(2, 0, t_kv).unwrap();

    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    for _ in 0..3 {
        let _ = q.flash_attention_mxfp8kv(&k_q, &v_q, &ks, &vs, scale, true).unwrap();
    }
    stream.synchronize().unwrap();
    let it = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..it {
        let _ = q.flash_attention_mxfp8kv(&k_q, &v_q, &ks, &vs, scale, true).unwrap();
    }
    stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / it as f64;
    // Прочитанные KV-байты: K+V (по 1 байту) + скейлы (1/32) на nkv голов.
    let kv_bytes = (b * nkv * t_kv * hd * 2) as f64 * (1.0 + 1.0 / 32.0);
    eprintln!(
        "[mxfp8 kv {label}] Tq={t_q} Tkv={t_kv} {:.3} ms/call, {:.1} GB/s eff, x16 слоёв = {:.1} ms/ток",
        dt * 1e3,
        kv_bytes / dt / 1e9,
        dt * 1e3 * 16.0
    );
}

#[test]
#[ignore]
fn bench_decode_tq1_47k() {
    bench(1, 47000, "decode");
}

#[test]
#[ignore]
fn bench_verify_tq2_47k() {
    bench(2, 47000, "verify");
}

#[test]
#[ignore]
fn bench_decode_tq1_8k() {
    bench(1, 8192, "decode");
}

// Префилл: чанк 256 токенов в хвосте контекста (Tkv = позиция + 256).
#[test]
#[ignore]
fn bench_prefill_tq256_8k() {
    bench(256, 8192, "prefill");
}

#[test]
#[ignore]
fn bench_prefill_tq256_24k() {
    bench(256, 24576, "prefill");
}

#[test]
#[ignore]
fn bench_prefill_tq256_47k() {
    bench(256, 47000, "prefill");
}
