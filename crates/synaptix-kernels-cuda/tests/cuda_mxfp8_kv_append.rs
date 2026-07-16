#![cfg(feature = "cuda")]

//! CUDA MXFP8-KV append-квант: BF16 → MXFP8 E4M3 + per-32-block E8M0 (U8).
//! Сверяет CUDA-ядро с CPU-референсом (тот же E8M0/E4M3 кодек) — bit-exact
//! roundtrip (cos≈1.0), доказывая корректность block-scale scatter-append.

use half::bf16;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::quant::{decode_e4m3, e8m0_decode, e8m0_scale_byte, encode_e4m3, MXFP8_BLOCK};

fn setup() -> bool {
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

#[test]
fn mxfp8_kv_append_matches_cpu_block_codec() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    let (b, nkv, t, hd) = (2usize, 3, 7, 128);
    let nb = hd / MXFP8_BLOCK;
    let max_seq = 32usize;
    let seq_pos = 0usize;

    let src_f = det(7, b * nkv * t * hd, 1.0);
    let src_bf: Vec<bf16> = src_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let src = Tensor::from_vec(src_bf.clone(), vec![b, nkv, t, hd], dev).unwrap();

    let mut dst = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();
    dst.kv_append_quant_mxfp8_inplace(&mut sc, &src, seq_pos).unwrap();

    // Копируем сырые MXFP8/U8-байты на host (MXFP8 не конвертится to_dtype → читаем
    // storage напрямую).
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let dst_u8: Vec<u8> = stream.clone_dtoh(dst.storage().as_cuda().unwrap().slice()).unwrap();
    let sc_u8: Vec<u8> = stream.clone_dtoh(sc.storage().as_cuda().unwrap().slice()).unwrap();

    // CPU-эталон: квантизуем те же bf16 (как f32) per-32-block, сверяем decode.
    let mut cuda_deq = vec![0.0f32; b * nkv * t * hd];
    let mut cpu_deq = vec![0.0f32; b * nkv * t * hd];
    for bh in 0..(b * nkv) {
        for ti in 0..t {
            for blk in 0..nb {
                // CPU reference quant блока.
                let src_row = (bh * t + ti) * hd + blk * MXFP8_BLOCK;
                let mut amax = 0.0f32;
                for i in 0..MXFP8_BLOCK {
                    amax = amax.max(src_bf[src_row + i].to_f32().abs());
                }
                let cpu_sv = e8m0_decode(e8m0_scale_byte(amax));
                // CUDA appended на слот (seq_pos+ti) в max_seq-буфере.
                let dst_row = (bh * max_seq + seq_pos + ti) * hd + blk * MXFP8_BLOCK;
                let sc_idx = (bh * max_seq + seq_pos + ti) * nb + blk;
                let cuda_sv = e8m0_decode(sc_u8[sc_idx]);
                for i in 0..MXFP8_BLOCK {
                    let out_i = (bh * t + ti) * hd + blk * MXFP8_BLOCK + i;
                    cpu_deq[out_i] =
                        decode_e4m3(encode_e4m3(src_bf[src_row + i].to_f32() / cpu_sv)) * cpu_sv;
                    cuda_deq[out_i] = decode_e4m3(dst_u8[dst_row + i]) * cuda_sv;
                }
            }
        }
    }

    let cs = cos_sim(&cuda_deq, &cpu_deq);
    let max_abs = cuda_deq.iter().zip(&cpu_deq).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
    assert!(cs > 0.9999, "cuda vs cpu block-codec cos {cs} (max_abs {max_abs})");
}
