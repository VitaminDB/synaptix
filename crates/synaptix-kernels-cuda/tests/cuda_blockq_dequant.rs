//! Деквант одноблобных форматов на карте бит в бит с CPU-эталоном
//! `synaptix_core::quant`: все типы весов ggml (случайные валидные блоки,
//! шкалы f16 в разумном диапазоне) и SQ1…SQ8 (энкодер эталона), выход F16 и
//! BF16. Модуль собирается с --fmad=false, поэтому равенство — точное.

use cudarc::driver::CudaSlice;
use half::{bf16, f16};
use synaptix_core::dtype::DType;
use synaptix_core::quant::{block_row_bytes, dequant_row_f32, sq, GgmlType};
use synaptix_kernels_cuda::elementwise::blockq::{blockq_dequant, BlockqDequantKernels};

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
    /// f16 с экспонентой в [-8, 2]: произведения не улетают в inf.
    fn sane_f16(&mut self) -> [u8; 2] {
        let sign = ((self.next() & 1) as u16) << 15;
        let exp = ((7 + self.next() % 11) as u16) << 10;
        let man = (self.next() % 1024) as u16;
        (sign | exp | man).to_le_bytes()
    }
}

/// Смещения f16-полей шкал в блоке (как в фикстуре gguf-py).
fn f16_fields(t: GgmlType) -> &'static [usize] {
    use GgmlType::*;
    match t {
        Q4_0 | Q5_0 | Q8_0 | Q3K | Iq2Xxs | Iq2Xs | Iq2S | Iq3Xxs | Iq3S | Iq1S | Iq4Nl | Iq4Xs | Q1_0 | Q2_0 => {
            match t {
                Q3K => &[108],
                _ => &[0],
            }
        }
        Q4_1 | Q5_1 | Q8_1 | Q4K | Q5K => &[0, 2],
        Q2K => &[80, 82],
        Q6K => &[208],
        Tq1_0 => &[52],
        Tq2_0 => &[64],
        _ => &[],
    }
}

fn random_blocks(t: GgmlType, nblk: usize, rng: &mut Lcg) -> Vec<u8> {
    let bb = t.block_bytes();
    let mut v: Vec<u8> = (0..nblk * bb).map(|_| rng.byte()).collect();
    for b in 0..nblk {
        let blk = &mut v[b * bb..(b + 1) * bb];
        for &off in f16_fields(t) {
            blk[off..off + 2].copy_from_slice(&rng.sane_f16());
        }
        match t {
            GgmlType::Iq1M => blk[55] = (blk[55] & 0x0F) | 0x30,
            GgmlType::Mxfp4 => blk[0] = 110 + (rng.next() % 30) as u8,
            GgmlType::Q8K => blk[0..4].copy_from_slice(&rng.f32().to_le_bytes()),
            _ => {}
        }
    }
    v
}

fn cpu_ref(dtype: DType, blob: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let rb = block_row_bytes(dtype, k).unwrap();
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        dequant_row_f32(dtype, &blob[r * rb..(r + 1) * rb], k, &mut out[r * k..(r + 1) * k]).unwrap();
    }
    out
}

fn check(dtype: DType, blob: &[u8], rows: usize, k: usize) {
    let Some(ctx) = synaptix_core::device::cuda::get(0).ok() else {
        eprintln!("нет CUDA — пропуск");
        return;
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let want = cpu_ref(dtype, blob, rows, k);
    let src: CudaSlice<u8> = stream.clone_htod(blob).unwrap();

    let kf = BlockqDequantKernels::for_context(&ctx).unwrap();
    let mut out: CudaSlice<u8> = stream.alloc_zeros(rows * k * 2).unwrap();
    blockq_dequant(&kf, &stream, dtype, &src, &mut out, rows as u32, k as u32).unwrap();
    let got: Vec<u8> = stream.clone_dtoh(&out).unwrap();
    let mut bad = 0;
    for i in 0..rows * k {
        let g = f16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let w = f16::from_f32(want[i]);
        if g.to_bits() != w.to_bits() {
            if bad < 5 {
                eprintln!("{dtype:?} f16 [{i}]: gpu {g} ({:#06x}) vs cpu {w} ({:#06x}) из {}", g.to_bits(), w.to_bits(), want[i]);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{dtype:?}: {bad} расхождений в F16");

    let kb = BlockqDequantKernels::for_context_bf16(&ctx).unwrap();
    let mut out_b: CudaSlice<u8> = stream.alloc_zeros(rows * k * 2).unwrap();
    blockq_dequant(&kb, &stream, dtype, &src, &mut out_b, rows as u32, k as u32).unwrap();
    let got: Vec<u8> = stream.clone_dtoh(&out_b).unwrap();
    let mut bad = 0;
    for i in 0..rows * k {
        let g = bf16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let w = bf16::from_f32(want[i]);
        if g.to_bits() != w.to_bits() {
            if bad < 5 {
                eprintln!("{dtype:?} bf16 [{i}]: gpu {g} vs cpu {w} из {}", want[i]);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{dtype:?}: {bad} расхождений в BF16");
}

#[test]
fn ggml_weight_formats_match_cpu() {
    let mut rng = Lcg(0x5EED_2026_09_22);
    let rows = 5usize;
    for t in GgmlType::ALL {
        if !t.is_weight_format() {
            continue;
        }
        let be = t.block_elems();
        let k = if be >= 32 { 3 * be } else { 96 };
        let blocks_per_row = k / be;
        let blob = random_blocks(t, rows * blocks_per_row, &mut rng);
        check(DType::Ggml(t), &blob, rows, k);
    }
}

#[test]
fn activation_formats_match_cpu() {
    // Q8_1/Q8_K — форматы активаций, но ядро деквантует и их.
    let mut rng = Lcg(77);
    for t in [GgmlType::Q8_1, GgmlType::Q8K] {
        let be = t.block_elems();
        let k = 2 * be;
        let blob = random_blocks(t, 3 * (k / be), &mut rng);
        check(DType::Ggml(t), &blob, 3, k);
    }
}

#[test]
fn sq_all_bits_match_cpu() {
    let mut rng = Lcg(4242);
    let (rows, k) = (4usize, 800usize); // хвостовой супер-блок неполный (800 = 3·256 + 32)
    let x: Vec<f32> = (0..rows * k).map(|_| rng.f32() * 3.0).collect();
    for bits in 1..=8u8 {
        let blob = sq::quantize_matrix(bits, &x, rows, k).unwrap();
        check(DType::Sq { bits }, &blob, rows, k);
    }
}

#[test]
fn sq_zero_and_edge_blocks() {
    // Нулевой блоб (d = dmin = 0) и блоб из одних 0xFF (макс. шкалы, q = qmax).
    let (rows, k) = (2usize, 512usize);
    for bits in [1u8, 3, 4, 8] {
        let rb = sq::row_bytes(bits, k);
        check(DType::Sq { bits }, &vec![0u8; rows * rb], rows, k);
        let mut blob = vec![0xFFu8; rows * rb];
        // d/dmin = 0xFFFF — NaN в f16; берём крупную, но конечную шкалу.
        for r in 0..rows {
            for sb in 0..k / 256 {
                let o = r * rb + sb * sq::super_block_bytes(bits);
                blob[o..o + 2].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
                blob[o + 2..o + 4].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
            }
        }
        check(DType::Sq { bits }, &blob, rows, k);
    }
}
