//! NVFP4 GEMV на CPU прямо по перемешанной раскладке GPU-зеркала.
//!
//! Эксперты MoE в RAM лежат так, как их читают тензорные ядра карты
//! (`QuantWeight::nvfp4_repack_host`): тайл = 16 строк × 64 значения K,
//! 512 байт подряд, внутри — 16 строк по 32 байта (полубайт на значение,
//! младший — чётное K). Масштабы — E4M3 на каждые 16 значений в swizzle-
//! раскладке (`nvfp4_scale_index`); четыре масштаба 64-элементного куска
//! строки лежат четырьмя байтами подряд. Глобального масштаба нет:
//! `w = E2M1[q] · e4m3(s)`.
//!
//! Счёт целочисленный: значения E2M1, умноженные на 2, — целые
//! {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}; активация квантуется в int8 блоками по
//! 16 (масштаб на блок), и на каждый блок выходит `Σ w2·xq · e4m3(s)·sx/2`.
//! Для `vpdpbusd` (u8 × s8) вес берётся по модулю, знак переносится на
//! активацию (`sign_epi8`).

use super::dequant::nvfp4_sf_inner_dim;

/// E2M1 × 2 по коду полубайта.
const W2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn e4m3_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = super::decode_e4m3(i as u8);
        }
        t
    })
}

/// Активация, квантованная под [`gemv_shuf`]: int8 блоками по 16 и
/// разложенная по кускам 64 на чётные (`lo`) и нечётные (`hi`) позиции —
/// в том же порядке, что полубайты веса.
pub struct ActQ {
    pub k: usize,
    lo: Vec<i8>,
    hi: Vec<i8>,
    /// Масштаб блока 16, уже поделённый на 2 (вес хранится как E2M1·2).
    sx: Vec<f32>,
}

impl ActQ {
    pub fn new(x: &[f32]) -> Self {
        let k = x.len();
        assert!(k % 64 == 0, "nvfp4 cpu: K={k} не кратно 64");
        let mut q = vec![0i8; k];
        let mut sx = vec![0f32; k / 16];
        for (b, blk) in x.chunks_exact(16).enumerate() {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            if amax > 0.0 {
                let s = amax / 127.0;
                let inv = 1.0 / s;
                for (o, v) in q[b * 16..b * 16 + 16].iter_mut().zip(blk) {
                    *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
                sx[b] = s * 0.5;
            }
        }
        let mut lo = vec![0i8; k / 2];
        let mut hi = vec![0i8; k / 2];
        for c in 0..k / 64 {
            for j in 0..32 {
                lo[c * 32 + j] = q[c * 64 + 2 * j];
                hi[c * 32 + j] = q[c * 64 + 2 * j + 1];
            }
        }
        Self { k, lo, hi, sx }
    }
}

/// Индекс первого из четырёх масштабов куска `chunk` строки `row`.
#[inline]
fn scale_base(row: usize, chunk: usize, sf_inner: usize) -> usize {
    let tile_row = row >> 7;
    let local = row & 127;
    (chunk * 4 + tile_row * sf_inner) * 128 + (local & 31) * 16 + (local >> 5) * 4
}

/// `out[r] = Σ_k W[r, k]·x[k]` для строк `[16·mb0, 16·mb1)` (`mb` — номер
/// 16-строчного тайла). `packed` — перемешанная раскладка `N·K/2` байт,
/// `scales` — swizzle-масштабы, `out` — по строке на выход (`out.len()` ≥
/// числа строк диапазона, `out[0]` — строка `16·mb0`).
pub fn gemv_shuf(packed: &[u8], scales: &[u8], n: usize, act: &ActQ, mb0: usize, mb1: usize, out: &mut [f32]) {
    let k = act.k;
    debug_assert!(n % 16 == 0 && packed.len() >= n * k / 2);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            let vnni = std::is_x86_feature_detected!("avxvnni");
            // SAFETY: наборы инструкций проверены выше.
            unsafe {
                if vnni {
                    x86::gemv_vnni(packed, scales, n, act, mb0, mb1, out);
                } else {
                    x86::gemv_avx2(packed, scales, n, act, mb0, mb1, out);
                }
            }
            return;
        }
    }
    gemv_scalar(packed, scales, n, act, mb0, mb1, out);
}

pub fn gemv_scalar(packed: &[u8], scales: &[u8], _n: usize, act: &ActQ, mb0: usize, mb1: usize, out: &mut [f32]) {
    let k = act.k;
    let chunks = k / 64;
    let sf = nvfp4_sf_inner_dim(k);
    let t = e4m3_table();
    for mb in mb0..mb1 {
        for r in 0..16 {
            let row = mb * 16 + r;
            let mut acc = 0f32;
            for c in 0..chunks {
                let tile = &packed[(mb * chunks + c) * 512 + r * 32..][..32];
                let sb = scale_base(row, c, sf);
                for b in 0..4 {
                    let mut d = 0i32;
                    for j in 8 * b..8 * b + 8 {
                        let byte = tile[j];
                        d += W2[(byte & 15) as usize] as i32 * act.lo[c * 32 + j] as i32;
                        d += W2[(byte >> 4) as usize] as i32 * act.hi[c * 32 + j] as i32;
                    }
                    acc += d as f32 * t[scales[sb + b] as usize] * act.sx[c * 4 + b];
                }
            }
            out[(mb - mb0) * 16 + r] = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn lut() -> __m256i {
        _mm256_setr_epi8(
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6,
            -8, -12,
        )
    }

    macro_rules! gemv_body {
        ($name:ident, $feat:literal, $dot:ident) => {
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(
                packed: &[u8],
                scales: &[u8],
                _n: usize,
                act: &ActQ,
                mb0: usize,
                mb1: usize,
                out: &mut [f32],
            ) {
                let k = act.k;
                let chunks = k / 64;
                let sf = nvfp4_sf_inner_dim(k);
                let t = e4m3_table();
                let lut = lut();
                let m4 = _mm256_set1_epi8(0x0f);
                let pk = packed.as_ptr();
                for mb in mb0..mb1 {
                    let mut acc = [_mm256_setzero_ps(); 16];
                    for c in 0..chunks {
                        let xl = _mm256_loadu_si256(act.lo.as_ptr().add(c * 32) as *const __m256i);
                        let xh = _mm256_loadu_si256(act.hi.as_ptr().add(c * 32) as *const __m256i);
                        let sx = act.sx.as_ptr().add(c * 4);
                        let (s0, s1, s2, s3) = (*sx, *sx.add(1), *sx.add(2), *sx.add(3));
                        let tile = pk.add((mb * chunks + c) * 512);
                        for r in 0..16 {
                            let b = _mm256_loadu_si256(tile.add(r * 32) as *const __m256i);
                            let lo = _mm256_and_si256(b, m4);
                            let hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m4);
                            let wl = _mm256_shuffle_epi8(lut, lo);
                            let wh = _mm256_shuffle_epi8(lut, hi);
                            let d = $dot(wl, xl, wh, xh);
                            let sb = scale_base(mb * 16 + r, c, sf);
                            let sc = scales.as_ptr().add(sb);
                            let (w0, w1, w2, w3) = (
                                t[*sc as usize] * s0,
                                t[*sc.add(1) as usize] * s1,
                                t[*sc.add(2) as usize] * s2,
                                t[*sc.add(3) as usize] * s3,
                            );
                            let sv = _mm256_setr_ps(w0, w0, w1, w1, w2, w2, w3, w3);
                            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), sv, acc[r]);
                        }
                    }
                    for r in 0..16 {
                        out[(mb - mb0) * 16 + r] = hsum(acc[r]);
                    }
                }
            }
        };
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// Σ по четвёркам байт: |w|·sign(x, w), u8 × s8 → i32 (VNNI).
    #[inline(always)]
    unsafe fn dot_vnni(wl: __m256i, xl: __m256i, wh: __m256i, xh: __m256i) -> __m256i {
        let z = _mm256_setzero_si256();
        let d = _mm256_dpbusd_avx_epi32(z, _mm256_abs_epi8(wl), _mm256_sign_epi8(xl, wl));
        _mm256_dpbusd_avx_epi32(d, _mm256_abs_epi8(wh), _mm256_sign_epi8(xh, wh))
    }

    /// То же на AVX2: maddubs (пары → i16, |w|·|x| ≤ 12·127 — без насыщения)
    /// и madd с единицами (пары i16 → i32).
    #[inline(always)]
    unsafe fn dot_avx2(wl: __m256i, xl: __m256i, wh: __m256i, xh: __m256i) -> __m256i {
        let ones = _mm256_set1_epi16(1);
        let pl = _mm256_maddubs_epi16(_mm256_abs_epi8(wl), _mm256_sign_epi8(xl, wl));
        let ph = _mm256_maddubs_epi16(_mm256_abs_epi8(wh), _mm256_sign_epi8(xh, wh));
        _mm256_madd_epi16(_mm256_add_epi16(pl, ph), ones)
    }

    gemv_body!(gemv_vnni, "avx2,fma,avxvnni", dot_vnni);
    gemv_body!(gemv_avx2, "avx2,fma", dot_avx2);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn make(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
        let w = noise(seed, n * k);
        let (p, s) = crate::quant::encode::quantize_nvfp4_rows(&w, n, k).expect("квант");
        let mut shuf = vec![0u8; p.len()];
        synaptix_core::tensor::quant::QuantWeight::nvfp4_repack_host(&p, &mut shuf, n, k).expect("перемешивание");
        // Эталон — деквантованный вес (то, что видит GPU).
        let mut wd = vec![0f32; n * k];
        for r in 0..n {
            crate::quant::dequant::nvfp4_dequant_row(&p[r * k / 2..(r + 1) * k / 2], &s, r, k, &mut wd[r * k..(r + 1) * k]);
        }
        (wd, shuf, s)
    }

    #[test]
    fn shuffled_gemv_matches_dequantized() {
        for (n, k) in [(1280usize, 2560usize), (2560, 640), (128, 64)] {
            let (wd, shuf, s) = make(n, k, n as u64 + k as u64);
            let x = noise(99, k);
            let act = ActQ::new(&x);
            let reference: Vec<f32> = (0..n).map(|r| (0..k).map(|j| wd[r * k + j] * x[j]).sum()).collect();
            let mut simd = vec![0f32; n];
            gemv_shuf(&shuf, &s, n, &act, 0, n / 16, &mut simd);
            let mut scalar = vec![0f32; n];
            gemv_scalar(&shuf, &s, n, &act, 0, n / 16, &mut scalar);
            let norm = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
            let rel = |a: &[f32]| a.iter().zip(&reference).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt() / norm;
            let (es, ec) = (rel(&simd), rel(&scalar));
            let agree = simd.iter().zip(&scalar).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("nvfp4 cpu N={n} K={k}: simd rel {es:.2e}, scalar rel {ec:.2e}, simd−scalar {agree:.2e}");
            // Ошибка — от int8-кванта активации (~0,4 %), не от веса.
            assert!(es < 1e-2 && ec < 1e-2, "N={n} K={k}");
            assert!(agree < 1e-3 * norm.max(1.0), "N={n} K={k}: simd и скаляр разошлись на {agree}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use rayon::prelude::*;

    /// Скорость на формах экспертов Qwen4Exp из DRAM: 192 эксперта (~530 МБ,
    /// больше L3), «слой» — 3 эксперта: gate_up [1280, 2560] → silu·up →
    /// down [2560, 640]. `cargo test --release -p synaptix-kernels-cpu
    /// nvfp4dot::bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn expert_layer_throughput() {
        let (h, i, ne) = (2560usize, 640usize, 192usize);
        let gu_bytes = 2 * i * h / 2;
        let dn_bytes = h * i / 2;
        let gu_sc = nvfp4_sf_inner_dim(h) * 2 * i;
        let dn_sc = nvfp4_sf_inner_dim(i) * h;
        let mut rng = 1u64;
        let mut fill = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    // масштабы E4M3 в разумном диапазоне, полубайты — любые
                    (rng >> 24) as u8 & 0x77
                })
                .collect()
        };
        let gu: Vec<u8> = fill(gu_bytes * ne);
        let dn: Vec<u8> = fill(dn_bytes * ne);
        let gs: Vec<u8> = fill(gu_sc * ne);
        let ds: Vec<u8> = fill(dn_sc * ne);
        let x: Vec<f32> = (0..h).map(|j| ((j * 37 % 101) as f32 - 50.0) / 50.0).collect();
        for per_layer in [2usize, 3, 4] {
            let layers = 400;
            let t = std::time::Instant::now();
            let mut sink = 0f32;
            for l in 0..layers {
                let ids: Vec<usize> = (0..per_layer).map(|j| (l * 7 + j * 61) % ne).collect();
                let act = ActQ::new(&x);
                // фаза 1: gate_up всех экспертов, тайлы по 4
                let mut y = vec![0f32; per_layer * 2 * i];
                let tiles = 2 * i / 16;
                y.par_chunks_mut(64).enumerate().for_each(|(ci, out)| {
                    let e = ci * 4 / tiles;
                    let mb0 = (ci * 4) % tiles;
                    let id = ids[e];
                    gemv_shuf(&gu[id * gu_bytes..][..gu_bytes], &gs[id * gu_sc..][..gu_sc], 2 * i, &act, mb0, mb0 + 4, out);
                });
                // silu·up и фаза 2: down
                let mut acts = Vec::with_capacity(per_layer);
                for e in 0..per_layer {
                    let g = &y[e * 2 * i..e * 2 * i + i];
                    let u = &y[e * 2 * i + i..(e + 1) * 2 * i];
                    let hh: Vec<f32> = g.iter().zip(u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
                    acts.push(ActQ::new(&hh));
                }
                let mut z = vec![0f32; per_layer * h];
                let tiles = h / 16;
                z.par_chunks_mut(128).enumerate().for_each(|(ci, out)| {
                    let e = ci * 8 / tiles;
                    let mb0 = (ci * 8) % tiles;
                    let id = ids[e];
                    gemv_shuf(&dn[id * dn_bytes..][..dn_bytes], &ds[id * dn_sc..][..dn_sc], h, &acts[e], mb0, mb0 + 8, out);
                });
                sink += z[0];
            }
            let dt = t.elapsed().as_secs_f64() / layers as f64;
            let bytes = per_layer * (gu_bytes + dn_bytes + gu_sc + dn_sc);
            eprintln!(
                "{per_layer} эксперта на слой: {:.0} мкс, {:.1} ГБ/с (sink {sink:.1})",
                dt * 1e6,
                bytes as f64 / dt / 1e9
            );
        }
    }
}
