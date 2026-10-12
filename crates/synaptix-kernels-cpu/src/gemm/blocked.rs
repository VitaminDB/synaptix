//! Блочный параллельный SGEMM поверх SIMD-микроядер (AVX2/FMA, NEON).
//!
//! Выход режется на тайлы `MB × NB`; каждый тайл — независимая rayon-задача,
//! которая вызывает микроядро `C += A·B` на своём прямоугольнике (C предварительно
//! обнулён). Для `linear` (`out = x·wᵀ`, вес `[N, K]`) задача сначала упаковывает
//! свою полосу весов в `[K, NB]` — транспонированная панель (K·NB·4 байт, ≈256 КБ
//! при K=1024) живёт в L2 и переиспользуется всеми строками тайла, вместо
//! перечитывания всей матрицы весов на каждую строку входа.

use rayon::prelude::*;

const MB: usize = 48;
const NB: usize = 64;

#[derive(Clone, Copy)]
pub struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline]
    pub fn new(p: *mut f32) -> SendPtr {
        SendPtr(p)
    }
    #[inline]
    pub fn ptr(self) -> *mut f32 {
        self.0
    }
    #[inline]
    pub fn at(self, off: usize) -> SendPtr {
        SendPtr(unsafe { self.0.add(off) })
    }
}

/// `C += A·B` на прямоугольнике (A — `[m, k]` со страйдом lda, B — `[k, n]`
/// со страйдом ldb, C — `[m, n]` со страйдом ldc).
#[inline]
fn microkernel(m: usize, n: usize, k: usize, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        super::avx2::sgemm_avx2(m, n, k, a, lda, b, ldb, c, ldc);
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::sgemm_neon(m, n, k, a, lda, b, ldb, c, ldc);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        super::avx2::sgemm_naive_fallback(m, n, k, a, lda, b, ldb, c, ldc);
    }
}

/// Строки микроядра (MR) и колонки (NR): тайлы дополняются до кратных, чтобы
/// микроядро не уходило в скалярный хвост.
#[cfg(target_arch = "x86_64")]
const MR: usize = super::avx2::MR;
#[cfg(target_arch = "x86_64")]
const NR: usize = super::avx2::NR;
#[cfg(not(target_arch = "x86_64"))]
const MR: usize = 8;
#[cfg(not(target_arch = "x86_64"))]
const NR: usize = 16;

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Один тайл: `out[rows × cols] = A[rows × k] · Bp[k × colsp]`, где `Bp` —
/// упакованная панель шириной `colsp` (кратна NR, хвост нулевой). Строки A
/// дополняются до кратных MR копией; результат пишется через локальный буфер.
#[allow(clippy::too_many_arguments)]
fn tile(rows: usize, cols: usize, k: usize, a: &[f32], lda: usize, bp: &[f32], colsp: usize, out: SendPtr, ldo: usize) {
    let rowsp = round_up(rows, MR);
    let apad: Vec<f32>;
    let (a_use, lda_use): (&[f32], usize) = if rowsp == rows {
        (a, lda)
    } else {
        let mut v = vec![0f32; rowsp * k];
        for r in 0..rows {
            v[r * k..(r + 1) * k].copy_from_slice(&a[r * lda..r * lda + k]);
        }
        apad = v;
        (&apad[..], k)
    };
    let mut cbuf = vec![0f32; rowsp * colsp];
    microkernel(rowsp, colsp, k, a_use, lda_use, bp, colsp, &mut cbuf, colsp);
    for r in 0..rows {
        // SAFETY: каждая задача пишет только свой непересекающийся тайл.
        unsafe {
            std::ptr::copy_nonoverlapping(cbuf.as_ptr().add(r * colsp), out.0.add(r * ldo), cols);
        }
    }
}

/// `C = A·B` (row-major, непрерывные): A `[m, k]`, B `[k, n]`, C `[m, n]`.
pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let cp = SendPtr(c.as_mut_ptr());
    (0..n.div_ceil(NB)).into_par_iter().for_each(|jb| {
        let c0 = jb * NB;
        let cols = NB.min(n - c0);
        let colsp = round_up(cols, NR);
        let mut bp = vec![0f32; k * colsp];
        for p in 0..k {
            bp[p * colsp..p * colsp + cols].copy_from_slice(&b[p * n + c0..p * n + c0 + cols]);
        }
        (0..m.div_ceil(MB)).into_par_iter().for_each(|ib| {
            let r0 = ib * MB;
            let rows = MB.min(m - r0);
            let out = cp.at(r0 * n + c0);
            tile(rows, cols, k, &a[r0 * k..], k, &bp, colsp, out, n);
        });
    });
}

pub fn linear(m: usize, n: usize, k: usize, x: &[f32], w: &[f32], out: &mut [f32]) {
    let op = SendPtr(out.as_mut_ptr());
    (0..n.div_ceil(NB)).into_par_iter().for_each(|jb| {
        let c0 = jb * NB;
        let cols = NB.min(n - c0);
        linear_panel(m, cols, k, x, &w[c0 * k..(c0 + cols) * k], op.at(c0), n);
    });
}

pub fn linear_panel(m: usize, cols: usize, k: usize, x: &[f32], w_rows: &[f32], out: SendPtr, ldo: usize) {
    let colsp = round_up(cols, NR);
    let mut bp = vec![0f32; k * colsp];
    for (jj, row) in w_rows[..cols * k].chunks_exact(k).enumerate() {
        for (p, &v) in row.iter().enumerate() {
            bp[p * colsp + jj] = v;
        }
    }
    (0..m.div_ceil(MB)).into_par_iter().for_each(|ib| {
        let r0 = ib * MB;
        let rows = MB.min(m - r0);
        tile(rows, cols, k, &x[r0 * k..], k, &bp, colsp, out.at(r0 * ldo), ldo);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                (*seed % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn sgemm_and_linear_match_naive() {
        let mut seed = 99;
        for &(m, n, k) in &[(1, 1, 1), (7, 13, 5), (64, 64, 64), (130, 70, 33), (200, 1024, 64), (9, 200, 300)] {
            let a = rnd(m * k, &mut seed);
            let b = rnd(k * n, &mut seed);
            let w: Vec<f32> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect(); // w = bᵀ
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
                }
            }
            let mut c = vec![f32::NAN; m * n];
            sgemm(m, n, k, &a, &b, &mut c);
            let mut l = vec![f32::NAN; m * n];
            linear(m, n, k, &a, &w, &mut l);
            for idx in 0..m * n {
                assert!((c[idx] - want[idx]).abs() < 1e-3 * (1.0 + want[idx].abs()), "sgemm {m}x{n}x{k} @{idx}");
                assert!((l[idx] - want[idx]).abs() < 1e-3 * (1.0 + want[idx].abs()), "linear {m}x{n}x{k} @{idx}");
            }
        }
    }
}
