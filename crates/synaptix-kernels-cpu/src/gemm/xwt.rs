//! `out = x·wᵀ` для Linear: вес `[N, K]` читается как есть, без транспонирования
//! и упаковки (раньше `blocked::linear` перекладывал весь вес скалярным циклом на
//! каждом вызове — у OmniVoice это была треть времени синтеза).
//!
//! Микроядро — блок R строк `x` × C строк `w` (до 4×3: 12 ymm-аккумуляторов + 4
//! загрузки `x` + 3 загрузки `w` = 19 значений, но загрузки `w` сразу уходят в FMA,
//! так что живых регистров ≤ 16), скалярные произведения по 8 элементов K, в конце —
//! горизонтальные суммы. Вес F32/F16/BF16 переводится в F32 при загрузке (F16C /
//! сдвиг для BF16) — памяти читается вдвое меньше, чем у F32-веса.
//!
//! Кэш: K режется на куски `KC` (строки `x` куска — в L1, полоса весов — в L2),
//! частичные суммы копятся в локальном F32-тайле задачи. Задача — `RB` строк × `CB`
//! колонок, их достаточно много для 24 потоков и каждая — десятки микросекунд.

use rayon::prelude::*;

/// Строки `x` и `w` в микроядре.
const MR: usize = 4;
const NR: usize = 3;
/// Тайл задачи: строки × колонки.
const RB: usize = 32;
const CB: usize = 48;
/// Кусок K.
const KC: usize = 512;

/// Тип веса для [`linear`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WType {
    F32,
    F16,
    BF16,
}

impl WType {
    fn bytes(self) -> usize {
        match self {
            WType::F32 => 4,
            WType::F16 | WType::BF16 => 2,
        }
    }
}

/// Есть ли быстрый путь на этой машине (AVX2 + FMA + F16C).
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[derive(Clone, Copy)]
struct Ptr<T>(*const T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
#[derive(Clone, Copy)]
struct PtrMut(*mut f32);
unsafe impl Send for PtrMut {}
unsafe impl Sync for PtrMut {}

/// `out[m, n] = Σ_k x[m, k]·w[n, k]`. `x` — F32 `[m, k]` сплошняком, `w` — байты
/// веса `[n, k]` типа `wt`, `out` — F32 `[m, n]`. Вызывать только при [`available`].
pub fn linear(m: usize, n: usize, k: usize, x: &[f32], w: &[u8], wt: WType, out: &mut [f32]) {
    assert!(out.len() >= m * n);
    // SAFETY: длина `out` проверена выше.
    unsafe { linear_ld(m, n, k, x, w, wt, out.as_mut_ptr(), n) }
}

/// Как [`linear`], но строки выхода — с шагом `ldo` (полоса колонок большего
/// выхода). `out` должен вмещать `(m − 1)·ldo + n` элементов.
///
/// # Safety
/// `out` указывает на буфер не меньше `(m − 1)·ldo + n` F32, который никто
/// не читает и не пишет на время вызова.
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_ld(m: usize, n: usize, k: usize, x: &[f32], w: &[u8], wt: WType, out: *mut f32, ldo: usize) {
    assert!(x.len() >= m * k && w.len() >= n * k * wt.bytes() && ldo >= n);
    if m == 0 || n == 0 {
        return;
    }
    let (xp, wp, op) = (Ptr(x.as_ptr()), Ptr(w.as_ptr()), PtrMut(out));
    let (rt, ct) = (m.div_ceil(RB), n.div_ceil(CB));
    (0..rt * ct).into_par_iter().for_each(move |t| {
        // Колонки — внешний индекс: соседние задачи делят полосу весов.
        let (cb, rb) = (t / rt, t % rt);
        let (r0, c0) = (rb * RB, cb * CB);
        let (rows, cols) = (RB.min(m - r0), CB.min(n - c0));
        // SAFETY: задачи пишут непересекающиеся тайлы `out`, входы только читаются.
        unsafe { task(xp, wp, wt, op, k, ldo, r0, rows, c0, cols) };
    });
}

#[allow(clippy::too_many_arguments)]
unsafe fn task(x: Ptr<f32>, w: Ptr<u8>, wt: WType, out: PtrMut, k: usize, n: usize, r0: usize, rows: usize, c0: usize, cols: usize) {
    let mut acc = [0f32; RB * CB];
    let mut k0 = 0;
    while k0 < k {
        let kc = KC.min(k - k0);
        let mut i = 0;
        while i < rows {
            let r = MR.min(rows - i);
            let mut j = 0;
            while j < cols {
                let c = NR.min(cols - j);
                let xs = x.0.add((r0 + i) * k + k0);
                let ws = w.0.add(((c0 + j) * k + k0) * wt.bytes());
                let dst = acc.as_mut_ptr().add(i * CB + j);
                block(r, c, xs, k, ws, k * wt.bytes(), wt, kc, dst, CB);
                j += c;
            }
            i += r;
        }
        k0 += kc;
    }
    for i in 0..rows {
        std::ptr::copy_nonoverlapping(acc.as_ptr().add(i * CB), out.0.add((r0 + i) * n + c0), cols);
    }
}

/// Блок `r × c` (r ≤ 4, c ≤ 3): `dst[i·ldd + j] += Σ x[i]·w[j]` по `kc` элементам.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn block(r: usize, c: usize, x: *const f32, ldx: usize, w: *const u8, ldw: usize, wt: WType, kc: usize, dst: *mut f32, ldd: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! go {
            ($R:literal, $C:literal) => {
                match wt {
                    WType::F32 => avx::kern::<$R, $C, 0>(x, ldx, w, ldw, kc, dst, ldd),
                    WType::F16 => avx::kern::<$R, $C, 1>(x, ldx, w, ldw, kc, dst, ldd),
                    WType::BF16 => avx::kern::<$R, $C, 2>(x, ldx, w, ldw, kc, dst, ldd),
                }
            };
        }
        match (r, c) {
            (4, 3) => go!(4, 3),
            (4, 2) => go!(4, 2),
            (4, 1) => go!(4, 1),
            (3, 3) => go!(3, 3),
            (3, 2) => go!(3, 2),
            (3, 1) => go!(3, 1),
            (2, 3) => go!(2, 3),
            (2, 2) => go!(2, 2),
            (2, 1) => go!(2, 1),
            (1, 3) => go!(1, 3),
            (1, 2) => go!(1, 2),
            _ => go!(1, 1),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (r, c, x, ldx, w, ldw, wt, kc, dst, ldd);
        unreachable!("xwt::linear без AVX2");
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    /// 8 элементов веса в F32: `W` = 0 — F32, 1 — F16, 2 — BF16.
    #[inline(always)]
    unsafe fn load_w<const W: u8>(p: *const u8, idx: usize) -> __m256 {
        match W {
            0 => _mm256_loadu_ps((p as *const f32).add(idx)),
            1 => _mm256_cvtph_ps(_mm_loadu_si128((p as *const u16).add(idx) as *const __m128i)),
            _ => {
                let h = _mm_loadu_si128((p as *const u16).add(idx) as *const __m128i);
                _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
            }
        }
    }

    #[inline(always)]
    unsafe fn w_scalar<const W: u8>(p: *const u8, idx: usize) -> f32 {
        match W {
            0 => *(p as *const f32).add(idx),
            1 => half::f16::from_bits(*(p as *const u16).add(idx)).to_f32(),
            _ => half::bf16::from_bits(*(p as *const u16).add(idx)).to_f32(),
        }
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn kern<const R: usize, const C: usize, const W: u8>(
        x: *const f32,
        ldx: usize,
        w: *const u8,
        ldw: usize,
        kc: usize,
        dst: *mut f32,
        ldd: usize,
    ) {
        let wsz = if W == 0 { 4 } else { 2 };
        let ldw_el = ldw / wsz;
        let mut acc = [[_mm256_setzero_ps(); C]; R];
        let k8 = kc / 8 * 8;
        let mut p = 0;
        while p < k8 {
            let mut wv = [_mm256_setzero_ps(); C];
            for (j, v) in wv.iter_mut().enumerate() {
                *v = load_w::<W>(w, j * ldw_el + p);
            }
            for (i, row) in acc.iter_mut().enumerate() {
                let xv = _mm256_loadu_ps(x.add(i * ldx + p));
                for (a, wj) in row.iter_mut().zip(wv.iter()) {
                    *a = _mm256_fmadd_ps(xv, *wj, *a);
                }
            }
            p += 8;
        }
        for (i, row) in acc.iter().enumerate() {
            for (j, a) in row.iter().enumerate() {
                let mut s = hsum(*a);
                for q in k8..kc {
                    s += *x.add(i * ldx + q) * w_scalar::<W>(w, j * ldw_el + q);
                }
                *dst.add(i * ldd + j) += s;
            }
        }
    }
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
    fn matches_naive_all_types() {
        if !available() {
            return;
        }
        let mut seed = 7;
        for &(m, n, k) in &[(1, 1, 1), (1, 7, 9), (5, 13, 17), (33, 50, 515), (64, 97, 1030), (3, 200, 64), (130, 49, 8)] {
            let x = rnd(m * k, &mut seed);
            let wf = rnd(n * k, &mut seed);
            for wt in [WType::F32, WType::F16, WType::BF16] {
                let (bytes, wq): (Vec<u8>, Vec<f32>) = match wt {
                    WType::F32 => (bytemuck::cast_slice(&wf).to_vec(), wf.clone()),
                    WType::F16 => {
                        let h: Vec<half::f16> = wf.iter().map(|&v| half::f16::from_f32(v)).collect();
                        (bytemuck::cast_slice(&h).to_vec(), h.iter().map(|v| v.to_f32()).collect())
                    }
                    WType::BF16 => {
                        let h: Vec<half::bf16> = wf.iter().map(|&v| half::bf16::from_f32(v)).collect();
                        (bytemuck::cast_slice(&h).to_vec(), h.iter().map(|v| v.to_f32()).collect())
                    }
                };
                let mut out = vec![f32::NAN; m * n];
                linear(m, n, k, &x, &bytes, wt, &mut out);
                for i in 0..m {
                    for j in 0..n {
                        let want: f32 = (0..k).map(|p| x[i * k + p] * wq[j * k + p]).sum();
                        let got = out[i * n + j];
                        assert!((got - want).abs() < 1e-3 * (1.0 + want.abs()), "{wt:?} {m}x{n}x{k} [{i},{j}]: {got} vs {want}");
                    }
                }
            }
        }
    }
}
