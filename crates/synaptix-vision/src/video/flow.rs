//! Dense optical flow — Farnebäck (CPU, без deps) + bilinear backward warp.
//!
//! Вход/выход:
//! * `frame*` — `[B, C, H, W]` F32 на CPU; если `C > 1`, кадры усредняются по
//!   каналам в grayscale (Farnebäck работает по интенсивности).
//! * `flow`   — `[B, 2, H, W]` F32: канал 0 — `dx`, канал 1 — `dy`.
//!   Семантика как у OpenCV `calcOpticalFlowFarneback`:
//!   `frame1(y, x) ≈ frame2(y + dy, x + dx)`.
//!
//! Реализация Farnebäck следует постановке G. Farnebäck, *“Two-frame motion
//! estimation based on polynomial expansion”* (SCIA 2003) — sec. 4–5. На каждом
//! уровне пирамиды:
//!   1. Для каждого пикселя оба кадра аппроксимируются квадратичной формой
//!      `f(p) ≈ pᵀ A p + bᵀ p + c` через гауссо-взвешенный 2D полиномиальный
//!      разлет в окне `winsize`.
//!   2. Из условия `f₂(p + d) = f₁(p)` получаем систему `Ā · d = Δb̄`, где
//!      `Ā(p) = (A₁(p) + A₂(p + d_prev)) / 2`, `Δb̄(p) = -(b₂(p + d_prev) − b₁(p))/2`.
//!   3. Регуляризация — box-filter усреднение элементов нормальных уравнений
//!      `AᵀA` / `Aᵀb` по окну `winsize`, после чего 2×2 систему решаем
//!      аналитически.
//!   4. Поток уточняется `iterations` раз, затем апскейлится на следующий
//!      уровень пирамиды.
//!
//! RAFT здесь не реализован — он требует обученного чекпоинта.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

/// Dense Farnebäck optical flow. Возвращает `[B, 2, H, W]` F32.
///
/// * `pyr_scale` — коэффициент уменьшения между уровнями пирамиды (типично 0.5).
/// * `levels`    — число уровней пирамиды; `1` — без пирамиды.
/// * `winsize`   — размер окна (нечётный, ≥ 3); используется и для полиномиального
///   разлета, и для усреднения нормальных уравнений.
/// * `iterations` — число итераций уточнения потока на каждом уровне.
pub fn optical_flow_farneback(
    frame1: &Tensor,
    frame2: &Tensor,
    pyr_scale: f32,
    levels: usize,
    winsize: usize,
    iterations: usize,
) -> Result<Tensor> {
    if frame1.dtype() != DType::F32 || frame2.dtype() != DType::F32 {
        return Err(SynaptixError::Other(
            "optical_flow_farneback: только F32".into(),
        ));
    }
    if frame1.device() != Device::Cpu || frame2.device() != Device::Cpu {
        return Err(SynaptixError::Other(
            "optical_flow_farneback: только CPU".into(),
        ));
    }
    if frame1.dims() != frame2.dims() {
        return Err(SynaptixError::shape_mismatch(frame1.dims(), frame2.dims()));
    }
    let (b, c, h, w) = frame1.dims4()?;
    if h < 4 || w < 4 {
        return Err(SynaptixError::Other(format!(
            "optical_flow_farneback: слишком маленькое разрешение {h}×{w}"
        )));
    }
    let pyr_scale = pyr_scale.clamp(0.05, 0.95);
    let levels = levels.max(1);
    let winsize = (winsize.max(3)) | 1; // odd
    let iterations = iterations.max(1);

    // Хост-буферы кадров → grayscale на батч.
    let f1 = frame1.contiguous()?.reshape((b * c * h * w,))?.to_vec1::<f32>()?;
    let f2 = frame2.contiguous()?.reshape((b * c * h * w,))?.to_vec1::<f32>()?;

    let mut out_flow = vec![0.0f32; b * 2 * h * w];

    for bi in 0..b {
        let off = bi * c * h * w;
        let g1 = to_grayscale(&f1[off..off + c * h * w], c, h, w);
        let g2 = to_grayscale(&f2[off..off + c * h * w], c, h, w);

        let (dx, dy) = farneback_pyramid(&g1, &g2, h, w, pyr_scale, levels, winsize, iterations);

        // [2, H, W] записываем в выходной батч
        let out_off = bi * 2 * h * w;
        out_flow[out_off..out_off + h * w].copy_from_slice(&dx);
        out_flow[out_off + h * w..out_off + 2 * h * w].copy_from_slice(&dy);
    }

    Tensor::from_vec(out_flow, (b, 2, h, w), Device::Cpu)
}

/// RAFT optical flow — требует обученного чекпоинта (encoder + context net + GRU);
/// без него корректного потока вернуть нельзя. Возвращаем явный
/// `Unsupported`, чтобы caller подключил веса через отдельный pathway.
pub fn optical_flow_raft(_frame1: &Tensor, _frame2: &Tensor) -> Result<Tensor> {
    Err(SynaptixError::Unsupported(
        "optical_flow_raft: требует обученного чекпоинта (RAFT encoder + GRU); используйте optical_flow_farneback для bottom-up CPU-варианта",
    ))
}

/// Backward warp кадра по полю потока через bilinear sampling.
///
/// `frame: [B, C, H, W]` F32, `flow: [B, 2, H, W]` F32 (канал 0 = `dx`, 1 = `dy`).
/// Граница — clamp-to-edge (как `cv2.remap(..., BORDER_REPLICATE)`).
///
/// Возвращает `[B, C, H, W]` F32: `out(y, x) = sample(frame, y + dy, x + dx)`.
pub fn warp_with_flow(frame: &Tensor, flow: &Tensor) -> Result<Tensor> {
    if frame.dtype() != DType::F32 || flow.dtype() != DType::F32 {
        return Err(SynaptixError::Other("warp_with_flow: только F32".into()));
    }
    if frame.device() != Device::Cpu || flow.device() != Device::Cpu {
        return Err(SynaptixError::Other("warp_with_flow: только CPU".into()));
    }
    let (b, c, h, w) = frame.dims4()?;
    let (bf, two, hf, wf) = flow.dims4()?;
    if bf != b || two != 2 || hf != h || wf != w {
        return Err(SynaptixError::shape_mismatch(&[b, 2, h, w], flow.dims()));
    }

    let frame_host = frame.contiguous()?.reshape((b * c * h * w,))?.to_vec1::<f32>()?;
    let flow_host = flow.contiguous()?.reshape((b * 2 * h * w,))?.to_vec1::<f32>()?;

    let mut out = vec![0.0f32; b * c * h * w];

    for bi in 0..b {
        let frame_off = bi * c * h * w;
        let flow_off = bi * 2 * h * w;
        let out_off = bi * c * h * w;
        let dx = &flow_host[flow_off..flow_off + h * w];
        let dy = &flow_host[flow_off + h * w..flow_off + 2 * h * w];

        for ch in 0..c {
            let f_ch = &frame_host[frame_off + ch * h * w..frame_off + (ch + 1) * h * w];
            let o_ch = &mut out[out_off + ch * h * w..out_off + (ch + 1) * h * w];
            for y in 0..h {
                for x in 0..w {
                    let sx = x as f32 + dx[y * w + x];
                    let sy = y as f32 + dy[y * w + x];
                    o_ch[y * w + x] = sample_bilinear_clamp(f_ch, h, w, sx, sy);
                }
            }
        }
    }

    Tensor::from_vec(out, (b, c, h, w), Device::Cpu)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn to_grayscale(src: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    if c == 1 {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; h * w];
    let inv_c = 1.0 / c as f32;
    for ch in 0..c {
        let plane = &src[ch * h * w..(ch + 1) * h * w];
        for i in 0..h * w {
            out[i] += plane[i] * inv_c;
        }
    }
    out
}

#[inline]
fn sample_bilinear_clamp(img: &[f32], h: usize, w: usize, sx: f32, sy: f32) -> f32 {
    let x0f = sx.floor();
    let y0f = sy.floor();
    let fx = sx - x0f;
    let fy = sy - y0f;
    let x0 = (x0f as isize).clamp(0, w as isize - 1) as usize;
    let y0 = (y0f as isize).clamp(0, h as isize - 1) as usize;
    let x1 = ((x0f as isize) + 1).clamp(0, w as isize - 1) as usize;
    let y1 = ((y0f as isize) + 1).clamp(0, h as isize - 1) as usize;
    let v00 = img[y0 * w + x0];
    let v01 = img[y0 * w + x1];
    let v10 = img[y1 * w + x0];
    let v11 = img[y1 * w + x1];
    let v0 = v00 * (1.0 - fx) + v01 * fx;
    let v1 = v10 * (1.0 - fx) + v11 * fx;
    v0 * (1.0 - fy) + v1 * fy
}

#[inline]
fn sample_bilinear_clamp_3<const N: usize>(
    planes: &[&[f32]; N],
    h: usize,
    w: usize,
    sx: f32,
    sy: f32,
) -> [f32; N] {
    let x0f = sx.floor();
    let y0f = sy.floor();
    let fx = sx - x0f;
    let fy = sy - y0f;
    let x0 = (x0f as isize).clamp(0, w as isize - 1) as usize;
    let y0 = (y0f as isize).clamp(0, h as isize - 1) as usize;
    let x1 = ((x0f as isize) + 1).clamp(0, w as isize - 1) as usize;
    let y1 = ((y0f as isize) + 1).clamp(0, h as isize - 1) as usize;
    let mut out = [0.0f32; N];
    for k in 0..N {
        let v00 = planes[k][y0 * w + x0];
        let v01 = planes[k][y0 * w + x1];
        let v10 = planes[k][y1 * w + x0];
        let v11 = planes[k][y1 * w + x1];
        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;
        out[k] = v0 * (1.0 - fy) + v1 * fy;
    }
    out
}

// ---------------------------------------------------------------------------
// pyramid
// ---------------------------------------------------------------------------

// Размер окна полиномиального разложения. Должен быть значительно меньше
// длины волны сигнала, иначе квадратичная аппроксимация бессмысленна. Это
// соответствует параметру `poly_n` в OpenCV; мы держим его маленьким и
// независимым от `winsize` (который служит окном регуляризации).
const POLY_N: usize = 5;

/// Ограничивает `winsize` так, чтобы он влезал в картинку (нечётный, ≥ 3,
/// ≤ min(h, w) / 2 · 2 + 1).
fn level_winsize(winsize: usize, h: usize, w: usize) -> usize {
    let limit = (h.min(w) / 2 * 2).saturating_sub(1).max(3);
    let w_clamped = winsize.min(limit);
    (w_clamped | 1).max(3)
}

fn farneback_pyramid(
    f1: &[f32],
    f2: &[f32],
    h: usize,
    w: usize,
    pyr_scale: f32,
    levels: usize,
    winsize: usize,
    iterations: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Собираем пирамиду (level 0 = original, последний = самый грубый).
    let mut pyr1: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(levels);
    let mut pyr2: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(levels);
    pyr1.push((f1.to_vec(), h, w));
    pyr2.push((f2.to_vec(), h, w));
    // Нижний уровень должен оставаться достаточно большим, чтобы окно
    // регуляризации `winsize` имело смысл (иначе box-фильтр на маленькой
    // картинке клеймит всё на границу и AᵀA «вырождается»).
    let min_dim = (winsize * 2).max(16);
    for _ in 1..levels {
        let (prev1, ph, pw) = pyr1.last().unwrap();
        let new_h = ((*ph as f32) * pyr_scale).round() as usize;
        let new_w = ((*pw as f32) * pyr_scale).round() as usize;
        if new_h < min_dim || new_w < min_dim {
            break;
        }
        // gaussian blur + downsample
        let blurred1 = gaussian_blur(prev1, *ph, *pw, 1.0);
        let blurred2 = gaussian_blur(&pyr2.last().unwrap().0, *ph, *pw, 1.0);
        let down1 = downsample_bilinear(&blurred1, *ph, *pw, new_h, new_w);
        let down2 = downsample_bilinear(&blurred2, *ph, *pw, new_h, new_w);
        pyr1.push((down1, new_h, new_w));
        pyr2.push((down2, new_h, new_w));
    }

    // Снизу-вверх: начинаем с самого грубого.
    let (top1, top_h, top_w) = pyr1.last().unwrap();
    let (top2, _, _) = pyr2.last().unwrap();
    let mut dx = vec![0.0f32; top_h * top_w];
    let mut dy = vec![0.0f32; top_h * top_w];
    let top_winsize = level_winsize(winsize, *top_h, *top_w);
    farneback_single_level(top1, top2, *top_h, *top_w, top_winsize, iterations, &mut dx, &mut dy);

    // Поднимаемся, апскейля поток (значения тоже масштабируются по 1/pyr_scale).
    for level in (0..pyr1.len() - 1).rev() {
        let (_, prev_h, prev_w) = pyr1[level + 1];
        let (_, cur_h, cur_w) = pyr1[level];
        let sx_ratio = cur_w as f32 / prev_w as f32;
        let sy_ratio = cur_h as f32 / prev_h as f32;
        dx = upsample_flow(&dx, prev_h, prev_w, cur_h, cur_w, sx_ratio);
        dy = upsample_flow(&dy, prev_h, prev_w, cur_h, cur_w, sy_ratio);
        let (cf1, ch_l, cw_l) = &pyr1[level];
        let (cf2, _, _) = &pyr2[level];
        let lw = level_winsize(winsize, *ch_l, *cw_l);
        farneback_single_level(cf1, cf2, *ch_l, *cw_l, lw, iterations, &mut dx, &mut dy);
    }

    (dx, dy)
}

// ---------------------------------------------------------------------------
// single-level Farnebäck
// ---------------------------------------------------------------------------

fn farneback_single_level(
    f1: &[f32],
    f2: &[f32],
    h: usize,
    w: usize,
    winsize: usize,
    iterations: usize,
    dx: &mut Vec<f32>,
    dy: &mut Vec<f32>,
) {
    // Polynomial expansion с маленьким окном POLY_N — отдельно от регуляризационного.
    let r1 = poly_expansion(f1, h, w, POLY_N);
    let r2 = poly_expansion(f2, h, w, POLY_N);

    for _iter in 0..iterations {
        // Соберём 5 каналов нормальных уравнений: m0..m4.
        let mut m = vec![0.0f32; 5 * h * w];
        let r2_bx = &r2[0 * h * w..1 * h * w];
        let r2_by = &r2[1 * h * w..2 * h * w];
        let r2_axx = &r2[2 * h * w..3 * h * w];
        let r2_ayy = &r2[3 * h * w..4 * h * w];
        let r2_axy = &r2[4 * h * w..5 * h * w];

        let planes2: [&[f32]; 5] = [r2_bx, r2_by, r2_axx, r2_ayy, r2_axy];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let cur_dx = dx[idx];
                let cur_dy = dy[idx];

                // Сэмплируем коэффициенты f2 в (x+dx, y+dy).
                let sx = x as f32 + cur_dx;
                let sy = y as f32 + cur_dy;
                let s = sample_bilinear_clamp_3::<5>(&planes2, h, w, sx, sy);
                let (b2x, b2y, a2xx, a2yy, a2xy) = (s[0], s[1], s[2], s[3], s[4]);

                let b1x = r1[0 * h * w + idx];
                let b1y = r1[1 * h * w + idx];
                let a1xx = r1[2 * h * w + idx];
                let a1yy = r1[3 * h * w + idx];
                let a1xy = r1[4 * h * w + idx];

                // Ā = (A1 + A2)/2
                let axx = 0.5 * (a1xx + a2xx);
                let ayy = 0.5 * (a1yy + a2yy);
                let axy = 0.5 * (a1xy + a2xy);

                // Уравнение: H̃ · d_new = (b₁ - b₂_sampled) + H̃ · d_old,
                // где H̃ = (H₁ + H₂_sampled)/2 — усреднённый Гессиан, b — линейный
                // коэффициент полинома. Здесь axx/ayy/axy уже содержат
                // полный Гессиан (`2·c₃, 2·c₄, c₅`), поэтому множителя 0.5
                // на (b₁ - b₂) нет.
                let dbx = (b1x - b2x) + axx * cur_dx + axy * cur_dy;
                let dby = (b1y - b2y) + axy * cur_dx + ayy * cur_dy;

                // Нормальные уравнения AᵀA · d = Aᵀ δb.
                m[0 * h * w + idx] = axx * axx + axy * axy;
                m[1 * h * w + idx] = axx * axy + axy * ayy;
                m[2 * h * w + idx] = axy * axy + ayy * ayy;
                m[3 * h * w + idx] = axx * dbx + axy * dby;
                m[4 * h * w + idx] = axy * dbx + ayy * dby;
            }
        }

        // Регуляризация — box-фильтр по winsize на каждом канале.
        for ch in 0..5 {
            let filtered = box_filter(&m[ch * h * w..(ch + 1) * h * w], h, w, winsize);
            m[ch * h * w..(ch + 1) * h * w].copy_from_slice(&filtered);
        }

        // Решаем 2×2 на каждый пиксель.
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let g11 = m[0 * h * w + idx];
                let g12 = m[1 * h * w + idx];
                let g22 = m[2 * h * w + idx];
                let h1 = m[3 * h * w + idx];
                let h2 = m[4 * h * w + idx];
                // Регуляризация: добавляем небольшой λ·I к ĀᵀĀ. λ берём
                // относительным от диагональных энергий, чтобы быть инвариантным к
                // масштабу пикселей.
                let lambda = 1e-4 * (g11.max(g22)).max(1e-10);
                let g11r = g11 + lambda;
                let g22r = g22 + lambda;
                let det = g11r * g22r - g12 * g12;
                if det > 1e-30 {
                    let inv = 1.0 / det;
                    dx[idx] = (g22r * h1 - g12 * h2) * inv;
                    dy[idx] = (-g12 * h1 + g11r * h2) * inv;
                }
                // Иначе — поток оставляем прежним (полностью вырожденная область).
            }
        }
    }
}

// ---------------------------------------------------------------------------
// polynomial expansion
// ---------------------------------------------------------------------------

/// Гауссо-взвешенное полиномиальное разложение второго порядка.
/// Возвращает плотный массив `[5, H, W]` коэффициентов
/// `(b_x, b_y, 2·A_xx, 2·A_yy, 2·A_xy)`.
/// (Множители 2 учтены сразу — что соответствует второй производной
/// `∂²f/∂x² = 2·A_xx` и упрощает дальнейшие выкладки.)
fn poly_expansion(img: &[f32], h: usize, w: usize, winsize: usize) -> Vec<f32> {
    let half = (winsize / 2) as isize;
    let sigma = (winsize as f32) / 6.0 + 0.5; // эвристика OpenCV
    let two_sig2 = 2.0 * sigma * sigma;

    // Веса и моменты 1D базиса по индексу i ∈ [-half..=half].
    let n = (2 * half + 1) as usize;
    let mut wts = vec![0.0f32; n];
    let mut sum_w = 0.0f32;
    for k in 0..n {
        let i = k as isize - half;
        let v = (-((i * i) as f32) / two_sig2).exp();
        wts[k] = v;
        sum_w += v;
    }
    for v in wts.iter_mut() {
        *v /= sum_w;
    }

    // Моменты по 1D базису:  m0 = Σ w, m2 = Σ w·i², m4 = Σ w·i⁴.
    let mut s0 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s4 = 0.0f32;
    for k in 0..n {
        let i = (k as isize - half) as f32;
        s0 += wts[k];
        s2 += wts[k] * i * i;
        s4 += wts[k] * i * i * i * i;
    }

    // Матрица G для 2D-базиса {1, x, y, x², y², xy} в сепарабельной форме.
    // С учётом нулевых нечётных моментов отличны от нуля только:
    //   M[1,1] = M[2,2] = s0*s2
    //   M[5,5] = s2*s2
    //   M[3,3] = s0*s4    M[4,4] = s0*s4
    //   M[3,4] = M[4,3] = s2*s2
    //   M[0,0] = s0*s0    M[0,3] = M[3,0] = s0*s2    M[0,4] = M[4,0] = s0*s2
    // Отсюда:
    //   b_x = m_x / (s0·s2),  b_y = m_y / (s0·s2)
    //   A_xy = m_xy / (s2·s2)
    //   (a, A_xx, A_yy) — из 3×3 блока [{1, x², y²}], решаем явно ниже.
    let block = [
        [s0 * s0, s0 * s2, s0 * s2],
        [s0 * s2, s0 * s4, s2 * s2],
        [s0 * s2, s2 * s2, s0 * s4],
    ];
    let inv_block = invert_3x3(block);

    let inv_bxy = 1.0 / (s0 * s2);
    let inv_axy = 1.0 / (s2 * s2);

    // Считаем 6 моментов на пиксель через сепарабельные горизонтальные/вертикальные свёртки.
    // Шаг 1: по строкам — 3 канала: t0 = Σ_dx w·f, t1 = Σ_dx w·dx·f, t2 = Σ_dx w·dx²·f.
    let mut t0 = vec![0.0f32; h * w];
    let mut t1 = vec![0.0f32; h * w];
    let mut t2 = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            let mut a2 = 0.0f32;
            for k in 0..n {
                let dx = k as isize - half;
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                let v = img[y * w + xx];
                let wk = wts[k];
                a0 += wk * v;
                a1 += wk * (dx as f32) * v;
                a2 += wk * (dx as f32) * (dx as f32) * v;
            }
            t0[y * w + x] = a0;
            t1[y * w + x] = a1;
            t2[y * w + x] = a2;
        }
    }

    // Шаг 2: по столбцам с теми же весами — получаем 6 моментов.
    let mut m_1 = vec![0.0f32; h * w]; // Σ w(y) w(x) · f
    let mut m_x = vec![0.0f32; h * w]; // Σ w·x·f
    let mut m_y = vec![0.0f32; h * w]; // Σ w·y·f
    let mut m_x2 = vec![0.0f32; h * w];
    let mut m_y2 = vec![0.0f32; h * w];
    let mut m_xy = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut a_1 = 0.0f32;
            let mut a_x = 0.0f32;
            let mut a_y = 0.0f32;
            let mut a_x2 = 0.0f32;
            let mut a_y2 = 0.0f32;
            let mut a_xy = 0.0f32;
            for k in 0..n {
                let dy = k as isize - half;
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let wk = wts[k];
                let dyf = dy as f32;
                a_1 += wk * t0[yy * w + x];
                a_x += wk * t1[yy * w + x];
                a_y += wk * dyf * t0[yy * w + x];
                a_x2 += wk * t2[yy * w + x];
                a_y2 += wk * dyf * dyf * t0[yy * w + x];
                a_xy += wk * dyf * t1[yy * w + x];
            }
            m_1[y * w + x] = a_1;
            m_x[y * w + x] = a_x;
            m_y[y * w + x] = a_y;
            m_x2[y * w + x] = a_x2;
            m_y2[y * w + x] = a_y2;
            m_xy[y * w + x] = a_xy;
        }
    }

    // 5 каналов: b_x, b_y, A_xx (умножен на 2), A_yy (×2), A_xy (×2).
    let mut out = vec![0.0f32; 5 * h * w];
    let (bx_off, by_off, axx_off, ayy_off, axy_off) = (0, h * w, 2 * h * w, 3 * h * w, 4 * h * w);
    for i in 0..h * w {
        out[bx_off + i] = m_x[i] * inv_bxy;
        out[by_off + i] = m_y[i] * inv_bxy;
        // решаем 3×3 для (a, A_xx, A_yy) — нас интересуют только последние две.
        let bvec = [m_1[i], m_x2[i], m_y2[i]];
        let axx = inv_block[1][0] * bvec[0] + inv_block[1][1] * bvec[1] + inv_block[1][2] * bvec[2];
        let ayy = inv_block[2][0] * bvec[0] + inv_block[2][1] * bvec[1] + inv_block[2][2] * bvec[2];
        out[axx_off + i] = 2.0 * axx;
        out[ayy_off + i] = 2.0 * ayy;
        out[axy_off + i] = m_xy[i] * inv_axy; // уже соответствует ∂²f/∂x∂y
    }
    out
}

fn invert_3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv = 1.0 / det;
    let mut out = [[0.0f32; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv;
    out[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
    out[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
    out[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv;
    out[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
    out
}

// ---------------------------------------------------------------------------
// filters & resampling
// ---------------------------------------------------------------------------

fn box_filter(src: &[f32], h: usize, w: usize, winsize: usize) -> Vec<f32> {
    let half = (winsize / 2) as isize;
    let mut tmp = vec![0.0f32; h * w];
    // horizontal
    for y in 0..h {
        let mut acc = 0.0f32;
        for k in -half..=half {
            let xx = k.clamp(0, w as isize - 1) as usize;
            acc += src[y * w + xx];
        }
        tmp[y * w] = acc;
        for x in 1..w {
            let add_x = (x as isize + half).clamp(0, w as isize - 1) as usize;
            let rem_x = (x as isize - half - 1).clamp(0, w as isize - 1) as usize;
            acc += src[y * w + add_x] - src[y * w + rem_x];
            tmp[y * w + x] = acc;
        }
    }
    // vertical
    let mut out = vec![0.0f32; h * w];
    let win = (2 * half + 1) as f32;
    for x in 0..w {
        let mut acc = 0.0f32;
        for k in -half..=half {
            let yy = k.clamp(0, h as isize - 1) as usize;
            acc += tmp[yy * w + x];
        }
        out[x] = acc / (win * win);
        for y in 1..h {
            let add_y = (y as isize + half).clamp(0, h as isize - 1) as usize;
            let rem_y = (y as isize - half - 1).clamp(0, h as isize - 1) as usize;
            acc += tmp[add_y * w + x] - tmp[rem_y * w + x];
            out[y * w + x] = acc / (win * win);
        }
    }
    out
}

fn gaussian_blur(src: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let radius = ((sigma * 3.0).ceil() as isize).max(1);
    let n = (2 * radius + 1) as usize;
    let two_sig2 = 2.0 * sigma * sigma;
    let mut k = vec![0.0f32; n];
    let mut s = 0.0f32;
    for i in 0..n {
        let d = i as isize - radius;
        let v = (-((d * d) as f32) / two_sig2).exp();
        k[i] = v;
        s += v;
    }
    for v in k.iter_mut() {
        *v /= s;
    }

    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0f32;
            for i in 0..n {
                let dx = i as isize - radius;
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                a += k[i] * src[y * w + xx];
            }
            tmp[y * w + x] = a;
        }
    }
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0f32;
            for i in 0..n {
                let dy = i as isize - radius;
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                a += k[i] * tmp[yy * w + x];
            }
            out[y * w + x] = a;
        }
    }
    out
}

fn downsample_bilinear(src: &[f32], src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_h * dst_w];
    if dst_h == 0 || dst_w == 0 {
        return out;
    }
    let sx_ratio = src_w as f32 / dst_w as f32;
    let sy_ratio = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        let sy = (y as f32 + 0.5) * sy_ratio - 0.5;
        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * sx_ratio - 0.5;
            out[y * dst_w + x] = sample_bilinear_clamp(src, src_h, src_w, sx, sy);
        }
    }
    out
}

fn upsample_flow(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_h * dst_w];
    let sx_ratio = src_w as f32 / dst_w as f32;
    let sy_ratio = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        let sy = (y as f32 + 0.5) * sy_ratio - 0.5;
        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * sx_ratio - 0.5;
            out[y * dst_w + x] = sample_bilinear_clamp(src, src_h, src_w, sx, sy) * scale;
        }
    }
    out
}
