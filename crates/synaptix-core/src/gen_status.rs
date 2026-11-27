//! Этап генерации (видео, картинки, музыка) для индикатора службы: снаружи
//! видны только шаги денойза, а долгие этапы — кодирование текста, загрузка
//! DiT, декодер VAE — проходили молча. Архитектура пишет этап и раскладку
//! блоков (сколько на карте); служба добавляет их к идущей работе. Одна
//! генерация на устройство — статус общий на процесс.

use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenStatus {
    /// Этап словами: «кодирование текста», «загрузка DiT», «денойз», «декодер VAE».
    pub phase: String,
    /// Блоки (слои) модели этапа: на карте и всего; `None` — неизвестно.
    pub blocks: Option<(u32, u32)>,
}

static STATUS: Mutex<Option<GenStatus>> = Mutex::new(None);

/// Превью генерации: маленькая картинка текущей оценки результата (RGB8,
/// строки сверху вниз) и шаг, на котором она снята.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preview {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub step: usize,
    pub total: usize,
}

static PREVIEW: Mutex<Option<Preview>> = Mutex::new(None);

/// Новое превью (прежнее, если его не забрали, заменяется).
pub fn set_preview(p: Preview) {
    *PREVIEW.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
}

/// Коэффициенты линейной проекции латента в RGB из байтов f32 LE: `3·(c+1)`
/// чисел (веса и смещение на канал цвета); другое число байт — `None`.
pub fn parse_factors(bytes: &[u8], c: usize) -> Option<Vec<f32>> {
    (bytes.len() == 3 * (c + 1) * 4).then(|| bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Коэффициенты блочной проекции латента в RGB: окно `kernel×kernel` токенов
/// (`1` или `3`) → блок `scale×scale` пикселей (`1`, `2`, `4`, `8`).
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewFactors {
    pub kernel: usize,
    pub scale: usize,
    pub weights: Vec<f32>,
}

/// Коэффициенты блочной проекции «окно `k×k` токенов → `scale×scale` пикселей»
/// из сырых байтов f32 LE: `3·scale²·(k²·c + 1)` чисел (`k ∈ {1, 3}`,
/// `scale ∈ {1, 2, 4, 8}`). Однозначно определяется по длине при любом `c ≥ 1`.
pub fn parse_block_factors(bytes: &[u8], c: usize) -> Option<PreviewFactors> {
    if c == 0 || bytes.is_empty() || bytes.len() % 4 != 0 {
        return None;
    }
    let floats = bytes.len() / 4;
    for kernel in [1usize, 3usize] {
        let stride = kernel * kernel * c + 1;
        for scale in [1usize, 2, 4, 8] {
            if floats == 3 * scale * scale * stride {
                let weights = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                return Some(PreviewFactors { kernel, scale, weights });
            }
        }
    }
    None
}

/// Каждые сколько шагов снимать превью (`SYN_GEN_PREVIEW_EVERY`, 0 — выключить).
pub fn preview_every(default: usize) -> usize {
    std::env::var("SYN_GEN_PREVIEW_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let ac = a.chunks_exact(4);
    let bc = b.chunks_exact(4);
    let ar = ac.remainder();
    let br = bc.remainder();
    for (x, y) in ac.zip(bc) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// Превью линейной проекцией: `x0` — `h·w` строк по `c` каналов (оценка
/// результата в латенте), `rgb = W·x + b` с обрезкой в [0, 1].
pub fn linear_preview(x0: &[f32], c: usize, width: usize, height: usize, factors: &[f32], step: usize, total: usize) {
    linear_preview_blocks(
        x0,
        c,
        width,
        height,
        &PreviewFactors { kernel: 1, scale: 1, weights: factors.to_vec() },
        step,
        total,
    );
}

/// Превью блочной проекцией («токен 1×1» или «окно 3×3 токенов» → блок
/// `scale×scale` пикселей): `x0` — `height·width` строк по `c` каналов,
/// результат — `(width·scale) × (height·scale)` RGB8.
#[allow(clippy::too_many_arguments)]
pub fn linear_preview_blocks(
    x0: &[f32],
    c: usize,
    width: usize,
    height: usize,
    factors: &PreviewFactors,
    step: usize,
    total: usize,
) {
    let kernel = factors.kernel;
    let scale = factors.scale;
    let in_dim = kernel * kernel * c;
    let stride = in_dim + 1;
    if (kernel != 1 && kernel != 3)
        || scale == 0
        || width == 0
        || height == 0
        || factors.weights.len() != 3 * scale * scale * stride
        || x0.len() != width * height * c
    {
        return;
    }
    let out_w = width * scale;
    let out_h = height * scale;
    let mut rgb = vec![0u8; out_w * out_h * 3];
    let mut win = if kernel == 3 { vec![0.0f32; in_dim] } else { Vec::new() };
    let weights = &factors.weights;
    for ty in 0..height {
        for tx in 0..width {
            let feat: &[f32] = if kernel == 1 {
                &x0[(ty * width + tx) * c..(ty * width + tx + 1) * c]
            } else {
                for ky in 0..3 {
                    let ny = (ty + ky).saturating_sub(1).min(height - 1);
                    for kx in 0..3 {
                        let nx = (tx + kx).saturating_sub(1).min(width - 1);
                        let src = (ny * width + nx) * c;
                        let dst = (ky * 3 + kx) * c;
                        win[dst..dst + c].copy_from_slice(&x0[src..src + c]);
                    }
                }
                &win
            };
            for dy in 0..scale {
                let row_off = ((ty * scale + dy) * out_w + tx * scale) * 3;
                for dx in 0..scale {
                    let out_off = row_off + dx * 3;
                    let base = (dy * scale + dx) * 3;
                    for ch in 0..3 {
                        let w = &weights[(base + ch) * stride..(base + ch + 1) * stride];
                        let y = w[in_dim] + dot_f32(feat, &w[..in_dim]);
                        rgb[out_off + ch] = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
    }
    set_preview(Preview { width: out_w, height: out_h, rgb, step, total });
}

/// Забрать превью, если появилось новое.
pub fn take_preview() -> Option<Preview> {
    PREVIEW.lock().unwrap_or_else(|e| e.into_inner()).take()
}

fn with(f: impl FnOnce(&mut Option<GenStatus>)) {
    f(&mut STATUS.lock().unwrap_or_else(|e| e.into_inner()));
}

/// Новый этап; раскладка блоков прошлого этапа сбрасывается.
pub fn phase(p: impl Into<String>) {
    let p = p.into();
    with(|s| *s = Some(GenStatus { phase: p, blocks: None }));
}

/// Раскладка блоков текущего этапа: `vram` на карте из `total`.
pub fn blocks(vram: usize, total: usize) {
    with(|s| s.get_or_insert_with(Default::default).blocks = Some((vram as u32, total as u32)));
}

pub fn clear() {
    with(|s| *s = None);
    let _ = take_preview();
}

pub fn get() -> Option<GenStatus> {
    STATUS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Снимает статус по выходу из генерации (в том числе по ошибке).
pub struct Scope;

impl Scope {
    pub fn new(first: impl Into<String>) -> Self {
        phase(first);
        Scope
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_and_scope() {
        {
            let _s = Scope::new("кодирование текста");
            blocks(36, 50);
            assert_eq!(get(), Some(GenStatus { phase: "кодирование текста".into(), blocks: Some((36, 50)) }));
            phase("денойз");
            assert_eq!(get().unwrap().blocks, None);
            // Проекция: один пиксель, два канала; R = x0 + x1, G = смещение 0.5, B = 0.
            linear_preview(&[0.25, 0.5], 2, 1, 1, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 3, 8);
            let p = take_preview().unwrap();
            assert_eq!((p.rgb, p.step, p.total), (vec![191, 128, 0], 3, 8));
        }
        assert_eq!(get(), None);
    }

    #[test]
    fn block_projection_2x2() {
        // 1 токен (c = 1, x = 0.5) → блок 2×2 пикселя с разными константами/весами.
        let mut f = vec![0.0f32; 3 * 4 * 2];
        // (dy=0, dx=0): R = 1.0 * x = 0.5 → 128
        f[0] = 1.0;
        // (dy=0, dx=1): G = 1.0 → 255
        f[(3 + 1) * 2 + 1] = 1.0;
        // (dy=1, dx=0): B = 0.5 * x + 0.25 = 0.5 → 128
        f[(6 + 2) * 2] = 0.5;
        f[(6 + 2) * 2 + 1] = 0.25;
        // (dy=1, dx=1): R = 0.25, G = 0.5, B = 1.0
        f[(9 + 0) * 2 + 1] = 0.25;
        f[(9 + 1) * 2 + 1] = 0.5;
        f[(9 + 2) * 2 + 1] = 1.0;

        let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        let parsed = parse_block_factors(&bytes, 1).unwrap();
        assert_eq!((parsed.kernel, parsed.scale), (1, 2));
        assert_eq!(parsed.weights, f);

        linear_preview_blocks(&[0.5], 1, 1, 1, &parsed, 1, 4);
        let p = take_preview().unwrap();
        assert_eq!((p.width, p.height, p.step, p.total), (2, 2, 1, 4));
        assert_eq!(p.rgb, vec![128, 0, 0, 0, 255, 0, 0, 0, 128, 64, 128, 255]);

        // Окно 3×3 (c = 1 → 10 чисел на выходной канал), 2×1 латент, scale = 1:
        // R берёт правого соседа (слот ky=1,kx=2 → индекс 5), G = центр (индекс 4), B = 0.
        let mut f3 = vec![0.0f32; 3 * 10];
        f3[5] = 1.0; // R = right neighbor
        f3[10 + 4] = 1.0; // G = center
        let bytes3: Vec<u8> = f3.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p3 = parse_block_factors(&bytes3, 1).unwrap();
        assert_eq!((p3.kernel, p3.scale), (3, 1));
        linear_preview_blocks(&[0.2, 0.8], 1, 2, 1, &p3, 2, 4);
        let out3 = take_preview().unwrap();
        // Токен (0,0): центр 0.2, правый сосед 0.8 → R=204, G=51, B=0.
        // Токен (0,1): центр 0.8, правый сосед (edge pad) 0.8 → R=204, G=204, B=0.
        assert_eq!(out3.rgb, vec![204, 51, 0, 204, 204, 0]);
    }
}
