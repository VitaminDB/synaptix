//! Canny edge detection — препроцессор control-сигналов (IC-LoRA union-control:
//! контурная карта ведёт видео-генерацию, как ComfyUI Canny-нода).
//!
//! Классический пайплайн: grayscale (BT.601) → Gaussian 5×5 (σ=1.4) → Sobel 3×3 →
//! non-maximum suppression (4 направления) → двойной порог → гистерезис (слабые
//! рёбра живут только рядом с сильными). Пороги `low`/`high` — доли от
//! max-нормированной величины градиента (типично 0.1/0.2).

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::error::{IoError, Result};

/// Canny по серому полю `gray` `[h·w]` (значения [0,1]) → бинарные рёбра `[h·w]`
/// (0.0 / 1.0). `low`/`high` ∈ (0,1] — пороги относительно max градиента.
pub fn canny_gray(gray: &[f32], h: usize, w: usize, low: f32, high: f32) -> Vec<f32> {
    assert_eq!(gray.len(), h * w);
    // 1) Gaussian 5×5, σ≈1.4 (нормированное ядро), separable 1D [1,4,7,4,1]-подобное
    let k: [f32; 5] = {
        let sigma = 1.4f32;
        let mut k = [0f32; 5];
        let mut s = 0f32;
        for (i, kv) in k.iter_mut().enumerate() {
            let x = i as f32 - 2.0;
            *kv = (-x * x / (2.0 * sigma * sigma)).exp();
            s += *kv;
        }
        for kv in k.iter_mut() {
            *kv /= s;
        }
        k
    };
    let clamp = |v: isize, n: usize| -> usize { v.clamp(0, n as isize - 1) as usize };
    // горизонтальный проход
    let mut tmp = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (i, kv) in k.iter().enumerate() {
                let xx = clamp(x as isize + i as isize - 2, w);
                acc += kv * gray[y * w + xx];
            }
            tmp[y * w + x] = acc;
        }
    }
    // вертикальный проход
    let mut blur = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (i, kv) in k.iter().enumerate() {
                let yy = clamp(y as isize + i as isize - 2, h);
                acc += kv * tmp[yy * w + x];
            }
            blur[y * w + x] = acc;
        }
    }
    // 2) Sobel
    let mut mag = vec![0f32; h * w];
    let mut dir = vec![0u8; h * w]; // 0:гориз, 1:45°, 2:верт, 3:135°
    let mut max_mag = 0f32;
    for y in 0..h {
        for x in 0..w {
            let p = |dy: isize, dx: isize| -> f32 {
                blur[clamp(y as isize + dy, h) * w + clamp(x as isize + dx, w)]
            };
            let gx = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
            let m = (gx * gx + gy * gy).sqrt();
            mag[y * w + x] = m;
            if m > max_mag {
                max_mag = m;
            }
            // направление → 4 бина (угол градиента)
            let ang = gy.atan2(gx).to_degrees(); // [-180,180]
            let a = if ang < 0.0 { ang + 180.0 } else { ang }; // [0,180)
            dir[y * w + x] = if !(22.5..157.5).contains(&a) {
                0 // 0° → ребро вертикально, соседи слева/справа
            } else if a < 67.5 {
                1
            } else if a < 112.5 {
                2
            } else {
                3
            };
        }
    }
    if max_mag <= 0.0 {
        return vec![0f32; h * w];
    }
    // 3) non-maximum suppression
    let mut nms = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let m = mag[y * w + x];
            let (d1, d2): ((isize, isize), (isize, isize)) = match dir[y * w + x] {
                0 => ((0, -1), (0, 1)),
                1 => ((-1, 1), (1, -1)),
                2 => ((-1, 0), (1, 0)),
                _ => ((-1, -1), (1, 1)),
            };
            let n1 = mag[clamp(y as isize + d1.0, h) * w + clamp(x as isize + d1.1, w)];
            let n2 = mag[clamp(y as isize + d2.0, h) * w + clamp(x as isize + d2.1, w)];
            if m >= n1 && m >= n2 {
                nms[y * w + x] = m;
            }
        }
    }
    // 4) двойной порог + 5) гистерезис (DFS от сильных)
    let (tl, th) = (low * max_mag, high * max_mag);
    let mut out = vec![0u8; h * w]; // 0 нет, 1 слабое, 2 сильное
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..h * w {
        if nms[i] >= th {
            out[i] = 2;
            stack.push(i);
        } else if nms[i] >= tl {
            out[i] = 1;
        }
    }
    while let Some(i) = stack.pop() {
        let (y, x) = (i / w, i % w);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let (yy, xx) = (y as isize + dy, x as isize + dx);
                if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if out[j] == 1 {
                    out[j] = 2;
                    stack.push(j);
                }
            }
        }
    }
    out.iter().map(|&v| if v == 2 { 1.0 } else { 0.0 }).collect()
}

/// Canny по RGB-кадру `[3,H,W]` ([0,1]) → рёбра как RGB `[3,H,W]` (белые контуры
/// на чёрном, [0,1]). Тензор любого устройства; вычисление на CPU.
pub fn canny_rgb(frame: &Tensor, low: f32, high: f32) -> Result<Tensor> {
    let dims = frame.dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(IoError::Image(format!("canny ждёт [3,H,W], получено {dims:?}")));
    }
    let (h, w) = (dims[1], dims[2]);
    let dev = frame.device();
    let v: Vec<f32> = frame
        .to_dtype(DType::F32)
        .map_err(IoError::Core)?
        .flatten_all()
        .map_err(IoError::Core)?
        .to_vec1()
        .map_err(IoError::Core)?;
    // grayscale BT.601
    let mut gray = vec![0f32; h * w];
    for i in 0..h * w {
        gray[i] = 0.299 * v[i] + 0.587 * v[h * w + i] + 0.114 * v[2 * h * w + i];
    }
    let edges = canny_gray(&gray, h, w, low, high);
    let mut rgb = vec![0f32; 3 * h * w];
    rgb[..h * w].copy_from_slice(&edges);
    rgb[h * w..2 * h * w].copy_from_slice(&edges);
    rgb[2 * h * w..].copy_from_slice(&edges);
    Tensor::from_vec(rgb, vec![3, h, w], Device::Cpu)
        .and_then(|t| t.to_device(dev))
        .map_err(IoError::Core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canny_finds_square_edges() {
        // белый квадрат 8..24 на чёрном 32×32 → рёбра по периметру, не внутри
        let (h, w) = (32usize, 32usize);
        let mut g = vec![0f32; h * w];
        for y in 8..24 {
            for x in 8..24 {
                g[y * w + x] = 1.0;
            }
        }
        let e = canny_gray(&g, h, w, 0.1, 0.2);
        let total: f32 = e.iter().sum();
        assert!(total > 30.0, "слишком мало рёбер: {total}");
        // центр квадрата и дальний фон — без рёбер
        assert_eq!(e[16 * w + 16], 0.0);
        assert_eq!(e[2 * w + 2], 0.0);
        // окрестность границы (y=8) содержит ребро
        let near: f32 = (7..10).map(|y| e[y * w + 16]).sum();
        assert!(near > 0.0, "нет ребра у границы");
    }
}
