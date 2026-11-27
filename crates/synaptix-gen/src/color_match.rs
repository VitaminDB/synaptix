//! Подгонка цвета на стыке роликов, снятых с латентного хвоста.
//!
//! Кадры перекрытия нового ролика — тот же латент, что последние кадры
//! предыдущего, но VAE декодирует их в другом окружении, и яркость/оттенок
//! чуть уходят (заметный скачок на склейке). Хвост хранит средние и разброс
//! цвета последних кадров ролика; новый ролик сравнивает с ними свои кадры
//! перекрытия и подстраивает начало, плавно отпуская подгонку за `ramp` кадров.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Средние и стандартные отклонения R, G, B по кадрам `[3,H,W]` (0..1).
pub fn stats(frames: &[Tensor]) -> R<Option<[f32; 6]>> {
    let (mut s, mut q, mut n) = ([0f64; 3], [0f64; 3], 0f64);
    for f in frames {
        let d = f.dims();
        if d.len() != 3 || d[0] != 3 {
            return Ok(None);
        }
        let hw = d[1] * d[2];
        let v = f.to_device(Device::Cpu).map_err(err)?.to_dtype(DType::F32).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
        for c in 0..3 {
            for &x in &v[c * hw..(c + 1) * hw] {
                s[c] += x as f64;
                q[c] += (x as f64) * (x as f64);
            }
        }
        n += hw as f64;
    }
    if n == 0.0 {
        return Ok(None);
    }
    let mut out = [0f32; 6];
    for c in 0..3 {
        let m = s[c] / n;
        out[c] = m as f32;
        out[3 + c] = (q[c] / n - m * m).max(0.0).sqrt() as f32;
    }
    Ok(Some(out))
}

/// Масштаб и сдвиг канала `c` при весе подгонки `w` (0 — без изменений).
pub fn channel_affine(src: &[f32; 6], dst: &[f32; 6], c: usize, w: f32) -> (f32, f32) {
    let g = if src[3 + c] > 1e-4 { (dst[3 + c] / src[3 + c]).clamp(0.8, 1.25) } else { 1.0 };
    (1.0 + w * (g - 1.0), w * (dst[c] - src[c] * g))
}

/// Вес подгонки кадра `i`: 1 в начале, плавно к 0 к кадру `ramp`.
pub fn weight(i: usize, ramp: usize) -> f32 {
    if ramp == 0 || i >= ramp {
        return 0.0;
    }
    let t = i as f32 / ramp as f32;
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// Подогнать начало ролика: цвет `src` (кадры перекрытия нового ролика) → `dst`
/// (конец предыдущего), с затуханием за `ramp` кадров.
pub fn match_start(frames: &mut [Tensor], src: &[f32; 6], dst: &[f32; 6], ramp: usize) -> R<()> {
    for (i, f) in frames.iter_mut().enumerate().take(ramp) {
        let w = weight(i, ramp);
        if w <= 0.0 {
            break;
        }
        let mut ch = Vec::with_capacity(3);
        for c in 0..3 {
            let (a, b) = channel_affine(src, dst, c, w);
            ch.push(f.narrow(0, c, 1).map_err(err)?.mul_scalar(a).map_err(err)?.add_scalar(b).map_err(err)?);
        }
        let refs: Vec<&Tensor> = ch.iter().collect();
        *f = Tensor::cat(&refs, 0).map_err(err)?.clamp(0.0, 1.0).map_err(err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_maps_src_to_dst() {
        let src = [0.40, 0.5, 0.5, 0.10, 0.1, 0.1];
        let dst = [0.42, 0.49, 0.5, 0.11, 0.1, 0.1];
        let (a, b) = channel_affine(&src, &dst, 0, 1.0);
        assert!((0.40 * a + b - 0.42).abs() < 1e-6);
        let (a, b) = channel_affine(&src, &dst, 0, 0.0);
        assert_eq!((a, b), (1.0, 0.0));
    }

    #[test]
    fn weight_ramps_down() {
        assert_eq!(weight(0, 48), 1.0);
        assert!(weight(24, 48) > 0.4 && weight(24, 48) < 0.6);
        assert_eq!(weight(48, 48), 0.0);
    }

    #[test]
    fn stats_and_match_on_frames() {
        synaptix_kernels_cpu::ensure_registered();
        let f = Tensor::from_vec(vec![0.4f32; 3 * 4], vec![3, 2, 2], Device::Cpu).unwrap();
        let s = stats(std::slice::from_ref(&f)).unwrap().unwrap();
        assert!((s[0] - 0.4).abs() < 1e-6 && s[3] < 1e-6);
        let mut fr = vec![f];
        match_start(&mut fr, &s, &[0.45, 0.4, 0.4, 0.0, 0.0, 0.0], 10).unwrap();
        let v = fr[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 0.45).abs() < 1e-5);
    }
}
