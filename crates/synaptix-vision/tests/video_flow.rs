use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_vision::{optical_flow_farneback, optical_flow_raft, warp_with_flow};

fn setup() {
    ensure_registered();
}

/// Синусоидальная картинка — хорошо для проверки optical flow, т.к. имеет
/// различимые градиенты во всех направлениях.
fn make_sine_image(h: usize, w: usize, freq_x: f32, freq_y: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let val = 0.5
                + 0.4
                    * (2.0 * std::f32::consts::PI * freq_x * x as f32).sin()
                    * (2.0 * std::f32::consts::PI * freq_y * y as f32).cos();
            v.push(val);
        }
    }
    v
}

/// Сдвиг картинки на целое (dx, dy) с краевыми пикселями (replicate).
fn shift_image(src: &[f32], h: usize, w: usize, dx: i32, dy: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as i32 - dx).max(0).min(w as i32 - 1)) as usize;
            let sy = ((y as i32 - dy).max(0).min(h as i32 - 1)) as usize;
            out[y * w + x] = src[sy * w + sx];
        }
    }
    out
}

#[test]
fn warp_with_zero_flow_is_identity() {
    setup();
    let h = 8;
    let w = 8;
    let data: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
    let frame = Tensor::from_vec(data.clone(), (1, 1, h, w), Device::Cpu).unwrap();
    let flow = Tensor::zeros((1, 2, h, w), synaptix_core::dtype::DType::F32, Device::Cpu).unwrap();
    let out = warp_with_flow(&frame, &flow).unwrap();
    let out_vec = out.reshape((h * w,)).unwrap().to_vec1::<f32>().unwrap();
    for i in 0..h * w {
        assert!((out_vec[i] - data[i]).abs() < 1e-6, "mismatch at {i}");
    }
}

#[test]
fn warp_with_constant_flow_shifts_pattern() {
    setup();
    let h = 16;
    let w = 16;
    // Pattern с outer-зоной нулей, чтобы избежать clamping на границах.
    let mut data = vec![0.0f32; h * w];
    for y in 4..12 {
        for x in 4..12 {
            data[y * w + x] = 1.0;
        }
    }
    let frame = Tensor::from_vec(data.clone(), (1, 1, h, w), Device::Cpu).unwrap();

    // flow = (+2, +1)  ⇒  out(y, x) = src(y + 1, x + 2): «окно» сдвигается
    // влево-вверх на (2, 1).
    let mut flow_buf = vec![0.0f32; 2 * h * w];
    let dx_off = 0;
    let dy_off = h * w;
    for i in 0..h * w {
        flow_buf[dx_off + i] = 2.0;
        flow_buf[dy_off + i] = 1.0;
    }
    let flow = Tensor::from_vec(flow_buf, (1, 2, h, w), Device::Cpu).unwrap();
    let out = warp_with_flow(&frame, &flow).unwrap();
    let out_vec = out.reshape((h * w,)).unwrap().to_vec1::<f32>().unwrap();

    // Проверяем сдвинутый патч.
    for y in 3..11 {
        for x in 2..10 {
            assert!(
                (out_vec[y * w + x] - 1.0).abs() < 1e-6,
                "expected 1.0 at ({y},{x}), got {}",
                out_vec[y * w + x]
            );
        }
    }
}

#[test]
fn farneback_recovers_constant_translation() {
    setup();
    let h = 64;
    let w = 64;
    // Разные частоты по осям — иначе A_xx == A_yy и Гессиан вырожден
    // (det(AᵀA) ≡ 0, поток восстановить нельзя).
    let src = make_sine_image(h, w, 1.0 / 10.0, 1.0 / 16.0);
    // Сдвиг pattern'а: frame2 = shift(frame1, +2, +1) ⇒ flow в семантике
    // OpenCV (frame1(p) ≈ frame2(p + d)) равен (+2, +1).
    let shifted = shift_image(&src, h, w, 2, 1);

    let f1 = Tensor::from_vec(src, (1, 1, h, w), Device::Cpu).unwrap();
    let f2 = Tensor::from_vec(shifted, (1, 1, h, w), Device::Cpu).unwrap();

    let flow = optical_flow_farneback(&f1, &f2, 0.5, 3, 13, 5).unwrap();
    let flow_v = flow.reshape((2 * h * w,)).unwrap().to_vec1::<f32>().unwrap();
    let dx = &flow_v[0..h * w];
    let dy = &flow_v[h * w..2 * h * w];

    // Средние во внутренней зоне (отбрасываем границы winsize/2).
    let pad = 12;
    let mut sum_dx = 0.0f32;
    let mut sum_dy = 0.0f32;
    let mut cnt = 0usize;
    for y in pad..h - pad {
        for x in pad..w - pad {
            sum_dx += dx[y * w + x];
            sum_dy += dy[y * w + x];
            cnt += 1;
        }
    }
    let mean_dx = sum_dx / cnt as f32;
    let mean_dy = sum_dy / cnt as f32;
    let err_dx = (mean_dx - 2.0).abs();
    let err_dy = (mean_dy - 1.0).abs();
    assert!(
        err_dx < 0.5 && err_dy < 0.5,
        "expected mean flow ≈ (2.0, 1.0), got ({mean_dx:.3}, {mean_dy:.3})"
    );
}

#[test]
fn farneback_zero_flow_on_identical_frames() {
    setup();
    let h = 32;
    let w = 32;
    let src = make_sine_image(h, w, 1.0 / 8.0, 1.0 / 8.0);
    let f1 = Tensor::from_vec(src.clone(), (1, 1, h, w), Device::Cpu).unwrap();
    let f2 = Tensor::from_vec(src, (1, 1, h, w), Device::Cpu).unwrap();
    let flow = optical_flow_farneback(&f1, &f2, 0.5, 1, 11, 3).unwrap();
    let flow_v = flow.reshape((2 * h * w,)).unwrap().to_vec1::<f32>().unwrap();
    // Поток должен быть пренебрежимо мал везде.
    let pad = 8;
    for y in pad..h - pad {
        for x in pad..w - pad {
            let dx = flow_v[y * w + x];
            let dy = flow_v[h * w + y * w + x];
            assert!(dx.abs() < 1e-2 && dy.abs() < 1e-2, "non-zero flow ({dx},{dy}) at ({y},{x})");
        }
    }
}

#[test]
fn raft_returns_unsupported_error() {
    setup();
    let h = 8;
    let w = 8;
    let f1 = Tensor::zeros((1, 1, h, w), synaptix_core::dtype::DType::F32, Device::Cpu).unwrap();
    let f2 = Tensor::zeros((1, 1, h, w), synaptix_core::dtype::DType::F32, Device::Cpu).unwrap();
    let err = optical_flow_raft(&f1, &f2).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("optical_flow_raft") || msg.contains("RAFT"));
}
