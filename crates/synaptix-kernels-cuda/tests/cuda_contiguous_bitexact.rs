//! `contiguous()` обязан копировать байты, а не пропускать их через
//! арифметику. Раньше strided-копия на CUDA шла через `x * 1 + 0`, и `-0.0`
//! превращался в `+0.0`: срез стопки экспертов, нарезанный на GPU, расходился
//! с тем же срезом, нарезанным на CPU, а квант из бандла — с квантом,
//! посчитанным на лету.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered as ensure_cpu;
use synaptix_kernels_cuda::ensure_registered as ensure_cuda;

fn setup() -> bool {
    ensure_cpu();
    ensure_cuda();
    synaptix_core::device::cuda::get(0).is_ok()
}

/// Сырые биты тензора с хоста.
fn bits16(t: &Tensor) -> Vec<u16> {
    let cpu = t.to_device(Device::Cpu).unwrap();
    let storage = cpu.storage_arc();
    storage
        .as_cpu()
        .unwrap()
        .as_bytes()
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bits32(t: &Tensor) -> Vec<u32> {
    let cpu = t.to_device(Device::Cpu).unwrap();
    let storage = cpu.storage_arc();
    storage
        .as_cpu()
        .unwrap()
        .as_bytes()
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `[E, N, K]` из F16-битов: у эксперта `e` нулевой элемент — `-0.0`.
fn stack_f16() -> (Vec<u8>, usize, usize, usize) {
    let (e, n, k) = (3usize, 2usize, 4usize);
    let mut raw: Vec<u8> = Vec::with_capacity(e * n * k * 2);
    for i in 0..e * n * k {
        let bits: u16 = if i % (n * k) == 0 { 0x8000 } else { 0x3C00 + (i as u16 % 16) };
        raw.extend_from_slice(&bits.to_le_bytes());
    }
    (raw, e, n, k)
}

#[test]
fn narrowed_slice_keeps_the_sign_of_zero() {
    if !setup() {
        return;
    }
    let (raw, e, n, k) = stack_f16();
    let cpu_t = Tensor::from_raw_slice(&raw, vec![e, n, k], DType::F16, Device::Cpu).unwrap();
    let gpu_t = cpu_t.to_device(Device::Cuda(0)).unwrap();

    // Срез 0 лежит в начале буфера и копируется memcpy'ем; срезы дальше идут
    // через strided-путь — раньше расходились именно они.
    for i in 0..e {
        let on_cpu = bits16(&cpu_t.narrow(0, i, 1).unwrap().contiguous().unwrap());
        let on_gpu = bits16(&gpu_t.narrow(0, i, 1).unwrap().contiguous().unwrap());
        assert_eq!(on_cpu, on_gpu, "срез {i}: CUDA-копия разошлась с CPU");
        assert_eq!(on_gpu[0], 0x8000, "срез {i}: -0.0 стал {:#06x}", on_gpu[0]);
    }
}

#[test]
fn transposed_copy_keeps_the_sign_of_zero_f32() {
    if !setup() {
        return;
    }
    // Транспонирование — второй strided-путь той же копии.
    let vals: Vec<f32> = vec![-0.0, 1.0, 2.0, -0.0, 3.0, 4.0];
    let cpu_t = Tensor::from_vec(vals, (2usize, 3usize), Device::Cpu).unwrap();
    let gpu_t = cpu_t.to_device(Device::Cuda(0)).unwrap();

    let on_cpu = bits32(&cpu_t.transpose(0, 1).unwrap().contiguous().unwrap());
    let on_gpu = bits32(&gpu_t.transpose(0, 1).unwrap().contiguous().unwrap());
    assert_eq!(on_cpu, on_gpu, "транспонированная CUDA-копия разошлась с CPU");
    assert_eq!(on_gpu[0], 0x8000_0000, "-0.0 стал {:#010x}", on_gpu[0]);
}

/// Affine остаётся affine: identity-op не должен был подменить умножение.
#[test]
fn affine_still_scales() {
    if !setup() {
        return;
    }
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3usize,), Device::Cuda(0)).unwrap();
    let y = t.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap();
    let got = y.to_device(Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got, vec![3.0, 5.0, 7.0]);
}
