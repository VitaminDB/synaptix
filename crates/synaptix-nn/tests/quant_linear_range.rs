use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;

const N: usize = 128;
const K: usize = 128;
const M: usize = 8;

fn cuda() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let d = Device::Cuda(0);
    Tensor::zeros(vec![1], DType::F32, d).ok().map(|_| d)
}

fn weight(dev: Device) -> Tensor {
    let w: Vec<f32> = (0..N * K)
        .map(|i| ((i * 37 % 211) as f32 / 211.0 - 0.5) * 0.25)
        .collect();
    Tensor::from_vec(w, vec![N, K], dev).unwrap().to_dtype(DType::BF16).unwrap()
}

fn activation(dev: Device, amax: f32) -> Tensor {
    let mut v: Vec<f32> = (0..M * K)
        .map(|i| ((i * 53 % 97) as f32 / 97.0 - 0.5) * 2.0)
        .collect();
    v[0] = amax;
    v[M * K - 1] = -amax;
    Tensor::from_vec(v, vec![M, K], dev).unwrap().to_dtype(DType::BF16).unwrap()
}

fn max_abs(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .unwrap()
        .into_iter()
        .fold(0.0f32, |a, b| a.max(b.abs()))
}

fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
    let va = a.to_dtype(DType::F32).and_then(|x| x.flatten_all()).and_then(|x| x.to_vec1::<f32>()).unwrap();
    let vb = b.to_dtype(DType::F32).and_then(|x| x.flatten_all()).and_then(|x| x.to_vec1::<f32>()).unwrap();
    let scale = vb.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    va.iter().zip(&vb).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn quant_linear_survives_activations_beyond_f16_range() {
    let Some(dev) = cuda() else { return };
    let w = weight(dev);
    let dense = QuantLinear::dense(w.clone(), None).unwrap();
    let q = QuantLinear::build(w, None, DType::NVFP4, DType::BF16).unwrap();
    assert!(q.is_quant(), "NVFP4 не собрался на {N}x{K}");

    let mut errs = Vec::new();
    for amax in [1.0e3f32, 6.0e4, 5.4e5, 4.0e6, 1.0e8] {
        let x = activation(dev, amax);
        let y = q.forward(&x).unwrap();
        let peak = max_abs(&y);
        assert!(
            peak.is_finite() && peak > 0.0,
            "amax {amax:.1e}: выход не конечен (peak {peak}) — активация переполнила F16"
        );
        let e = rel_err(&y, &dense.forward(&x).unwrap());
        eprintln!("[quant-range] amax {amax:.1e} · peak {peak:.3e} · rel {e:.4}");
        errs.push(e);
    }

    let lo = errs.iter().cloned().fold(f32::MAX, f32::min);
    let hi = errs.iter().cloned().fold(0.0f32, f32::max);
    assert!(hi < 0.5, "ошибка квантования {hi:.3} слишком велика даже для входа с выбросом");
    assert!(
        hi <= lo * 1.5 + 1e-6,
        "ошибка зависит от масштаба входа ({lo:.4}..{hi:.4}) — предмасштабирование должно быть точным"
    );
}

#[test]
fn prescale_does_not_change_small_activations() {
    let Some(dev) = cuda() else { return };
    let w = weight(dev);
    let x = activation(dev, 1.0);
    let q = QuantLinear::build(w, None, DType::NVFP4, DType::BF16).unwrap();
    let a = q.forward(&x).unwrap();
    let b = q.forward(&x).unwrap();
    assert_eq!(rel_err(&a, &b), 0.0, "путь без масштабирования недетерминирован");
}
