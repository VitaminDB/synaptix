use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn rnd(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn param(v: &[f32], shape: &[usize], dev: Device, dt: DType) -> R<Tensor> {
    let t = synaptix_core::backend::with_param_upload(|| Tensor::from_vec(v.to_vec(), shape.to_vec(), dev))?;
    Ok(t.to_dtype(dt)?)
}

fn data(v: &[f32], shape: &[usize], dev: Device, dt: DType) -> R<Tensor> {
    Ok(Tensor::from_vec(v.to_vec(), shape.to_vec(), dev)?.to_dtype(dt)?)
}

fn worst_cos(a: &[f32], b: &[f32], row: usize) -> (f64, f32) {
    let mut worst = 1.0f64;
    let mut maxd = 0f32;
    for (ra, rb) in a.chunks(row).zip(b.chunks(row)) {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in ra.iter().zip(rb) {
            d += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
            maxd = maxd.max((x - y).abs());
        }
        let c = if na == 0.0 && nb == 0.0 { 1.0 } else { d / (na.sqrt() * nb.sqrt()).max(1e-30) };
        worst = worst.min(c);
    }
    (worst, maxd)
}

fn model(x: &Tensor, p: &[Tensor]) -> R<Tensor> {
    let (conv_w, conv_b, w1, b1, ln_w, ln_b, w2) = (&p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6]);
    let h = synaptix_ops::conv::conv1d(x, conv_w, Some(conv_b), 2, 1)?;
    let h = synaptix_ops::activation::gelu_exact(&h)?;
    let h = h.transpose(1, 2)?.contiguous()?;
    let h = h.narrow(1, 1, h.dims()[1] - 2)?;
    let h = synaptix_ops::norm::layer_norm(&h, Some(ln_w), Some(ln_b), 1e-5)?;
    let h = h.linear_bias_residual(w1, Some(b1), None)?;
    let d = h.dims().to_vec();
    let q = h.reshape(vec![d[0], d[1], 4, d[2] / 4])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
    let a = synaptix_ops::attention::softmax::scaled_dot_attention(&q, &q, &q, 0.25, None)?;
    let a = a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(d.clone())?;
    let y = a.linear_bias_residual(w2, None, Some(&h))?;
    Ok(y.mean_keepdim(2)?.broadcast_sub(&y)?.sqr()?.broadcast_add(&y)?)
}

fn run(dev: Device, dt: DType, b: usize, c: usize, l: usize, hdim: usize) -> R<Vec<f32>> {
    let mut s = 7u64;
    let x = data(&rnd(&mut s, b * c * l, 1.0), &[b, c, l], dev, dt)?;
    let p = vec![
        param(&rnd(&mut s, hdim * c * 3, 0.2), &[hdim, c, 3], dev, dt)?,
        param(&rnd(&mut s, hdim, 0.1), &[hdim], dev, dt)?,
        param(&rnd(&mut s, hdim * hdim, 1.0 / (hdim as f32).sqrt()), &[hdim, hdim], dev, dt)?,
        param(&rnd(&mut s, hdim, 0.1), &[hdim], dev, dt)?,
        param(&rnd(&mut s, hdim, 1.0).iter().map(|v| 1.0 + v * 0.1).collect::<Vec<_>>(), &[hdim], dev, dt)?,
        param(&rnd(&mut s, hdim, 0.1), &[hdim], dev, dt)?,
        param(&rnd(&mut s, hdim * hdim, 1.0 / (hdim as f32).sqrt()), &[hdim, hdim], dev, dt)?,
    ];
    let mut out = Vec::new();
    for i in 0..3 {
        let t = std::time::Instant::now();
        let y = model(&x, &p)?;
        let v = y.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        println!("  {dev:?} {dt:?} прогон {i}: {:.1} мс", t.elapsed().as_secs_f64() * 1e3);
        out = v;
    }
    Ok(out)
}

fn main() -> R<()> {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (b, c, l, h) = (*a.first().unwrap_or(&1), *a.get(1).unwrap_or(&80), *a.get(2).unwrap_or(&200), *a.get(3).unwrap_or(&256));
    synaptix_kernels_cpu::ensure_registered();
    let info = synaptix_kernels_npu::register()?;
    println!("NPU: {} ({})", info.model.name, info.model.id);
    let cpu = run(Device::Cpu, DType::F32, b, c, l, h)?;
    let npu = run(Device::Npu(0), DType::F16, b, c, l, h)?;
    let (cos, maxd) = worst_cos(&cpu, &npu, h);
    println!("{:?}", synaptix_kernels_npu::stats());
    println!("сверка CPU f32 ↔ NPU f16 по строкам: худший cos {cos:.6}, макс |Δ| {maxd:.4}");
    if cos < 0.999 {
        return Err("расхождение".into());
    }
    Ok(())
}
