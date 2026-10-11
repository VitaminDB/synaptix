use synaptix_asr_whisper::{WhisperEncoder, WhisperWeights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn run(path: &str, dev: Device, dt: DType, layer: usize, x: &[f32], d: usize) -> R<Vec<(&'static str, Vec<f32>)>> {
    let w = WhisperWeights::open(path, dev, dt)?;
    let enc = WhisperEncoder::load(&w)?;
    let xt = Tensor::from_vec(x.to_vec(), (1, x.len() / d, d), dev)?.to_dtype(dt)?;
    let mut out = Vec::new();
    for (n, t) in enc.trace_layer(layer, &xt)? {
        out.push((n, t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?));
    }
    Ok(out)
}

fn main() -> R<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (path, layer, input) = (&a[0], a[1].parse::<usize>()?, &a[2]);
    let rows: Vec<usize> = a[3].split(',').map(|s| s.parse().unwrap()).collect();
    let x: Vec<f32> = std::fs::read(input)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let d = 1280;
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_npu::register()?;
    let cpu = run(path, Device::Cpu, DType::F32, layer, &x, d)?;
    let npu = run(path, Device::Npu(0), DType::F16, layer, &x, d)?;
    for ((n, c), (_, p)) in cpu.iter().zip(&npu) {
        let w = c.len() / (x.len() / d);
        let (mut md, mut mx, mut at) = (0f32, 0f32, (0, 0));
        let mut cos_min = 1f64;
        for &r in &rows {
            let (cr, pr) = (&c[r * w..(r + 1) * w], &p[r * w..(r + 1) * w]);
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for j in 0..w {
                let e = (cr[j] - pr[j]).abs();
                if e > md {
                    md = e;
                    at = (r, j);
                }
                mx = mx.max(cr[j].abs());
                dot += cr[j] as f64 * pr[j] as f64;
                na += (cr[j] as f64).powi(2);
                nb += (pr[j] as f64).powi(2);
            }
            cos_min = cos_min.min(dot / (na.sqrt() * nb.sqrt()).max(1e-30));
        }
        let (r, j) = at;
        println!("{n:5}: cos≥{cos_min:.5} max|x| {mx:8.2} max|Δ| {md:8.3} (строка {r}, канал {j}: CPU {:.3}, NPU {:.3})", c[r * w + j], p[r * w + j]);
    }
    Ok(())
}
