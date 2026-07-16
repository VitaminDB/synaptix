use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

fn lcg_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            u * scale
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "dump" => {
            let path = &args[2];
            let mut all: Vec<f32> = Vec::new();
            for (ci, co, d, h, w) in [
                (128usize, 128usize, 12usize, 22usize, 40usize),
                (256, 256, 9, 88, 160),
                (512, 512, 7, 44, 80),
                (128, 512, 5, 22, 40),
            ] {
                let x = Tensor::from_vec(lcg_fill(ci * d * h * w, 7, 1.0), vec![1, ci, d, h, w], Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(dev)?;
                let wt = Tensor::from_vec(
                    lcg_fill(co * ci * 27, 13, 0.05), vec![co, ci, 3, 3, 3], Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(dev)?;
                let bias = Tensor::from_vec(lcg_fill(co, 29, 0.5), vec![co], Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(dev)?;
                let y = x.conv3d(&wt, Some(&bias), (1, 1, 1), (1, 1, 1))?;
                let n = y.dims().iter().product::<usize>();
                let yf = y.to_device(Device::Cpu)?.to_dtype(DType::F32)?.reshape(vec![n])?;
                all.extend_from_slice(&yf.to_vec1::<f32>()?);
                eprintln!("форма {ci}->{co} d{d} {h}x{w}: out {:?}", y.dims());
            }
            let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, bytes)?;
            eprintln!("dump {} ({} значений)", path, all.len());
        }
        "cmp" => {
            let a: Vec<f32> = std::fs::read(&args[2])?
                .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let b: Vec<f32> = std::fs::read(&args[3])?
                .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(a.len(), b.len());
            let mut max_abs = 0f32;
            let mut max_at = 0usize;
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            let mut diff_cnt = 0usize;
            for i in 0..a.len() {
                let d = (a[i] - b[i]).abs();
                if d > max_abs { max_abs = d; max_at = i; }
                if d > 0.0 { diff_cnt += 1; }
                dot += a[i] as f64 * b[i] as f64;
                na += (a[i] as f64).powi(2);
                nb += (b[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            println!(
                "n={} diff_cnt={} ({:.3}%) max_abs={:.6} @{} (a={:.5} b={:.5}) cos={:.9}",
                a.len(), diff_cnt, 100.0 * diff_cnt as f64 / a.len() as f64,
                max_abs, max_at, a[max_at], b[max_at], cos
            );
        }
        "pn" => {
            for (c, d, h, w) in [(128usize, 4usize, 22usize, 40usize), (512, 7, 88, 160), (1024, 3, 22, 40), (1024, 14, 17, 22), (256, 5, 13, 7)] {
                let x = Tensor::from_vec(lcg_fill(c * d * h * w, 17, 2.0), vec![1, c, d, h, w], Device::Cpu)?
                    .to_dtype(DType::BF16)?.to_device(dev)?;
                for silu in [false, true] {
                    let yf = x.pixel_norm_fused(1e-8, silu)?;
                    let xf = x.to_dtype(DType::F32)?;
                    let ms = xf.sqr()?.mean_keepdim(1)?;
                    let den = ms.add_scalar(1e-8)?.sqrt()?;
                    let mut yr = xf.broadcast_div(&den)?.to_dtype(DType::BF16)?;
                    if silu { yr = yr.silu()?; }
                    let n = c * d * h * w;
                    let a = yf.to_device(Device::Cpu)?.to_dtype(DType::F32)?.reshape(vec![n])?.to_vec1::<f32>()?;
                    let b = yr.to_device(Device::Cpu)?.to_dtype(DType::F32)?.reshape(vec![n])?.to_vec1::<f32>()?;
                    let mut max_abs = 0f32;
                    let mut diff = 0usize;
                    for i in 0..n {
                        let dd = (a[i] - b[i]).abs();
                        if dd > max_abs { max_abs = dd; }
                        if dd > 0.0 { diff += 1; }
                    }
                    println!("pn c={c} {d}x{h}x{w} silu={silu}: diff={diff}/{n} max_abs={max_abs:.6}");
                }
            }
        }
        other => return Err(format!("режим {other}? dump|cmp|pn").into()),
    }
    Ok(())
}
