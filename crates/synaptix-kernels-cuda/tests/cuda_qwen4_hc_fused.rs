//! Слитый gated-residual Qwen4Exp (hc_down + hc_up) против плотного эталона
//! по деквантованным NVFP4-весам.
use synaptix_core::backend::HcLaunch;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn hc_fused_matches_dense_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !synaptix_kernels_cuda::caps::DeviceCaps::for_ordinal(0).map(|c| c.fp4_mma()).unwrap_or(false) {
        eprintln!("пропуск: перемешанная NVFP4-копия строится только с FP4 MMA");
        return;
    }
    let dev = Device::Cuda(0);
    let (hc, h, r) = (4usize, 2560usize, 320usize);
    let k = hc * h;
    for rows in [1usize, 2, 3] {
        let f16 = |v: Vec<f32>, dims: Vec<usize>| Tensor::from_vec(v, dims, Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
        let x = f16(noise(1 + rows as u64, rows * k, 3.0), vec![rows, k]);
        let norm_w = f16(noise(2, k, 0.5).iter().map(|v| 1.0 + v).collect(), vec![k]);
        let wd = f16(noise(3, r * k, 0.05), vec![r, k]);
        let wu = f16(noise(4, k * r, 0.2), vec![k, r]);
        let wi = f16(noise(5, hc * k, 0.05), vec![hc, k]);
        let qd = wd.quantize_to_nvfp4().unwrap();
        let qu = wu.quantize_to_nvfp4().unwrap();
        let (dd, du) = (host(&qd.dequantize(DType::F16).unwrap()), host(&qu.dequantize(DType::F16).unwrap()));
        qd.ensure_shuffled().unwrap();
        qu.ensure_shuffled().unwrap();
        let (dwa, dsa) = qd.shuffled_addr().unwrap();
        let (uwa, usa) = qu.shuffled_addr().unwrap();
        let mix = Tensor::zeros(vec![rows, r], DType::F32, dev).unwrap();
        let norms = Tensor::zeros(vec![rows, hc], DType::F32, dev).unwrap();
        let inj = Tensor::zeros(vec![rows, hc], DType::F16, dev).unwrap();
        let mixed = Tensor::zeros(vec![rows, h], DType::F16, dev).unwrap();
        let a = |t: &Tensor| t.device_address().unwrap();
        let args = HcLaunch {
            x: a(&x),
            norm_w: a(&norm_w),
            down_w: dwa,
            down_s: dsa,
            inj_w: a(&wi),
            up_w: uwa,
            up_s: usa,
            mix: a(&mix),
            norms: a(&norms),
            inj_out: a(&inj),
            mixed: a(&mixed),
            rows: rows as u32,
            hc: hc as u32,
            h: h as u32,
            r: r as u32,
            eps: 1e-6,
            _pad: 0.0,
        };
        synaptix_core::tensor::ops::qwen4_hc_fused(dev, &args).unwrap();
        if std::env::var("HC_BENCH").is_ok() {
            synaptix_core::device::cuda::synchronize(0).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..500 {
                synaptix_core::tensor::ops::qwen4_hc_fused(dev, &args).unwrap();
            }
            synaptix_core::device::cuda::synchronize(0).unwrap();
            eprintln!("hc_fused rows={rows}: {:.1} мкс на вызов (оба ядра)", t0.elapsed().as_secs_f64() * 1e6 / 500.0);
        }
        // Эталон на хосте (f64).
        let (xh, nwh, wih) = (host(&x), host(&norm_w), host(&wi));
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let (got_mixed, got_inj) = (host(&mixed), host(&inj));
        let (mut e_m, mut n_m, mut e_i, mut n_i) = (0f64, 0f64, 0f64, 0f64);
        for t in 0..rows {
            let xr = &xh[t * k..(t + 1) * k];
            let mut normed = vec![0f64; k];
            for g in 0..hc {
                let ss: f64 = xr[g * h..(g + 1) * h].iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let inv = 1.0 / (ss / h as f64 + 1e-6).sqrt();
                for i in 0..h {
                    normed[g * h + i] = xr[g * h + i] as f64 * inv * nwh[g * h + i] as f64;
                }
            }
            let m: Vec<f64> = (0..r)
                .map(|row| {
                    let v: f64 = (0..k).map(|j| dd[row * k + j] as f64 * normed[j]).sum::<f64>() / hc as f64;
                    v * sig(v)
                })
                .collect();
            for i in 0..h {
                let mut acc = 0f64;
                for c in 0..hc {
                    let row = c * h + i;
                    let up: f64 = (0..r).map(|j| du[row * r + j] as f64 * m[j]).sum();
                    acc += sig(up) * normed[row];
                }
                let want = acc / hc as f64;
                e_m += (got_mixed[t * h + i] as f64 - want).powi(2);
                n_m += want * want;
            }
            for c in 0..hc {
                let v: f64 = (0..k).map(|j| wih[c * k + j] as f64 * normed[j]).sum::<f64>() / hc as f64;
                let want = 2.0 * sig(v);
                e_i += (got_inj[t * hc + c] as f64 - want).powi(2);
                n_i += want * want;
            }
        }
        let (rm, ri) = ((e_m / n_m).sqrt(), (e_i / n_i).sqrt());
        eprintln!("hc_fused rows={rows}: mixed отн. {rm:.2e}, inject отн. {ri:.2e}");
        assert!(rm < 2e-3 && ri < 2e-3, "rows={rows}");
    }
}

#[test]
fn shared_expert_fused_matches_dense_reference() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if !synaptix_kernels_cuda::caps::DeviceCaps::for_ordinal(0).map(|c| c.fp4_mma()).unwrap_or(false) {
        return;
    }
    let dev = Device::Cuda(0);
    let (h, i) = (2560usize, 640usize);
    for rows in [1usize, 2] {
        let f16 = |v: Vec<f32>, dims: Vec<usize>| Tensor::from_vec(v, dims, Device::Cpu).unwrap().to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
        let x = f16(noise(11 + rows as u64, rows * h, 2.0), vec![rows, h]);
        let router = Tensor::from_vec(noise(12, h, 0.05), vec![1, h], dev).unwrap();
        let q = |seed: u64, n: usize, k: usize, sc: f32| {
            let w = f16(noise(seed, n * k, sc), vec![n, k]).quantize_to_nvfp4().unwrap();
            let d = host(&w.dequantize(DType::F16).unwrap());
            w.ensure_shuffled().unwrap();
            (w, d)
        };
        let (qg, dg) = q(13, i, h, 0.05);
        let (qu, du) = q(14, i, h, 0.05);
        let (qd, dd) = q(15, h, i, 0.1);
        let hb = Tensor::zeros(vec![rows, i], DType::F32, dev).unwrap();
        let sg = Tensor::zeros(vec![rows], DType::F32, dev).unwrap();
        let out = Tensor::zeros(vec![rows, h], DType::F16, dev).unwrap();
        let a = |t: &Tensor| t.device_address().unwrap();
        let ((gw, gs), (uw, us), (dw, ds)) = (qg.shuffled_addr().unwrap(), qu.shuffled_addr().unwrap(), qd.shuffled_addr().unwrap());
        let args = synaptix_core::backend::SeLaunch {
            x: a(&x), gate_w: gw, gate_s: gs, up_w: uw, up_s: us, down_w: dw, down_s: ds, router: a(&router),
            h: a(&hb), sg: a(&sg), out: a(&out), rows: rows as u32, hidden: h as u32, inter: i as u32, _pad: 0,
        };
        synaptix_core::tensor::ops::moe_shared_fused(dev, &args).unwrap();
        if std::env::var("HC_BENCH").is_ok() {
            synaptix_core::device::cuda::synchronize(0).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..500 {
                synaptix_core::tensor::ops::moe_shared_fused(dev, &args).unwrap();
            }
            synaptix_core::device::cuda::synchronize(0).unwrap();
            eprintln!("se_fused rows={rows}: {:.1} мкс на вызов", t0.elapsed().as_secs_f64() * 1e6 / 500.0);
        }
        let (xh, rh, got) = (host(&x), host(&router), host(&out));
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let (mut e, mut n) = (0f64, 0f64);
        for t in 0..rows {
            let xr = &xh[t * h..(t + 1) * h];
            let hv: Vec<f64> = (0..i)
                .map(|r| {
                    let g: f64 = (0..h).map(|k| dg[r * h + k] as f64 * xr[k] as f64).sum();
                    let u: f64 = (0..h).map(|k| du[r * h + k] as f64 * xr[k] as f64).sum();
                    g * sig(g) * u
                })
                .collect();
            let s = sig((0..h).map(|k| rh[k] as f64 * xr[k] as f64).sum());
            for o in 0..h {
                let want = s * (0..i).map(|k| dd[o * i + k] as f64 * hv[k]).sum::<f64>();
                e += (got[t * h + o] as f64 - want).powi(2);
                n += want * want;
            }
        }
        let rel = (e / n).sqrt();
        eprintln!("se_fused rows={rows}: отн. {rel:.2e}");
        assert!(rel < 2e-3, "rows={rows}");
    }
}
