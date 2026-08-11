//! Свежий бенч GEMM на формах LTX-2.3 DiT (video stream): best_cu bf16 (TN linear),
//! nvfp4, mxfp8 (on-the-fly квант активации — как делает QuantLinear в проде).
//! TFLOP/s = 2*M*N*K / time. NVRTC-компиляция вырезана прогревом (warmup).
//! Запускать ПО ОДНОМУ процессу (см. память: параллель = OOM-фриз).

use std::time::Instant;
use synaptix_core::device::{self, Device};
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn sync() {
    device::cuda::synchronize(0).unwrap();
}

fn tflops(m: usize, n: usize, k: usize, dt: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / dt / 1e12
}

fn fill(shape: Vec<usize>, dt: DType, dev: Device) -> Tensor {
    Tensor::ones(shape, dt, dev).unwrap().mul_scalar(0.02).unwrap()
}

// SYN_BENCH_POWER=1: после тайминга гоняем ядро back-to-back ~1с и сэмплим
// nvidia-smi (power.draw, clocks.sm) — на laptop-GPU ватты=клок=перф (DVFS),
// без этого «TFLOPS» не говорит, упёрлись мы в мощность или в ядро.
fn measure_power<F: FnMut()>(mut f: F) -> Option<(f64, f64)> {
    if std::env::var("SYN_BENCH_POWER").as_deref() != Ok("1") {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let h = std::thread::spawn(move || {
        let mut pw: Vec<f64> = Vec::new();
        let mut ck: Vec<f64> = Vec::new();
        while !s2.load(Ordering::Relaxed) {
            if let Ok(out) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits"])
                .output()
            {
                let t = String::from_utf8_lossy(&out.stdout);
                let v: Vec<f64> = t.trim().split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() == 2 {
                    pw.push(v[0]);
                    ck.push(v[1]);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(80));
        }
        // первый сэмпл — разгон, выкидываем при наличии запаса
        let cut = if pw.len() > 2 { 1 } else { 0 };
        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        (avg(&pw[cut..]), avg(&ck[cut..]))
    });
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 1.0 {
        f();
        if t0.elapsed().as_secs_f64() > 1.0 {
            break;
        }
    }
    sync();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    h.join().ok()
}

fn time_loop<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    // SYN_BENCH_DOBENCH=1 — протокол triton.do_bench (как у qutlass-якоря):
    // L2-flush (запись 256MB) между итерациями, тайминг CUDA-СОБЫТИЯМИ —
    // ровно как triton (чистое GPU-время ядра, без латентности launch/sync;
    // wall+sync штрафовал нас ~4-6мкс/ячейку — перекос против якоря).
    let dobench = std::env::var("SYN_BENCH_DOBENCH").map(|v| v == "1").unwrap_or(false);
    if dobench {
        let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
        // triton.do_bench 1-в-1: (1) флаш-буфер 256MB аллоцируется ОДИН раз,
        // в цикле только GPU-memset (= cache.zero_()) — аллокация в цикле
        // роняла duty-cycle → клок к 180MHz; (2) БЕЗ sync между итерациями —
        // события копятся, один sync в конце: per-iteration sync рвал
        // GPU-очередь → DVFS ронял DRAM-клок (наш 53.7μs vs 49.7 у cuBLAS
        // на ИДЕНТИЧНОМ по локу ядре).
        let mk_ev = || {
            stream
                .context()
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap()
        };
        let evs_a: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
        let evs_b: Vec<_> = (0..iters).map(|_| mk_ev()).collect();
        // Флаш SM-ЯДРОМ (= triton cache.zero_()), НЕ memset_zeros: cuMemsetD8
        // идёт через copy-engine → SM простаивает 285мкс/итерацию → DVFS
        // держит SM-клок ниже и ядро меряется замедленным (ядро-vs-ядро при
        // равном клоке мы ≥ cuBLAS: 44.5 vs 45.7мкс @ 2.0GHz, ncu free-clock).
        let flush_src = Tensor::ones(vec![64 * 1024 * 1024], DType::F32, Device::Cuda(0)).unwrap();
        for _ in 0..warmup {
            f();
        }
        sync();
        for i in 0..iters {
            // флаш SM-ядром 256MB read + 256MB write (triton-аналог; см. выше)
            let fz = flush_src.mul_scalar(0.0).unwrap();
            std::hint::black_box(&fz);
            evs_a[i].record(&stream).unwrap();
            f();
            evs_b[i].record(&stream).unwrap();
        }
        sync();
        let total: f64 = (0..iters)
            .map(|i| evs_a[i].elapsed_ms(&evs_b[i]).unwrap() as f64 / 1e3)
            .sum();
        return total / iters as f64;
    }
    for _ in 0..warmup {
        f();
    }
    sync();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    sync();
    t.elapsed().as_secs_f64() / iters as f64
}

fn dtype_on(name: &str) -> bool {
    match std::env::var("SYN_BENCH_DTYPES") {
        Ok(v) => v.split(',').any(|d| d.trim() == name),
        Err(_) => true,
    }
}

fn run(m: usize, n: usize, k: usize) {
    let dev = Device::Cuda(0);
    let iters = std::env::var("SYN_BENCH_IT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| (8.0e11 / (2.0 * m as f64 * n as f64 * k as f64)).clamp(8.0, 50.0) as usize);

    // --- bf16 dense (TN linear: x[m,k] @ W[n,k]ᵀ) ---
    let x_bf = fill(vec![m, k], DType::BF16, dev);
    let w_bf = fill(vec![n, k], DType::BF16, dev);
    let t_bf = if dtype_on("bf16") {
        Some(time_loop(
            || {
                let y = x_bf.linear(&w_bf).unwrap();
                std::hint::black_box(&y);
            },
            10,
            iters,
        ))
    } else {
        None
    };

    // --- f16 dense (TN linear, gemm_f16tn) ---
    let x_f16 = fill(vec![m, k], DType::F16, dev);
    let w_f16 = fill(vec![n, k], DType::F16, dev);
    let t_f16 = if dtype_on("f16") {
        Some(time_loop(
            || {
                let y = x_f16.linear(&w_f16).unwrap();
                std::hint::black_box(&y);
            },
            10,
            iters,
        ))
    } else {
        None
    };

    // --- nvfp4 (on-the-fly act-quant) ---
    let t_nv = if dtype_on("nvfp4") && n % 64 == 0 && k % 64 == 0 {
        match w_f16.quantize_to_nvfp4() {
            Ok(qw) => Some(time_loop(
                || {
                    let y = x_f16.linear_quant(&qw).unwrap();
                    std::hint::black_box(&y);
                },
                10,
                iters,
            )),
            Err(_) => None,
        }
    } else {
        None
    };

    // --- mxfp8 (on-the-fly act-quant) ---
    let t_mx = if dtype_on("mxfp8") && k % 32 == 0 {
        match w_f16.quantize_to_mxfp8() {
            Ok(qw) => Some(time_loop(
                || {
                    let y = x_f16.linear_quant(&qw).unwrap();
                    std::hint::black_box(&y);
                },
                10,
                iters,
            )),
            Err(_) => None,
        }
    } else {
        None
    };

    let fmt = |o: Option<f64>| {
        o.map(|d| format!("{:7.1}", tflops(m, n, k, d)))
            .unwrap_or_else(|| "    n/a".to_string())
    };
    println!(
        "M={m:6} N={n:6} K={k:6} | bf16 {} | f16 {} | nvfp4 {} | mxfp8 {}  ({iters} it)",
        fmt(t_bf),
        fmt(t_f16),
        fmt(t_nv),
        fmt(t_mx),
    );
    // мощность/клок sustained по каждому dtype (SYN_BENCH_POWER=1)
    let pw_bf = if dtype_on("bf16") {
        measure_power(|| {
            let y = x_bf.linear(&w_bf).unwrap();
            std::hint::black_box(&y);
        })
    } else {
        None
    };
    if let Some((w_avg, c_avg)) = pw_bf {
        let pw_f16 = if dtype_on("f16") {
            measure_power(|| {
                let y = x_f16.linear(&w_f16).unwrap();
                std::hint::black_box(&y);
            })
            .unwrap_or((0.0, 0.0))
        } else {
            (0.0, 0.0)
        };
        let pw_nv = if dtype_on("nvfp4") {
            w_f16.quantize_to_nvfp4().ok().and_then(|qw| {
                measure_power(|| {
                    let y = x_f16.linear_quant(&qw).unwrap();
                    std::hint::black_box(&y);
                })
            })
        } else {
            None
        };
        let pw_mx = if dtype_on("mxfp8") {
            w_f16.quantize_to_mxfp8().ok().and_then(|qw| {
                measure_power(|| {
                    let y = x_f16.linear_quant(&qw).unwrap();
                    std::hint::black_box(&y);
                })
            })
        } else {
            None
        };
        let f = |o: Option<(f64, f64)>| {
            o.map(|(w, c)| format!("{w:5.1}W {c:4.0}MHz"))
                .unwrap_or_else(|| "      n/a".into())
        };
        println!(
            "    power      | bf16 {w_avg:5.1}W {c_avg:4.0}MHz | f16 {:5.1}W {:4.0}MHz | nvfp4 {} | mxfp8 {}",
            pw_f16.0, pw_f16.1, f(pw_nv), f(pw_mx),
        );
    }
}

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    // LTX inference крутится под no-grad → run_linear использует backend.linear
    // (best_cu gemm_bf16 TN), а НЕ matmul-fallback (gemm_wmma NN). Без этого guard
    // бенч мерил бы не тот путь.
    let _ng = synaptix_core::grad::NoGradGuard::new();
    // SYN_BENCH_ENQ="M N K [iters]": чистая хост-стоимость enqueue (wall/it цикла
    // x.linear БЕЗ sync — GPU-очередь глубокая, хост не ждёт). Для сравнения с
    // torch-enqueue (зазор record→launch входит в событийное окно do_bench).
    if let Ok(spec) = std::env::var("SYN_BENCH_ENQ") {
        let v: Vec<usize> = spec.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() >= 3 {
            let (m, n, k) = (v[0], v[1], v[2]);
            // ≤512: под лимитом pending-очереди CUDA — иначе cuLaunchKernel
            // блокируется и wall/it превращается в GPU-время ядра.
            let iters = v.get(3).copied().unwrap_or(512);
            let dev = Device::Cuda(0);
            let x = fill(vec![m, k], DType::BF16, dev);
            let w = fill(vec![n, k], DType::BF16, dev);
            for _ in 0..50 {
                let y = x.linear(&w).unwrap();
                std::hint::black_box(&y);
            }
            sync();
            let t = Instant::now();
            for _ in 0..iters {
                let y = x.linear(&w).unwrap();
                std::hint::black_box(&y);
            }
            let host = t.elapsed().as_secs_f64() / iters as f64;
            sync();
            println!("ENQ M={m} N={n} K={k}: host enqueue {:.2}us/it ({iters} it)", host * 1e6);
            return;
        }
    }
    // single-shot режим для ncu: SYN_BENCH_ONE="M N K" → один прогон только bf16.
    if let Ok(spec) = std::env::var("SYN_BENCH_ONE") {
        let v: Vec<usize> = spec.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() == 3 {
            let (m, n, k) = (v[0], v[1], v[2]);
            let dev = Device::Cuda(0);
            let x = fill(vec![m, k], DType::BF16, dev);
            let w = fill(vec![n, k], DType::BF16, dev);
            for _ in 0..5 {
                let y = x.linear(&w).unwrap();
                std::hint::black_box(&y);
            }
            sync();
            eprintln!("SYN_BENCH_ONE done M={m} N={n} K={k}");
            return;
        }
    }
    // Корректность fast-квантизатора активаций: SYN_BENCH_CHECK_QUANT=1 —
    // linear_quant (nvfp4/mxfp8) со старым (set_*_quant_slow(true)) vs новым
    // квантизатором; та же арифметика → per-row max|Δ| должен быть 0.
    if std::env::var("SYN_BENCH_CHECK_QUANT").as_deref() == Ok("1") {
        let dev = Device::Cuda(0);
        println!("CHECK quant fast vs slow — per-row max|Δ| (ожидаем 0):");
        for (m, n, k) in [(2048usize, 4096usize, 4096usize), (11960, 4096, 4096), (4992, 16384, 4096), (2048, 4096, 16384)] {
            let x = Tensor::randn(vec![m, k], Device::Cpu).unwrap().to_device(dev).unwrap().mul_scalar(0.1).unwrap().to_dtype(DType::F16).unwrap();
            let w = Tensor::randn(vec![n, k], Device::Cpu).unwrap().to_device(dev).unwrap().mul_scalar(0.1).unwrap().to_dtype(DType::F16).unwrap();
            let qw = w.quantize_to_nvfp4().unwrap();
            synaptix_kernels_cuda::elementwise::quant::set_nvfp4_quant_slow(true);
            let ys = x.linear_quant(&qw).unwrap().to_dtype(DType::F32).unwrap();
            synaptix_kernels_cuda::elementwise::quant::set_nvfp4_quant_slow(false);
            let yf = x.linear_quant(&qw).unwrap().to_dtype(DType::F32).unwrap();
            std::env::set_var("SYN_NVFP4_NO_BIGTILE", "1");
            let yo = x.linear_quant(&qw).unwrap().to_dtype(DType::F32).unwrap();
            std::env::remove_var("SYN_NVFP4_NO_BIGTILE");
            let diff = ys.sub(&yf).unwrap().abs().unwrap();
            let worst = diff.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let dbt = yo.sub(&yf).unwrap().abs().unwrap();
            let worst_bt = dbt.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            println!("  nvfp4 M={m:6} N={n:6} K={k:6}: quant-fast per-row max|Δ|={worst:.6}  bigtile per-row max|Δ|={worst_bt:.6}");
            let qw8 = w.quantize_to_mxfp8().unwrap();
            synaptix_kernels_cuda::elementwise::quant::set_mxfp8_quant_slow(true);
            let ys8 = x.linear_quant(&qw8).unwrap().to_dtype(DType::F32).unwrap();
            synaptix_kernels_cuda::elementwise::quant::set_mxfp8_quant_slow(false);
            let yf8 = x.linear_quant(&qw8).unwrap().to_dtype(DType::F32).unwrap();
            let d8 = ys8.sub(&yf8).unwrap().abs().unwrap();
            let worst8 = d8.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            println!("  mxfp8 M={m:6} N={n:6} K={k:6}: quant-fast per-row max|Δ|={worst8:.6}");
        }
        return;
    }
    // Однократный квант-прогон для ncu/клок-мониторинга:
    // SYN_BENCH_ONE_Q="M N K nvfp4|mxfp8 [iters]".
    if let Ok(spec) = std::env::var("SYN_BENCH_ONE_Q") {
        let v: Vec<&str> = spec.split_whitespace().collect();
        if v.len() >= 4 {
            let (m, n, k) = (v[0].parse().unwrap(), v[1].parse().unwrap(), v[2].parse().unwrap());
            let iters: usize = v.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
            let dev = Device::Cuda(0);
            let x = fill(vec![m, k], DType::F16, dev);
            let w = fill(vec![n, k], DType::F16, dev);
            let qw = if v[3] == "nvfp4" {
                w.quantize_to_nvfp4().unwrap()
            } else {
                w.quantize_to_mxfp8().unwrap()
            };
            for _ in 0..3 {
                let y = x.linear_quant(&qw).unwrap();
                std::hint::black_box(&y);
            }
            sync();
            let t = Instant::now();
            for i in 0..iters {
                let y = x.linear_quant(&qw).unwrap();
                std::hint::black_box(&y);
                if i % 8 == 7 {
                    sync();
                }
            }
            sync();
            eprintln!(
                "SYN_BENCH_ONE_Q done {spec}: {:.1} TF",
                tflops(m, n, k, t.elapsed().as_secs_f64() / iters as f64)
            );
            return;
        }
    }
    // Sustained-loop для мониторинга клоков/power: SYN_BENCH_LOOP="M N K SECS".
    if let Ok(spec) = std::env::var("SYN_BENCH_LOOP") {
        let v: Vec<usize> = spec.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() == 4 {
            let (m, n, k, secs) = (v[0], v[1], v[2], v[3]);
            let dev = Device::Cuda(0);
            let x = fill(vec![m, k], DType::BF16, dev);
            let w = fill(vec![n, k], DType::BF16, dev);
            for _ in 0..10 {
                let y = x.linear(&w).unwrap();
                std::hint::black_box(&y);
            }
            sync();
            let t0 = Instant::now();
            let mut iters = 0usize;
            while t0.elapsed().as_secs_f64() < secs as f64 {
                let y = x.linear(&w).unwrap();
                std::hint::black_box(&y);
                iters += 1;
                if iters % 8 == 0 {
                    sync();
                }
            }
            sync();
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "SYN_BENCH_LOOP M={m} N={n} K={k}: {iters} it, {:.1} TFLOP/s sustained",
                tflops(m, n, k, dt)
            );
            return;
        }
    }
    // Корректность конфига (SYN_BENCH_CHECK_CFG=s2|s4|...) vs default — per-row max|Δ|
    // (тот же mma f32-acc → bit-точно, ожидаем 0) + row-consistency (строка i не
    // зависит от M: считаем M и M-стрип, сравниваем общие строки).
    if let Ok(cfg) = std::env::var("SYN_BENCH_CHECK_CFG") {
        let shapes: Vec<(usize, usize, usize)> = vec![
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (192, 4096, 4096),
            (32, 16384, 4096),
            (64, 16384, 4096),
            (192, 16384, 4096),
            (32, 4096, 16384),
            (64, 4096, 16384),
            (192, 4096, 16384),
            (256, 4096, 16384),
            (2048, 4096, 4096),
            (26520, 4096, 4096),
        ];
        let dev = Device::Cuda(0);
        println!("CHECK cfg={cfg} vs default — per-row max|Δ| (ожидаем 0) + row-consistency:");
        for (m, n, k) in shapes {
            let x = Tensor::randn(vec![m, k], Device::Cpu).unwrap().to_device(dev).unwrap().mul_scalar(0.1).unwrap().to_dtype(DType::BF16).unwrap();
            let w = Tensor::randn(vec![n, k], Device::Cpu).unwrap().to_device(dev).unwrap().mul_scalar(0.1).unwrap().to_dtype(DType::BF16).unwrap();
            synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::set_bf16_cfg_override(None);
            let yd = x.linear(&w).unwrap().to_dtype(DType::F32).unwrap();
            synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::set_bf16_cfg_override(Some(&cfg));
            let yn = x.linear(&w).unwrap().to_dtype(DType::F32).unwrap();
            let m_sub = (m / 3).max(1);
            let x_sub = x.narrow(0, 0, m_sub).unwrap().contiguous().unwrap();
            let yn_sub = x_sub.linear(&w).unwrap().to_dtype(DType::F32).unwrap();
            synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::set_bf16_cfg_override(None);
            // Независимый f32-референс (ядро gemm_f32, другой загрузчик) — ловит
            // системные баги общего bf16-загрузчика, которые cfg-vs-default скрыл бы.
            let x32 = x.to_dtype(DType::F32).unwrap();
            let w32 = w.to_dtype(DType::F32).unwrap();
            let yref = x32.linear(&w32).unwrap();
            let diff = yd.sub(&yn).unwrap().abs().unwrap();
            let perrow = diff.max([1usize]).unwrap();
            let worst = perrow.max_all().unwrap().to_scalar::<f32>().unwrap();
            let rd = yn.narrow(0, 0, m_sub).unwrap().sub(&yn_sub).unwrap().abs().unwrap();
            let rc_worst = rd.max_all().unwrap().to_scalar::<f32>().unwrap();
            let refd = yn.sub(&yref).unwrap().abs().unwrap();
            let ref_worst = refd.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let scale = yd.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            println!(
                "  M={m:6} N={n:6} K={k:6}: vs-default per-row max|Δ|={worst:.6}  row-consist(M={m_sub}) max|Δ|={rc_worst:.6}  vs-f32ref per-row max|Δ|={ref_worst:.6}  scale={scale:.3}",
            );
        }
        return;
    }
    println!("synaptix LTX-DiT GEMM (best_cu) — TFLOP/s (выше = лучше), on-the-fly act-quant, NVRTC excluded:");
    println!("  формы: attn(4096,4096) | ff_up(16384,4096) | ff_down(4096,16384); M = video-токены");
    let shapes: Vec<(usize, usize)> = match std::env::var("SYN_BENCH_SHAPE") {
        Ok(s) => {
            let v: Vec<usize> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() == 2 { vec![(v[0], v[1])] } else { vec![(4096, 4096), (16384, 4096), (4096, 16384)] }
        }
        Err(_) => vec![(4096, 4096), (16384, 4096), (4096, 16384)],
    };
    let ms: Vec<usize> = match std::env::var("SYN_BENCH_MS") {
        Ok(s) => s.split_whitespace().filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![4080, 4096, 4992, 32640],
    };
    for (n, k) in shapes {
        println!("--- N={n} K={k} ---");
        for &m in &ms {
            run(m, n, k);
        }
    }
}
