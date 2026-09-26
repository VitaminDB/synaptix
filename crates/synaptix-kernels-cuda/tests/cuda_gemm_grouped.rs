//! Групповой GEMM экспертов по портируемой таблице (`ExpertTable::gemm_grouped_dense`)
//! против плотного эталона по каждому сегменту: деквантованный вес × та же
//! активация — расхождение только от порядка суммирования и округления выхода.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::{ExpertTable, QuantWeight};
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, len: usize) -> Vec<f32> {
    let mut st = seed;
    (0..len)
        .map(|_| {
            st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

fn case(quant: DType, act: DType, n: usize, k: usize, seg_rows: &[usize]) {
    let dev = Device::Cuda(0);
    let experts = seg_rows.len();
    let weights: Vec<QuantWeight> = (0..experts)
        .map(|e| {
            Tensor::from_vec(noise(100 + e as u64, n * k), vec![n, k], dev)
                .and_then(|t| t.to_dtype(DType::F16))
                .and_then(|t| t.quantize_to(quant))
                .unwrap()
        })
        .collect();
    let refs: Vec<&QuantWeight> = weights.iter().collect();
    let table = ExpertTable::build_dense(&refs).expect("портируемая таблица");

    // Эксперты в обратном порядке: сегменты не обязаны идти по номеру.
    let rows: usize = seg_rows.iter().sum();
    let mut segments = Vec::new();
    let mut at = 0u32;
    for (i, r) in seg_rows.iter().enumerate() {
        let e = (experts - 1 - i) as u32;
        segments.push((e, at, at + *r as u32));
        at += *r as u32;
    }
    let x = Tensor::from_vec(noise(7, rows * k), vec![rows, k], dev)
        .and_then(|t| t.to_dtype(act))
        .unwrap();
    let got = host(&table.gemm_grouped_dense(&x, &segments).expect("групповой GEMM"));

    let mut want = Vec::with_capacity(rows * n);
    for &(e, s, t) in &segments {
        if s == t {
            continue;
        }
        let xs = x.narrow(0, s as usize, (t - s) as usize).and_then(|t| t.contiguous()).unwrap();
        // MXFP8 деквантуется только в F16; E4M3·2^k точно представим и в BF16.
        let wd = weights[e as usize]
            .dequantize(act)
            .or_else(|_| weights[e as usize].dequantize(DType::F16).and_then(|t| t.to_dtype(act)))
            .unwrap();
        let wt = wd.transpose(0, 1).and_then(|t| t.contiguous()).unwrap();
        want.extend(host(&xs.matmul(&wt).unwrap()));
    }
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(&want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    let l2 = (num / den.max(1e-12)).sqrt();
    println!("{quant:?} {act:?} n={n} k={k} segs={seg_rows:?} l2_rel={l2:.2e} ‖ref‖²={den:.3e}");
    assert!(den > 1.0, "эталон пустой");
    assert!(l2 < 0.01, "{quant:?}/{act:?}: групповой GEMM разошёлся с эталоном: {l2}");
}

#[test]
fn grouped_gemm_matches_dequant_matmul() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    // Сегменты: пустой, короче тайла, ровно тайл, больше тайла; N не кратно 64.
    let segs = [0usize, 5, 64, 130, 1];
    for quant in [DType::NVFP4, DType::MXFP8, DType::Sq { bits: 4 }] {
        for act in [DType::BF16, DType::F16] {
            case(quant, act, 208, 256, &segs);
        }
    }
    // K кратно 32, но не 64 (хвост BK), N чётное не кратное 16.
    case(DType::MXFP8, DType::BF16, 202, 288, &segs);
    case(DType::MXFP8, DType::BF16, 1408, 2816, &[37, 200, 64]);
    case(DType::Sq { bits: 4 }, DType::BF16, 2816, 768, &[3, 129]);
}

/// Сбор строк в загрузке тайла (`x_rows`): тот же ответ, что у GEMM по
/// заранее собранной копии (строки повторяются, как у k слотов токена).
#[test]
fn grouped_gemm_gathers_rows() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (n, k, tokens) = (192usize, 256usize, 40usize);
    let weights: Vec<QuantWeight> = (0..3)
        .map(|e| {
            Tensor::from_vec(noise(300 + e, n * k), vec![n, k], dev)
                .and_then(|t| t.to_dtype(DType::F16))
                .and_then(|t| t.quantize_to(DType::MXFP8))
                .unwrap()
        })
        .collect();
    let refs: Vec<&QuantWeight> = weights.iter().collect();
    let table = ExpertTable::build_dense(&refs).unwrap();
    let rows: Vec<u32> = (0..100u32).map(|i| (i * 7 + i / 3) % tokens as u32).collect();
    let segments = [(2u32, 0u32, 30u32), (0, 30, 95), (1, 95, 100)];
    let x_host = noise(9, tokens * k);
    let x = Tensor::from_vec(x_host.clone(), vec![tokens, k], dev).and_then(|t| t.to_dtype(DType::BF16)).unwrap();
    let gathered_host: Vec<f32> =
        rows.iter().flat_map(|r| x_host[*r as usize * k..(*r as usize + 1) * k].to_vec()).collect();
    let gathered = Tensor::from_vec(gathered_host, vec![rows.len(), k], dev)
        .and_then(|t| t.to_dtype(DType::BF16))
        .unwrap();
    let idx = Tensor::from_vec::<_, u32>(rows.clone(), vec![rows.len()], dev).unwrap();
    let a = host(&table.gemm_grouped_dense_rows(&x, Some(&idx), &segments, false).unwrap());
    let b = host(&table.gemm_grouped_dense(&gathered, &segments).unwrap());
    assert_eq!(a.len(), b.len());
    assert!(b.iter().any(|v| *v != 0.0));
    assert_eq!(a, b, "сбор строк в ядре разошёлся с собранной копией");
}

/// Замер: групповой GEMM (128 экспертов по 128 строк, форма gate_up Gemma-4)
/// против плотного BF16 GEMM той же формы. `--ignored --nocapture`.
#[test]
#[ignore]
fn grouped_gemm_bench() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (n, k, experts, per) = (1408usize, 2816usize, 128usize, 128usize);
    let fmt = match std::env::var("GG_FMT").as_deref() {
        Ok("mxfp8") => DType::MXFP8,
        Ok("sq4") => DType::Sq { bits: 4 },
        _ => DType::NVFP4,
    };
    let dense = Tensor::from_vec(noise(1, n * k), vec![n, k], dev).and_then(|t| t.to_dtype(DType::F16)).unwrap();
    let w = dense.quantize_to(fmt).unwrap();
    let weights: Vec<&QuantWeight> = (0..experts).map(|_| &w).collect();
    let table = ExpertTable::build_dense(&weights).unwrap();
    let r = experts * per;
    let segments: Vec<(u32, u32, u32)> =
        (0..experts).map(|e| (e as u32, (e * per) as u32, ((e + 1) * per) as u32)).collect();
    let x = Tensor::from_vec(noise(2, r * k), vec![r, k], dev).and_then(|t| t.to_dtype(DType::BF16)).unwrap();
    let wt = dense.to_dtype(DType::BF16).and_then(|t| t.transpose(0, 1)).and_then(|t| t.contiguous()).unwrap();
    let sync = || synaptix_core::stream::Stream::default_for(dev).unwrap().sync().unwrap();
    let time = |f: &dyn Fn()| {
        f();
        sync();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            f();
        }
        sync();
        t0.elapsed().as_secs_f64() / 10.0
    };
    let tg = time(&|| {
        table.gemm_grouped_dense(&x, &segments).unwrap();
    });
    let t8 = time(&|| {
        table.gemm_grouped_dense_rows(&x, None, &segments, true).unwrap();
    });
    let td = time(&|| {
        x.matmul(&wt).unwrap();
    });
    let flop = 2.0 * r as f64 * n as f64 * k as f64;
    println!(
        "{fmt:?}: grouped {:.3} мс ({:.1} TFLOPS), fp8 {:.3} мс ({:.1} TFLOPS), dense bf16 {:.3} мс ({:.1} TFLOPS)",
        tg * 1e3,
        flop / tg / 1e12,
        t8 * 1e3,
        flop / t8 / 1e12,
        td * 1e3,
        flop / td / 1e12
    );
}

fn l2_rel(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    assert!(den > 1.0, "эталон пустой");
    (num / den).sqrt()
}

/// FP8 MMA (sm_89+): MXFP8-вес против эталона «деквант MXFP8-активации ×
/// деквант веса» — те же значения, расхождение только от порядка суммы;
/// NVFP4/SQ4 (вес переквантуется в MXFP8) — против плотного эталона в
/// пределах шума кванта. Сбор строк по индексам — тем же путём.
#[test]
fn grouped_gemm_fp8() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let caps = synaptix_kernels_cuda::caps::DeviceCaps::for_ordinal(0).unwrap();
    if !caps.fp8_mma() {
        eprintln!("нет FP8 MMA — пропуск");
        return;
    }
    // Мелкие сегменты — тайл 64, крупные — 128 (выбор gg8_bm).
    for segs_rows in [vec![0usize, 5, 64, 70, 1], vec![130usize, 200, 97]] {
        fp8_case(&caps, &segs_rows);
    }
}

fn fp8_case(_caps: &synaptix_kernels_cuda::caps::DeviceCaps, segs_rows: &[usize]) {
    let dev = Device::Cuda(0);
    let (n, k, tokens) = (208usize, 288usize, 90usize);
    let total: usize = segs_rows.iter().sum();
    let rows: Vec<u32> = (0..total as u32).map(|i| (i * 11 + i / 5) % tokens as u32).collect();
    let mut segments = Vec::new();
    let mut at = 0u32;
    for (i, r) in segs_rows.iter().enumerate() {
        segments.push(((segs_rows.len() - 1 - i) as u32, at, at + *r as u32));
        at += *r as u32;
    }
    assert_eq!(at as usize, rows.len());
    let x_host = noise(21, tokens * k);
    for act in [DType::BF16, DType::F16] {
        let x = Tensor::from_vec(x_host.clone(), vec![tokens, k], dev).and_then(|t| t.to_dtype(act)).unwrap();
        // MXFP8-активация, развёрнутая обратно: ровно то, что видит ядро.
        let xq = x.to_dtype(DType::F16).and_then(|t| t.quantize_to_mxfp8()).unwrap();
        let xq_host = host(&xq.dequantize(DType::F16).unwrap());
        let x_host_act = host(&x);
        let idx = Tensor::from_vec::<_, u32>(rows.clone(), vec![rows.len()], dev).unwrap();
        for quant in [DType::MXFP8, DType::NVFP4, DType::Sq { bits: 4 }] {
            let kq = if quant == DType::MXFP8 { k } else { 256 };
            let (xs_host, xd_host): (Vec<f32>, Vec<f32>) = if kq == k {
                (xq_host.clone(), x_host_act.clone())
            } else {
                continue_k(&xq_host, &x_host_act, tokens, k, kq)
            };
            let xk = Tensor::from_vec(xd_host.clone(), vec![tokens, kq], dev).and_then(|t| t.to_dtype(act)).unwrap();
            let weights: Vec<QuantWeight> = (0..segs_rows.len())
                .map(|e| {
                    Tensor::from_vec(noise(500 + e as u64, n * kq), vec![n, kq], dev)
                        .and_then(|t| t.to_dtype(DType::F16))
                        .and_then(|t| t.quantize_to(quant))
                        .unwrap()
                })
                .collect();
            let refs: Vec<&QuantWeight> = weights.iter().collect();
            let table = ExpertTable::build_dense(&refs).unwrap();
            let got = host(&table.gemm_grouped_dense_rows(&xk, Some(&idx), &segments, true).unwrap());
            // Эталон по строкам пар: активация (MXFP8 или плотная) × деквант веса.
            let reference = |src: &[f32]| -> Vec<f32> {
                let mut out = Vec::new();
                for &(e, s, t) in &segments {
                    if s == t {
                        continue;
                    }
                    let g: Vec<f32> = rows[s as usize..t as usize]
                        .iter()
                        .flat_map(|r| src[*r as usize * kq..(*r as usize + 1) * kq].to_vec())
                        .collect();
                    let xs = Tensor::from_vec(g, vec![(t - s) as usize, kq], dev).unwrap();
                    let wd = weights[e as usize]
                        .dequantize(DType::F16)
                        .and_then(|t| t.to_dtype(DType::F32))
                        .and_then(|t| t.transpose(0, 1))
                        .and_then(|t| t.contiguous())
                        .unwrap();
                    out.extend(host(&xs.matmul(&wd).unwrap()));
                }
                out
            };
            let exact = l2_rel(&got, &reference(&xs_host));
            let dense = l2_rel(&got, &reference(&xd_host));
            println!("FP8 {quant:?} {act:?} {segs_rows:?}: к MXFP8-эталону {exact:.2e}, к плотному {dense:.3}");
            if quant == DType::MXFP8 {
                assert!(exact < 5e-3, "{quant:?}/{act:?}: FP8 GEMM разошёлся с MXFP8-эталоном: {exact}");
            }
            assert!(dense < 0.1, "{quant:?}/{act:?}: FP8 GEMM дальше шума кванта: {dense}");
        }
    }
}

/// Первые `kq` столбцов обеих матриц (SQ/NVFP4 требуют K кратно 256/64).
fn continue_k(a: &[f32], b: &[f32], rows: usize, k: usize, kq: usize) -> (Vec<f32>, Vec<f32>) {
    let cut = |m: &[f32]| (0..rows).flat_map(|r| m[r * k..r * k + kq].to_vec()).collect::<Vec<f32>>();
    (cut(a), cut(b))
}

/// Замер портируемого GEMV (M = 1) на большом весе: ГБ/с по байтам веса.
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn blockq_gemv_bandwidth_bench() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (n, k) = (16384usize, 8192usize);
    for fmt in [DType::NVFP4, DType::MXFP8, DType::Sq { bits: 4 }] {
        let w = Tensor::from_vec(noise(3, n * k), vec![n, k], dev)
            .and_then(|t| t.to_dtype(DType::F16))
            .and_then(|t| t.quantize_to(fmt))
            .unwrap();
        let bytes = w.packed_arc().unwrap().byte_len() + w.scales_opt().map_or(0, |s| s.byte_len());
        let x = Tensor::from_vec(noise(4, k), vec![1, k], dev).and_then(|t| t.to_dtype(DType::BF16)).unwrap();
        let sync = || synaptix_core::stream::Stream::default_for(dev).unwrap().sync().unwrap();
        let run = || {
            Tensor::dec_gemv_grouped_dense(&[&w], &x).unwrap();
        };
        run();
        sync();
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            run();
        }
        sync();
        let t = t0.elapsed().as_secs_f64() / 20.0;
        println!("{fmt:?}: {:.1} мкс, {:.0} ГБ/с", t * 1e6, bytes as f64 / t / 1e9);
    }
}
