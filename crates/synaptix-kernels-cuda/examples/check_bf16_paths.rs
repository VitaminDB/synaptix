//! Полный гейт качества BF16/F16 GEMM (best_cu): все диспетч-зоны × границы.
//!
//! Гейты (per-row max|Δ|, НЕ cos):
//!   A. эквивалентность путей: дефолт-диспетч vs CFG-форсы — bit-exact
//!      (split-K зоны: ≤ неск. ULP — другой порядок суммирования чанков);
//!   B. row-consistency: выход строки не зависит от M (полный M vs M-стрип);
//!   B2. chunked-vs-single: y(x) построчно равен y(x[..m1]) ++ y(x[m1..]);
//!   C. vs f32-референс (gemm_f32, независимый загрузчик): rel ≤ 0.02;
//!   D. bias+residual (linear_bias_residual) vs f32-композиция.
//! Запускать ОДНИМ процессом (env-кэши других ручек не трогаем).
#![cfg(feature = "cuda")]

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const FAIL_REL: f32 = 0.02;
const WARN_REL: f32 = 0.008;

fn per_row_max(a: &Tensor, b: &Tensor) -> f32 {
    let a32 = a.to_dtype(DType::F32).unwrap();
    let b32 = b.to_dtype(DType::F32).unwrap();
    let d = a32.sub(&b32).unwrap().abs().unwrap();
    d.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
}

// допуск Δ между нашими путями: 0 (bit-exact), кроме зоны b256-сплита —
// порядок суммирования чанков даёт ~1 ULP bf16 (= scale/256 на пару ULP).
fn path_tol(m: usize, n: usize, k: usize, scale: f32) -> f32 {
    let split_b256 = (193..=256).contains(&m) && n % 128 == 0 && n < 16384 && k >= 8192;
    if split_b256 {
        scale / 64.0
    } else {
        0.0
    }
}

fn cfgs_for(m: usize) -> &'static [&'static str] {
    if m <= 192 {
        &["s3", "s4", "s64s4", "s64s6"]
    } else if m <= 2048 {
        &["s3", "s5", "s6", "b256s4"]
    } else {
        &["s5", "b256s4", "b256ts4"]
    }
}

struct Stats {
    cells: usize,
    checks: usize,
    warns: usize,
    fails: usize,
}

#[allow(clippy::too_many_arguments)]
fn check_cell(
    m: usize,
    n: usize,
    k: usize,
    dt: DType,
    x_cpu: &Tensor,
    w_cpu: &Tensor,
    with_epilogue: bool,
    st: &mut Stats,
) {
    let dev = Device::Cuda(0);
    let tag = format!("{dt:?} M={m:5} N={n:5} K={k:5}");
    let x32 = x_cpu.narrow(0, 0, m).unwrap().contiguous().unwrap();
    let w32 = w_cpu.clone();
    let x = x32.to_dtype(dt).unwrap();
    let w = w32.to_dtype(dt).unwrap();
    // f32-референс из ТЕХ ЖЕ квантованных входов (изолирует ошибку compute
    // от ошибки квантования данных в bf16/f16).
    let xq32 = x.to_dtype(DType::F32).unwrap();
    let wq32 = w.to_dtype(DType::F32).unwrap();
    let y32 = xq32.linear(&wq32).unwrap();
    let scale = y32.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);

    let y_def = x.linear(&w).unwrap();
    let y_def32 = y_def.to_dtype(DType::F32).unwrap();
    st.cells += 1;

    // C: vs f32-референс
    let d_ref = per_row_max(&y_def32, &y32);
    st.checks += 1;
    let rel = d_ref / scale;
    if rel > FAIL_REL {
        st.fails += 1;
        println!("FAIL C  {tag}: vs-f32 rel={rel:.4} (Δ={d_ref:.4} scale={scale:.2})");
    } else if rel > WARN_REL {
        st.warns += 1;
        println!("warn C  {tag}: vs-f32 rel={rel:.4}");
    }

    // A: эквивалентность путей (дефолт vs CFG-форсы)
    let tol = path_tol(m, n, k, scale);
    for cfg in cfgs_for(m) {
        synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::set_bf16_cfg_override(Some(*cfg));
        let y_cfg = x.linear(&w).unwrap();
        synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::set_bf16_cfg_override(None);
        let d = per_row_max(&y_def, &y_cfg);
        st.checks += 1;
        if d > tol {
            st.fails += 1;
            println!("FAIL A  {tag}: default vs {cfg}: Δ={d:.6} (tol={tol:.6})");
        }
    }

    // B: row-consistency (страховка класса «chunked теряет контекст»).
    // m_sub==1 идёт GEMV-путём (warp-reduction, другой порядок суммирования) —
    // не bit-равен GEMM-пути, но в f32-бюджете (гейт C на M=1 это проверяет
    // отдельно) → rel-допуск вместо 0.
    let m_sub = (m / 3).max(1);
    let x_sub = x.narrow(0, 0, m_sub).unwrap().contiguous().unwrap();
    let y_sub = x_sub.linear(&w).unwrap();
    let d_rc = per_row_max(&y_def.narrow(0, 0, m_sub).unwrap().contiguous().unwrap(), &y_sub);
    st.checks += 1;
    let tol_rc = if m_sub == 1 {
        tol.max(2.0 * FAIL_REL * scale)
    } else {
        tol.max(path_tol(m_sub, n, k, scale))
    };
    if d_rc > tol_rc {
        st.fails += 1;
        println!("FAIL B  {tag}: row-consist(M={m_sub}): Δ={d_rc:.6} (tol={tol_rc:.6})");
    }

    // B2: chunked-vs-single
    if m >= 4 {
        let m1 = m / 2;
        let xa = x.narrow(0, 0, m1).unwrap().contiguous().unwrap();
        let xb = x.narrow(0, m1, m - m1).unwrap().contiguous().unwrap();
        let ya = xa.linear(&w).unwrap();
        let yb = xb.linear(&w).unwrap();
        let da = per_row_max(&y_def.narrow(0, 0, m1).unwrap().contiguous().unwrap(), &ya);
        let db = per_row_max(&y_def.narrow(0, m1, m - m1).unwrap().contiguous().unwrap(), &yb);
        st.checks += 1;
        let tol_ch = tol.max(path_tol(m1, n, k, scale)).max(path_tol(m - m1, n, k, scale));
        if da.max(db) > tol_ch {
            st.fails += 1;
            println!("FAIL B2 {tag}: chunked(M={m1}+{}): Δ={:.6} (tol={tol_ch:.6})", m - m1, da.max(db));
        }
    }

    // D: bias+residual эпилог (другие маршруты: TMA отпадает, сплиты → reduce)
    if with_epilogue && m >= 2 {
        let b32 = Tensor::randn(vec![n], Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
            .mul_scalar(0.1)
            .unwrap();
        let r32 = Tensor::randn(vec![m, n], Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
            .mul_scalar(0.1)
            .unwrap();
        let b = b32.to_dtype(dt).unwrap();
        let r = r32.to_dtype(dt).unwrap();
        let y_e = x.linear_bias_residual(&w, Some(&b), Some(&r)).unwrap();
        let bq32 = b.to_dtype(DType::F32).unwrap();
        let rq32 = r.to_dtype(DType::F32).unwrap();
        let y_e_ref = y32
            .add(&bq32.broadcast_as(vec![m, n]).unwrap().contiguous().unwrap())
            .unwrap()
            .add(&rq32)
            .unwrap();
        let d_e = per_row_max(&y_e.to_dtype(DType::F32).unwrap(), &y_e_ref);
        st.checks += 1;
        let rel_e = d_e / scale;
        if rel_e > FAIL_REL {
            st.fails += 1;
            println!("FAIL D  {tag}: bias+residual rel={rel_e:.4}");
        }
    }
}

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let _ng = synaptix_core::grad::NoGradGuard::new();
    // (N, K, полная ли M-сетка, эпилог-гейт)
    let nk: Vec<(usize, usize, bool)> = vec![
        (4096, 4096, true),
        (16384, 4096, true),
        (4096, 16384, true),
        (4096, 64, false),
        (4096, 96, false),
        (4096, 4060, false),
        (4224, 4096, false),
        (4192, 4096, false),
        (1056, 4096, false),
        (160, 8192, false),
    ];
    let ms_full: Vec<usize> = vec![
        1, 2, 3, 8, 16, 31, 32, 33, 48, 63, 64, 65, 96, 97, 127, 128, 129, 160, 191, 192, 193,
        224, 255, 256, 257, 320, 384, 511, 512, 513, 1024, 2048, 4000, 4080, 4096, 4992,
    ];
    let ms_short: Vec<usize> = vec![1, 5, 32, 33, 64, 97, 128, 192, 256, 257, 512];
    let mut st = Stats { cells: 0, checks: 0, warns: 0, fails: 0 };
    for (n, k, full) in nk {
        let ms = if full { &ms_full } else { &ms_short };
        let m_max = *ms.iter().max().unwrap();
        println!("=== N={n} K={k} (M до {m_max}) ===");
        let dev = Device::Cuda(0);
        let x_cpu = Tensor::randn(vec![m_max, k], Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
            .mul_scalar(0.1)
            .unwrap();
        let w_cpu = Tensor::randn(vec![n, k], Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
            .mul_scalar(0.1)
            .unwrap();
        for &m in ms {
            for dt in [DType::BF16, DType::F16] {
                let epi = full && matches!(m, 32 | 256 | 4992);
                check_cell(m, n, k, dt, &x_cpu, &w_cpu, epi, &mut st);
            }
        }
    }
    println!(
        "\nИТОГ: ячеек {}, проверок {}, warn {}, FAIL {}",
        st.cells, st.checks, st.warns, st.fails
    );
    if st.fails == 0 {
        println!("ВСЕ ГЕЙТЫ PASS ✅");
    } else {
        std::process::exit(1);
    }
}
