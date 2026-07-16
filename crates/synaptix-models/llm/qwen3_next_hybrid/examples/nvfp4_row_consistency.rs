//! Строковая согласованность NVFP4-GEMM по M: корректный GEMM даёт
//! out[i,:] = x[i,:] @ Wᵀ независимо от общего M. Берём X[0:M], считаем Y_big;
//! берём X[off:M] (меньше строк), считаем Y_small; строки [off:M] обязаны
//! совпасть. Если нет → GEMM строко-несогласован (= chunked-prefill баг).
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//! --example nvfp4_row_consistency
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n).map(|_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
    }).collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn case(dev: Device, n: usize, k: usize, m_big: usize, off: usize) {
    // вес [N,K] F16 → NVFP4
    let wf = det(0xA53F ^ (n as u64).wrapping_mul(131) ^ (k as u64), n * k, 0.5);
    let w_t = Tensor::from_vec(wf, vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let qw = w_t.quantize_to_nvfp4().unwrap();
    // активация [m_big,K] F16
    let xf = det(0x5151, m_big * k, 0.4);
    let x_full = Tensor::from_vec(xf.clone(), vec![m_big, k], dev).unwrap().to_dtype(DType::F16).unwrap();

    // Y_big = X[0:m_big] @ Wᵀ
    let y_big = x_full.linear_quant(&qw).unwrap();
    let y_big_h = host(&y_big);

    // Y_small = X[off:m_big] @ Wᵀ  (m_small = m_big-off строк)
    let m_small = m_big - off;
    let x_small = Tensor::from_vec(xf[off * k..].to_vec(), vec![m_small, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let y_small = x_small.linear_quant(&qw).unwrap();
    let y_small_h = host(&y_small);

    // сравнить строки [off:m_big] big vs [0:m_small] small
    let mut maxd = 0.0f32;
    let mut sumd = 0.0f32;
    for r in 0..m_small {
        for c in 0..n {
            let a = y_big_h[(off + r) * n + c];
            let b = y_small_h[r * n + c];
            let d = (a - b).abs();
            maxd = maxd.max(d);
            sumd += d;
        }
    }
    let mean = sumd / (m_small * n) as f32;
    eprintln!("N={n} K={k} | Y_big(M={m_big})[{off}:] vs Y_small(M={m_small}) → max_abs={maxd:.4} mean_abs={mean:.5} {}",
        if maxd < 0.05 { "OK строко-согласован" } else { "◄── НЕСОГЛАСОВАН (баг!)" });
}

// Проверка: строка `row` выхода НЕ зависит от общего M (корректный GEMM
// строко-независим). Берём один X[0:600], считаем Y при разных M (все > row),
// сравниваем Y[row] между ними.
fn row_independence(dev: Device, n: usize, k: usize, row: usize, ms: &[usize]) {
    let wf = det(0xA53F ^ (n as u64).wrapping_mul(131) ^ (k as u64), n * k, 0.5);
    let qw = Tensor::from_vec(wf, vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap().quantize_to_nvfp4().unwrap();
    let xf = det(0x7777, ms.iter().max().unwrap() * k, 0.4);
    let mut refrow: Option<Vec<f32>> = None;
    let mut maxd = 0.0f32;
    for &m in ms {
        let x = Tensor::from_vec(xf[..m * k].to_vec(), vec![m, k], dev).unwrap().to_dtype(DType::F16).unwrap();
        let y = host(&x.linear_quant(&qw).unwrap());
        let r = y[row * n..(row + 1) * n].to_vec();
        match &refrow {
            None => refrow = Some(r),
            Some(rr) => {
                let d = rr.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                maxd = maxd.max(d);
            }
        }
    }
    eprintln!("N={n} K={k} строка[{row}] при M={ms:?} → max_abs между M = {maxd:.4} {}",
        if maxd < 0.05 { "OK M-независима" } else { "◄── M-ЗАВИСИМА (баг!)" });
}

// Для строки `row`: печатает ||Y_M[row] - Y_ref[row]|| для каждого M, где Y_ref =
// F32-эталон (dequant W·X). Показывает, какие M-ветки NVFP4-GEMM корректны.
fn row_vs_ref(dev: Device, n: usize, k: usize, row: usize, ms: &[usize]) {
    let wf = det(0xA53F ^ (n as u64).wrapping_mul(131) ^ (k as u64), n * k, 0.5);
    let w_t = Tensor::from_vec(wf.clone(), vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let qw = w_t.quantize_to_nvfp4().unwrap();
    let maxm = *ms.iter().max().unwrap();
    let xf = det(0x7777, maxm * k, 0.4);
    // F32-эталон строки row: dequant-вес из NVFP4 обратно (через linear_quant на 1 строке).
    // Берём «истину» как результат при M=row+1 минимального... вместо этого считаем
    // F16-dense эталон x[row]·Wᵀ напрямую (W f16, x f16) — высокоточный ориентир.
    let xrow: Vec<f32> = xf[row * k..(row + 1) * k].to_vec();
    let mut yref = vec![0.0f32; n];
    for (j, yr) in yref.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for kk in 0..k { acc += xrow[kk] * wf[j * k + kk]; }
        *yr = acc;
    }
    eprint!("N={n} K={k} строка[{row}] vs F16-dense эталон:");
    for &m in ms {
        let x = Tensor::from_vec(xf[..m * k].to_vec(), vec![m, k], dev).unwrap().to_dtype(DType::F16).unwrap();
        let y = host(&x.linear_quant(&qw).unwrap());
        let r = &y[row * n..(row + 1) * n];
        let d = r.iter().zip(&yref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprint!(" M={m}:{d:.3}");
    }
    eprintln!();
}

// Dense F16 GEMM (как in_proj_a/b: x[M,K]@W[N,K]ᵀ). Строковая согласованность по M.
fn case_dense(dev: Device, n: usize, k: usize, m_big: usize, off: usize) {
    let wf = det(0xBEEF ^ (n as u64) ^ (k as u64).wrapping_mul(7), n * k, 0.5);
    let w = Tensor::from_vec(wf, vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let xf = det(0x1234, m_big * k, 0.4);
    let x_full = Tensor::from_vec(xf.clone(), vec![m_big, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let y_big = host(&x_full.linear(&w).unwrap());
    let m_small = m_big - off;
    let x_small = Tensor::from_vec(xf[off * k..].to_vec(), vec![m_small, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let y_small = host(&x_small.linear(&w).unwrap());
    let mut maxd = 0.0f32;
    for r in 0..m_small { for c in 0..n {
        maxd = maxd.max((y_big[(off + r) * n + c] - y_small[r * n + c]).abs());
    }}
    eprintln!("DENSE N={n} K={k} | Y(M={m_big})[{off}:] vs Y(M={m_small}) → max_abs={maxd:.4} {}",
        if maxd < 0.05 { "OK" } else { "◄── M-НЕСОГЛАСОВАН (баг a/b!)" });
}

// rms_norm_fused row-consistency: вход [1,M,H] F16, rms по H. Сравниваем строку
// (тот же токен) при M=850 vs M=338. Per-token норма ДОЛЖНА быть M-независима.
fn case_rmsnorm(dev: Device, h: usize, m_big: usize, off: usize) {
    let wf = det(0xDEAD, h, 0.6);
    let w = Tensor::from_vec(wf, vec![h], dev).unwrap().to_dtype(DType::F16).unwrap();
    let xf = det(0xCAFE, m_big * h, 0.7);
    let x_full = Tensor::from_vec(xf.clone(), vec![1, m_big, h], dev).unwrap().to_dtype(DType::F16).unwrap();
    let y_big = host(&x_full.rms_norm_fused(&w, 1e-6, false).unwrap());
    let m_small = m_big - off;
    let x_small = Tensor::from_vec(xf[off * h..].to_vec(), vec![1, m_small, h], dev).unwrap().to_dtype(DType::F16).unwrap();
    let y_small = host(&x_small.rms_norm_fused(&w, 1e-6, false).unwrap());
    let mut maxd = 0.0f32;
    for r in 0..m_small { for c in 0..h {
        maxd = maxd.max((y_big[(off + r) * h + c] - y_small[r * h + c]).abs());
    }}
    eprintln!("RMSNORM H={h} | Y(M={m_big})[{off}:] vs Y(M={m_small}) → max_abs={maxd:.5} {}",
        if maxd < 1e-4 { "OK M-независим" } else { "◄── M-ЗАВИСИМ (баг rms_norm!)" });
}

// КЛЮЧЕВОЕ: chunk1 rows 0-511 (M=512) vs single rows 0-511 (M=850). ОДИН и тот же
// вход x[0:512], сравниваем строку r<512 при M=512 vs M=850. case() раньше сравнивал
// только ХВОСТ — chunk1-префикс НЕ проверялся. Если расходится → in_proj M-несогласован
// для chunk1, что пропагирует в scan-state и ломает chunked-prefill.
fn case_prefix(dev: Device, n: usize, k: usize, m_small: usize, m_big: usize) {
    let wf = det(0xA53F ^ (n as u64).wrapping_mul(131) ^ (k as u64), n * k, 0.5);
    let qw = Tensor::from_vec(wf, vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap().quantize_to_nvfp4().unwrap();
    let xf = det(0x7777, m_big * k, 0.4);
    let y_small = host(&Tensor::from_vec(xf[..m_small * k].to_vec(), vec![m_small, k], dev).unwrap().to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap());
    let y_big = host(&Tensor::from_vec(xf[..m_big * k].to_vec(), vec![m_big, k], dev).unwrap().to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap());
    let mut maxd = 0.0f32; let mut bad_row = usize::MAX;
    for r in 0..m_small { for c in 0..n {
        let d = (y_small[r * n + c] - y_big[r * n + c]).abs();
        if d > maxd { maxd = d; bad_row = r; }
    }}
    eprintln!("PREFIX N={n} K={k} | Y(M={m_small})[r] vs Y(M={m_big})[r], r<{m_small} → max_abs={maxd:.4} @row{bad_row} {}",
        if maxd < 0.05 { "OK" } else { "◄── chunk1 M-НЕСОГЛАСОВАН (баг!)" });
}

fn main() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    eprintln!("=== pick_nvfp4 plan по M (N=5120,K=5120) ===");
    for m in [128u32, 256, 338, 512, 600, 850, 1024] {
        eprintln!("  M={m}: {:?}", synaptix_kernels_cuda::gemm::dispatch::pick_nvfp4(m, 5120, 5120));
    }
    eprintln!("=== row 59/60 vs F16-эталон по M (кто неверен) ===");
    row_vs_ref(dev, 5120, 5120, 59, &[128, 256, 338, 512, 600, 850, 1024]);
    row_vs_ref(dev, 5120, 5120, 60, &[128, 256, 338, 512, 600, 850, 1024]);
    eprintln!("=== CHUNK1 PREFIX row-consistency (M=512 vs M=850, те же строки 0-511) ===");
    case_prefix(dev, 10240, 5120, 512, 850); // in_proj_qkv conv_dim
    case_prefix(dev, 5120, 5120, 512, 850);
    case_prefix(dev, 5120, 5120, 338, 850);  // chunk2-стиль (M=338)
    case_prefix(dev, 5120, 5120, 512, 600);  // M=512 vs M=600
    case_prefix(dev, 5120, 5120, 600, 850);  // M=600 vs M=850 (оба не-кратны 128)
    eprintln!("=== RMS_NORM row-consistency (вход в in_proj) ===");
    case_rmsnorm(dev, 5120, 850, 512); // chunk2 M=338
    case_rmsnorm(dev, 5120, 600, 512); // chunk2 M=88
    case_rmsnorm(dev, 5120, 827, 512); // chunk2 M=315
    case_rmsnorm(dev, 5120, 1024, 512); // chunk2 M=512
    // in_proj_a/b — Dense F16, N=48. Проверяем M-согласованность (chunk2 vs single).
    eprintln!("=== DENSE a/b (N=48) ===");
    case_dense(dev, 48, 5120, 850, 512); // chunk2 M=338
    case_dense(dev, 48, 5120, 600, 512); // chunk2 M=88
    case_dense(dev, 48, 5120, 827, 512); // chunk2 M=315
    case_dense(dev, 48, 5120, 1024, 512); // chunk2 M=512
    // КЛЮЧЕВОЕ: chunk2 M=315 vs single M=827 (T=827 падает), M=88 vs 600 (T=600 ок),
    // для attention-N. case() = Y(M_big)[off:] vs Y(M_small) — те же токены.
    eprintln!("=== chunk2 vs single хвост, разные K (M=850 vs 338, M=600 vs 88) ===");
    // in_proj_a/b N=48(num_v_heads)! dt N=48. conv_dim=10240, value_dim=6144.
    for &(n, k) in &[(48usize,5120usize),(64,5120),(96,5120),(10240,5120),(6144,5120),(5120,6144)] {
        eprintln!("N={n} K={k}:");
        case(dev, n, k, 850, 512); // chunk2 M=338
        case(dev, n, k, 600, 512); // chunk2 M=88
    }
    eprintln!("=== старое (linear/mlp) ===");
    // После gate Full off — проверяем ВСЕ проекционные N при M=512(chunk1) vs 600(single).
    // attention: q=12288(gate), kv=1024, o-in=6144; linear in_proj conv_dim; mlp=17408.
    eprintln!("=== row-consistency M=[257,512,600,827] после Full-off, разные N (K=5120) ===");
    for &n in &[1024usize, 6144, 12288, 5120, 17408, 7680] {
        row_independence(dev, n, 5120, 100, &[257, 512, 600, 827]);
        row_vs_ref(dev, n, 5120, 100, &[512, 600]);
    }
    // o_proj: K=6144 (nh*hd), N=hidden=5120
    eprintln!("=== o_proj K=6144 ===");
    row_independence(dev, 5120, 6144, 100, &[257, 512, 600, 827]);
    // Форма qwen3.6-hybrid in_proj_qkv: K=hidden=5120, N=conv_dim≈5120.
    // Воспроизводим чанк-границы: single M=600 vs chunk2 off=512 (M_small=88).
    case(dev, 5120, 5120, 600, 512);  // chunk2 M=88 (<128 ветка)
    case(dev, 5120, 5120, 850, 512);  // chunk2 M=338
    case(dev, 5120, 5120, 827, 512);  // chunk2 M=315
    case(dev, 5120, 5120, 1024, 512); // chunk2 M=512 (=128*4)
    // MLP-форма: N=intermediate=17408
    case(dev, 17408, 5120, 600, 512);
    case(dev, 17408, 5120, 850, 512);
}
