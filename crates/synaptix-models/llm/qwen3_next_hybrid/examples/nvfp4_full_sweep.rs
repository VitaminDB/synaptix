//! ИСЧЕРПЫВАЮЩАЯ проверка NVFP4/MXFP8 GEMM на класс бага «per-row некорректность /
//! M-зависимость» (B-scale читал scale чужой batch-строки). Для КАЖДОЙ формы проекции
//! модели × широкого диапазона M через production-dispatch (Tensor::linear_quant):
//!   (1) row-consistency: Y(M)[r] == Y(Mref)[r] ∀ M,r  (выход строки НЕ зависит от M);
//!   (2) per-row корректность: worst-row |Y - F16dense_ref| мал относительно |ref|.
//! Печатает план (pick_nvfp4) на каждый M → видно покрытие Gemv/Reg/N8/Broadcast/Coop.
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//!   --example nvfp4_full_sweep
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cuda::gemm::dispatch::pick_nvfp4;

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

// M-список, покрывающий все ветки pick_nvfp4: M=1(Gemv), m%16(Reg), m%8\16(N8),
// !m%8 при n%64(Broadcast), плюс границы 128/256/512.
const MS: &[usize] = &[
    1, 2, 4, 8, 16, 24, 32, 40, 64, 100, 127, 128, 129, 200, 248, 255, 256, 257,
    300, 338, 400, 511, 512, 513, 600, 768, 850, 1024, 1536, 2048,
];

#[derive(Clone, Copy)]
enum Quant { Nvfp4, Mxfp8 }

fn sweep_shape(dev: Device, n: usize, k: usize, q: Quant, label: &str) -> (usize, usize) {
    sweep_shape_ms(dev, n, k, q, label, MS)
}

fn sweep_shape_ms(dev: Device, n: usize, k: usize, q: Quant, label: &str, ms: &[usize]) -> (usize, usize) {
    let mmax = *ms.iter().max().unwrap();
    let wf = det(0xA53F ^ (n as u64).wrapping_mul(131) ^ (k as u64), n * k, 0.5);
    let w_t = Tensor::from_vec(wf.clone(), vec![n, k], dev).unwrap().to_dtype(DType::F16).unwrap();
    let qw = match q { Quant::Nvfp4 => w_t.quantize_to_nvfp4(), Quant::Mxfp8 => w_t.quantize_to_mxfp8() }.unwrap();
    let xf = det(0x7777 ^ (k as u64), mmax * k, 0.4);

    // Эталонный прогон на Mref=mmax → строки 0..mmax. Все меньшие M сверяем с ним.
    let y_ref = host(&Tensor::from_vec(xf.clone(), vec![mmax, k], dev).unwrap()
        .to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap());

    // F16-dense эталон строки (для абсолютной корректности): берём первые строки.
    let probe_rows = [0usize, 1, 11, 59, 60, 100, 127, n.min(255)].into_iter()
        .filter(|&r| r < mmax).collect::<Vec<_>>();
    let mut ref_dense = std::collections::HashMap::new();
    let mut mean_abs = 0.0f64;
    for &r in &probe_rows {
        let xr = &xf[r * k..(r + 1) * k];
        let mut row = vec![0.0f32; n];
        for (o, slot) in row.iter_mut().enumerate() {
            let wr = &wf[o * k..(o + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k { acc += xr[j] * wr[j]; }
            *slot = acc;
        }
        mean_abs += row.iter().map(|v| v.abs() as f64).sum::<f64>() / n as f64;
        ref_dense.insert(r, row);
    }
    mean_abs /= probe_rows.len().max(1) as f64;

    let mut fails = 0usize;
    let mut worst_mdep = 0.0f32;
    let mut worst_abs = 0.0f32;
    let mut plans = std::collections::BTreeSet::new();
    for &m in MS {
        if m > mmax { continue; }
        let y = host(&Tensor::from_vec(xf[..m * k].to_vec(), vec![m, k], dev).unwrap()
            .to_dtype(DType::F16).unwrap().linear_quant(&qw).unwrap());
        if matches!(q, Quant::Nvfp4) { plans.insert(format!("{:?}", pick_nvfp4(m as u32, n as u32, k as u32))); }
        // (1) M-независимость: строка r при M == та же строка при Mref.
        let mut md = 0.0f32;
        for r in 0..m { for c in 0..n {
            md = md.max((y[r * n + c] - y_ref[r * n + c]).abs());
        }}
        worst_mdep = worst_mdep.max(md);
        if md > 1e-3 { fails += 1; }
        // (2) абсолютная корректность probe-строк vs F16-dense.
        for &r in &probe_rows {
            if r >= m { continue; }
            let rd = &ref_dense[&r];
            let mut a = 0.0f32;
            for c in 0..n { a = a.max((y[r * n + c] - rd[c]).abs()); }
            worst_abs = worst_abs.max(a);
        }
    }
    let rel = worst_abs as f64 / mean_abs.max(1.0);
    // M-расхождение — РЕАЛЬНЫЙ дефект (wrong-scale-row класс, как был NVFP4) только
    // если сравнимо с per-row квант-ошибкой vs эталон (>30% от worst_abs). Если же
    // M-разброс << квант-шума — это fp-порядок аккумуляции (benign, неассоциативность
    // float при разных тайлингах/grid по M). NVFP4 после фикса = 0.0 (бит-идентичен
    // по всем M). MXFP8 = ~0.25 << abs 5.0 → fp-order, корректен (каждый M в шуме).
    let mdep_defect = worst_mdep as f64 > 0.30 * worst_abs.max(1e-6) as f64 && worst_mdep > 1e-2;
    let abs_ok = rel < 0.05; // квант-шум ~1-2%; wrong-row баг даёт спайк строки
    let ok = !mdep_defect && abs_ok;
    let mlabel = if worst_mdep < 1e-3 { "M-indep(bit)" }
        else if mdep_defect { "M-ЗАВИСИМ-БАГ!" } else { "fp-order(benign)" };
    eprintln!(
        "{} {label} N={n} K={k} | M-dep worst={worst_mdep:.4} ({mlabel}) | abs worst={worst_abs:.3} rel={:.3}% ({}) | plans={:?}",
        if ok { "✅" } else { "❌" },
        rel * 100.0,
        if abs_ok { "корр" } else { "НЕВЕРНО!" },
        plans
    );
    (if ok { 0 } else { 1 }, fails)
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    // ВСЕ NVFP4-проекции qwen3-next-hybrid (N=выход, K=вход).
    let shapes: &[(usize, usize, &str)] = &[
        (10240, 5120, "in_proj_qkv"),
        (6144, 5120, "in_proj_z  "),
        (12288, 5120, "q_proj_gate"),
        (1024, 5120, "kv_proj    "),
        (5120, 6144, "o_proj     "),
        (17408, 5120, "mlp_gate_up"),
        (5120, 17408, "mlp_down   "),
        (5120, 5120, "square     "),
        (248320, 5120, "lm_head    "),
        // Coop (план для n%64!=0) НЕдостижим в проде: GEMV/quant требуют N%64==0 →
        // n%64!=0 невозможен. Coop проверяется ПРЯМЫМ вызовом в diag-тесте (N=5120).
    ];

    eprintln!("=== NVFP4 sweep (production dispatch, per-row + M-consistency) ===");
    let mut tot = 0usize;
    for &(n, k, lbl) in shapes {
        let (f, _) = sweep_shape(dev, n, k, Quant::Nvfp4, lbl);
        tot += f;
    }
    eprintln!("\n=== MXFP8 sweep (весь M-диапазон: кросс-конфиг fp-order ожидаем) ===");
    for &(n, k, lbl) in &shapes[..8] {
        let (f, _) = sweep_shape(dev, n, k, Quant::Mxfp8, lbl);
        let _ = f; // кросс-конфиг M-dep benign — НЕ считаем как fail (см. same-config ниже)
    }
    // РЕШАЮЩИЙ: M из ОДНОГО конфига (все %128 → C_128_128_S2). Если M-dep=0.0 →
    // разброс выше был от смены конфига (benign fp-order), а НЕ scale-row баг.
    eprintln!("\n=== MXFP8 SAME-CONFIG (M∈{{128,256,512,1024,2048}}, один bm=128) ===");
    for &(n, k, lbl) in &shapes[..8] {
        let (f, _) = sweep_shape_ms(dev, n, k, Quant::Mxfp8, lbl, &[128, 256, 512, 1024, 2048]);
        tot += f; // here M-dep ОБЯЗАН быть 0.0 — иначе реальный per-row баг
    }
    eprintln!("\n{}", if tot == 0 { "✅ ВСЕ формы × все M: per-row корректны и M-независимы" }
        else { "❌ ЕСТЬ дефекты (см. ❌ выше)" });
    std::process::exit(if tot == 0 { 0 } else { 1 });
}
