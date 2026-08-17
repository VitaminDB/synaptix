//! MXFP8 linear для N вне 128-кратности (lm_head 202048×6656 у muse-glimmer).
//!
//! Голова `N - N%128` считается tiled-ядром, хвост (<128 строк; при K%128!=0 —
//! весь N) идёт полосами деквантa, куски сшиваются в out pitched-копиями.
//! Проверяем не точность, а СШИВКУ: каждый кусок обязан совпасть бит в бит с
//! отдельным прогоном ровно той же формы — иначе смещения колонок поехали.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn gen(len: usize, step: usize, period: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * step) % period) as f32 / period as f32 - 0.5) * 0.5)
        .collect()
}

/// `a @ Wᵀ` через MXFP8-квант веса; возвращает выход [m, n] построчно.
fn linear_mxfp8(w: &[f32], n: usize, k: usize, a: &Tensor, dev: Device) -> Vec<f32> {
    let wt = Tensor::from_vec(w.to_vec(), vec![n, k], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let qw = wt.quantize_to_mxfp8().unwrap();
    a.linear_quant(&qw)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Колонки `[c0, c0+width)` каждой из `m` строк выхода шириной `n`.
fn cols(v: &[f32], m: usize, n: usize, c0: usize, width: usize) -> Vec<f32> {
    (0..m)
        .flat_map(|r| v[r * n + c0..r * n + c0 + width].to_vec())
        .collect()
}

#[test]
fn mxfp8_split_n_stitching() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    // 1 МиБ на полосу деквантa: при k=544 даёт 963 строки → 3 полосы на N=2000.
    // Ставим до первого linear_quant — бюджет читается один раз за процесс.
    std::env::set_var("SYN_MXFP8_DEQ_MB", "1");
    let dev = Device::Cuda(0);

    // ── голова (tiled) + хвост (деквант): N = 16*128 + 64, K кратно 512 ──
    let (n, k, m) = (2112usize, 6656usize, 15usize);
    let (head, tail) = (2048usize, 64usize);
    let w = gen(n * k, 31, 199);
    let a = Tensor::from_vec(gen(m * k, 17, 173), vec![m, k], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let full = linear_mxfp8(&w, n, k, &a, dev);
    let only_head = linear_mxfp8(&w[..head * k], head, k, &a, dev);
    let only_tail = linear_mxfp8(&w[head * k..], tail, k, &a, dev);
    assert_eq!(
        cols(&full, m, n, 0, head),
        only_head,
        "голова N (tiled) не совпала с отдельным прогоном N={head}"
    );
    assert_eq!(
        cols(&full, m, n, head, tail),
        only_tail,
        "хвост N (деквант) не совпал с отдельным прогоном N={tail}"
    );

    // ── K вне 128-кратности: головы нет, весь N идёт полосами ──
    let (n2, k2, m2) = (2000usize, 544usize, 9usize);
    let w2 = gen(n2 * k2, 23, 191);
    let a2 = Tensor::from_vec(gen(m2 * k2, 13, 167), vec![m2, k2], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let full2 = linear_mxfp8(&w2, n2, k2, &a2, dev);
    let chunk = 1024 * 1024 / (k2 * 2); // = 963, как считает mxfp8_deq_budget_bytes
    assert!(chunk < n2, "форма должна давать несколько полос");
    let mut c0 = 0usize;
    while c0 < n2 {
        let width = chunk.min(n2 - c0);
        let part = linear_mxfp8(&w2[c0 * k2..(c0 + width) * k2], width, k2, &a2, dev);
        assert_eq!(
            cols(&full2, m2, n2, c0, width),
            part,
            "полоса [{c0}, {}) не совпала с отдельным прогоном",
            c0 + width
        );
        c0 += width;
    }
}
