//! Attention по таблице блоков против сборки выбранных позиций гатером.
//!
//! Ядро читает KV прямо по индексам блоков, эталон — тот же расчёт после
//! материализации выбранных позиций в отдельный буфер. Значения совпадают не
//! бит в бит (разный порядок суммирования и другой софтмакс-проход), поэтому
//! сверка идёт по относительной норме.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const NH: usize = 8;
const NKV: usize = 2;
const D: usize = 128;
const CAP: usize = 256;
const RATIO: usize = 4;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.5
        })
        .collect()
}

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("на хост")
}

/// Эталон: собрать выбранные позиции в `[B, NKV, M, D]` и посчитать внимание
/// обычным путём.
#[allow(clippy::too_many_arguments)]
fn gathered_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rows: &[(Vec<u32>, (u32, u32))],
    device: Device,
    nh: usize,
    nkv: usize,
    d: usize,
) -> Vec<f32> {
    let b = rows.len();
    let m = rows[0].0.len() * RATIO + rows[0].1 .1 as usize;
    let mut idx = Vec::with_capacity(b * nkv * m);
    for (blocks, (from, len)) in rows {
        for head in 0..nkv {
            let base = (head * CAP) as u32;
            for blk in blocks {
                for j in 0..RATIO as u32 {
                    idx.push(base + blk * RATIO as u32 + j);
                }
            }
            for j in 0..*len {
                idx.push(base + from + j);
            }
        }
    }
    let idx = Tensor::from_vec(idx, vec![b * nkv * m], device).expect("индексы");
    let gather = |src: &Tensor| {
        src.reshape(vec![nkv * CAP, d])
            .and_then(|t| t.embed_gather(&idx))
            .and_then(|t| t.reshape(vec![b, nkv, m, d]))
            .expect("гатер")
    };
    let k_sel = gather(k);
    let v_sel = gather(v);
    let q4 = q.reshape(vec![b, nh, 1, d]).expect("q");
    let out = q4
        .flash_attention(&k_sel, &v_sel, 1.0 / (d as f32).sqrt(), false)
        .expect("flash");
    host(&out)
}

fn check(dtype: DType, tol: f32) {
    check_shape(dtype, tol, NH, NKV, D);
}

fn check_shape(dtype: DType, tol: f32, nh: usize, nkv: usize, d: usize) {
    let device = Device::Cuda(0);
    let b = 5usize;
    let nb = 6usize;

    let q = Tensor::from_vec::<_, f32>(noise(1, b * nh * d), vec![b, nh, d], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("q");
    let k = Tensor::from_vec::<_, f32>(noise(2, nkv * CAP * d), vec![nkv, CAP, d], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("k");
    let v = Tensor::from_vec::<_, f32>(noise(3, nkv * CAP * d), vec![nkv, CAP, d], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("v");

    // У всех запросов одинаковая длина набора — иначе эталон не собрать одной
    // формой; сами блоки и хвост у каждого свои.
    let rows: Vec<(Vec<u32>, (u32, u32))> = (0..b)
        .map(|i| {
            let blocks: Vec<u32> = (0..nb).map(|j| ((i * 7 + j * 5) % (CAP / RATIO)) as u32).collect();
            let mut blocks = blocks;
            blocks.sort_unstable();
            blocks.dedup();
            while blocks.len() < nb {
                let next = (blocks.last().copied().unwrap_or(0) + 1) % (CAP / RATIO) as u32;
                if !blocks.contains(&next) {
                    blocks.push(next);
                }
                blocks.sort_unstable();
            }
            (blocks, ((i as u32 % 3) * RATIO as u32 + 40, 3))
        })
        .collect();

    let want = gathered_reference(&q, &k, &v, &rows, device, nh, nkv, d);

    let table: Vec<u32> = rows.iter().flat_map(|(b, _)| b.iter().copied()).collect();
    let table = Tensor::from_vec(table, vec![b, nb], device).expect("таблица");
    let tail_from: Vec<u32> = rows.iter().map(|(_, (f, _))| *f).collect();
    let tail_len: Vec<u32> = rows.iter().map(|(_, (_, l))| *l).collect();
    let tail_from = Tensor::from_vec(tail_from, vec![b], device).expect("хвост");
    let tail_len = Tensor::from_vec(tail_len, vec![b], device).expect("длина хвоста");

    let got = q
        .flash_attention_blocks(&k, &v, &table, &tail_from, &tail_len, RATIO, 1.0 / (d as f32).sqrt())
        .expect("ядро по блокам");
    let got = host(&got);

    assert_eq!(got.len(), want.len());
    let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|x| (*x as f64).powi(2)).sum();
    let rel = (num / den.max(1e-12)).sqrt() as f32;
    eprintln!("{dtype:?}: блочное ядро против сборки rel_l2={rel:.3e}");
    assert!(rel < tol, "{dtype:?}: разошлось на {rel:.3e}");
}

/// Форма большой модели: голова 256 и двенадцать q-голов на kv-голову — там
/// состояние всех голов в регистры не влезает, и ядро берёт их пачками.
#[test]
fn blocks_match_gathered_wide_heads() {
    if !ready() {
        return;
    }
    check_shape(DType::F16, 4e-3, 24, 2, 256);
}

#[test]
fn blocks_match_gathered_f32() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(DType::F32, 1e-5);
}

#[test]
fn blocks_match_gathered_f16() {
    if !ready() {
        return;
    }
    check(DType::F16, 4e-3);
}
