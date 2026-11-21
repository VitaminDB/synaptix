//! Zero-copy декод MoE (`moe_zc`): эксперты в закреплённой RAM, кэш слотов
//! VRAM на карте. Таблицы шага (`stage`) обязаны давать в индексных GEMV
//! бит в бит тот же ответ, что таблица по резидентным весам
//! (`ExpertTable::build_dense`): те же ядра, те же байты — меняется лишь,
//! откуда они читаются (слот VRAM или RAM).
//!
//! Форматы: NVFP4 и MXFP8 (packed + scales, слот из четырёх блобов) и SQ4
//! (одноблобный, прежний путь). Слотов меньше экспертов: шаги со случайными
//! idx проходят попадания, промахи с вытеснением, «нет свободного слота»
//! (пар больше слотов) и неверный idx.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::memory::pinned::PinnedBuf;
use synaptix_core::tensor::quant::{ExpertTable, QuantWeight};
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::moe_zc::{ExpertBlobs, ZcLayer, ZcPool};

const E: usize = 128;
const H: usize = 256;
const I: usize = 256;

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn f(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
    }
}

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    (0..n).map(|_| r.f()).collect()
}

/// Стопка экспертов одной матрицы: packed и scales всех экспертов подряд в
/// закреплённой RAM + резидентные копии на карте (эталон).
struct Stack {
    packed: PinnedBuf,
    scales: Option<PinnedBuf>,
    p_bytes: usize,
    s_bytes: usize,
    resident: Vec<QuantWeight>,
}

impl Stack {
    fn build(dtype: DType, n: usize, k: usize, seed: u64) -> Stack {
        let dev = Device::Cuda(0);
        let (p_bytes, s_bytes) = synaptix_core::quant::matrix_blob_bytes(dtype, n, k).expect("размеры");
        let mut packed = PinnedBuf::new(E * p_bytes);
        let mut scales = (s_bytes > 0).then(|| PinnedBuf::new(E * s_bytes));
        let mut resident = Vec::with_capacity(E);
        for e in 0..E {
            let w = noise(seed * 1000 + e as u64, n * k);
            use synaptix_kernels_cpu::quant::encode as enc;
            let (pb, sb) = match dtype {
                DType::NVFP4 => enc::quantize_nvfp4_rows(&w, n, k).map(|(p, s)| (p, Some(s))),
                DType::MXFP8 => enc::quantize_mxfp8_rows(&w, n, k).map(|(p, s)| (p, Some(s))),
                DType::Sq { bits } => enc::quantize_sq_rows(bits, &w, n, k).map(|p| (p, None)),
                _ => unreachable!(),
            }
            .expect("квант");
            assert_eq!(pb.len(), p_bytes, "{dtype:?}: размер packed");
            assert_eq!(sb.as_ref().map_or(0, |s| s.len()), s_bytes, "{dtype:?}: размер scales");
            packed.as_mut_slice()[e * p_bytes..(e + 1) * p_bytes].copy_from_slice(&pb);
            let res = if let (Some(sc), Some(sb)) = (scales.as_mut(), sb) {
                sc.as_mut_slice()[e * s_bytes..(e + 1) * s_bytes].copy_from_slice(&sb);
                // Резидент — из тех же байт, что в RAM.
                let pt = Tensor::from_raw_slice(&pb, vec![pb.len()], DType::U8, dev).unwrap();
                let stt = Tensor::from_raw_slice(&sb, vec![sb.len()], DType::U8, dev).unwrap();
                QuantWeight::new(pt.storage_arc(), stt.storage_arc(), dtype, n, k).unwrap()
            } else {
                let pt = Tensor::from_raw_slice(&pb, vec![pb.len()], DType::U8, dev).unwrap();
                QuantWeight::new_block(pt.storage_arc(), dtype, n, k).unwrap()
            };
            resident.push(res);
        }
        Stack { packed, scales, p_bytes, s_bytes, resident }
    }

    fn addr(&self, e: usize) -> (u64, u64) {
        let p = self.packed.as_ptr() as u64 + (e * self.p_bytes) as u64;
        let s = self.scales.as_ref().map_or(0, |b| b.as_ptr() as u64 + (e * self.s_bytes) as u64);
        (p, s)
    }

    fn table(&self) -> ExpertTable {
        let refs: Vec<&QuantWeight> = self.resident.iter().collect();
        ExpertTable::build_dense(&refs).expect("эталонная таблица")
    }
}

fn raw(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).and_then(|t| t.flatten_all()).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()).unwrap()
}

/// Прогон одного формата: `slots` слотов, шаги — (пар, сколько из них
/// неверных idx, повтор прошлого шага).
fn run(dtype: DType, slots: usize, min_slots: usize, steps: &[(usize, usize, bool)]) {
    let dev = Device::Cuda(0);
    let gu = Stack::build(dtype, 2 * I, H, 1);
    let dn = Stack::build(dtype, H, I, 2);
    let blobs: Vec<ExpertBlobs> = (0..E)
        .map(|e| {
            let (g, gs) = gu.addr(e);
            let (d, ds) = dn.addr(e);
            ExpertBlobs { gu: g, gu_s: gs, dn: d, dn_s: ds }
        })
        .collect();
    let pool = ZcPool::new(dev);
    pool.set_min_slots(min_slots);
    pool.set_slots_per_layer(slots);
    // Затравка: первые 10 экспертов «частые» — сразу в слотах.
    let seed: Vec<f32> = (0..E).map(|e| if e < 10 { 100.0 - e as f32 } else { 0.0 }).collect();
    let layer = ZcLayer::build_from_blobs(&blobs, dtype, (2 * I, H), (H, I), &pool, Some(&seed))
        .expect("сборка")
        .expect("слой собран");
    assert_eq!(layer.split_scales(), gu.s_bytes > 0);
    assert_eq!(layer.slots(), slots);
    let ref_gu = gu.table();
    let ref_dn = dn.table();

    let mut rng = Rng(0xC0FFEE ^ dtype_tag(dtype));
    let mut prev: Vec<u32> = Vec::new();
    let mut total_pairs = 0u64;
    let mut forced_hits = 0u64;
    for (si, &(pairs, bad, repeat)) in steps.iter().enumerate() {
        let ids: Vec<u32> = if repeat {
            prev.clone()
        } else {
            let mut v: Vec<u32> = (0..pairs).map(|_| (rng.next() % E as u64) as u32).collect();
            for b in 0..bad {
                v[pairs - 1 - b] = E as u32 + 7 + b as u32; // неверный idx
            }
            v
        };
        let (h0, p0) = layer.stats();
        let idx = Tensor::from_vec::<_, u32>(ids.clone(), vec![ids.len()], dev).unwrap();
        let x = Tensor::from_vec::<_, f32>(noise(500 + si as u64, ids.len() * H), vec![ids.len(), H], dev)
            .and_then(|t| t.to_dtype(DType::F16))
            .unwrap();
        let hh = Tensor::from_vec::<_, f32>(noise(900 + si as u64, ids.len() * I), vec![ids.len(), I], dev)
            .and_then(|t| t.to_dtype(DType::F16))
            .unwrap();
        let (gu_t, dn_t, iota) = layer.stage(&pool, &idx).expect("stage");
        let got_gu = gu_t.gemv_indexed_dense(&iota, &x, true).expect("zc gate_up");
        let got_dn = dn_t.gemv_indexed_dense(&iota, &hh, true).expect("zc down");
        // Эталон — только по верным парам (неверные эталон прочитать не может).
        let valid: Vec<usize> = (0..ids.len()).filter(|p| (ids[*p] as usize) < E).collect();
        let vids: Vec<u32> = valid.iter().map(|p| ids[*p]).collect();
        let vidx = Tensor::from_vec::<_, u32>(vids.clone(), vec![vids.len()], dev).unwrap();
        let vrows = Tensor::from_vec::<_, u32>(valid.iter().map(|p| *p as u32).collect(), vec![valid.len()], dev).unwrap();
        let vx = x.index_select(0, &vrows).unwrap();
        let vh = hh.index_select(0, &vrows).unwrap();
        let want_gu = ref_gu.gemv_indexed_dense(&vidx, &vx, true).unwrap();
        let want_dn = ref_dn.gemv_indexed_dense(&vidx, &vh, true).unwrap();
        let (g, d) = (raw(&got_gu), raw(&got_dn));
        let (wg, wd) = (raw(&want_gu), raw(&want_dn));
        for (j, p) in valid.iter().enumerate() {
            let (n_gu, n_dn) = (2 * I, H);
            let a = &g[p * n_gu..(p + 1) * n_gu];
            let b = &wg[j * n_gu..(j + 1) * n_gu];
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{dtype:?} шаг {si}: gate_up пары {p} (эксперт {}) расходится", ids[*p]);
            let a = &d[p * n_dn..(p + 1) * n_dn];
            let b = &wd[j * n_dn..(j + 1) * n_dn];
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{dtype:?} шаг {si}: down пары {p} (эксперт {}) расходится", ids[*p]);
        }
        assert!(g.iter().chain(d.iter()).all(|v| v.is_finite()), "{dtype:?} шаг {si}: не конечные значения (неверный idx?)");
        let (h1, p1) = layer.stats();
        let mut uniq = vids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(p1 - p0, ids.len() as u64, "{dtype:?} шаг {si}: счётчик выборов");
        assert!(h1 - h0 <= uniq.len() as u64, "{dtype:?} шаг {si}: попаданий {} > уникальных {}", h1 - h0, uniq.len());
        if repeat && uniq.len() <= slots {
            // Повтор шага, влезшего в слоты: все уникальные — попадания.
            assert_eq!(h1 - h0, uniq.len() as u64, "{dtype:?} шаг {si}: повтор должен попасть целиком");
            forced_hits += uniq.len() as u64;
        }
        total_pairs += ids.len() as u64;
        println!("{dtype:?} шаг {si}: пар {}, уникальных {}, попаданий {}", ids.len(), uniq.len(), h1 - h0);
        prev = ids;
    }
    let (h, p) = layer.stats();
    assert_eq!(p, total_pairs);
    assert!(h >= forced_hits && h < p, "{dtype:?}: попаданий {h} из {p}");
    println!("{dtype:?}: слотов {slots}, попаданий {h} из {p} ({:.0} %)", 100.0 * h as f64 / p as f64);
}

fn dtype_tag(d: DType) -> u64 {
    match d {
        DType::NVFP4 => 1,
        DType::MXFP8 => 2,
        _ => 3,
    }
}

/// Шаги: декод (8 и 32 пары — варп-версия часов), повторы (попадания),
/// пачка в 100 пар при 64 слотах (последовательная версия, «нет свободного
/// слота»), неверные idx.
fn steps_main() -> Vec<(usize, usize, bool)> {
    vec![
        (8, 0, false),
        (8, 0, true),
        (32, 0, false),
        (32, 0, true),
        (8, 1, false),
        (100, 0, false),
        (100, 3, false),
        (32, 2, false),
        (24, 0, false),
        (24, 0, true),
        (64, 0, false),
        (8, 0, false),
    ]
}

#[test]
fn zc_nvfp4_matches_resident_bitwise() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    run(DType::NVFP4, 64, 64, &steps_main());
}

#[test]
fn zc_mxfp8_matches_resident_bitwise() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    run(DType::MXFP8, 64, 64, &steps_main());
}

#[test]
fn zc_sq4_one_blob_unchanged() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    run(DType::Sq { bits: 4 }, 64, 64, &steps_main());
}

/// 16 слотов: «нет свободного слота» в варп-версии (32 пары > 16 слотов).
#[test]
fn zc_split_few_slots_warp_overflow() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    let steps = vec![(32, 0, false), (32, 1, false), (8, 0, false), (8, 0, true), (32, 0, false), (100, 2, false), (16, 0, false), (16, 0, true)];
    run(DType::NVFP4, 16, 16, &steps);
    run(DType::MXFP8, 16, 16, &steps);
}

/// Упреждающая подкачка (второй стрим) и роутер на карте: prefetch по входу
/// поднимает top-k в слоты, следующий `stage_routed` по тому же входу
/// обязан попасть во все уникальные и дать эталонный ответ бит в бит.
#[test]
fn zc_split_prefetch_and_routed() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    let dev = Device::Cuda(0);
    for dtype in [DType::NVFP4, DType::MXFP8] {
        let gu = Stack::build(dtype, 2 * I, H, 3);
        let dn = Stack::build(dtype, H, I, 4);
        let blobs: Vec<ExpertBlobs> = (0..E)
            .map(|e| {
                let (g, gs) = gu.addr(e);
                let (d, ds) = dn.addr(e);
                ExpertBlobs { gu: g, gu_s: gs, dn: d, dn_s: ds }
            })
            .collect();
        let pool = ZcPool::new(dev);
        pool.set_slots_per_layer(64);
        let layer = ZcLayer::build_from_blobs(&blobs, dtype, (2 * I, H), (H, I), &pool, None).unwrap().unwrap();
        let (ref_gu, ref_dn) = (gu.table(), dn.table());
        let router = Tensor::from_vec::<_, f32>(noise(77, E * H), vec![E, H], dev).unwrap();
        for (si, rows) in [1usize, 4, 2].into_iter().enumerate() {
            let k = 8;
            let x = Tensor::from_vec::<_, f32>(noise(300 + si as u64, rows * H), vec![rows, H], dev)
                .and_then(|t| t.to_dtype(DType::F16))
                .unwrap();
            layer.prefetch(&pool, &router, &x, k, 64).expect("prefetch");
            let (h0, _) = layer.stats();
            let idx = Tensor::zeros(vec![rows * k], DType::U32, dev).unwrap();
            let wts = Tensor::zeros(vec![rows * k], DType::F32, dev).unwrap();
            let (gu_t, dn_t, iota) = layer.stage_routed(&pool, &idx, Some((&router, &x, &wts))).expect("stage_routed");
            let rows_idx: Vec<u32> = (0..rows * k).map(|p| (p / k) as u32).collect();
            let rt = Tensor::from_vec::<_, u32>(rows_idx, vec![rows * k], dev).unwrap();
            let xr = x.index_select(0, &rt).unwrap();
            let hh = Tensor::from_vec::<_, f32>(noise(700 + si as u64, rows * k * I), vec![rows * k, I], dev)
                .and_then(|t| t.to_dtype(DType::F16))
                .unwrap();
            let got = (raw(&gu_t.gemv_indexed_dense(&iota, &xr, true).unwrap()), raw(&dn_t.gemv_indexed_dense(&iota, &hh, true).unwrap()));
            let want = (raw(&ref_gu.gemv_indexed_dense(&idx, &xr, true).unwrap()), raw(&ref_dn.gemv_indexed_dense(&idx, &hh, true).unwrap()));
            assert!(got.0.iter().zip(&want.0).all(|(a, b)| a.to_bits() == b.to_bits()), "{dtype:?} шаг {si}: gate_up");
            assert!(got.1.iter().zip(&want.1).all(|(a, b)| a.to_bits() == b.to_bits()), "{dtype:?} шаг {si}: down");
            let ids = idx.to_device(Device::Cpu).and_then(|t| t.to_vec1::<u32>()).unwrap();
            let mut u = ids.clone();
            u.sort_unstable();
            u.dedup();
            let (h1, _) = layer.stats();
            println!("{dtype:?} упреждение, строк {rows}: уникальных {}, попаданий {}", u.len(), h1 - h0);
            assert_eq!(h1 - h0, u.len() as u64, "{dtype:?} шаг {si}: после упреждения всё должно попасть");
        }
    }
}

/// Префилл: групповой GEMM по таблицам шага (запись = уникальный эксперт
/// группы) против таблицы резидентов — бит в бит, в том числе для
/// экспертов, не влезших в слоты (читаются из RAM).
#[test]
fn zc_split_grouped_gemm_matches() {
    if !setup() {
        eprintln!("CUDA нет — пропуск");
        return;
    }
    let dev = Device::Cuda(0);
    for dtype in [DType::NVFP4, DType::MXFP8, DType::Sq { bits: 4 }] {
        let gu = Stack::build(dtype, 2 * I, H, 5);
        let dn = Stack::build(dtype, H, I, 6);
        let blobs: Vec<ExpertBlobs> = (0..E)
            .map(|e| {
                let (g, gs) = gu.addr(e);
                let (d, ds) = dn.addr(e);
                ExpertBlobs { gu: g, gu_s: gs, dn: d, dn_s: ds }
            })
            .collect();
        let pool = ZcPool::new(dev);
        pool.set_slots_per_layer(64);
        let layer = ZcLayer::build_from_blobs(&blobs, dtype, (2 * I, H), (H, I), &pool, None).unwrap().unwrap();
        let (ref_gu, ref_dn) = (gu.table(), dn.table());
        for fp8 in [false, true] {
            // 80 уникальных экспертов (> 64 слотов), по 1–4 строки на эксперта.
            let mut rng = Rng(42 + fp8 as u64);
            let mut ids: Vec<u32> = (0..E as u32).collect();
            for i in (1..ids.len()).rev() {
                ids.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            ids.truncate(80);
            let mut seg_local = Vec::new();
            let mut seg_ref = Vec::new();
            let mut r = 0u32;
            for (u, e) in ids.iter().enumerate() {
                let n = 1 + (rng.next() % 4) as u32;
                seg_local.push((u as u32, r, r + n));
                seg_ref.push((*e, r, r + n));
                r += n;
            }
            let x = Tensor::from_vec::<_, f32>(noise(1234, r as usize * H), vec![r as usize, H], dev).and_then(|t| t.to_dtype(DType::F16)).unwrap();
            let hh = Tensor::from_vec::<_, f32>(noise(4321, r as usize * I), vec![r as usize, I], dev).and_then(|t| t.to_dtype(DType::F16)).unwrap();
            let idx = Tensor::from_vec::<_, u32>(ids.clone(), vec![ids.len()], dev).unwrap();
            let (gu_t, dn_t, _) = layer.stage(&pool, &idx).unwrap();
            let got = (raw(&gu_t.gemm_grouped_dense_rows(&x, None, &seg_local, fp8).unwrap()), raw(&dn_t.gemm_grouped_dense_rows(&hh, None, &seg_local, fp8).unwrap()));
            let want = (raw(&ref_gu.gemm_grouped_dense_rows(&x, None, &seg_ref, fp8).unwrap()), raw(&ref_dn.gemm_grouped_dense_rows(&hh, None, &seg_ref, fp8).unwrap()));
            assert!(got.0.iter().zip(&want.0).all(|(a, b)| a.to_bits() == b.to_bits()), "{dtype:?} fp8={fp8}: gate_up");
            assert!(got.1.iter().zip(&want.1).all(|(a, b)| a.to_bits() == b.to_bits()), "{dtype:?} fp8={fp8}: down");
            println!("{dtype:?} fp8={fp8}: групповой GEMM {r} строк, 80 экспертов — бит в бит");
        }
    }
}
