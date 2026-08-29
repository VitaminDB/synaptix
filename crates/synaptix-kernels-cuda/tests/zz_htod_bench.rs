use std::sync::Arc;

use rayon::prelude::*;

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

const BODY: usize = 2 << 20;
const N: usize = 512;
/// Источник заведомо больше кэша процессора: тело эксперта приезжает из
/// страничного кэша, а не из L3, и копия там вдвое медленнее.
const SRC_BODIES: usize = 512;

#[test]
#[ignore]
fn htod_stage_paths() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let src = vec![7u8; BODY * SRC_BODIES];
    let stream = synaptix_core::device::cuda::default_stream(0).expect("поток");
    let total = (BODY * N) as f64 / (1u64 << 30) as f64;

    let mut dst = synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY);
    let t = std::time::Instant::now();
    for i in 0..N {
        let off = (i % SRC_BODIES) * BODY;
        dst.as_mut_slice()[..BODY].copy_from_slice(&src[off..off + BODY]);
    }
    let one = t.elapsed().as_secs_f64();
    eprintln!("memcpy в pinned одним потоком: {:.1} ГБ/с", total / one);

    let t = std::time::Instant::now();
    for i in 0..N {
        let off = (i % SRC_BODIES) * BODY;
        let s = &src[off..off + BODY];
        dst.as_mut_slice()[..BODY]
            .par_chunks_mut(BODY / 4)
            .zip(s.par_chunks(BODY / 4))
            .for_each(|(d, s)| d.copy_from_slice(s));
    }
    let par = t.elapsed().as_secs_f64();
    eprintln!("memcpy в pinned четырьмя потоками: {:.1} ГБ/с", total / par);

    let _guard = synaptix_core::device::cuda::PinnedStageGuard::new();
    let t = std::time::Instant::now();
    for i in 0..N {
        let off = (i % SRC_BODIES) * BODY;
        let d = synaptix_core::device::cuda::pinned_htod_tls(&stream, &src[off..off + BODY])
            .expect("подкачка");
        std::hint::black_box(&d);
    }
    stream.synchronize().expect("синк");
    let tls = t.elapsed().as_secs_f64();
    eprintln!("подкачка pinned_htod_tls (как у экспертов): {:.1} ГБ/с", total / tls);

    // Конвейер вручную: два закреплённых буфера и два готовых приёмника,
    // копия следующего тела идёт под DMA текущего. Показывает, что остаётся
    // от пути, если убрать аллокацию приёмника и создание события.
    for ways in [2usize, 4, 8] {
        let mut bufs: Vec<synaptix_core::memory::pinned::PinnedBuf> =
            (0..ways).map(|_| synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY)).collect();
        let mut dsts: Vec<_> =
            (0..ways).map(|_| unsafe { stream.alloc::<u8>(BODY) }.expect("приёмник")).collect();
        let evs: Vec<_> =
            (0..ways).map(|_| stream.context().new_event(None).expect("событие")).collect();
        let mut armed = vec![false; ways];
        let t = std::time::Instant::now();
        for i in 0..N {
            let b = i % ways;
            if armed[b] {
                evs[b].synchronize().expect("ожидание");
            }
            let off = (i % SRC_BODIES) * BODY;
            bufs[b].as_mut_slice()[..BODY].copy_from_slice(&src[off..off + BODY]);
            stream.memcpy_htod(&bufs[b].as_slice()[..BODY], &mut dsts[b]).expect("dma");
            evs[b].record(&stream).expect("запись события");
            armed[b] = true;
        }
        stream.synchronize().expect("синк");
        let pipe = t.elapsed().as_secs_f64();
        eprintln!("конвейер на {ways} буферах: {:.1} ГБ/с", total / pipe);
    }

    #[allow(unreachable_code)]
    if false {
        let mut bufs = [
            synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY),
            synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY),
        ];
        let mut dsts = [
            unsafe { stream.alloc::<u8>(BODY) }.expect("приёмник"),
            unsafe { stream.alloc::<u8>(BODY) }.expect("приёмник"),
        ];
        let evs = [
            stream.context().new_event(None).expect("событие"),
            stream.context().new_event(None).expect("событие"),
        ];
        let mut armed = [false, false];
        let t = std::time::Instant::now();
        for i in 0..N {
            let b = i & 1;
            if armed[b] {
                evs[b].synchronize().expect("ожидание");
            }
            let off = (i % SRC_BODIES) * BODY;
            bufs[b].as_mut_slice()[..BODY].copy_from_slice(&src[off..off + BODY]);
            stream.memcpy_htod(&bufs[b].as_slice()[..BODY], &mut dsts[b]).expect("dma");
            evs[b].record(&stream).expect("запись события");
            armed[b] = true;
        }
        stream.synchronize().expect("синк");
        let pipe = t.elapsed().as_secs_f64();
        eprintln!("конвейер вручную (буфер и приёмник готовы): {:.1} ГБ/с", total / pipe);
    }

    // Подкачка врозь: у каждого потока свои закреплённые буферы, тела разные.
    // Одному потоку копия упирается в память раньше, чем в DMA.
    for ways in [2usize, 4] {
        let src = &src;
        let t = std::time::Instant::now();
        std::thread::scope(|sc| {
            for w in 0..ways {
                sc.spawn(move || {
                    let stream = synaptix_core::device::cuda::default_stream(0).expect("поток");
                    let _guard = synaptix_core::device::cuda::PinnedStageGuard::new();
                    for i in (w..N).step_by(ways) {
                        let off = (i % SRC_BODIES) * BODY;
                        let d = synaptix_core::device::cuda::pinned_htod_tls(
                            &stream,
                            &src[off..off + BODY],
                        )
                        .expect("подкачка");
                        std::hint::black_box(&d);
                    }
                    stream.synchronize().expect("синк");
                });
            }
        });
        let par = t.elapsed().as_secs_f64();
        eprintln!("подкачка {ways} потоками: {:.1} ГБ/с", total / par);
    }

    let dev: Arc<_> = stream.clone();
    let mut pinned = synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY);
    pinned.as_mut_slice()[..BODY].copy_from_slice(&src[..BODY]);
    let t = std::time::Instant::now();
    for _ in 0..N {
        let mut d = unsafe { dev.alloc::<u8>(BODY) }.expect("alloc");
        dev.memcpy_htod(&pinned.as_slice()[..BODY], &mut d).expect("dma");
    }
    dev.synchronize().expect("синк");
    let dma = t.elapsed().as_secs_f64();
    eprintln!("чистая DMA из готового pinned: {:.1} ГБ/с", total / dma);
}
