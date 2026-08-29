use std::sync::Arc;

use rayon::prelude::*;

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

const BODY: usize = 2 << 20;
const N: usize = 512;

#[test]
#[ignore]
fn htod_stage_paths() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let src = vec![7u8; BODY * 8];
    let stream = synaptix_core::device::cuda::default_stream(0).expect("поток");
    let total = (BODY * N) as f64 / (1u64 << 30) as f64;

    let mut dst = synaptix_core::memory::pinned::PinnedBuf::new_uninit(BODY);
    let t = std::time::Instant::now();
    for i in 0..N {
        let off = (i % 8) * BODY;
        dst.as_mut_slice()[..BODY].copy_from_slice(&src[off..off + BODY]);
    }
    let one = t.elapsed().as_secs_f64();
    eprintln!("memcpy в pinned одним потоком: {:.1} ГБ/с", total / one);

    let t = std::time::Instant::now();
    for i in 0..N {
        let off = (i % 8) * BODY;
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
        let off = (i % 8) * BODY;
        let d = synaptix_core::device::cuda::pinned_htod_tls(&stream, &src[off..off + BODY])
            .expect("подкачка");
        std::hint::black_box(&d);
    }
    stream.synchronize().expect("синк");
    let tls = t.elapsed().as_secs_f64();
    eprintln!("подкачка pinned_htod_tls (как у экспертов): {:.1} ГБ/с", total / tls);

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
