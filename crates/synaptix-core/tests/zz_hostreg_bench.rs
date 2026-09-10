//! Бенч подкачки экспертов из mmap-бандла: staging через pinned-буфер потока
//! (как сейчас у `FetchJob`) против прямой DMA из страниц страничного кэша,
//! зарегистрированных `cuMemHostRegister(READ_ONLY)`.
//!
//! `SYN_HOSTREG_BUNDLE=<путь .syn> cargo test -p synaptix-core --release
//! --test zz_hostreg_bench -- --ignored --nocapture`

use std::sync::Arc;

use cudarc::driver::sys;
use synaptix_core::device::cuda;

const BODY: usize = 2_800_000; // ≈ эксперт qwen3.8-flash-next
const REGION: usize = 4 << 30;
const OFFSET: usize = 16 << 30;

struct Mmap {
    ptr: *mut u8,
    len: usize,
}
impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut _, self.len) };
    }
}

fn mmap_file(path: &str) -> Mmap {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path).expect("open");
    let len = f.metadata().expect("meta").len() as usize;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            f.as_raw_fd(),
            0,
        )
    };
    assert!(ptr != libc::MAP_FAILED, "mmap");
    Mmap { ptr: ptr as *mut u8, len }
}

fn gbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64 / secs
}

/// DMA тел подряд из `src` в кольцо приёмников, событие на буфер.
fn dma_bodies(stream: &Arc<cudarc::driver::CudaStream>, src: &[u8], ways: usize) -> f64 {
    let n = src.len() / BODY;
    let mut dsts: Vec<_> =
        (0..ways).map(|_| unsafe { stream.alloc::<u8>(BODY) }.expect("dst")).collect();
    let evs: Vec<_> = (0..ways).map(|_| stream.context().new_event(None).expect("ev")).collect();
    let mut armed = vec![false; ways];
    let t = std::time::Instant::now();
    for i in 0..n {
        let b = i % ways;
        if armed[b] {
            evs[b].synchronize().expect("wait");
        }
        stream.memcpy_htod(&src[i * BODY..(i + 1) * BODY], &mut dsts[b]).expect("dma");
        evs[b].record(stream).expect("rec");
        armed[b] = true;
    }
    stream.synchronize().expect("sync");
    gbs(n * BODY, t.elapsed().as_secs_f64())
}

#[test]
#[ignore]
fn hostreg_vs_staging() {
    let Ok(path) = std::env::var("SYN_HOSTREG_BUNDLE") else {
        eprintln!("SYN_HOSTREG_BUNDLE не задан — пропуск");
        return;
    };
    let Ok(stream) = cuda::default_stream(0) else {
        eprintln!("CUDA недоступна — пропуск");
        return;
    };
    let dev = stream.context();
    let ro = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED)
        .expect("attr");
    eprintln!("READ_ONLY_HOST_REGISTER_SUPPORTED = {ro}");

    let map = mmap_file(&path);
    assert!(map.len >= OFFSET + REGION, "бандл меньше {} ГБ", (OFFSET + REGION) >> 30);
    let region = unsafe { std::slice::from_raw_parts(map.ptr.add(OFFSET), REGION) };

    // Прогрев страничного кэша: как в бою, где бандл уже читали.
    let t = std::time::Instant::now();
    let mut acc = 0u64;
    for i in (0..REGION).step_by(4096) {
        acc = acc.wrapping_add(region[i] as u64);
    }
    eprintln!("прогрев страниц: {:.2} с (acc {acc})", t.elapsed().as_secs_f64());

    // 1. Текущий путь: pinned_htod_tls (memcpy в pinned-буфер потока + DMA).
    {
        let _g = cuda::PinnedStageGuard::new();
        let n = REGION / BODY;
        let t = std::time::Instant::now();
        for i in 0..n {
            let d = cuda::pinned_htod_tls(&stream, &region[i * BODY..(i + 1) * BODY]).expect("tls");
            std::hint::black_box(&d);
        }
        stream.synchronize().expect("sync");
        eprintln!("staging pinned_htod_tls, 1 поток: {:.1} ГБ/с", gbs(n * BODY, t.elapsed().as_secs_f64()));
    }
    // То же в 4 потока (как FetchJob::run через rayon).
    {
        let n = REGION / BODY;
        let t = std::time::Instant::now();
        std::thread::scope(|s| {
            for w in 0..4 {
                let stream = stream.clone();
                s.spawn(move || {
                    let _g = cuda::PinnedStageGuard::new();
                    for i in (w..n).step_by(4) {
                        let d = cuda::pinned_htod_tls(&stream, &region[i * BODY..(i + 1) * BODY])
                            .expect("tls");
                        std::hint::black_box(&d);
                    }
                });
            }
        });
        stream.synchronize().expect("sync");
        eprintln!("staging pinned_htod_tls, 4 потока: {:.1} ГБ/с", gbs(n * BODY, t.elapsed().as_secs_f64()));
    }

    // 2. Регистрация страниц mmap как pinned и прямая DMA.
    let t = std::time::Instant::now();
    let rc = unsafe {
        sys::cuMemHostRegister_v2(
            region.as_ptr() as *mut _,
            REGION,
            sys::CU_MEMHOSTREGISTER_READ_ONLY,
        )
    };
    eprintln!(
        "cuMemHostRegister(READ_ONLY) {} ГБ: {:?} за {:.2} с",
        REGION >> 30,
        rc,
        t.elapsed().as_secs_f64()
    );
    if rc != sys::CUresult::CUDA_SUCCESS {
        let rc2 = unsafe {
            sys::cuMemHostRegister_v2(region.as_ptr() as *mut _, REGION, 0)
        };
        eprintln!("cuMemHostRegister(0): {rc2:?}");
        if rc2 != sys::CUresult::CUDA_SUCCESS {
            return;
        }
    }
    for ways in [2usize, 4, 8] {
        eprintln!("DMA из зарегистрированных страниц, {ways} приёмников: {:.1} ГБ/с", dma_bodies(&stream, region, ways));
    }
    // Тот же путь, что в бою: alloc приёмника из пула + memcpy_htod + событие.
    {
        let n = REGION / BODY;
        let t = std::time::Instant::now();
        for i in 0..n {
            let mut dst = unsafe { cuda::alloc_bytes_uninit(&stream, BODY) }.expect("alloc");
            stream.memcpy_htod(&region[i * BODY..(i + 1) * BODY], &mut dst).expect("dma");
            std::hint::black_box(&dst);
        }
        stream.synchronize().expect("sync");
        eprintln!("DMA из зарегистрированных + alloc из пула, 1 поток: {:.1} ГБ/с", gbs(n * BODY, t.elapsed().as_secs_f64()));
    }
    let rc = unsafe { sys::cuMemHostUnregister(region.as_ptr() as *mut _) };
    eprintln!("cuMemHostUnregister: {rc:?}");
}
