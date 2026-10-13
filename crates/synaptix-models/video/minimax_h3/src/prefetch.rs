//! Прогрев page cache под mmap весов: пока одни тензоры едут на карту,
//! следующие подкачиваются с диска — чтение перекрывается с H2D. Байты не
//! меняются, только страницы попадают в память раньше, чем их тронет загрузка.

use std::sync::atomic::{AtomicUsize, Ordering};

const THREADS: usize = 16;
const CHUNK: usize = 64 << 20;
const MADV_POPULATE_READ: i32 = 22;

/// Подкачать срезы mmap: `MADV_POPULATE_READ` (Linux 5.14+, ядро читает
/// крупными запросами) кусками по 64 МБ в 16 потоках; не вышло — касание по
/// байту на страницу.
pub fn populate(ranges: &[&[u8]]) {
    let chunks: Vec<&[u8]> = ranges.iter().flat_map(|b| b.chunks(CHUNK)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..THREADS.min(chunks.len()) {
            sc.spawn(|| loop {
                let Some(c) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let a0 = (c.as_ptr() as usize) & !4095;
                let len = c.as_ptr() as usize + c.len() - a0;
                if unsafe { libc::madvise(a0 as *mut libc::c_void, len, MADV_POPULATE_READ) } != 0 {
                    let mut acc = 0u8;
                    for off in (0..c.len()).step_by(4096) {
                        acc ^= unsafe { std::ptr::read_volatile(c.as_ptr().add(off)) };
                    }
                    std::hint::black_box(acc);
                }
            });
        }
    });
}
