use crate::tensor::storage::CpuBuf;

pub fn alloc_zeros_cpu(n_bytes: usize) -> CpuBuf { CpuBuf::alloc_zeros(n_bytes) }

pub fn alloc_uninit_cpu(n_bytes: usize) -> CpuBuf {
    if let Some(v) = pool::take(n_bytes) {
        return CpuBuf { bytes: v, pooled: true };
    }
    let mut v = Vec::with_capacity(n_bytes);
    #[allow(clippy::uninit_vec)]
    unsafe {
        v.set_len(n_bytes);
    }
    CpuBuf { bytes: v, pooled: n_bytes >= pool::MIN }
}

/// Вернуть системе буферы пула (конец запроса: между запросами память не держим).
pub fn release_cpu_pool() {
    pool::clear();
}

/// Пул крупных CPU-буферов: освобождённый буфер выхода операции (≥ 1 МБ) не отдаётся
/// системе, а ждёт следующего выделения того же размера. Без него каждая операция над
/// большими активациями платила `munmap` при освобождении и page fault на каждой странице
/// нового буфера (Whisper large на телефоне: ~45 мс на слой энкодера). Объём пула —
/// `SYNAPTIX_CPU_POOL_MB` (по умолчанию 256, 0 — выключен); один буфер — не больше 1/4 пула.
pub(crate) mod pool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    pub const MIN: usize = 1 << 20;

    struct Pool {
        bufs: HashMap<usize, Vec<Vec<u8>>>,
        bytes: usize,
    }

    static POOL: Mutex<Option<Pool>> = Mutex::new(None);

    fn limit() -> usize {
        static L: OnceLock<usize> = OnceLock::new();
        *L.get_or_init(|| {
            std::env::var("SYNAPTIX_CPU_POOL_MB").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(256) << 20
        })
    }

    pub fn take(n: usize) -> Option<Vec<u8>> {
        if n < MIN || limit() == 0 {
            return None;
        }
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let p = g.as_mut()?;
        let v = p.bufs.get_mut(&n)?.pop()?;
        p.bytes -= v.capacity();
        let mut v = v;
        // SAFETY: ёмкость == n; содержимое — прежние байты (буфер «uninit» по контракту).
        unsafe { v.set_len(n) };
        Some(v)
    }

    pub fn clear() {
        let old = POOL.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(old);
    }

    pub fn put(v: Vec<u8>) {
        let cap = v.capacity();
        let lim = limit();
        if cap < MIN || lim == 0 || cap > lim / 4 {
            return;
        }
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let p = g.get_or_insert_with(|| Pool { bufs: HashMap::new(), bytes: 0 });
        if p.bytes + cap > lim {
            return;
        }
        p.bytes += cap;
        p.bufs.entry(cap).or_default().push(v);
    }
}
