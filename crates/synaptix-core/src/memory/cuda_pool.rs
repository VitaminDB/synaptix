
use std::sync::atomic::{AtomicUsize, Ordering};

static POOL_ALLOC: AtomicUsize = AtomicUsize::new(0);
static POOL_PEAK: AtomicUsize = AtomicUsize::new(0);
static POOL_TRIM_THRESHOLD: AtomicUsize = AtomicUsize::new(0);

/// Топ живых аллокаций (bytes, count), убыв. по суммарному объёму.
pub fn live_alloc_top(_n: usize) -> Vec<(usize, isize)> {
    Vec::new()
}

pub fn record_cuda_alloc(bytes: usize) {
    let prev = POOL_ALLOC.fetch_add(bytes, Ordering::Relaxed);
    let new_total = prev + bytes;
    let mut peak = POOL_PEAK.load(Ordering::Relaxed);
    while new_total > peak {
        match POOL_PEAK.compare_exchange_weak(peak, new_total, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(cur) => peak = cur,
        }
    }
}

pub fn record_cuda_free(bytes: usize) {
    POOL_ALLOC.fetch_sub(bytes.min(POOL_ALLOC.load(Ordering::Relaxed)), Ordering::Relaxed);
}

pub fn cuda_allocated_bytes() -> usize { POOL_ALLOC.load(Ordering::Relaxed) }
pub fn cuda_peak_bytes() -> usize { POOL_PEAK.load(Ordering::Relaxed) }
pub fn cuda_allocated_mb() -> f64 { cuda_allocated_bytes() as f64 / 1_048_576.0 }
pub fn cuda_peak_mb() -> f64 { cuda_peak_bytes() as f64 / 1_048_576.0 }
pub fn reset_peak() { POOL_PEAK.store(cuda_allocated_bytes(), Ordering::Relaxed); }

/// Порог в байтах — сколько памяти оставить в default-mempool после trim'a. По умолчанию 0 =
/// агрессивный trim (вернуть драйверу всё, что не in-use). Используется в [`trim_cuda_mempool`].
/// Программный сеттер [`set_trim_threshold`] задаёт PyTorch-стиль удержания free-list (пул
/// копит и реюзает вместо возврата ОС; постоянные тримы-в-ноль пересоздавали сегменты
/// и пере-чередовали живые с транзиентами → нераздаваемое решето на длинных T).
pub fn set_trim_threshold(bytes: usize) { POOL_TRIM_THRESHOLD.store(bytes, Ordering::Relaxed); }
pub fn trim_threshold() -> usize {
    POOL_TRIM_THRESHOLD.load(Ordering::Relaxed)
}

/// Вернуть драйверу неиспользуемые блоки default-mempool устройства 0.
///
/// На CPU-сборке (`not(feature = "cuda")`) — no-op (счётчики не трогаем — они отражают наш
/// учёт, а не реальный heap). На CUDA-сборке вызывает `cuMemPoolTrimTo(pool, threshold)`,
/// где threshold = [`trim_threshold()`].

pub fn trim_cuda_mempool() {
    let _ = trim_cuda_mempool_device(0);
}

/// То же, что [`trim_cuda_mempool`], но для конкретного device-ordinal и с пробросом ошибки.
pub fn trim_cuda_mempool_device(ordinal: usize) -> crate::error::Result<()> {
    use crate::error::SynaptixError;
    let ctx = crate::device::cuda::get(ordinal)?;
    ctx.bind_to_thread()
        .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
    let pool = unsafe { cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device()) }
        .map_err(|e| SynaptixError::Cuda(format!("cuDeviceGetDefaultMemPool: {e:?}")))?;
    unsafe { cudarc::driver::result::mem_pool::trim_to(pool, trim_threshold()) }
        .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolTrimTo: {e:?}")))?;
    Ok(())
}

/// Полный трим независимо от текущего trim-threshold (временно сбрасывает в 0):
/// для граничных компактов, когда в горячей фазе пул работает с высоким
/// threshold (держит free-list стабильным), а на границе нужно вернуть всё.
pub fn hard_trim_cuda_mempool_device(ordinal: usize) -> crate::error::Result<()> {
    let saved = trim_threshold();
    set_trim_threshold(0);
    let r = trim_cuda_mempool_device(ordinal);
    set_trim_threshold(saved);
    let _ = trim_graph_mem_device(ordinal);
    r
}

/// Вернуть драйверу НЕиспользуемую память graph mem pool (alloc-ноды CUDA-graph
/// живут в отдельном пуле драйвера; после Drop графов physical-страницы висят
/// зарезервированными — обычный cuMemPoolTrimTo их не видит, VAE-бюджет
/// занижается → другой тайлинг). Память живых графов не трогается.
pub fn trim_graph_mem_device(ordinal: usize) -> crate::error::Result<()> {
    use crate::error::SynaptixError;
    let ctx = crate::device::cuda::get(ordinal)?;
    ctx.bind_to_thread()
        .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
    let r = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(ctx.cu_device()) };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(SynaptixError::Cuda(format!("cuDeviceGraphMemTrim: {r:?}")));
    }
    Ok(())
}

/// Полный трим default-пула И weights-пула (перед фазами, меряющими свободную
/// VRAM, — иначе удержанные пулами блоки занижают бюджет недетерминированно).
pub fn hard_trim_all_pools_device(ordinal: usize) -> crate::error::Result<()> {
    let _ = crate::device::cuda::synchronize_all(ordinal);
    let r = hard_trim_cuda_mempool_device(ordinal);
    let _ = crate::device::cuda::trim_weights_pool(ordinal);
    r
}




/// (reserved, used) байт default-пула: RESERVED = удержано у драйвера (вкл.
/// свободные сегменты), USED = живые с точки зрения пула. reserved≫used =
/// фрагментация сегментов; used≫наш live-учёт = неучтённые аллокации.
pub fn cuda_mempool_stats(ordinal: usize) -> crate::error::Result<(u64, u64)> {
    use crate::error::SynaptixError;
    use cudarc::driver::sys;
    let ctx = crate::device::cuda::get(ordinal)?;
    ctx.bind_to_thread()
        .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
    let pool = unsafe { cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device()) }
        .map_err(|e| SynaptixError::Cuda(format!("cuDeviceGetDefaultMemPool: {e:?}")))?;
    let get = |attr: sys::CUmemPool_attribute| -> crate::error::Result<u64> {
        let mut v: u64 = 0;
        unsafe {
            cudarc::driver::result::mem_pool::get_attribute(
                pool,
                attr,
                &mut v as *mut u64 as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolGetAttribute: {e:?}")))?;
        Ok(v)
    };
    Ok((
        get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
    ))
}
