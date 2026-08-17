//! Device-скретчи ядер с trim-on-OOM.
//!
//! Ядра берут workspace напрямую у cudarc (`stream.alloc_zeros`), минуя
//! [`crate::cuda_backend::CudaBackend::alloc_zeros`] — а вместе с ним и его
//! страховку от фрагментации пула. Цена: `RELEASE_THRESHOLD=MAX`
//! (см. `synaptix_core::device::cuda`) заставляет async-пул ДЕРЖАТЬ
//! освобождённые блоки, поэтому на префилле длинного промпта пул пухнет
//! (workspace GDN-скана ≈60 МБ на чанк × десятки чанков) и драйвер отдаёт
//! `CUDA_ERROR_OUT_OF_MEMORY` при живых единицах процентов VRAM: на 27B/24 ГБ
//! промпт в 8 тысяч токенов падал в `alloc chunk_scan ws` при `pool
//! reserved 4.7 ГБ / used 51 МБ`.
//!
//! Протокол ретрая — тот же, что в `CudaBackend::alloc_zeros`: sync ВСЕХ
//! стримов (`cuMemFreeAsync` исполняется в порядке своего стрима, до sync
//! trim не видит pending-frees) → `cuMemPoolTrimTo(0)` → до 5 попыток с
//! эскалацией паузы. На happy-path — ровно одна ветка `match`.

use std::sync::Arc;

use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError, ValidAsZeroBits};

/// Сколько раз пробовать после первого trim'а. Фрагментация транзиентна:
/// соседние frees на других стримах подходят с задержкой.
const RETRIES: u32 = 5;

fn is_oom(e: &DriverError) -> bool {
    matches!(e.0, CUresult::CUDA_ERROR_OUT_OF_MEMORY)
}

/// Печать крупных скретчей по `SYNAPTIX_TRACE_ALLOC_MIN` — эти аллокации идут
/// мимо учёта `cuda_pool`, и без трассы их не видно ни в `live_alloc_top`, ни в
/// `[TRACE_ALLOC_MIN]`.
fn trace(bytes: usize) {
    if bytes >= synaptix_core::memory::cuda_pool::trace_alloc_min() {
        eprintln!("[TRACE_WS_ALLOC] {bytes}");
    }
}

/// Sync всех стримов + возврат пулом освобождённых блоков драйверу.
/// `attempt` — номер попытки, задаёт паузу перед sync'ом.
fn reclaim(ordinal: usize, attempt: u32) {
    if attempt > 0 {
        std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
    }
    let _ = synaptix_core::device::cuda::synchronize_all(ordinal);
    let _ = synaptix_core::memory::cuda_pool::trim_pools_on_oom(ordinal);
}

/// Диагностика провалившегося скретча: кто держит VRAM в момент OOM'а.
/// Печатается один раз на исчерпание — ровно как `[OOM_TOP]`/`[OOM_SUM]` в
/// [`crate::cuda_backend`], чтобы post-mortem не требовал пересборки.
fn report_oom(ordinal: usize, what: &str, bytes: usize) {
    let (free, total) = synaptix_core::device::cuda::mem_info(ordinal).unwrap_or((0, 0));
    let (rsv, used) =
        synaptix_core::memory::cuda_pool::cuda_mempool_stats(ordinal).unwrap_or((0, 0));
    let (wrsv, wused) =
        synaptix_core::device::cuda::weights_pool_stats(ordinal).unwrap_or((0, 0));
    let gb = |x: f64| x / 1e9;
    eprintln!(
        "[WS_OOM] {what}({bytes} B) live={:.2}GB free={:.2}GB total={:.2}GB \
         pool_rsv={:.2}GB pool_used={:.2}GB weights_rsv={:.2}GB weights_used={:.2}GB",
        gb(synaptix_core::memory::cuda_pool::cuda_allocated_bytes() as f64),
        gb(free as f64),
        gb(total as f64),
        gb(rsv as f64),
        gb(used as f64),
        gb(wrsv as f64),
        gb(wused as f64)
    );
    for (sz, count) in synaptix_core::memory::cuda_pool::live_alloc_top(12) {
        eprintln!("[WS_OOM_TOP] {sz:>12} B \u{00d7} {count} = {:.2}GB", gb(sz as f64 * count as f64));
    }
}

/// `alloc_zeros`/`alloc` с trim-ретраем на OOM. Подмешивается к
/// `Arc<CudaStream>`, поэтому call-site меняется на один префикс `ws_`.
///
/// Разделение по времени жизни важно для пулов (см.
/// `synaptix_core::device::cuda::activations_pool`):
/// * `ws_*` — скретч НА ВЫЗОВ ядра → пул активаций (крутится и триммится);
/// * `cache_*` — то, что ядро держит МЕЖДУ вызовами (shuffled-веса,
///   MXFP8-скретчи, [`WsBuf`]) → default-пул, рядом с весами: churn'а нет, а
///   в пуле активаций такие блоки мешали бы его триму.
pub trait WsAlloc {
    fn ws_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError>;

    /// # Safety
    /// Как `CudaStream::alloc`: память не инициализирована, читать её до
    /// записи ядром нельзя.
    unsafe fn ws_alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError>;

    /// Долгоживущий кэш ядра — из default-пула (см. док трейта).
    fn cache_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError>;

    /// Как [`WsAlloc::cache_alloc_zeros`], но без зануления.
    ///
    /// # Safety
    /// Как `CudaStream::alloc`.
    unsafe fn cache_alloc_uninit<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError>;
}

impl WsAlloc for Arc<CudaStream> {
    fn ws_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        trace(len * std::mem::size_of::<T>());
        let first = match synaptix_core::device::cuda::alloc_act_zeros::<T>(self, len) {
            Ok(b) => return Ok(b),
            Err(e) if is_oom(&e) => e,
            Err(e) => return Err(e),
        };
        let ord = self.context().ordinal();
        for attempt in 0..RETRIES {
            reclaim(ord, attempt);
            if let Ok(b) = synaptix_core::device::cuda::alloc_act_zeros::<T>(self, len) {
                return Ok(b);
            }
        }
        report_oom(ord, "ws_alloc_zeros", len * std::mem::size_of::<T>());
        Err(first)
    }

    fn cache_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        trace(len * std::mem::size_of::<T>());
        let first = match self.alloc_zeros::<T>(len) {
            Ok(b) => return Ok(b),
            Err(e) if is_oom(&e) => e,
            Err(e) => return Err(e),
        };
        let ord = self.context().ordinal();
        for attempt in 0..RETRIES {
            reclaim(ord, attempt);
            if let Ok(b) = self.alloc_zeros::<T>(len) {
                return Ok(b);
            }
        }
        report_oom(ord, "cache_alloc_zeros", len * std::mem::size_of::<T>());
        Err(first)
    }

    unsafe fn cache_alloc_uninit<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        trace(len * std::mem::size_of::<T>());
        let first = match self.alloc::<T>(len) {
            Ok(b) => return Ok(b),
            Err(e) if is_oom(&e) => e,
            Err(e) => return Err(e),
        };
        let ord = self.context().ordinal();
        for attempt in 0..RETRIES {
            reclaim(ord, attempt);
            if let Ok(b) = self.alloc::<T>(len) {
                return Ok(b);
            }
        }
        report_oom(ord, "cache_alloc_uninit", len * std::mem::size_of::<T>());
        Err(first)
    }

    unsafe fn ws_alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError> {
        trace(len * std::mem::size_of::<T>());
        let first = match unsafe { synaptix_core::device::cuda::alloc_act_uninit::<T>(self, len) } {
            Ok(b) => return Ok(b),
            Err(e) if is_oom(&e) => e,
            Err(e) => return Err(e),
        };
        let ord = self.context().ordinal();
        for attempt in 0..RETRIES {
            reclaim(ord, attempt);
            if let Ok(b) = unsafe { synaptix_core::device::cuda::alloc_act_uninit::<T>(self, len) } {
                return Ok(b);
            }
        }
        report_oom(ord, "ws_alloc", len * std::mem::size_of::<T>());
        Err(first)
    }
}

/// Скретч ядра, живущий МЕЖДУ вызовами: растёт под запрос, зануляет
/// затребованный префикс, блок держит за собой.
///
/// Зачем: на префилле оркестраторы GDN брали ~24 свежих буфера НА СЛОЙ НА ЧАНК
/// (48 linear-слоёв × десятки чанков = десятки тысяч аллокаций 2-45 МБ). Пул с
/// `RELEASE_THRESHOLD=MAX` их не отдаёт, а веса LLM живут в ТОМ ЖЕ пуле, так
/// что `cuMemPoolTrimTo` не возвращает сегменты (в каждом лежит вес) — free-list
/// превращается в решето из дыр по 2-5 МБ, и промпт в 8k токенов падал с
/// `CUDA_ERROR_OUT_OF_MEMORY` на скретче в 6 МБ при 6.6 ГБ свободного внутри
/// пула. Кэш убирает churn: после первого чанка аллокаций нет вообще.
///
/// Семантика для вызывающего не меняется: `fit_zeros` отдаёт буфер, у которого
/// первые `len` элементов — нули, как у `alloc_zeros`.
pub struct WsBuf<T> {
    buf: Option<CudaSlice<T>>,
}

impl<T> Default for WsBuf<T> {
    fn default() -> Self {
        Self { buf: None }
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> WsBuf<T> {
    /// Отдаёт буфер РОВНО на `len` элементов, зануленный целиком — то есть
    /// неотличимый от свежего `alloc_zeros(len)`. Длина именно точная, а не
    /// «не меньше»: часть ядер выводит размер работы из `slice.len()`, и
    /// буфер-с-запасом менял бы им геометрию. На префилле все чанки одного
    /// размера, поэтому реаллокация случается только на смене формы.
    pub fn fit_zeros(
        &mut self,
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<&mut CudaSlice<T>, DriverError> {
        let need = len.max(1);
        let grow = match &self.buf {
            Some(b) => b.len() != need,
            None => true,
        };
        if grow {
            // Старый блок отдаём пулу ДО нового alloc'а — иначе на границе
            // роста в VRAM одновременно живут оба.
            self.buf = None;
            self.buf = Some(stream.cache_alloc_zeros::<T>(need)?);
            return Ok(self.buf.as_mut().unwrap());
        }
        let buf = self.buf.as_mut().unwrap();
        stream.memset_zeros(buf)?;
        Ok(buf)
    }

    /// Байт под блоком (0, если не аллоцирован) — для отчёта в
    /// `release_device_caches`.
    pub fn bytes(&self) -> usize {
        self.buf.as_ref().map(|b| b.len() * std::mem::size_of::<T>()).unwrap_or(0)
    }

    /// Отдать блок пулу.
    pub fn release(&mut self) -> usize {
        let n = self.bytes();
        self.buf = None;
        n
    }
}
