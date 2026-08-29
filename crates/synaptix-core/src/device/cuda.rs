mod inner {
    use crate::error::{Result, SynaptixError};
    use cudarc::driver::{CudaContext, CudaStream};
    pub use cudarc::driver::{CudaGraph, CudaStream as Stream};
    use once_cell::sync::Lazy;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    static REGISTRY: Lazy<RwLock<HashMap<usize, Arc<CudaContext>>>> =
        Lazy::new(|| RwLock::new(HashMap::new()));

    static STREAMS: Lazy<RwLock<HashMap<usize, Arc<CudaStream>>>> =
        Lazy::new(|| RwLock::new(HashMap::new()));

    // Отдельный персистентный stream для offload-загрузки весов (H2D), чтобы
    // copy-engine перекрывался с compute на default_stream (префетч блоков LTX).
    static LOADER_STREAMS: Lazy<RwLock<HashMap<usize, Arc<CudaStream>>>> =
        Lazy::new(|| RwLock::new(HashMap::new()));

    thread_local! {
        // Override stream'а для alloc/H2D на текущем потоке. loader-поток ставит сюда
        // loader_stream → from_raw_slice/alloc_zeros идут не на default (overlap).
        static ALLOC_STREAM_OVERRIDE: std::cell::RefCell<Option<Arc<CudaStream>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub fn get(ordinal: usize) -> Result<Arc<CudaContext>> {
        if let Some(ctx) = REGISTRY.read().get(&ordinal).cloned() {
            return Ok(ctx);
        }
        let mut w = REGISTRY.write();
        if let Some(ctx) = w.get(&ordinal).cloned() {
            return Ok(ctx);
        }
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| SynaptixError::Cuda(format!("CudaContext::new({ordinal}): {e:?}")))?;
        // RELEASE_THRESHOLD=MAX: async-mempool УДЕРЖИВАЕТ освобождённые блоки для
        // реюза вместо возврата драйверу на каждом sync (дефолт=0). Иначе под VRAM-
        // давлением (резидентные 23GB + активации) каждая реаллокация активации
        // re-acquire'ит с OS → ~26× медленнее (транспозы attention: 0.6мс→15.8мс).
        // Downside (пул держит память) страхуется trim-on-OOM в alloc_zeros/uninit.
        if let Err(e) = ctx.bind_to_thread() {
            return Err(SynaptixError::Cuda(format!("bind_to_thread({ordinal}): {e:?}")));
        }
        if let Ok(pool) =
            unsafe { cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device()) }
        {
            let mut thr: u64 = u64::MAX;
            let _ = unsafe {
                cudarc::driver::result::mem_pool::set_attribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut thr as *mut u64 as *mut core::ffi::c_void,
                )
            };
        }
        w.insert(ordinal, ctx.clone());
        Ok(ctx)
    }

    /// Единый персистентный stream на context (кэшируется). Это НЕ NULL
    /// default-stream (`ctx.default_stream()`), а явный `new_stream()` —
    /// NULL-stream нельзя `begin_capture` (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`),
    /// а весь decode-граф (P6.3) захватывается именно на этом stream'е. Все
    /// kernel-launch'и / alloc'и / memcpy идут на него → внутри-stream порядок
    /// сохранён (других stream'ов нет), capture валиден.
    pub fn default_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
        if let Some(s) = STREAMS.read().get(&ordinal).cloned() {
            return Ok(s);
        }
        let mut w = STREAMS.write();
        if let Some(s) = w.get(&ordinal).cloned() {
            return Ok(s);
        }
        let ctx = get(ordinal)?;
        let s = ctx
            .new_stream()
            .map_err(|e| SynaptixError::Cuda(format!("new_stream({ordinal}): {e:?}")))?;
        w.insert(ordinal, s.clone());
        Ok(s)
    }

    /// Персистентный loader-stream (отдельно от default) для overlap H2D с compute.
    pub fn loader_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
        if let Some(s) = LOADER_STREAMS.read().get(&ordinal).cloned() {
            return Ok(s);
        }
        let mut w = LOADER_STREAMS.write();
        if let Some(s) = w.get(&ordinal).cloned() {
            return Ok(s);
        }
        let ctx = get(ordinal)?;
        let s = ctx
            .new_stream()
            .map_err(|e| SynaptixError::Cuda(format!("loader new_stream({ordinal}): {e:?}")))?;
        w.insert(ordinal, s.clone());
        Ok(s)
    }

    /// Compute-стрим для ядра по home-стриму его входа: loader-стрим (H2D весов)
    /// никогда не исполняет ядра — перенаправляем на default; остальные стримы
    /// (default, capture-стрим CUDA-графа) сохраняются как есть.
    pub fn compute_stream_for(src: &Arc<CudaStream>, ordinal: usize) -> Result<Arc<CudaStream>> {
        if let Some(l) = LOADER_STREAMS.read().get(&ordinal) {
            if Arc::ptr_eq(l, src) {
                return default_stream(ordinal);
            }
        }
        Ok(src.clone())
    }

    /// Stream для alloc/H2D на текущем потоке: override (если стоит) либо default.
    pub fn alloc_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
        if let Some(s) = ALLOC_STREAM_OVERRIDE.with(|c| c.borrow().clone()) {
            return Ok(s);
        }
        default_stream(ordinal)
    }

    /// Поставить/снять override alloc-stream'а на текущем потоке (loader-поток).
    pub fn set_alloc_stream(s: Option<Arc<CudaStream>>) {
        ALLOC_STREAM_OVERRIDE.with(|c| *c.borrow_mut() = s);
    }

    // Двойной pinned staging-чанк для offload-H2D: memcpy mmap→pinned чанка i+1
    // перекрывается с DMA чанка i (раньше последовательный memcpy→H2D→sync давал
    // эфф. ~8-10 GB/s — стрим блоков LTX не прятался за compute, stream-wait
    // 1.6-3s/шаг). Pending-event на буфер живёт МЕЖДУ вызовами → конвейер сквозной
    // через тензоры (без пер-вызовного sync). Глобальный (один loader-поток за раз).
    const STAGE_CHUNK: usize = 32 << 20;
    struct StagePipe {
        bufs: [crate::memory::pinned::PinnedBuf; 2],
        pending: [Option<cudarc::driver::CudaEvent>; 2],
        next: usize,
    }
    static PINNED_STAGE: Lazy<parking_lot::Mutex<StagePipe>> = Lazy::new(|| {
        parking_lot::Mutex::new(StagePipe {
            bufs: [
                crate::memory::pinned::PinnedBuf::new(0),
                crate::memory::pinned::PinnedBuf::new(0),
            ],
            pending: [None, None],
            next: 0,
        })
    });

    /// Многопоточный memcpy (page cache → pinned): однопоточный ~10 GB/s упирался
    /// раньше PCIe (~45 GB/s pinned-H2D); 4-8 потоков дают 25-40 GB/s.
    fn par_copy(dst: &mut [u8], src: &[u8]) {
        let n = src.len();
        if n < (4 << 20) {
            dst[..n].copy_from_slice(src);
            return;
        }
        let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(8);
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (d, sr) in dst[..n].chunks_mut(chunk).zip(src.chunks(chunk)) {
                s.spawn(move || d.copy_from_slice(sr));
            }
        });
    }

    // Ленивое pinned-зеркало host-stream квант-блоков (ключ = ptr CPU-Vec'а
    // весов; стабилен и иммутабелен до Drop DiT): первый своп копирует
    // RAM→pinned, дальше DMA ~45GB/s без staging. Включается ТОЛЬКО на время
    // fetch-блока (set_pin_mirror) — чужие from_vec-буферы (переиспользуемые
    // ptr!) сюда не попадают.
    static PIN_MIRROR: Lazy<RwLock<Option<std::collections::HashMap<usize, crate::memory::pinned::PinnedBuf>>>> =
        Lazy::new(|| RwLock::new(None));

    /// RAII-гард зеркала (host-stream offload): включает кэш, Drop освобождает
    /// pinned-копии (~размер квант-весов, LTX mxfp8 ≈ 24GB).
    pub struct PinMirrorGuard;
    impl PinMirrorGuard {
        pub fn new() -> Self {
            *PIN_MIRROR.write() = Some(std::collections::HashMap::new());
            Self
        }
    }
    impl Default for PinMirrorGuard {
        fn default() -> Self {
            Self::new()
        }
    }
    impl Drop for PinMirrorGuard {
        fn drop(&mut self) {
            *PIN_MIRROR.write() = None;
        }
    }

    thread_local! {
        static PIN_MIRROR_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    /// Вкл/выкл зеркало для H2D на текущем потоке (ТОЛЬКО вокруг fetch блока).
    pub fn set_pin_mirror(on: bool) {
        PIN_MIRROR_ON.with(|c| c.set(on));
    }

    /// H2D через зеркало: `Some` если зеркало активно и включено на потоке
    /// (miss → копия RAM→pinned 1×). DMA асинхронна на `stream` (источник
    /// персистентен до Drop гарда).
    /// Отдельный CUDA-пул для стрим-весов (release-threshold = MAX → пул копит
    /// и идеально реюзает стабильные size-классы карусели блоков). Изолирует
    /// тысячи мелких (5-50MB) вес-аллокаций от default-пула активаций: их смесь
    /// дробила free-list в нераздаваемое решето (nvfp4-19s: 11GB свободно
    /// внутри пула, куска 134MB нет, никакие trim-стратегии не лечат).
    fn weights_pool(ord: usize) -> Result<cudarc::driver::sys::CUmemoryPool> {
        use cudarc::driver::sys;
        static POOLS: Lazy<RwLock<std::collections::HashMap<usize, usize>>> =
            Lazy::new(|| RwLock::new(std::collections::HashMap::new()));
        if let Some(&p) = POOLS.read().get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let mut wr = POOLS.write();
        if let Some(&p) = wr.get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let ctx = get(ord)?;
        ctx.bind_to_thread()
            .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.__bindgen_anon_1.id = ord as i32;
        let pool = unsafe { cudarc::driver::result::mem_pool::create(&props) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolCreate(weights): {e:?}")))?;
        let thr: u64 = u64::MAX;
        unsafe {
            let _ = cudarc::driver::result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &thr as *const u64 as *mut std::ffi::c_void,
            );
        }
        wr.insert(ord, pool as usize);
        Ok(pool)
    }

    /// Пул АКТИВАЦИЙ — отдельный от default'ного, где живут веса.
    ///
    /// Зачем: async-пул с `RELEASE_THRESHOLD=MAX` (см. [`get`]) не отдаёт
    /// освобождённые блоки драйверу, а страницы, в которых лежит хоть один
    /// живой блок, `cuMemPoolTrimTo` вернуть не может. Пока веса (17-19 ГБ
    /// резидента) и транзиенты префилла (≈20 МБ на токен трафика аллокаций)
    /// делят один пул, его free-list за проход по промпту деградирует
    /// («решето»): на 27B/24 ГБ пул прибавлял ~0.5 МБ на токен промпта и на
    /// 8k токенов упирался в OOM при неизменных 17.4 ГБ живого — а trim не
    /// возвращал НИЧЕГО, потому что каждая страница держала вес.
    ///
    /// С раздельными пулами транзиенты крутятся в своём: между чанками он
    /// пуст, поэтому trim-on-OOM реально возвращает всю его резервацию, и
    /// длина промпта перестаёт определять пик. Веса остаются в default'ном
    /// (их карусель size-класcов стабильна).
    fn activations_pool(ord: usize) -> Result<cudarc::driver::sys::CUmemoryPool> {
        use cudarc::driver::sys;
        static POOLS: Lazy<RwLock<std::collections::HashMap<usize, usize>>> =
            Lazy::new(|| RwLock::new(std::collections::HashMap::new()));
        if let Some(&p) = POOLS.read().get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let mut wr = POOLS.write();
        if let Some(&p) = wr.get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let ctx = get(ord)?;
        ctx.bind_to_thread()
            .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.__bindgen_anon_1.id = ord as i32;
        let pool = unsafe { cudarc::driver::result::mem_pool::create(&props) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolCreate(activations): {e:?}")))?;
        let thr: u64 = u64::MAX;
        unsafe {
            let _ = cudarc::driver::result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &thr as *const u64 as *mut std::ffi::c_void,
            );
        }
        wr.insert(ord, pool as usize);
        Ok(pool)
    }

    pub fn activations_pool_stats(ordinal: usize) -> Result<(u64, u64)> {
        use cudarc::driver::sys;
        let pool = activations_pool(ordinal)?;
        let mut rsv: u64 = 0;
        let mut used: u64 = 0;
        unsafe {
            let _ = cudarc::driver::result::mem_pool::get_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut rsv as *mut u64 as *mut std::ffi::c_void,
            );
            let _ = cudarc::driver::result::mem_pool::get_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used as *mut u64 as *mut std::ffi::c_void,
            );
        }
        Ok((rsv, used))
    }

    /// Вернуть драйверу всё, что пул активаций не держит живым.
    pub fn trim_activations_pool(ordinal: usize) -> Result<()> {
        let pool = activations_pool(ordinal)?;
        unsafe { cudarc::driver::result::mem_pool::trim_to(pool, 0) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolTrimTo(activations): {e:?}")))
    }

    /// Пул ЭКСПЕРТОВ MoE — третий, отдельно и от весов, и от активаций.
    ///
    /// Зачем: эксперты кэша (см. `synaptix_llm_common::moe::ExpertCache`) —
    /// единственные веса, которые приходят и уходят пачками по ходу
    /// генерации. Пока их packed/scales и перемешанные копии лежали в
    /// default-пуле рядом с резидентными весами модели, вытеснение эксперта
    /// НИЧЕГО не возвращало драйверу: `cuMemPoolTrimTo` не отдаёт страницу, в
    /// которой остался хоть один живой блок, а страницы были перемешаны с
    /// весами. Наблюдали ровно это: default-пул reserved 17.3 ГБ / used 10.9,
    /// кэш ужат вдвое — и всё равно `alloc_uninit(219 МБ)` падал в OOM при
    /// 6.4 ГБ «свободного» внутри пула.
    ///
    /// В своём пуле у экспертов ровно два-три size-класса, набор живёт и
    /// умирает целиком, поэтому `trim_experts_pool` реально возвращает
    /// освобождённое драйверу — и активациям префилла есть куда расти.
    fn experts_pool(ord: usize) -> Result<cudarc::driver::sys::CUmemoryPool> {
        use cudarc::driver::sys;
        static POOLS: Lazy<RwLock<std::collections::HashMap<usize, usize>>> =
            Lazy::new(|| RwLock::new(std::collections::HashMap::new()));
        if let Some(&p) = POOLS.read().get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let mut wr = POOLS.write();
        if let Some(&p) = wr.get(&ord) {
            return Ok(p as sys::CUmemoryPool);
        }
        let ctx = get(ord)?;
        ctx.bind_to_thread()
            .map_err(|e| SynaptixError::Cuda(format!("bind_to_thread: {e:?}")))?;
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.__bindgen_anon_1.id = ord as i32;
        let pool = unsafe { cudarc::driver::result::mem_pool::create(&props) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolCreate(experts): {e:?}")))?;
        let thr: u64 = u64::MAX;
        unsafe {
            let _ = cudarc::driver::result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &thr as *const u64 as *mut std::ffi::c_void,
            );
        }
        wr.insert(ord, pool as usize);
        Ok(pool)
    }

    /// (reserved, used) байт пула экспертов.
    pub fn experts_pool_stats(ordinal: usize) -> Result<(u64, u64)> {
        use cudarc::driver::sys;
        let pool = experts_pool(ordinal)?;
        let mut rsv: u64 = 0;
        let mut used: u64 = 0;
        unsafe {
            let _ = cudarc::driver::result::mem_pool::get_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut rsv as *mut u64 as *mut std::ffi::c_void,
            );
            let _ = cudarc::driver::result::mem_pool::get_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used as *mut u64 as *mut std::ffi::c_void,
            );
        }
        Ok((rsv, used))
    }

    /// Вернуть драйверу всё, что пул экспертов не держит живым.
    pub fn trim_experts_pool(ordinal: usize) -> Result<()> {
        let pool = experts_pool(ordinal)?;
        unsafe { cudarc::driver::result::mem_pool::trim_to(pool, 0) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolTrimTo(experts): {e:?}")))
    }

    thread_local! {
        /// Идёт подкачка/репак эксперта MoE на этом потоке → аллокации в
        /// пул экспертов (приоритетнее флага загрузки весов).
        static EXPERTS_ALLOC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Пометить поток как поднимающий эксперта; возвращает прежнее значение.
    pub fn set_experts_alloc(on: bool) -> bool {
        EXPERTS_ALLOC.with(|c| c.replace(on))
    }

    pub fn experts_alloc() -> bool {
        EXPERTS_ALLOC.with(|c| c.get())
    }

    /// RAII вокруг [`set_experts_alloc`]: всё, что аллоцируется внутри, ложится
    /// в пул экспертов. Ставится на подкачку эксперта из бандла и на построение
    /// его перемешанной копии — на всё, что вытесняется вместе с экспертом.
    pub struct ExpertsAllocGuard {
        prev: bool,
    }

    impl ExpertsAllocGuard {
        pub fn new() -> Self {
            Self { prev: set_experts_alloc(true) }
        }

        /// Только для CUDA-устройства; на CPU — пустышка (флаг не трогаем).
        pub fn for_device(device: crate::device::Device) -> Option<Self> {
            matches!(device, crate::device::Device::Cuda(_)).then(Self::new)
        }
    }

    impl Default for ExpertsAllocGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ExpertsAllocGuard {
        fn drop(&mut self) {
            set_experts_alloc(self.prev);
        }
    }

    thread_local! {
        /// Идёт загрузка весов на этом потоке → аллокации в default-пул.
        static WEIGHTS_ALLOC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Пометить поток как загружающий веса; возвращает прежнее значение
    /// (восстановить через повторный вызов — см. [`WeightsAllocGuard`]).
    pub fn set_weights_alloc(on: bool) -> bool {
        WEIGHTS_ALLOC.with(|c| c.replace(on))
    }

    /// RAII-обёртка над [`set_weights_alloc`]: всё, что аллоцируется в её
    /// области видимости, ложится в default-пул (веса и транзиенты
    /// деквантизации), а не в пул активаций.
    pub struct WeightsAllocGuard {
        prev: bool,
        trim_ord: Option<usize>,
    }

    impl WeightsAllocGuard {
        pub fn new() -> Self {
            Self { prev: set_weights_alloc(true), trim_ord: None }
        }

        /// Как [`Self::new`], но на выходе из области видимости отдаёт драйверу
        /// свободное в staging-пуле: у квантованных моделей это все временные
        /// bf16-буферы загрузки (гигабайты), и без трима они так и остаются
        /// зарезервированными — при том что KV-рингу их не видно.
        pub fn for_device(device: crate::device::Device) -> Self {
            let ord = match device {
                crate::device::Device::Cuda(o) => Some(o),
                _ => None,
            };
            Self { prev: set_weights_alloc(true), trim_ord: ord }
        }
    }

    impl Default for WeightsAllocGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for WeightsAllocGuard {
        fn drop(&mut self) {
            set_weights_alloc(self.prev);
            if self.prev {
                return; // вложенный guard — тримит внешний
            }
            if let Some(ord) = self.trim_ord {
                let _ = synchronize_all(ord);
                let _ = trim_weights_pool(ord);
            }
        }
    }

    static GRAPH_CAPTURING: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Взводится на время `cuStreamBeginCapture`/`EndCapture`: аллокации под
    /// capture становятся узлами графа, и пул для них выбирает драйвер —
    /// свой пул тут не навязываем.
    pub fn set_graph_capturing(on: bool) {
        GRAPH_CAPTURING.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn graph_capturing() -> bool {
        GRAPH_CAPTURING.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Аварийный выключатель разделения пулов: `SYN_ACT_POOL=0` возвращает
    /// прежнее поведение (всё в default-пуле).
    fn act_pool_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SYN_ACT_POOL").as_deref() != Ok("0"))
    }

    fn alloc_in_activations_pool() -> bool {
        act_pool_enabled() && !WEIGHTS_ALLOC.with(|c| c.get()) && !graph_capturing()
    }

    /// Аллокация под АКТИВАЦИИ: из пула активаций, если он включён и мы не в
    /// загрузке весов / не под graph-capture; иначе — обычный stream-alloc
    /// (default-пул). Память НЕ инициализирована.
    ///
    /// # Safety
    /// Как `CudaStream::alloc`: читать до записи нельзя.
    pub unsafe fn alloc_act_uninit<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        let ord = stream.context().ordinal();
        // Эксперт MoE — в свой пул: он вытесняется пачками, и только отдельный
        // пул отдаёт освобождённое драйверу (см. `experts_pool`).
        if experts_alloc() && act_pool_enabled() && !graph_capturing() {
            if let Ok(pool) = experts_pool(ord) {
                let bytes = len.max(1) * std::mem::size_of::<T>();
                let ptr = unsafe {
                    cudarc::driver::result::mem_pool::alloc_async(pool, bytes, stream.cu_stream())
                }?;
                return Ok(unsafe { stream.upgrade_device_ptr::<T>(ptr, len) });
            }
        }
        if !alloc_in_activations_pool() {
            return stream.alloc::<T>(len);
        }
        let Ok(pool) = activations_pool(ord) else {
            return stream.alloc::<T>(len);
        };
        let bytes = len.max(1) * std::mem::size_of::<T>();
        let ptr = unsafe {
            cudarc::driver::result::mem_pool::alloc_async(pool, bytes, stream.cu_stream())
        }?;
        Ok(unsafe { stream.upgrade_device_ptr::<T>(ptr, len) })
    }

    /// Аллокация под H2D-байты: staging весов при загрузке или мелкие входы
    /// на инференсе (id токенов, device-скаляры).
    ///
    /// Под загрузкой ([`WeightsAllocGuard`]) — в weights-пул, а НЕ в
    /// default'ный: у квантованных моделей это временные bf16-буферы (по 170 МБ
    /// на MLP-вес), которые после кванта умирают. Пока они лежали рядом с
    /// готовыми nvfp4-весами, пул после загрузки держал ~3.7 ГБ нераздаваемой
    /// слабины (страницу с живым весом trim не вернёт) — ровно тех гигабайт,
    /// которых потом не хватало KV-рингу на 80k контекста. В отдельном пуле
    /// они умирают все вместе, и `trim_weights_pool` возвращает всё. У dense
    /// моделей в этом пуле остаются сами веса — тоже правильно: стабильная
    /// карусель size-класcов без транзиентов.
    ///
    /// # Safety
    /// Как `CudaStream::alloc`.
    pub unsafe fn alloc_bytes_uninit(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<u8>, cudarc::driver::DriverError> {
        if !act_pool_enabled() || graph_capturing() {
            return stream.alloc::<u8>(len);
        }
        let ord = stream.context().ordinal();
        // H2D эксперта (packed/scales из бандла) — в пул экспертов.
        if experts_alloc() {
            if let Ok(pool) = experts_pool(ord) {
                let ptr = unsafe {
                    cudarc::driver::result::mem_pool::alloc_async(
                        pool,
                        len.max(1),
                        stream.cu_stream(),
                    )
                }?;
                return Ok(unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) });
            }
        }
        let loading = WEIGHTS_ALLOC.with(|c| c.get());
        if loading {
            // Staging-буферы разных размеров оставляют в пуле по блоку на
            // size-класс, и за загрузку это набегает гигабайтами. Пока грузятся
            // только веса, это неважно (трим на выходе из
            // `WeightsAllocGuard`), но модель тянет за собой и компоненты
            // ПОСЛЕ основных весов — MTP-голову, DFlash-драфтер, vision-башню:
            // им квантоваться уже негде (проверено: DFlash Muse-Glimmer падал
            // с OOM на `quantize_mxfp8`). Поэтому крупные staging-аллокации
            // сначала возвращают пулу слабину.
            const BIG: usize = 32 * 1024 * 1024;
            const SLACK_LIMIT: u64 = 1536 * 1024 * 1024;
            if len >= BIG {
                if let Ok((rsv, used)) = weights_pool_stats(ord) {
                    if rsv.saturating_sub(used) > SLACK_LIMIT {
                        let _ = synchronize_all(ord);
                        let _ = trim_weights_pool(ord);
                    }
                }
            }
        }
        let pool = if loading {
            weights_pool(ord)
        } else {
            activations_pool(ord)
        };
        let Ok(pool) = pool else {
            return stream.alloc::<u8>(len);
        };
        let ptr = unsafe {
            cudarc::driver::result::mem_pool::alloc_async(pool, len.max(1), stream.cu_stream())
        }?;
        Ok(unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) })
    }

    /// Долгоживущий кэш ЯДРА (перемешанные веса, MXFP8-скретчи): default-пул,
    /// как раньше, — но под [`ExpertsAllocGuard`] уходит в пул экспертов, иначе
    /// перемешанная копия эксперта осела бы среди резидентных весов и держала
    /// бы их страницы от трима.
    ///
    /// # Safety
    /// Как `CudaStream::alloc`.
    pub unsafe fn alloc_cache_uninit<T: cudarc::driver::DeviceRepr>(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        if experts_alloc() && act_pool_enabled() && !graph_capturing() {
            let ord = stream.context().ordinal();
            if let Ok(pool) = experts_pool(ord) {
                let bytes = len.max(1) * std::mem::size_of::<T>();
                let ptr = unsafe {
                    cudarc::driver::result::mem_pool::alloc_async(pool, bytes, stream.cu_stream())
                }?;
                return Ok(unsafe { stream.upgrade_device_ptr::<T>(ptr, len) });
            }
        }
        stream.alloc::<T>(len)
    }

    /// [`alloc_cache_uninit`] + memset в нули.
    pub fn alloc_cache_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        if !experts_alloc() {
            return stream.alloc_zeros::<T>(len);
        }
        let mut buf = unsafe { alloc_cache_uninit::<T>(stream, len) }?;
        stream.memset_zeros(&mut buf)?;
        Ok(buf)
    }

    /// [`alloc_act_uninit`] + memset в нули (эквивалент `alloc_zeros`).
    pub fn alloc_act_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        let mut buf = unsafe { alloc_act_uninit::<T>(stream, len) }?;
        stream.memset_zeros(&mut buf)?;
        Ok(buf)
    }

    static LAYER_SYNC: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    pub fn set_layer_sync_mode(mode: u8) {
        LAYER_SYNC.store(mode, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn layer_sync_mode() -> u8 {
        LAYER_SYNC.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn layer_sync(ordinal: usize, is_prefill: bool) {
        let enabled = match layer_sync_mode() {
            1 => false,
            2 => true,
            _ => is_prefill,
        };
        if !enabled {
            return;
        }
        if let Ok(s) = default_stream(ordinal) {
            let _ = s.synchronize();
        }
    }

    pub fn weights_pool_stats(ordinal: usize) -> Result<(u64, u64)> {
        use cudarc::driver::sys;
        let pool = weights_pool(ordinal)?;
        let get = |attr: sys::CUmemPool_attribute| -> Result<u64> {
            let mut v: u64 = 0;
            unsafe {
                cudarc::driver::result::mem_pool::get_attribute(
                    pool,
                    attr,
                    &mut v as *mut u64 as *mut std::ffi::c_void,
                )
            }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolGetAttribute(weights): {e:?}")))?;
            Ok(v)
        };
        Ok((
            get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
        ))
    }

    pub fn trim_weights_pool(ordinal: usize) -> Result<()> {
        let pool = weights_pool(ordinal)?;
        unsafe { cudarc::driver::result::mem_pool::trim_to(pool, 0) }
            .map_err(|e| SynaptixError::Cuda(format!("cuMemPoolTrimTo(weights): {e:?}")))?;
        Ok(())
    }

    /// H2D вес-байтов через weights-пул: alloc из изолированного пула +
    /// memcpy_htod. Drop тензора вернёт блок в weights-пул (cuMemFreeAsync
    /// знает пул по указателю) — карусель реюзает свои классы.
    pub fn weights_htod(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<cudarc::driver::CudaSlice<u8>> {
        let ord = stream.context().ordinal();
        // Под [`ExpertsAllocGuard`] — в пул экспертов: pinned-зеркало MoE
        // ходит именно сюда, а вытесняться эксперт должен вместе со своим
        // пулом, иначе трим опять ничего не вернёт.
        let pool = if experts_alloc() { experts_pool(ord)? } else { weights_pool(ord)? };
        let ptr = unsafe {
            cudarc::driver::result::mem_pool::alloc_async(pool, bytes.len(), stream.cu_stream())
        }
        .map_err(|e| SynaptixError::Cuda(format!("alloc from weights-pool({}): {e:?}", bytes.len())))?;
        let mut sl = unsafe { stream.upgrade_device_ptr::<u8>(ptr, bytes.len()) };
        stream
            .memcpy_htod(bytes, &mut sl)
            .map_err(|e| SynaptixError::Cuda(format!("weights_htod memcpy: {e:?}")))?;
        Ok(sl)
    }

    pub fn pin_mirror_htod(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Option<Result<cudarc::driver::CudaSlice<u8>>> {
        if !PIN_MIRROR_ON.with(|c| c.get()) {
            return None;
        }
        let p = bytes.as_ptr() as usize;
        {
            let rd = PIN_MIRROR.read();
            let cache = rd.as_ref()?;
            if let Some(buf) = cache.get(&p) {
                debug_assert_eq!(buf.len(), bytes.len());
                return Some(weights_htod(stream, &buf.as_slice()[..bytes.len()]));
            }
        }
        let mut wr = PIN_MIRROR.write();
        let cache = wr.as_mut()?;
        let buf = cache.entry(p).or_insert_with(|| {
            let mut b = crate::memory::pinned::PinnedBuf::new_uninit(bytes.len());
            par_copy(b.as_mut_slice(), bytes);
            b
        });
        Some(weights_htod(stream, &buf.as_slice()[..bytes.len()]),
        )
    }

    thread_local! {
        static OFFLOAD_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    /// Вкл/выкл pinned-staging для H2D на текущем потоке (offload-загрузка весов).
    pub fn set_offload_pinned(on: bool) {
        OFFLOAD_PINNED.with(|c| c.set(on));
    }
    pub fn offload_pinned_enabled() -> bool {
        OFFLOAD_PINNED.with(|c| c.get())
    }

    // Персистентный pinned-кэш offload-весов (зеркало mmap-шардов ckpt в pinned
    // host-RAM): page cache не держит 46GB циклического стрима (use-once LRU →
    // перечитки с NVMe ~6GB/s), cuMemHostRegister ro-mmap → NOT_SUPPORTED.
    // Фоновые копировщики линейно зеркалят шард (NVMe-friendly) с водяным знаком
    // `ready`; запрос ниже знака → DMA из pinned (~45GB/s, async, без staging),
    // выше → фоллбэк staging (кэш догонит). Гейт по диапазонам шардов —
    // содержимое mmap иммутабельно → ptr ≡ контент.
    struct ShardPin {
        base: usize,
        len: usize,
        /// raw-ptr pinned-зеркала (запись фоном выше `ready`, чтение ниже);
        /// 0 = буфер ещё не аллоцирован (lazy: на первом проходе ПОСЛЕ снятия
        /// паузы — иначе 44GB pinned висели с старта и вместе с CPU-блоками
        /// bf16-Gemma host-stream выбивали RAM-лимит, SIGKILL).
        bptr: std::sync::atomic::AtomicUsize,
        ready: std::sync::atomic::AtomicUsize,
        buf: parking_lot::Mutex<Option<crate::memory::pinned::PinnedBuf>>,
    }
    struct OffloadPinCache {
        shards: Vec<ShardPin>,
        cancel: std::sync::atomic::AtomicBool,
        /// Пауза копировщиков (старт до text-encoding: NVMe нужен Gemma-load,
        /// конкуренция давала +2.5s — CLI резюмит после загрузки энкодера).
        paused: std::sync::atomic::AtomicBool,
    }
    static OFFLOAD_PIN_CACHE: Lazy<RwLock<Option<Arc<OffloadPinCache>>>> =
        Lazy::new(|| RwLock::new(None));

    /// `true` если pinned-кэш offload-весов уже активен (не создавать второй).
    pub fn offload_pin_cache_active() -> bool {
        OFFLOAD_PIN_CACHE.read().is_some()
    }

    /// RAII-гард pinned-кэша: стартует фоновое зеркалирование `ranges`
    /// (mmap-шарды ckpt) в pinned host-RAM (~размер ckpt, LTX 22B ≈ 43GB);
    /// Drop отменяет/джойнит копировщиков и освобождает зеркала.
    ///
    /// SAFETY-контракт: `ranges` обязаны жить дольше гарда (Drop джойнит фоновые
    /// потоки до возврата — после Drop чтений mmap нет).
    pub struct OffloadPinCacheGuard {
        cache: Arc<OffloadPinCache>,
        workers: Vec<std::thread::JoinHandle<()>>,
    }
    impl OffloadPinCacheGuard {
        pub fn new(ranges: &[&[u8]]) -> Self {
            Self::build(ranges, false, true)
        }

        /// Как [`OffloadPinCacheGuard::new`], но копировщики стартуют на паузе —
        /// NVMe остаётся text-encoder'у; продолжить — [`Self::resume`].
        pub fn new_paused(ranges: &[&[u8]]) -> Self {
            Self::build(ranges, true, true)
        }

        /// Как [`Self::new_paused`], но pinned-зеркала аллоцируются ЛЕНИВО на
        /// первом resume: для dense-Gemma host-stream (CPU-блоки 21.5GB + 44GB
        /// pinned сразу = RAM-OOM). Цена: cuMemHostAlloc(44GB)+старт копий
        /// попадают в text-encode окно (+13s) — потому только для dense.
        pub fn new_paused_lazy(ranges: &[&[u8]]) -> Self {
            Self::build(ranges, true, false)
        }

        pub fn resume(&self) {
            self.cache.paused.store(false, std::sync::atomic::Ordering::Release);
        }

        fn build(ranges: &[&[u8]], paused: bool, eager_alloc: bool) -> Self {
            let shards: Vec<ShardPin> = ranges
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let (bptr, buf) = if eager_alloc {
                        let b = crate::memory::pinned::PinnedBuf::new_uninit(s.len());
                        (b.as_ptr() as usize, Some(b))
                    } else {
                        (0, None)
                    };
                    ShardPin {
                        base: s.as_ptr() as usize,
                        len: s.len(),
                        bptr: std::sync::atomic::AtomicUsize::new(bptr),
                        ready: std::sync::atomic::AtomicUsize::new(0),
                        buf: parking_lot::Mutex::new(buf),
                    }
                })
                .collect();
            let cache = Arc::new(OffloadPinCache {
                shards,
                cancel: std::sync::atomic::AtomicBool::new(false),
                paused: std::sync::atomic::AtomicBool::new(paused),
            });
            *OFFLOAD_PIN_CACHE.write() = Some(cache.clone());
            let workers = (0..cache.shards.len())
                .map(|i| {
                    let c = cache.clone();
                    std::thread::spawn(move || {
                        use std::sync::atomic::Ordering;
                        let sh = &c.shards[i];
                        const PREFILL_CHUNK: usize = 256 << 20;
                        let mut off = 0usize;
                        while off < sh.len && !c.cancel.load(Ordering::Relaxed) {
                            if c.paused.load(Ordering::Acquire) {
                                std::thread::sleep(std::time::Duration::from_millis(20));
                                continue;
                            }
                            // lazy-аллокация зеркала на первом проходе после паузы
                            if sh.bptr.load(Ordering::Acquire) == 0 {
                                let b = crate::memory::pinned::PinnedBuf::new_uninit(sh.len);
                                sh.bptr.store(b.as_ptr() as usize, Ordering::Release);
                                *sh.buf.lock() = Some(b);
                            }
                            let bptr = sh.bptr.load(Ordering::Acquire);
                            let n = PREFILL_CHUNK.min(sh.len - off);
                            // SAFETY: src — mmap-шард (жив по контракту гарда, ro);
                            // dst — наше pinned-зеркало выше водяного знака (читатели
                            // не заходят выше `ready`).
                            let src = unsafe { std::slice::from_raw_parts((sh.base + off) as *const u8, n) };
                            let dst = unsafe { std::slice::from_raw_parts_mut((bptr + off) as *mut u8, n) };
                            par_copy(dst, src);
                            sh.ready.store(off + n, Ordering::Release);
                            // Скопированные mmap-страницы больше не нужны (читаем из
                            // зеркала; staging-фоллбэк ходит только ВЫШЕ знака) →
                            // отдаём их page cache'у сразу: иначе 46GB-скан выдавливает
                            // Gemma/прочее и hot-серия перечитывает их с NVMe.
                            let pg = 4096usize;
                            let a0 = (sh.base + off).div_ceil(pg) * pg;
                            let a1 = (sh.base + off + n) / pg * pg;
                            if a1 > a0 {
                                unsafe {
                                    let _ = libc::madvise(a0 as *mut libc::c_void, a1 - a0, libc::MADV_DONTNEED);
                                }
                            }
                            off += n;
                        }
                    })
                })
                .collect();
            Self { cache, workers }
        }
    }
    impl Drop for OffloadPinCacheGuard {
        fn drop(&mut self) {
            self.cache.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            for w in self.workers.drain(..) {
                let _ = w.join();
            }
            *OFFLOAD_PIN_CACHE.write() = None;
        }
    }

    /// H2D из pinned-кэша: `Some` если кэш активен, `bytes` внутри шарда и зеркало
    /// уже догнало этот диапазон. `None` → обычный путь (staging). Возвращённая
    /// DMA асинхронна на `stream` (источник персистентен до Drop гарда).
    pub fn offload_pin_cache_htod(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Option<Result<cudarc::driver::CudaSlice<u8>>> {
        let src = offload_pin_resolve(bytes)?;
        Some(weights_htod(stream, src))
    }

    /// Pinned-зеркало диапазона `bytes` (mmap-шард): `Some(слайс зеркала)` если
    /// кэш активен и догнал диапазон, иначе `None` (staging-путь).
    pub fn offload_pin_resolve(bytes: &[u8]) -> Option<&'static [u8]> {
        use std::sync::atomic::Ordering;
        let rd = OFFLOAD_PIN_CACHE.read();
        let cache = rd.as_ref()?;
        let p = bytes.as_ptr() as usize;
        for sh in &cache.shards {
            if p >= sh.base && p + bytes.len() <= sh.base + sh.len {
                let off = p - sh.base;
                if sh.ready.load(Ordering::Acquire) < off + bytes.len() {
                    return None; // зеркало ещё не догнало — staging, кэш догонит
                }
                let bptr = sh.bptr.load(Ordering::Acquire);
                if bptr == 0 {
                    return None; // буфер ещё не аллоцирован (lazy)
                }
                // SAFETY: диапазон ниже водяного знака — записан и стабилен.
                return Some(unsafe { std::slice::from_raw_parts((bptr + off) as *const u8, bytes.len()) });
            }
        }
        None
    }

    /// H2D `bytes` в регион существующего device-буфера по сырому указателю
    /// `dst_dptr` (слот-стриминг весов: фиксированные адреса под CUDA-graph).
    /// Источник: pinned-зеркало (прямая DMA) либо staging-конвейер pinned-чанков.
    /// Копия асинхронна на `stream`; вызывающий упорядочивает доступ к региону
    /// (события слота: запись только после завершения чтений предыдущего блока).
    ///
    /// SAFETY-контракт: `dst_dptr..dst_dptr+bytes.len()` — валидный регион живого
    /// device-буфера, эксклюзивный для записи на время операции.
    pub fn htod_into_region(stream: &Arc<CudaStream>, dst_dptr: u64, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if let Some(src) = offload_pin_resolve(bytes) {
            unsafe {
                cudarc::driver::result::memcpy_htod_async(dst_dptr, src, stream.cu_stream())
            }
            .map_err(|e| SynaptixError::Cuda(format!("htod_into_region pinned: {e:?}")))?;
            return Ok(());
        }
        let mut stage = PINNED_STAGE.lock();
        let mut off = 0usize;
        for chunk in bytes.chunks(STAGE_CHUNK) {
            let b = stage.next;
            stage.next ^= 1;
            if let Some(ev) = stage.pending[b].take() {
                ev.synchronize()
                    .map_err(|e| SynaptixError::Cuda(format!("htod_into_region event: {e:?}")))?;
            }
            if stage.bufs[b].len() < chunk.len() {
                stage.bufs[b] = crate::memory::pinned::PinnedBuf::new(STAGE_CHUNK.max(chunk.len()));
            }
            par_copy(stage.bufs[b].as_mut_slice(), chunk);
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    dst_dptr + off as u64,
                    &stage.bufs[b].as_slice()[..chunk.len()],
                    stream.cu_stream(),
                )
            }
            .map_err(|e| SynaptixError::Cuda(format!("htod_into_region memcpy: {e:?}")))?;
            stage.pending[b] = Some(
                stream
                    .record_event(None)
                    .map_err(|e| SynaptixError::Cuda(format!("htod_into_region record: {e:?}")))?,
            );
            off += chunk.len();
        }
        Ok(())
    }

    /// D2D `len` байт между регионами по сырым указателям, асинхронно на `stream`
    /// (слот-стриминг: F32 sst-таблицы из резидентного стора в слот).
    ///
    /// SAFETY-контракт: оба региона валидны и не перекрываются; вызывающий
    /// упорядочивает доступ (события слота).
    pub fn dtod_into_region(stream: &Arc<CudaStream>, dst_dptr: u64, src_dptr: u64, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst_dptr, src_dptr, len, stream.cu_stream())
        }
        .map_err(|e| SynaptixError::Cuda(format!("dtod_into_region: {e:?}")))
    }

    /// CUDA-событие слот-протокола (write-after-read анти-гонка ping-pong слотов
    /// весов): record на default-стриме ПОСЛЕ enqueue ядер, читающих слот; wait
    /// на loader-стриме ПЕРЕД перезаписью слота. Никогда не записанное событие
    /// считается завершённым (первый шаг). Не для использования под graph-capture.
    pub struct SlotEvent(cudarc::driver::CudaEvent);

    impl SlotEvent {
        pub fn new(ordinal: usize) -> Result<Self> {
            let ctx = get(ordinal)?;
            ctx.new_event(None)
                .map(Self)
                .map_err(|e| SynaptixError::Cuda(format!("SlotEvent new: {e:?}")))
        }

        /// cuEventRecord на default-стриме `ordinal` (compute).
        pub fn record_default(&self, ordinal: usize) -> Result<()> {
            let ds = default_stream(ordinal)?;
            self.0
                .record(&ds)
                .map_err(|e| SynaptixError::Cuda(format!("SlotEvent record: {e:?}")))
        }

        /// cuStreamWaitEvent на `stream` (loader перед записью слота).
        pub fn wait_on(&self, stream: &Arc<CudaStream>) -> Result<()> {
            stream
                .wait(&self.0)
                .map_err(|e| SynaptixError::Cuda(format!("SlotEvent wait: {e:?}")))
        }
    }

    /// D2H через конвейер pinned-чанков (зеркальный близнец [`pinned_htod`]):
    /// DMA чанка i+1 перекрывается с parallel-memcpy pinned→Vec чанка i.
    /// Pageable clone_dtoh давал 3-6GB/s — выгрузка 24GB квант-блоков LTX
    /// (host-stream) жгла 5-8s в DiT-load.
    ///
    /// TODO(квант-load, 2026-06-06): после D2H-фикса в DiT-load LTX остаётся
    /// ~18s (mxfp8) / ~7.6s (nvfp4) СКРЫТОГО времени внутри «генерации»:
    /// NVMe-чтение 44GB bf16 (~7.3s) + GPU-квант + view/parse. Два пути по −10s:
    /// (1) фоновый DiT-load параллельно text-encoding (нужен Arc<LtxCheckpoint>
    /// или thread::scope вокруг text-enc+генерации в CLI); (2) дисковый кэш
    /// квант-весов (.synq рядом с ckpt: грузить 12-24GB квантованных вместо
    /// 44GB + квант → load ≈ 3-5s; инвалидация по mtime/размеру ckpt).
    pub fn pinned_dtoh(
        stream: &Arc<CudaStream>,
        src: &cudarc::driver::CudaSlice<u8>,
    ) -> Result<Vec<u8>> {
        let n = src.len();
        let mut out: Vec<u8> = Vec::with_capacity(n);
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(n);
        }
        let mut stage = PINNED_STAGE.lock();
        let mut off = 0usize;
        let mut pending: Option<(usize, usize, usize)> = None; // (буфер, off, len)
        while off < n || pending.is_some() {
            if off < n {
                let len = STAGE_CHUNK.min(n - off);
                let b = stage.next;
                stage.next ^= 1;
                if let Some(ev) = stage.pending[b].take() {
                    ev.synchronize()
                        .map_err(|e| SynaptixError::Cuda(format!("pinned_dtoh event: {e:?}")))?;
                }
                if stage.bufs[b].len() < len {
                    stage.bufs[b] = crate::memory::pinned::PinnedBuf::new_uninit(STAGE_CHUNK.max(len));
                }
                // выкопировать ПРЕДЫДУЩИЙ чанк (его DMA уже дождались выше через
                // pending-событие его буфера на следующем витке) — здесь порядок:
                // сначала ждём свой буфер, затем запускаем DMA, копию делаем когда
                // буфер снова придёт в очередь. Для простоты: синхронный вариант
                // с перекрытием через два буфера ниже.
                let view = src.slice(off..off + len);
                stream
                    .memcpy_dtoh(&view, &mut stage.bufs[b].as_mut_slice()[..len])
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_dtoh dtoh: {e:?}")))?;
                let ev = stream
                    .record_event(None)
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_dtoh record: {e:?}")))?;
                stage.pending[b] = Some(ev);
                if let Some((pb, poff, plen)) = pending.take() {
                    if let Some(ev) = stage.pending[pb].take() {
                        ev.synchronize()
                            .map_err(|e| SynaptixError::Cuda(format!("pinned_dtoh event2: {e:?}")))?;
                    }
                    par_copy(&mut out[poff..poff + plen], &stage.bufs[pb].as_slice()[..plen]);
                }
                pending = Some((b, off, len));
                off += len;
            } else if let Some((pb, poff, plen)) = pending.take() {
                if let Some(ev) = stage.pending[pb].take() {
                    ev.synchronize()
                        .map_err(|e| SynaptixError::Cuda(format!("pinned_dtoh event3: {e:?}")))?;
                }
                par_copy(&mut out[poff..poff + plen], &stage.bufs[pb].as_slice()[..plen]);
            }
        }
        Ok(out)
    }

    thread_local! {
        /// Идёт подкачка мелких весов (эксперты MoE) → H2D через собственный
        /// pinned-буфер потока.
        static PINNED_TLS_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static PINNED_TLS_BUF: std::cell::RefCell<PinnedTls> = std::cell::RefCell::new(PinnedTls {
            bufs: [
                crate::memory::pinned::PinnedBuf::new(0),
                crate::memory::pinned::PinnedBuf::new(0),
            ],
            pending: [None, None],
            next: 0,
        });
    }

    /// До какого размера тело едет через буфер потока: крупное отдаём общему
    /// конвейеру, чтобы не держать десятки мегабайт pinned на каждом потоке.
    const PINNED_TLS_MAX: usize = 16 << 20;

    /// Пара буферов на поток: пока DMA читает один, копия следующего тела идёт
    /// во второй.
    struct PinnedTls {
        bufs: [crate::memory::pinned::PinnedBuf; 2],
        pending: [Option<cudarc::driver::CudaEvent>; 2],
        next: usize,
    }

    pub fn pinned_tls_enabled() -> bool {
        PINNED_TLS_ON.with(|c| c.get())
    }

    /// RAII-гард pinned-staging на потоке. Ставится вокруг подкачки экспертов:
    /// pageable-копия идёт через staging драйвера (~6 ГБ/с и общая очередь на
    /// процесс), а свой pinned-буфер даёт DMA без промежуточного копирования, и
    /// потоки предзагрузки перестают мешать друг другу.
    pub struct PinnedStageGuard {
        prev: bool,
    }

    impl PinnedStageGuard {
        pub fn new() -> Self {
            Self { prev: PINNED_TLS_ON.with(|c| c.replace(true)) }
        }
    }

    impl Default for PinnedStageGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for PinnedStageGuard {
        fn drop(&mut self) {
            PINNED_TLS_ON.with(|c| c.set(self.prev));
        }
    }

    pub fn pinned_htod_tls(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<cudarc::driver::CudaSlice<u8>> {
        if bytes.len() > PINNED_TLS_MAX {
            return pinned_htod(stream, bytes);
        }
        let mut dst = match unsafe { alloc_bytes_uninit(stream, bytes.len()) } {
            Ok(b) => b,
            Err(_) => {
                let ord = stream.context().ordinal();
                let _ = stream.synchronize();
                let _ = crate::memory::cuda_pool::trim_pools_on_oom(ord);
                unsafe { alloc_bytes_uninit(stream, bytes.len()) }.map_err(|e| {
                    SynaptixError::Cuda(format!(
                        "pinned_htod_tls alloc({}) after trim: {e:?}",
                        bytes.len()
                    ))
                })?
            }
        };
        PINNED_TLS_BUF.with(|cell| -> Result<()> {
            let mut slot = cell.borrow_mut();
            let b = slot.next;
            slot.next ^= 1;
            if let Some(ev) = slot.pending[b].take() {
                ev.synchronize()
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_htod_tls event: {e:?}")))?;
            }
            if slot.bufs[b].len() < bytes.len() {
                slot.bufs[b] = crate::memory::pinned::PinnedBuf::new_uninit(
                    bytes.len().next_power_of_two().max(1 << 22),
                );
            }
            slot.bufs[b].as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
            stream
                .memcpy_htod(&slot.bufs[b].as_slice()[..bytes.len()], &mut dst)
                .map_err(|e| SynaptixError::Cuda(format!("pinned_htod_tls memcpy: {e:?}")))?;
            slot.pending[b] = Some(
                stream
                    .record_event(None)
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_htod_tls record: {e:?}")))?,
            );
            Ok(())
        })?;
        Ok(dst)
    }

    /// H2D через конвейер pinned-чанков: parallel-memcpy чанка i+1 перекрыт с
    /// async-DMA чанка i (двойная буферизация, pending-event сквозь вызовы).
    /// ВОЗВРАЩАЕТСЯ ДО завершения DMA последнего чанка — копия упорядочена на
    /// `stream`, потребитель обязан синкать его перед использованием на другом
    /// stream'е (loader-путь это уже делает: `lsc.synchronize()` после fetch).
    /// Байты не меняются → bit-identical прежнему синхронному пути.
    pub fn pinned_htod(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<cudarc::driver::CudaSlice<u8>> {
        let mut stage = PINNED_STAGE.lock();
        let mut dst = match unsafe { alloc_bytes_uninit(stream, bytes.len()) } {
            Ok(b) => b,
            Err(_) => {
                // OOM: trim-ретрай как CudaBackend::alloc_* (фрагментация пула;
                // sync стримов — pending cuMemFreeAsync до sync trim не видит).
                let ord = stream.context().ordinal();
                let _ = stream.synchronize();
                if let Ok(ls) = loader_stream(ord) {
                    let _ = ls.synchronize();
                }
                let _ = crate::memory::cuda_pool::trim_pools_on_oom(ord);
                unsafe { alloc_bytes_uninit(stream, bytes.len()) }.map_err(|e| {
                    SynaptixError::Cuda(format!("pinned_htod alloc({}) after trim: {e:?}", bytes.len()))
                })?
            }
        };
        let mut off = 0usize;
        for chunk in bytes.chunks(STAGE_CHUNK) {
            let b = stage.next;
            stage.next ^= 1;
            // дождаться DMA, ещё читающей этот буфер (прошлый чанк/вызов)
            if let Some(ev) = stage.pending[b].take() {
                ev.synchronize()
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_htod event sync: {e:?}")))?;
            }
            if stage.bufs[b].len() < chunk.len() {
                stage.bufs[b] = crate::memory::pinned::PinnedBuf::new(STAGE_CHUNK.max(chunk.len()));
            }
            par_copy(stage.bufs[b].as_mut_slice(), chunk);
            {
                let mut view = dst.slice_mut(off..off + chunk.len());
                stream
                    .memcpy_htod(&stage.bufs[b].as_slice()[..chunk.len()], &mut view)
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_htod memcpy_htod: {e:?}")))?;
            }
            stage.pending[b] = Some(
                stream
                    .record_event(None)
                    .map_err(|e| SynaptixError::Cuda(format!("pinned_htod record_event: {e:?}")))?,
            );
            off += chunk.len();
        }
        Ok(dst)
    }

    pub fn synchronize(ordinal: usize) -> Result<()> {
        let stream = default_stream(ordinal)?;
        stream
            .synchronize()
            .map_err(|e| SynaptixError::Cuda(format!("cuda sync: {e:?}")))
    }

    /// Sync default + alloc + loader стримов. cuMemFreeAsync исполняется в
    /// порядке СВОЕГО стрима: тензоры из creation.rs (cat/zeros) живут на
    /// alloc_stream — sync только default оставляет их frees pending, trim
    /// пула их не видит (ложные OOM «after trim» при свободной памяти).
    pub fn synchronize_all(ordinal: usize) -> Result<()> {
        let ds = default_stream(ordinal)?;
        ds.synchronize()
            .map_err(|e| SynaptixError::Cuda(format!("cuda sync default: {e:?}")))?;
        if let Ok(s) = alloc_stream(ordinal) {
            let _ = s.synchronize();
        }
        if let Ok(s) = loader_stream(ordinal) {
            let _ = s.synchronize();
        }
        Ok(())
    }

    /// (free, total) байт VRAM на устройстве (cuMemGetInfo). Для авто-решения
    /// резидент-vs-offload по доступной памяти.
    pub fn mem_info(ordinal: usize) -> Result<(usize, usize)> {
        let ctx = get(ordinal)?;
        ctx.mem_get_info()
            .map_err(|e| SynaptixError::Cuda(format!("mem_get_info({ordinal}): {e:?}")))
    }
}

pub use inner::*;
