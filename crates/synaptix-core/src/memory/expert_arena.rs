//! Арена под резидентов MoE: крупные slab'ы, внутри — bump-суб-аллокация.
//!
//! Зачем. Эксперты живут в своём CUDA-мемпуле, и вытеснение возвращает их
//! память пулу — но `cuMemPoolTrimTo` отдаёт драйверу только сегменты, в
//! которых не осталось ни одной живой аллокации. Кэш вытесняет резидентов
//! вразнобой («часы» с битом обращения), поэтому в каждом сегменте оставался
//! кто-то живой, и трим возвращал ноль: измерено в
//! `synaptix-core/tests/experts_pool_fragmentation.rs` — освобождение 732 МБ
//! россыпью даёт драйверу 0 МБ. На живой сессии это выглядело как «фантомный
//! бюджет»: кэш ужимался 13.07 → 9.4 ГБ по своему учёту, `cuMemGetInfo`
//! показывал 40 МБ свободных, и следующий префилл падал с OOM на аллокации в
//! 205 МБ, имея рядом десять отдаваемых гигабайт.
//!
//! Что здесь. Пул отдаёт память не поштучно, а slab'ами по [`slab_bytes`];
//! эксперты нарезаются внутри slab'а bump-указателем. Слот сам по себе не
//! освобождается — освобождается slab целиком, когда из него вытеснены все
//! резиденты. Драйвер видит ровно одну аллокацию на slab, поэтому трим
//! возвращает её без остатка.
//!
//! Кто чем управляет: арена только раздаёт и считает; решение «какой slab
//! вытесняем» принимает кэш экспертов (`synaptix-llm-common::moe`), потому
//! что только он знает, кто из резидентов ещё нужен.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// Выравнивание суб-аллокации. CUDA-мемпул отдаёт 512-байтовые границы, ядра
/// (в том числе TMA) на большее не рассчитывают.
const ALIGN: usize = 512;

/// Границы всех slab'ов: быстрый фильтр для [`claim_free`], который зовётся
/// на КАЖДОМ освобождении в процессе. Внутрь мьютекса заходим, только когда
/// адрес действительно попал в диапазон арены.
static LO: AtomicU64 = AtomicU64::new(u64::MAX);
static HI: AtomicU64 = AtomicU64::new(0);

/// Размер slab'а. Больше — реже открываем новый и меньше огрызков, но грубее
/// шаг вытеснения: slab уходит целиком.
pub fn slab_bytes() -> usize {
    static V: Lazy<usize> = Lazy::new(|| {
        std::env::var("SYN_EXPERT_SLAB_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256)
            * (1 << 20)
    });
    *V
}

/// Выключатель: `SYN_EXPERT_ARENA=0` возвращает поштучные аллокации в пул.
pub fn enabled() -> bool {
    static V: Lazy<bool> = Lazy::new(|| std::env::var("SYN_EXPERT_ARENA").as_deref() != Ok("0"));
    *V
}

struct Slab {
    id: u64,
    ordinal: usize,
    base: u64,
    len: usize,
    /// Размер слота: у slab'а он один на всех. Эксперты модели одинаковы, так
    /// что разных размеров всего несколько (упакованные веса двух проекций и
    /// их масштабы), и слот освободившегося эксперта подходит следующему —
    /// ради этого slab и привязан к размеру.
    slot: usize,
    /// Свободные слоты по индексу. Пустой список и `bump == slots` означают,
    /// что slab занят целиком.
    free: Vec<u32>,
    /// Сколько слотов уже роздано хотя бы раз (граница нетронутого хвоста).
    bump: usize,
    /// Сколько байт держат живые слоты.
    live: usize,
    /// Владение блоком. `None` — блок вынут для освобождения вне мьютекса.
    buf: Option<cudarc::driver::CudaSlice<u8>>,
}

impl Slab {
    fn slots(&self) -> usize {
        self.len / self.slot
    }

    fn take_slot(&mut self) -> Option<u32> {
        if let Some(idx) = self.free.pop() {
            self.live += self.slot;
            return Some(idx);
        }
        if self.bump < self.slots() {
            let idx = self.bump as u32;
            self.bump += 1;
            self.live += self.slot;
            return Some(idx);
        }
        None
    }
}

#[derive(Default)]
struct State {
    slabs: Vec<Slab>,
    /// ptr суб-аллокации → (slab, слот). Нужна освобождению: оно знает только
    /// адрес.
    by_ptr: HashMap<u64, (u64, u32)>,
    next_id: u64,
}

static STATE: Lazy<Mutex<State>> = Lazy::new(|| Mutex::new(State::default()));

/// Сводка по арене для логов и планировщика VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaStats {
    pub slabs: usize,
    /// Сколько VRAM держат slab'ы (это и есть цена арены для драйвера).
    pub reserved: usize,
    /// Сколько роздано живым суб-аллокациям.
    pub live: usize,
}

pub fn stats() -> ArenaStats {
    let st = STATE.lock();
    ArenaStats {
        slabs: st.slabs.len(),
        reserved: st.slabs.iter().map(|s| s.len).sum(),
        live: st.slabs.iter().map(|s| s.live).sum(),
    }
}

/// Сколько VRAM держат slab'ы арены.
pub fn reserved_bytes() -> usize {
    STATE.lock().slabs.iter().map(|s| s.len).sum()
}

fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

fn widen_bounds(base: u64, len: usize) {
    LO.fetch_min(base, Ordering::AcqRel);
    HI.fetch_max(base + len as u64, Ordering::AcqRel);
}

/// Открыть slab под слоты размера `slot`. Вызывается под мьютексом.
fn open_slab(st: &mut State, ordinal: usize, slot: usize) -> Option<usize> {
    // Хук ставим при первом slab'е: до него в арене нет ни одного адреса, и
    // платить лишним вызовом на каждом освобождении в процессе незачем.
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| cudarc::driver::set_free_hook(claim_free));
    register_reclaimable();
    // Длину режем по слоту: хвост короче слота всё равно никому не достанется.
    let len = (slab_bytes() / slot).max(1) * slot;
    let stream = crate::device::cuda::default_stream(ordinal).ok()?;
    let pool = crate::device::cuda::experts_pool(ordinal).ok()?;
    let ptr = unsafe {
        cudarc::driver::result::mem_pool::alloc_async(pool, len, stream.cu_stream())
    }
    .ok()?;
    // Slab держим как обычный `CudaSlice`: его адрес НЕ попадёт в `claim_free`,
    // пока slab числится в `state` — освобождаем мы его сами, вне мьютекса.
    let buf = unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) };
    let id = st.next_id;
    st.next_id += 1;
    st.slabs.push(Slab {
        id,
        ordinal,
        base: ptr,
        len,
        slot,
        free: Vec::new(),
        bump: 0,
        live: 0,
        buf: Some(buf),
    });
    widen_bounds(ptr, len);
    Some(st.slabs.len() - 1)
}

/// Выдать `bytes` из арены. `None` — арена выключена, кусок слишком крупный
/// для slab'а или открыть slab не вышло; звавший идёт обычным путём в пул.
pub fn alloc(ordinal: usize, bytes: usize) -> Option<u64> {
    if !enabled() || bytes == 0 {
        return None;
    }
    let slot = align_up(bytes);
    // Крупные буферы — мимо арены: они и так занимают целые сегменты пула,
    // а slab с таким жильцом не опустеет никогда.
    if slot * 2 > slab_bytes() {
        return None;
    }
    let mut st = STATE.lock();
    // Слот из уже открытого slab'а этого размера — это и есть переиспользование
    // памяти вытесненных экспертов. Без него bump раздавал бы новый адрес на
    // каждую подкачку, и арена росла бы вслед за ПОТОКОМ, а не за числом
    // живых резидентов (наблюдали 16.5 ГБ slab'ов при 0.94 ГБ экспертов).
    let found = st
        .slabs
        .iter()
        .position(|s| s.ordinal == ordinal && s.slot == slot && (!s.free.is_empty() || s.bump < s.slots()));
    let idx = match found {
        Some(i) => i,
        None => open_slab(&mut st, ordinal, slot)?,
    };
    let slab = st.slabs.get_mut(idx)?;
    let slot_idx = slab.take_slot()?;
    let ptr = slab.base + slot_idx as u64 * slab.slot as u64;
    let id = slab.id;
    st.by_ptr.insert(ptr, (id, slot_idx));
    Some(ptr)
}

/// Хук `cudarc::driver::set_free_hook`: забирает себе освобождение адресов,
/// которые раздала арена. Драйвер о них не знает — `cuMemFreeAsync` на них
/// вернул бы `CUDA_ERROR_INVALID_VALUE`.
pub fn claim_free(ptr: cudarc::driver::sys::CUdeviceptr) -> bool {
    // Быстрый путь: почти все освобождения в процессе — не наши.
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return false;
    }
    let mut st = STATE.lock();
    let Some((id, slot_idx)) = st.by_ptr.remove(&ptr) else {
        return false;
    };
    if let Some(slab) = st.slabs.iter_mut().find(|s| s.id == id) {
        slab.live = slab.live.saturating_sub(slab.slot);
        slab.free.push(slot_idx);
    }
    true
}

/// Отдать драйверу slab'ы, из которых вытеснены все резиденты. Возвращает,
/// сколько байт отпущено.
pub fn release_empty(ordinal: usize) -> usize {
    if !enabled() {
        return 0;
    }
    let (freed, dead) = {
        let mut st = STATE.lock();
        let mut dead = Vec::new();
        let mut freed = 0usize;
        st.slabs.retain_mut(|s| {
            if s.ordinal != ordinal || s.live > 0 {
                return true;
            }
            freed += s.len;
            dead.push(s.buf.take());
            false
        });
        if st.slabs.is_empty() {
            LO.store(u64::MAX, Ordering::Release);
            HI.store(0, Ordering::Release);
        }
        (freed, dead)
    };
    // Дроп slab'а зовёт `claim_free` — он обязан увидеть мьютекс свободным и
    // НЕ найти адрес slab'а в карте, иначе блок останется у нас навсегда.
    drop(dead);
    if freed > 0 {
        let _ = crate::device::cuda::synchronize_all(ordinal);
        let _ = crate::device::cuda::trim_experts_pool(ordinal);
    }
    freed
}

/// Slab, которому принадлежит адрес.
pub fn slab_of(ptr: u64) -> Option<u64> {
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return None;
    }
    let st = STATE.lock();
    st.slabs
        .iter()
        .find(|s| ptr >= s.base && ptr < s.base + s.len as u64)
        .map(|s| s.id)
}

/// Slab'ы в порядке появления — кэшу, чтобы выбрать жертву (самый старый
/// набор экспертов).
pub fn slabs_by_age() -> Vec<u64> {
    let st = STATE.lock();
    let mut ids: Vec<u64> = st.slabs.iter().filter(|s| s.live > 0).map(|s| s.id).collect();
    ids.sort_unstable();
    ids
}

/// Пустые slab'ы — это отдаваемая память: на чужом OOM аллокатор попросит их
/// через реестр [`crate::memory::reclaim`] раньше, чем сдастся. Сам кэш
/// экспертов зарегистрирован там же и вытесняет резидентов; арена отдаёт лишь
/// то, из чего уже все ушли.
struct EmptySlabs;

impl crate::memory::reclaim::Reclaimable for EmptySlabs {
    fn reclaim(&self, device: crate::device::Device, _want: usize) -> usize {
        let crate::device::Device::Cuda(ord) = device else {
            return 0;
        };
        release_empty(ord)
    }

    fn reclaimable_bytes(&self, device: crate::device::Device) -> usize {
        let crate::device::Device::Cuda(ord) = device else {
            return 0;
        };
        let st = STATE.lock();
        st.slabs
            .iter()
            .filter(|s| s.ordinal == ord && s.live == 0)
            .map(|s| s.len)
            .sum()
    }
}

/// Зарегистрировать арену как отдаваемую память. Идемпотентно.
fn register_reclaimable() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        static KEEP: once_cell::sync::Lazy<std::sync::Arc<dyn crate::memory::reclaim::Reclaimable>> =
            once_cell::sync::Lazy::new(|| std::sync::Arc::new(EmptySlabs));
        crate::memory::reclaim::register(&KEEP);
    });
}

/// Распустить арену при выгрузке модели: отдаёт всё, что уже никем не занято.
///
/// Slab'ы с живыми резидентами остаются — их слоты розданы тензорам, которые
/// ещё не дропнуты, и вернуть такой блок драйверу значит подарить кому-то
/// use-after-free. Они уйдут следующим [`release_empty`], когда модель
/// действительно умрёт.
pub fn reset(ordinal: usize) -> usize {
    release_empty(ordinal)
}
