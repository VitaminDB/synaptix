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
    /// Сколько байт slab'а уже роздано (bump растёт только вперёд).
    bump: usize,
    /// Сколько байт держат живые суб-аллокации.
    live: usize,
    /// Владение блоком. `None` — блок вынут для освобождения вне мьютекса.
    buf: Option<cudarc::driver::CudaSlice<u8>>,
}

#[derive(Default)]
struct State {
    slabs: Vec<Slab>,
    /// ptr суб-аллокации → (slab, байт). Нужна, чтобы освобождение знало,
    /// у какого slab'а уменьшить счётчик живых.
    by_ptr: HashMap<u64, (u64, usize)>,
    /// Slab, в который идёт bump прямо сейчас.
    current: Option<u64>,
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

/// Открыть slab и сделать его текущим. Вызывается под мьютексом.
fn open_slab(st: &mut State, ordinal: usize, need: usize) -> Option<u64> {
    // Хук ставим при первом slab'е: до него в арене нет ни одного адреса, и
    // платить лишним вызовом на каждом освобождении в процессе незачем.
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| cudarc::driver::set_free_hook(claim_free));
    let len = slab_bytes().max(align_up(need));
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
    let base = ptr;
    st.slabs.push(Slab { id, ordinal, base, len, bump: 0, live: 0, buf: Some(buf) });
    st.current = Some(id);
    widen_bounds(base, len);
    Some(id)
}

/// Начать группу аллокаций, которая обязана лечь в один slab.
///
/// Эксперт — это несколько тензоров (packed + масштабы у каждой из двух
/// проекций), и вытеснение работает по slab'ам: если хвост эксперта уедет в
/// соседний slab, тот не опустеет, пока жив этот эксперт. `hint` — ожидаемый
/// размер эксперта целиком; при нехватке места в текущем slab'е открываем
/// новый заранее.
pub fn begin_group(ordinal: usize, hint: usize) {
    if !enabled() || hint == 0 {
        return;
    }
    let need = align_up(hint);
    if need > slab_bytes() {
        return;
    }
    let mut st = STATE.lock();
    let room = st
        .current
        .and_then(|id| st.slabs.iter().find(|s| s.id == id))
        .map(|s| s.len - s.bump)
        .unwrap_or(0);
    if room < need {
        open_slab(&mut st, ordinal, need);
    }
}

/// Id slab'а, в который сейчас идут аллокации.
pub fn current_slab() -> Option<u64> {
    STATE.lock().current
}

/// Выдать `bytes` из арены. `None` — арена выключена, кусок слишком крупный
/// для slab'а или открыть slab не вышло; звавший идёт обычным путём в пул.
pub fn alloc(ordinal: usize, bytes: usize) -> Option<u64> {
    if !enabled() || bytes == 0 {
        return None;
    }
    let need = align_up(bytes);
    // Крупные буферы — мимо арены: они и так занимают целые сегменты пула,
    // а slab с таким жильцом не опустеет никогда.
    if need * 2 > slab_bytes() {
        return None;
    }
    let mut st = STATE.lock();
    let fits = st
        .current
        .and_then(|id| st.slabs.iter().find(|s| s.id == id))
        .map(|s| s.len - s.bump >= need)
        .unwrap_or(false);
    if !fits {
        open_slab(&mut st, ordinal, need)?;
    }
    let id = st.current?;
    let slab = st.slabs.iter_mut().find(|s| s.id == id)?;
    let ptr = slab.base + slab.bump as u64;
    slab.bump += need;
    slab.live += need;
    st.by_ptr.insert(ptr, (id, need));
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
    let Some((id, bytes)) = st.by_ptr.remove(&ptr) else {
        return false;
    };
    if let Some(slab) = st.slabs.iter_mut().find(|s| s.id == id) {
        slab.live = slab.live.saturating_sub(bytes);
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
        let current = st.current;
        let mut dead = Vec::new();
        let mut freed = 0usize;
        st.slabs.retain_mut(|s| {
            // Текущий slab не трогаем: в него ещё идёт bump.
            if s.ordinal != ordinal || s.live > 0 || Some(s.id) == current {
                return true;
            }
            freed += s.len;
            dead.push(s.buf.take());
            false
        });
        if st.slabs.is_empty() {
            st.current = None;
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
/// набор экспертов). Текущий slab в список не попадает: вытеснять то, во что
/// прямо сейчас идёт запись, бессмысленно.
pub fn slabs_by_age() -> Vec<u64> {
    let st = STATE.lock();
    let current = st.current;
    let mut ids: Vec<u64> = st
        .slabs
        .iter()
        .filter(|s| Some(s.id) != current && s.live > 0)
        .map(|s| s.id)
        .collect();
    ids.sort_unstable();
    ids
}

/// Распустить арену при выгрузке модели: отдаёт всё, что уже никем не занято.
///
/// Slab'ы с живыми резидентами остаются — их адреса розданы тензорам, которые
/// ещё не дропнуты, и вернуть такой блок драйверу значит подарить кому-то
/// use-after-free. Они уйдут следующим [`release_empty`], когда модель
/// действительно умрёт.
pub fn reset(ordinal: usize) -> usize {
    let mut st = STATE.lock();
    if st.current.take().is_some() {
        // Текущий slab больше не текущий — иначе `release_empty` его не тронет.
    }
    drop(st);
    release_empty(ordinal)
}
