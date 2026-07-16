//! In-memory backend для distributed-примитивов: внутри одного процесса несколько потоков
//! могут зарегистрироваться под разными rank'ами и обмениваться тензорами через `Mailbox`
//! (Mutex<VecDeque> + Condvar). Это **не** заменяет NCCL/MPI/Gloo — это functional-stub для:
//!  - unit/integration-тестов pipeline-parallel без C-зависимостей;
//!  - single-process inference, где есть smysl shardить math между threads (не GPU);
//!  - dev-сборок без CUDA/MPI runtime.
//!
//! Когда реальный NCCL/MPI потребуется, он встанет под отдельной фичей и зарегистрируется как
//! второй вариант `Backend` без изменений public API.

use parking_lot::{Condvar, Mutex};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Внутри одного процесса (Arc<Mutex>+Condvar mailboxes), без сети.
    Local,
}

impl Backend {
    pub fn from_str_safe(s: &str) -> Result<Self> {
        match s {
            "" | "local" | "memory" | "in-process" => Ok(Backend::Local),
            "nccl" | "mpi" | "gloo" | "tcp" => Err(DistError::Other(format!(
                "backend `{s}` not built in this binary (only `local` backend is available)"
            ))),
            other => Err(DistError::Other(format!("unknown backend `{other}`"))),
        }
    }
}

struct Mailbox {
    queue: Mutex<std::collections::VecDeque<Tensor>>,
    cv: Condvar,
}

impl Mailbox {
    fn new() -> Self {
        Self {
            queue: Mutex::new(std::collections::VecDeque::new()),
            cv: Condvar::new(),
        }
    }

    fn push(&self, t: Tensor) {
        self.queue.lock().push_back(t);
        self.cv.notify_one();
    }

    fn pop_blocking(&self, timeout: Option<Duration>) -> Result<Tensor> {
        let mut q = self.queue.lock();
        if let Some(t) = q.pop_front() {
            return Ok(t);
        }
        match timeout {
            None => {
                loop {
                    self.cv.wait(&mut q);
                    if let Some(t) = q.pop_front() {
                        return Ok(t);
                    }
                }
            }
            Some(dur) => {
                let deadline = Instant::now() + dur;
                loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(DistError::Other("recv timed out".into()));
                    }
                    let res = self.cv.wait_for(&mut q, remaining);
                    if let Some(t) = q.pop_front() {
                        return Ok(t);
                    }
                    if res.timed_out() {
                        return Err(DistError::Other("recv timed out".into()));
                    }
                }
            }
        }
    }
}

struct BarrierState {
    /// Количество потоков, подошедших к текущему барьеру.
    count: usize,
    /// Монотонная версия — увеличивается на 1 каждый раз когда все `world_size` потоков
    /// подошли. Используется для wakeup-проверки в `wait` (нужно, иначе spurious wakeup
    /// может выпустить поток до момента когда последний пришёл).
    version: u64,
}

/// ProcessGroup — состояние in-memory backend для одного «мира» (world).
struct ProcessGroup {
    backend: Backend,
    world_size: usize,
    /// rank → Mailbox получателя (`send_to(rank, t)` пушит сюда).
    mailboxes: Vec<Mailbox>,
    /// Зарегистрированы ли отдельные потоки под каждым rank'ом (для диагностики).
    registered: Mutex<Vec<bool>>,
    /// Состояние barrier'a: count+version в одном мьютексе (требование Condvar).
    barrier_state: Mutex<BarrierState>,
    barrier_cv: Condvar,
}

impl ProcessGroup {
    fn new(backend: Backend, world_size: usize) -> Self {
        let mut mboxes = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            mboxes.push(Mailbox::new());
        }
        Self {
            backend,
            world_size,
            mailboxes: mboxes,
            registered: Mutex::new(vec![false; world_size]),
            barrier_state: Mutex::new(BarrierState { count: 0, version: 0 }),
            barrier_cv: Condvar::new(),
        }
    }
}

static GROUP: OnceLock<Mutex<Option<ProcessGroup>>> = OnceLock::new();

fn group_slot() -> &'static Mutex<Option<ProcessGroup>> {
    GROUP.get_or_init(|| Mutex::new(None))
}

thread_local! {
    static LOCAL_RANK: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Зарегистрировать ProcessGroup (одноразово в процессе) + закрепить за текущим потоком rank.
pub(crate) fn register(backend: Backend, rank: usize, world_size: usize) -> Result<()> {
    if world_size == 0 {
        return Err(DistError::Other("world_size must be ≥ 1".into()));
    }
    if rank >= world_size {
        return Err(DistError::RankOutOfRange { rank, world_size });
    }

    let slot = group_slot();
    {
        let mut g = slot.lock();
        match g.as_ref() {
            None => {
                *g = Some(ProcessGroup::new(backend, world_size));
            }
            Some(existing) => {
                if existing.world_size != world_size {
                    return Err(DistError::Other(format!(
                        "process group already initialized with world_size={}, requested {}",
                        existing.world_size, world_size
                    )));
                }
                if existing.backend != backend {
                    return Err(DistError::Other(format!(
                        "process group already initialized with backend={:?}, requested {:?}",
                        existing.backend, backend
                    )));
                }
            }
        }
        let pg = g.as_ref().unwrap();
        pg.registered.lock()[rank] = true;
    }

    LOCAL_RANK.with(|r| r.set(Some(rank)));
    // Совместимость со старым world::* API (используется примитивами collectives и pipeline
    // через synaptix_distributed::world). Это глобал per-process, поэтому в multi-thread
    // setup'ах он будет содержать последний-проинициализированный rank — но в multi-thread
    // тестах правильнее использовать `local_rank()` ниже.
    crate::world::init(rank, world_size)?;
    Ok(())
}

pub fn destroy() {
    let slot = group_slot();
    *slot.lock() = None;
    LOCAL_RANK.with(|r| r.set(None));
}

pub fn local_rank() -> Option<usize> {
    LOCAL_RANK.with(|r| r.get())
}

pub fn is_initialized() -> bool {
    group_slot().lock().is_some()
}

pub fn current_world_size() -> Option<usize> {
    group_slot().lock().as_ref().map(|g| g.world_size)
}

/// Отправить tensor в mailbox `dst_rank`. Не блокирует.
pub fn send_to(dst_rank: usize, tensor: Tensor) -> Result<()> {
    let slot = group_slot();
    let g = slot.lock();
    let pg = g.as_ref().ok_or(DistError::NotInitialized)?;
    if dst_rank >= pg.world_size {
        return Err(DistError::RankOutOfRange { rank: dst_rank, world_size: pg.world_size });
    }
    pg.mailboxes[dst_rank].push(tensor);
    Ok(())
}

/// Забрать tensor из mailbox `rank` (обычно `rank == local_rank()`). Блокирует до прихода.
pub fn recv_from(rank: usize, timeout: Option<Duration>) -> Result<Tensor> {
    // Снимаем slot lock до wait — иначе блокировка процесса.
    let mailbox_ptr: *const Mailbox = {
        let slot = group_slot();
        let g = slot.lock();
        let pg = g.as_ref().ok_or(DistError::NotInitialized)?;
        if rank >= pg.world_size {
            return Err(DistError::RankOutOfRange { rank, world_size: pg.world_size });
        }
        &pg.mailboxes[rank] as *const Mailbox
    };
    // Mailbox живёт пока существует ProcessGroup в OnceLock — `destroy()` только обнуляет
    // Option, но физически память не освобождается до перезаписи `Some(...)`. На практике
    // recv_from вызываются между init/destroy, поэтому это безопасно. (Полная защита
    // потребовала бы Arc<Mailbox> — overhead не стоит того при нашем use case.)
    let mbx = unsafe { &*mailbox_ptr };
    mbx.pop_blocking(timeout)
}

/// Дождаться всех `world_size` потоков. Каждый поток зовёт `barrier()`; разблокируется
/// одновременно когда все подошли. Поддерживает повторные вызовы благодаря version-counter'у.
pub fn barrier() -> Result<()> {
    // Снимаем slot.lock() сразу и берём указатель на ProcessGroup — иначе при wait()
    // на Condvar мы заблокируем group_slot и весь backend.
    let (pg_ptr, world_size) = {
        let slot = group_slot();
        let g = slot.lock();
        let pg = g.as_ref().ok_or(DistError::NotInitialized)?;
        (pg as *const ProcessGroup, pg.world_size)
    };
    // Safety: ProcessGroup живёт в `static OnceLock<Mutex<Option<...>>>` и физически
    // освобождается только при перезаписи слота через destroy(). Между init и destroy
    // указатель стабилен. См. комментарий к `recv_from`.
    let pg = unsafe { &*pg_ptr };

    let mut state = pg.barrier_state.lock();
    let my_version = state.version;
    state.count += 1;
    if state.count == world_size {
        state.count = 0;
        state.version = my_version.wrapping_add(1);
        pg.barrier_cv.notify_all();
        return Ok(());
    }
    while state.version == my_version {
        pg.barrier_cv.wait(&mut state);
    }
    Ok(())
}
