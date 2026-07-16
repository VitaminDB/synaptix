//! Public entry points для инициализации distributed-runtime.
//!
//! Поддерживается backend `"local"` (синонимы: `""`, `"memory"`, `"in-process"`) — in-memory
//! «мир» внутри одного процесса. Backend'ы `"nccl"/"mpi"/"gloo"/"tcp"` принимаются строкой
//! но возвращают `Err` (требуют отдельной фичи и C-зависимостей; см. handover).

use crate::backend::{self, Backend};
use crate::error::Result;

/// Зарегистрировать текущий поток в process group с заданным rank/world_size.
///
/// Idempotent на уровне ProcessGroup (первый вызов создаёт его, последующие проверяют
/// совместимость world_size и backend'a). Каждый поток должен звать с собственным rank'ом.
pub fn init_process_group(backend: &str, rank: usize, world_size: usize) -> Result<()> {
    let b = Backend::from_str_safe(backend)?;
    backend::register(b, rank, world_size)
}

/// Сбросить process group (для тестов). Параллельные `send_to/recv_from/barrier` могут
/// вернуть `NotInitialized` — звать только когда все потоки уже finish'нули.
pub fn destroy_process_group() {
    backend::destroy();
}

/// Барьер для всех зарегистрированных rank'ов. Блокирует до прихода всех.
pub fn barrier() -> Result<()> {
    backend::barrier()
}

/// Текущий rank потока (`None` если поток не зарегистрирован через `init_process_group`).
pub fn local_rank() -> Option<usize> {
    backend::local_rank()
}

/// Размер group (`None` если не инициализирована).
pub fn world_size() -> Option<usize> {
    backend::current_world_size()
}

/// True если хотя бы один поток вызвал `init_process_group`.
pub fn is_initialized() -> bool {
    backend::is_initialized()
}
