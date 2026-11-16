//! Allocator trait.

pub mod arena;
pub mod cuda_pool;
pub mod expert_arena;
pub mod host_mapped;
pub mod pinned;
pub mod reclaim;

/// Общий хук освобождения буферов CUDA: адрес может принадлежать закреплённой
/// RAM ([`host_mapped`]) или арене экспертов ([`expert_arena`]) — тогда
/// драйверу он не возвращается.
fn free_hook(ptr: cudarc::driver::sys::CUdeviceptr) -> bool {
    host_mapped::claim_free(ptr) || expert_arena::claim_free(ptr)
}

/// Поставить хук (один раз на процесс).
pub fn install_free_hook() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| cudarc::driver::set_free_hook(free_hook));
}
