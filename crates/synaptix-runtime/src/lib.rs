pub mod compute_dispatch;
pub mod flags;
pub mod flash_attn_mode;
pub mod graph_decode;
pub mod kv_prealloc;
pub mod layer_sync_mode;
pub mod mem_trace;
pub mod mempool;
pub mod nvfp4_flags;
pub mod telemetry;
pub mod thread_pool;

pub use flash_attn_mode::{flash_attn_mode, set_flash_attn_mode, FlashAttnMode};
pub use graph_decode::{graph_decode_enabled, set_graph_decode_enabled};
pub use kv_prealloc::{kv_prealloc_seq_len, set_kv_prealloc_seq_len};
pub use layer_sync_mode::{
    layer_sync_mode, layer_sync_should_apply, set_layer_sync_mode, LayerSyncMode,
};
pub use mem_trace::{mem_trace_enabled, set_mem_trace_enabled};
pub use nvfp4_flags::{
    nvfp4_gemv_enabled, nvfp4_mma_enabled, set_nvfp4_gemv_enabled, set_nvfp4_mma_enabled,
};
