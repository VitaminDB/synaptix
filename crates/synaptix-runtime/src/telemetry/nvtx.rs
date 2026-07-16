//! NVTX-range/mark — обвязка `cudarc::nvtx` под фичей `nvtx`. Без фичи — no-op.
//!
//! Дополнительная защита: если фича включена, но libnvToolsExt.so не установлена в системе
//! (типичный случай dev-сборки без Nsight), cudarc'у пришлось бы panic'ать при первом вызове.
//! Поэтому первая попытка NVTX-вызова обёрнута в `catch_unwind`; при провале флаг
//! `NVTX_AVAILABLE` фиксируется в `false` и все последующие вызовы — no-op.
//!
//! Пример:
//! ```ignore
//! let _r = NvtxRange::push("decode_step");
//! // ... GPU work ...
//! // drop(_r) → nvtxRangeEnd
//! ```

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "nvtx")]
const NVTX_UNKNOWN: u8 = 0;
#[cfg(feature = "nvtx")]
const NVTX_AVAILABLE: u8 = 1;
#[cfg(feature = "nvtx")]
const NVTX_MISSING: u8 = 2;

#[cfg(feature = "nvtx")]
static AVAILABILITY: AtomicU8 = AtomicU8::new(NVTX_UNKNOWN);

#[cfg(feature = "nvtx")]
fn probe_nvtx() -> bool {
    match AVAILABILITY.load(Ordering::Relaxed) {
        NVTX_AVAILABLE => return true,
        NVTX_MISSING => return false,
        _ => {}
    }
    // Пробуем mark с пустой меткой через catch_unwind. cudarc fallback-dynamic-loading
    // panic'нет если library не загружена. После probe-вызова cache'ируем результат.
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cudarc::nvtx::safe::mark("__synaptix_nvtx_probe__");
    }))
    .is_ok();
    AVAILABILITY.store(
        if ok { NVTX_AVAILABLE } else { NVTX_MISSING },
        Ordering::Relaxed,
    );
    ok
}

pub struct NvtxRange {
    name: String,
    #[cfg(feature = "nvtx")]
    inner: Option<cudarc::nvtx::safe::Range>,
}

impl NvtxRange {
    pub fn push(name: impl Into<String>) -> Self {
        let name = name.into();
        #[cfg(feature = "nvtx")]
        {
            let inner = if probe_nvtx() {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cudarc::nvtx::safe::scoped_range(&name)
                }))
                .ok()
            } else {
                None
            };
            return Self { name, inner };
        }
        #[cfg(not(feature = "nvtx"))]
        {
            Self { name }
        }
    }

    pub fn name(&self) -> &str { &self.name }

    /// Доступен ли реальный NVTX в текущем процессе. Без фичи `nvtx` — всегда `false`. С
    /// фичей — `true` только если cudarc успешно загрузил libnvToolsExt.so.
    pub fn is_available() -> bool {
        #[cfg(feature = "nvtx")]
        {
            probe_nvtx()
        }
        #[cfg(not(feature = "nvtx"))]
        {
            false
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRange {
    fn drop(&mut self) {
        // inner.drop() даст nvtxRangeEnd через cudarc.
        // Catch_unwind на drop не нужен: cudarc Range::drop сам по себе не panic'нет —
        // если probe не прошёл, inner = None и здесь ничего не происходит.
        let _ = self.inner.take();
    }
}

#[cfg(not(feature = "nvtx"))]
impl Drop for NvtxRange {
    fn drop(&mut self) {
        let _ = &self.name;
    }
}

/// Поставить мгновенную NVTX-метку (nvtxMarkA).
pub fn mark(message: &str) {
    #[cfg(feature = "nvtx")]
    {
        if probe_nvtx() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cudarc::nvtx::safe::mark(message);
            }));
        }
        return;
    }
    #[cfg(not(feature = "nvtx"))]
    {
        let _ = message;
    }
}
