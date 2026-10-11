mod backend;
mod record;
mod value;
mod view;

use once_cell::sync::OnceCell;
use synaptix_core::device::DeviceKind;
use synaptix_npu::{Npu, NpuInfo};

pub use backend::NpuBackend;
pub use record::{flush, stats, Stats};

static NPU: OnceCell<Box<dyn Npu>> = OnceCell::new();
static BACKEND: NpuBackend = NpuBackend { ordinal: 0 };

pub(crate) fn npu() -> &'static dyn Npu {
    NPU.get().expect("NPU не зарегистрирован: synaptix_kernels_npu::register()").as_ref()
}

pub fn register() -> Result<&'static NpuInfo, synaptix_npu::NpuError> {
    let n = NPU.get_or_try_init(synaptix_npu::open)?;
    synaptix_kernels_cpu::ensure_registered();
    synaptix_core::backend::registry::register_backend(DeviceKind::Npu, &BACKEND);
    Ok(n.info())
}

pub fn info() -> Option<&'static NpuInfo> {
    NPU.get().map(|n| n.info())
}
