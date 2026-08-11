pub mod facade;
pub mod prelude;

pub use synaptix_core::backend;
pub use synaptix_core::device::{self, Device, DeviceKind};
pub use synaptix_core::dtype::{DType, SynaptixScalar};
pub use synaptix_core::error::{Result, SynaptixError};
pub use synaptix_core::stream::Stream;
pub use synaptix_core::tensor::{self, Tensor};
pub use synaptix_core::tensor::layout::Layout;
pub use synaptix_core::tensor::shape::{Dim, IntoShape, Shape};

pub fn init() -> Result<()> {
    synaptix_kernels_cpu::ensure_registered();
    {
        synaptix_kernels_cuda::ensure_registered();
    }
    Ok(())
}
