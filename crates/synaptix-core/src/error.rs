use crate::device::Device;
use crate::dtype::DType;

pub type Result<T> = std::result::Result<T, SynaptixError>;

#[derive(Debug, thiserror::Error)]
pub enum SynaptixError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("device mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    DeviceMismatch { lhs: Device, rhs: Device },

    #[error("dim {dim} out of range for rank {rank}")]
    DimOutOfRange { dim: usize, rank: usize },

    #[error("rank mismatch: expected rank {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("cannot reshape {from:?} into {to:?}")]
    ReshapeMismatch { from: Vec<usize>, to: Vec<usize> },

    #[error("cannot broadcast {lhs:?} with {rhs:?}")]
    BroadcastMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("narrow out of bounds: dim={dim}, off={off}, len={len}, size={size}")]
    NarrowOutOfBounds { dim: usize, off: usize, len: usize, size: usize },

    #[error("non-contiguous tensor; call .contiguous() first")]
    NonContiguous,

    #[error("op `{0}` unsupported on this device/dtype combination")]
    Unsupported(&'static str),

    #[error("backend not registered for device {0:?} — did you call synaptix::init()?")]
    BackendNotRegistered(Device),

    #[error("cuda error: {0}")]
    Cuda(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl SynaptixError {
    pub fn dtype_mismatch(expected: DType, got: DType) -> Self {
        Self::DTypeMismatch { expected, got }
    }

    pub fn device_mismatch(lhs: Device, rhs: Device) -> Self {
        Self::DeviceMismatch { lhs, rhs }
    }

    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        Self::ShapeMismatch { expected: expected.to_vec(), got: got.to_vec() }
    }
}
