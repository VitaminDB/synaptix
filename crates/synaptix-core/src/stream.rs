use crate::device::Device;
use crate::error::{Result, SynaptixError};

#[derive(Clone)]
pub enum Stream {
    CpuNoop,
    Cuda(std::sync::Arc<cudarc::driver::CudaStream>),
}

impl Stream {
    pub fn default_for(device: Device) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Stream::CpuNoop),
            Device::Cuda(_ord) => {
                {
                    let s = crate::device::cuda::default_stream(_ord)?;
                    Ok(Stream::Cuda(s))
                }
            }
            _ => Err(SynaptixError::Unsupported("stream for this device")),
        }
    }

    pub fn sync(&self) -> Result<()> {
        match self {
            Stream::CpuNoop => Ok(()),
            Stream::Cuda(s) => s
                .synchronize()
                .map_err(|e| SynaptixError::Cuda(format!("stream sync: {e:?}"))),
        }
    }

    pub fn as_cuda(&self) -> Option<&std::sync::Arc<cudarc::driver::CudaStream>> {
        match self {
            Stream::Cuda(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stream::CpuNoop => write!(f, "Stream::CpuNoop"),
            Stream::Cuda(_) => write!(f, "Stream::Cuda(..)"),
        }
    }
}
