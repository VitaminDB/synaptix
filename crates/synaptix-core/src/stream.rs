use crate::device::Device;
use crate::error::{Result, SynaptixError};

#[derive(Clone)]
pub enum Stream {
    CpuNoop,
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<cudarc::driver::CudaStream>),
}

impl Stream {
    pub fn default_for(device: Device) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Stream::CpuNoop),
            Device::Cuda(_ord) => {
                #[cfg(feature = "cuda")]
                {
                    let s = crate::device::cuda::default_stream(_ord)?;
                    Ok(Stream::Cuda(s))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(SynaptixError::Unsupported("cuda stream without cuda feature"))
                }
            }
            _ => Err(SynaptixError::Unsupported("stream for this device")),
        }
    }

    pub fn sync(&self) -> Result<()> {
        match self {
            Stream::CpuNoop => Ok(()),
            #[cfg(feature = "cuda")]
            Stream::Cuda(s) => s
                .synchronize()
                .map_err(|e| SynaptixError::Cuda(format!("stream sync: {e:?}"))),
        }
    }

    #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
            Stream::Cuda(_) => write!(f, "Stream::Cuda(..)"),
        }
    }
}
