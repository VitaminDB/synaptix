use crate::device::Device;
use crate::error::{Result, SynaptixError};

pub fn get(_ordinal: usize) -> Result<Device> {
    Err(SynaptixError::Unsupported("metal backend"))
}
