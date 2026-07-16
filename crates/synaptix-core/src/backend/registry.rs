use crate::backend::Backend;
use crate::device::{Device, DeviceKind};
use crate::error::{Result, SynaptixError};
use once_cell::sync::Lazy;
use parking_lot::RwLock;

const NUM_KINDS: usize = 4;

static REGISTRY: Lazy<RwLock<[Option<&'static dyn Backend>; NUM_KINDS]>> =
    Lazy::new(|| RwLock::new([None; NUM_KINDS]));

fn kind_index(k: DeviceKind) -> usize {
    match k {
        DeviceKind::Cpu => 0,
        DeviceKind::Cuda => 1,
        DeviceKind::Metal => 2,
        DeviceKind::Wgpu => 3,
    }
}

pub fn register_backend(kind: DeviceKind, backend: &'static dyn Backend) {
    let mut w = REGISTRY.write();
    w[kind_index(kind)] = Some(backend);
}

pub fn backend_for(device: Device) -> Result<&'static dyn Backend> {
    let r = REGISTRY.read();
    match r[kind_index(device.kind())] {
        Some(b) => Ok(b),
        None => Err(SynaptixError::BackendNotRegistered(device)),
    }
}

pub fn is_registered(kind: DeviceKind) -> bool {
    REGISTRY.read()[kind_index(kind)].is_some()
}
