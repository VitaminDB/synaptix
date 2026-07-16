use std::sync::{Arc, Mutex};

use once_cell::sync::OnceCell;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::storage::Storage;

pub struct QuantWeight {
    packed: Mutex<Option<Arc<Storage>>>,
    scales: Arc<Storage>,
    dtype: DType,
    n: usize,
    k: usize,
    device: Device,
    shuffled: OnceCell<Arc<Storage>>,
}

impl QuantWeight {
    pub fn new(
        packed: Arc<Storage>,
        scales: Arc<Storage>,
        dtype: DType,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        if !dtype.is_quantized() {
            return Err(SynaptixError::Unsupported(
                "QuantWeight: dtype должен быть quantized (NVFP4/MXFP8/...)",
            ));
        }
        if packed.device() != scales.device() {
            return Err(SynaptixError::device_mismatch(
                packed.device(),
                scales.device(),
            ));
        }
        let device = packed.device();
        Ok(Self {
            packed: Mutex::new(Some(packed)),
            scales,
            dtype,
            n,
            k,
            device,
            shuffled: OnceCell::new(),
        })
    }

    pub fn from_tensors(packed: &Tensor, scales: &Tensor, n: usize, k: usize) -> Result<Self> {
        Self::new(
            packed.storage_arc(),
            scales.storage_arc(),
            packed.dtype(),
            n,
            k,
        )
    }

    pub fn packed_arc(&self) -> Option<Arc<Storage>> {
        self.packed.lock().unwrap().clone()
    }

    pub fn release_packed(&self) {
        *self.packed.lock().unwrap() = None;
    }

    pub fn scales(&self) -> &Storage {
        &self.scales
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Перенос packed+scales на `dev` (host-stream квант-весов: квантуем 1× на GPU →
    /// храним на CPU → стримим обратно по требованию). `shuffled`-кэш не переносится
    /// (лениво пере-инициализируется на новом устройстве). Байты не меняются →
    /// bit-identical с резидентным квантом.
    pub fn to_device(&self, dev: Device) -> Result<Self> {
        if self.device == dev {
            let packed = self.packed_arc().ok_or(SynaptixError::Unsupported(
                "QuantWeight::to_device: packed released",
            ))?;
            return Self::new(packed, self.scales.clone(), self.dtype, self.n, self.k);
        }
        let packed = self.packed_arc().ok_or(SynaptixError::Unsupported(
            "QuantWeight::to_device: packed released",
        ))?;
        let packed = Arc::new(crate::tensor::conversion::storage_to_device(&packed, dev)?);
        let scales = Arc::new(crate::tensor::conversion::storage_to_device(&self.scales, dev)?);
        Self::new(packed, scales, self.dtype, self.n, self.k)
    }

    pub fn shuffled(&self) -> Option<&Storage> {
        self.shuffled.get().map(|s| s.as_ref())
    }

    pub fn shuffled_or_try_init(
        &self,
        init: impl FnOnce() -> Result<Arc<Storage>>,
    ) -> Result<&Storage> {
        self.shuffled.get_or_try_init(init).map(|s| s.as_ref())
    }
}
