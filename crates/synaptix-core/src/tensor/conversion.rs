use std::sync::Arc;

use crate::device::Device;
use crate::dtype::SynaptixScalar;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::layout::Layout;
use crate::tensor::storage::Storage;
use crate::tensor::storage::CpuBuf;

impl Tensor {
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }
        if !self.is_contiguous() {
            return self.contiguous()?.to_device(device);
        }
        let n_bytes = self.dtype().bytes_for_numel(self.numel());
        let elem_bytes = size_helper::size_of_byte_block(self.dtype());
        let byte_offset = self.layout.offset() * elem_bytes;
        match (&*self.storage, device) {
            (Storage::Cpu(b), Device::Cuda(_ord)) => {
                {
                    // Вес-стрим (pin-mirror/offload-pinned активны) идёт через
                    // изолированный weights-пул + pinned-конвейер — тот же стек,
                    // что from_raw_slice (cuda_alloc_from_bytes). Прямой clone_htod
                    // раньше сыпал рой 5-50MB вес-аллокаций в default-пул и дробил
                    // его free-list до нераздаваемого решета (nvfp4-19s OOM).
                    let bytes = &b.as_bytes()[byte_offset..byte_offset + n_bytes];
                    let storage = crate::tensor::creation::cuda_alloc_from_bytes(device, bytes)?;
                    let layout = Layout::contiguous(self.shape().clone(), self.dtype());
                    Ok(Tensor::from_parts(Arc::new(storage), layout))
                }
            }
            (Storage::Cuda(b), Device::Cpu) => {
                {
                    let stream = b.stream().clone();
                    // offload-выгрузка → pinned-конвейер (pageable dtoh 3-6GB/s
                    // жёг 5-8s на 24GB квант-блоков LTX host-stream).
                    let bytes: Vec<u8> = if crate::device::cuda::offload_pinned_enabled()
                        && b.slice().len() >= (4 << 20)
                    {
                        crate::device::cuda::pinned_dtoh(&stream, b.slice())?
                    } else {
                        stream
                            .clone_dtoh(b.slice())
                            .map_err(|e| SynaptixError::Cuda(format!("clone_dtoh: {e:?}")))?
                    };
                    let storage = Storage::Cpu(CpuBuf::from_vec(bytes));
                    let layout = Layout::contiguous(self.shape().clone(), self.dtype());
                    Ok(Tensor::from_parts(Arc::new(storage), layout))
                }
            }
            _ => Err(SynaptixError::Unsupported("to_device: device combo")),
        }
    }
}

/// Побайтовый перенос Storage между устройствами (для стриминга квант-весов:
/// packed/scales — сырые байты без dtype-семантики). CPU→CUDA идёт через
/// `cuda_alloc_from_bytes` → alloc-stream текущего потока + pinned staging
/// (если включён `set_offload_pinned`) — как offload-загрузка dense-весов.
pub(crate) fn storage_to_device(s: &Storage, device: Device) -> Result<Storage> {
    match (s, device) {
        (Storage::Cpu(b), Device::Cuda(_ord)) => {
            {
                crate::tensor::creation::cuda_alloc_from_bytes(device, b.as_bytes())
            }
        }
        (Storage::Cuda(b), Device::Cpu) => {
            {
                let bytes: Vec<u8> = if crate::device::cuda::offload_pinned_enabled()
                    && b.slice().len() >= (4 << 20)
                {
                    crate::device::cuda::pinned_dtoh(&b.stream().clone(), b.slice())?
                } else {
                    b.stream()
                        .clone_dtoh(b.slice())
                        .map_err(|e| SynaptixError::Cuda(format!("storage clone_dtoh: {e:?}")))?
                };
                Ok(Storage::Cpu(CpuBuf::from_vec(bytes)))
            }
        }
        _ => Err(SynaptixError::Unsupported("storage_to_device: device combo")),
    }
}

mod size_helper {
    use crate::dtype::DType;
    pub fn size_of_byte_block(dtype: DType) -> usize {
        match dtype {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::MXFP8 => 1,
            DType::NVFP4 => 1,
        }
    }
}

impl Tensor {
    pub fn to_vec1<T: SynaptixScalar>(&self) -> Result<Vec<T>> {
        self.check_dtype::<T>()?;
        if self.rank() != 1 {
            return Err(SynaptixError::RankMismatch { expected: 1, got: self.rank() });
        }
        let host = self.to_host_typed::<T>()?;
        Ok(host)
    }

    pub fn to_vec2<T: SynaptixScalar>(&self) -> Result<Vec<Vec<T>>> {
        self.check_dtype::<T>()?;
        let (d0, d1) = self.dims2()?;
        let flat = self.to_host_typed::<T>()?;
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            out.push(flat[i * d1..(i + 1) * d1].to_vec());
        }
        Ok(out)
    }

    pub fn to_vec3<T: SynaptixScalar>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        self.check_dtype::<T>()?;
        let (d0, d1, d2) = self.dims3()?;
        let flat = self.to_host_typed::<T>()?;
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            let mut row = Vec::with_capacity(d1);
            for j in 0..d1 {
                let base = (i * d1 + j) * d2;
                row.push(flat[base..base + d2].to_vec());
            }
            out.push(row);
        }
        Ok(out)
    }

    pub fn to_scalar<T: SynaptixScalar>(&self) -> Result<T> {
        self.check_dtype::<T>()?;
        if self.numel() != 1 {
            return Err(SynaptixError::ShapeMismatch {
                expected: vec![1],
                got: self.dims().to_vec(),
            });
        }
        let host = self.to_host_typed::<T>()?;
        Ok(host[0])
    }

    fn check_dtype<T: SynaptixScalar>(&self) -> Result<()> {
        if self.dtype() != T::DTYPE {
            return Err(SynaptixError::dtype_mismatch(T::DTYPE, self.dtype()));
        }
        Ok(())
    }

    fn to_host_typed<T: SynaptixScalar>(&self) -> Result<Vec<T>> {
        if !self.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let numel = self.numel();
        let elem_bytes = std::mem::size_of::<T>();
        let offset_bytes = self.layout.offset() * elem_bytes;
        match &*self.storage {
            Storage::Cpu(b) => {
                let bytes = &b.as_bytes()[offset_bytes..offset_bytes + numel * elem_bytes];
                Ok(bytemuck::cast_slice(bytes).to_vec())
            }
            Storage::Cuda(b) => {
                {
                    let stream = b.stream.clone();
                    let bytes: Vec<u8> = stream
                        .clone_dtoh(&b.buf)
                        .map_err(|e| SynaptixError::Cuda(format!("clone_dtoh: {e:?}")))?;
                    let slice = &bytes[offset_bytes..offset_bytes + numel * elem_bytes];
                    Ok(bytemuck::cast_slice(slice).to_vec())
                }
            }
        }
    }
}
