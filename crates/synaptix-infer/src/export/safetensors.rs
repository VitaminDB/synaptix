use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::error::{InferError, Result};

fn to_st_dtype(d: DType) -> Result<safetensors::Dtype> {
    use safetensors::Dtype as S;
    Ok(match d {
        DType::F64 => S::F64,
        DType::F32 => S::F32,
        DType::F16 => S::F16,
        DType::BF16 => S::BF16,
        DType::U8 => S::U8,
        DType::U32 => S::U32,
        DType::I32 => S::I32,
        DType::I64 => S::I64,
        other => {
            return Err(InferError::Other(format!(
                "export safetensors: неподдерживаемый dtype {other:?} (квантованные форматы не сериализуются в safetensors)"
            )))
        }
    })
}

/// Владеющий байтами view одного тензора для `safetensors::serialize`.
struct OwnedView {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for OwnedView {
    fn dtype(&self) -> safetensors::Dtype { self.dtype }
    fn shape(&self) -> &[usize] { &self.shape }
    fn data(&self) -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
    fn data_len(&self) -> usize { self.data.len() }
}

/// Сериализовать тензоры в байты формата safetensors (CPU, contiguous).
pub fn serialize_tensors(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let mut views: Vec<(String, OwnedView)> = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        let cpu = t.to_device(Device::Cpu).map_err(InferError::Core)?;
        let cont = if cpu.is_contiguous() { cpu } else { cpu.contiguous().map_err(InferError::Core)? };
        let data = cont.to_bytes().map_err(InferError::Core)?;
        views.push((
            name.clone(),
            OwnedView { dtype: to_st_dtype(t.dtype())?, shape: t.dims().to_vec(), data },
        ));
    }
    safetensors::serialize(views, None).map_err(|e| InferError::Other(format!("safetensors serialize: {e}")))
}

/// Записать тензоры в `.safetensors` файл.
pub fn export_safetensors(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<()> {
    let bytes = serialize_tensors(tensors)?;
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| InferError::Other(format!("write safetensors: {e}")))
}
