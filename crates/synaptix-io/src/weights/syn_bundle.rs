use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::error::{IoError, Result};
use super::WeightLoader;

fn st_dtype_to_synaptix(dtype: safetensors::Dtype) -> Option<DType> {
    match dtype {
        safetensors::Dtype::F32  => Some(DType::F32),
        safetensors::Dtype::F16  => Some(DType::F16),
        safetensors::Dtype::BF16 => Some(DType::BF16),
        safetensors::Dtype::I32  => Some(DType::I32),
        safetensors::Dtype::I64  => Some(DType::I64),
        safetensors::Dtype::U8   => Some(DType::U8),
        safetensors::Dtype::U32  => Some(DType::U32),
        _                        => None,
    }
}

struct TensorMeta {
    dtype: Option<DType>,
    shape: Vec<usize>,
    off: usize,
    len: usize,
}

struct TensorIndex {
    by_name: HashMap<String, TensorMeta>,
    names: Vec<String>,
    prefix: Option<String>,
}

pub struct SynBundleLoader {
    bundle: Arc<Bundle>,
    component: Option<String>,
    default_device: Device,
    index: OnceLock<TensorIndex>,
}

impl SynBundleLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bundle = Bundle::open(path).map_err(|e| IoError::Bundle(e.to_string()))?;
        Ok(Self {
            bundle: Arc::new(bundle),
            component: None,
            default_device: Device::Cpu,
            index: OnceLock::new(),
        })
    }

    pub fn with_component(mut self, name: impl Into<String>) -> Self {
        self.component = Some(name.into());
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.default_device = device;
        self
    }

    fn st_bytes(&self) -> Result<(&[u8], Option<String>)> {
        match &self.component {
            Some(comp) => {
                let (bytes, prefix) = self.bundle.tensors_slice_for(comp)
                    .map_err(|e| IoError::Bundle(e.to_string()))?;
                Ok((bytes, prefix))
            }
            None => {
                let bytes = self.bundle.tensors_slice()
                    .map_err(|e| IoError::Bundle(e.to_string()))?;
                Ok((bytes, None))
            }
        }
    }

    fn build_index(&self) -> Result<TensorIndex> {
        let (bytes, prefix) = self.st_bytes()?;
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| IoError::Safetensors(e.to_string()))?;
        let base = bytes.as_ptr() as usize;
        let mut by_name = HashMap::with_capacity(st.len());
        let mut names = Vec::with_capacity(st.len());
        for name in st.names() {
            let tv = st.tensor(name)
                .map_err(|e| IoError::Safetensors(e.to_string()))?;
            let data = tv.data();
            by_name.insert(name.to_string(), TensorMeta {
                dtype: st_dtype_to_synaptix(tv.dtype()),
                shape: tv.shape().to_vec(),
                off: data.as_ptr() as usize - base,
                len: data.len(),
            });
            names.push(name.to_string());
        }
        Ok(TensorIndex { by_name, names, prefix })
    }

    fn index(&self) -> Result<&TensorIndex> {
        if let Some(i) = self.index.get() {
            return Ok(i);
        }
        let built = self.build_index()?;
        Ok(self.index.get_or_init(|| built))
    }

    fn resolve_key(name: &str, prefix: &Option<String>) -> String {
        match prefix {
            Some(pfx) if !name.starts_with(pfx.as_str()) => format!("{pfx}.{name}"),
            _ => name.to_string(),
        }
    }

    fn load_internal(&self, name: &str, device: Device, want_dtype: Option<DType>) -> Result<Tensor> {
        let (bytes, _) = self.st_bytes()?;
        let idx = self.index()?;

        let key = Self::resolve_key(name, &idx.prefix);
        let meta = idx.by_name.get(&key)
            .or_else(|| idx.by_name.get(name))
            .ok_or_else(|| IoError::Safetensors(format!("tensor not found: {name}")))?;

        let src_dtype = meta.dtype
            .ok_or_else(|| IoError::Safetensors(format!("unsupported dtype for {name}")))?;
        let slice = &bytes[meta.off..meta.off + meta.len];
        let tensor = Tensor::from_raw_slice(slice, meta.shape.clone(), src_dtype, device)
            .map_err(IoError::Core)?;

        match want_dtype {
            Some(d) if d != src_dtype => tensor.to_dtype(d).map_err(IoError::Core),
            _ => Ok(tensor),
        }
    }
}

impl WeightLoader for SynBundleLoader {
    fn load(&self, name: &str) -> Result<Tensor> {
        self.load_internal(name, self.default_device, None)
    }

    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        self.load_internal(name, device, Some(dtype))
    }

    fn names(&self) -> Vec<&str> {
        let Ok(idx) = self.index() else { return Vec::new(); };
        idx.names.iter().map(|s| s.as_str()).collect()
    }
}
