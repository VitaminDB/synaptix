use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
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

pub struct ShardInfo {
    pub path: PathBuf,
    pub names: Vec<String>,
}

pub struct StreamingLoader {
    shards: Vec<ShardInfo>,
    name_to_shard: HashMap<String, usize>,
    default_device: Device,
}

impl StreamingLoader {
    pub fn from_paths(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut shards = Vec::new();
        let mut name_to_shard = HashMap::new();

        for (i, p) in paths.iter().enumerate() {
            let path = p.as_ref().to_path_buf();
            let file = std::fs::File::open(&path).map_err(IoError::Io)?;
            let mmap = unsafe { Mmap::map(&file).map_err(IoError::Io)? };
            let data: &[u8] = &mmap;
            let st = SafeTensors::deserialize(data)
                .map_err(|e| IoError::Safetensors(e.to_string()))?;
            let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
            for n in &names {
                name_to_shard.insert(n.clone(), i);
            }
            shards.push(ShardInfo { path, names });
        }

        Ok(Self { shards, name_to_shard, default_device: Device::Cpu })
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(IoError::Io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        paths.sort();
        Self::from_paths(&paths)
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.default_device = device;
        self
    }

    pub fn shard_for(&self, name: &str) -> Option<&ShardInfo> {
        self.name_to_shard.get(name).map(|&i| &self.shards[i])
    }

    pub fn iter_layers<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&str, Tensor) -> Result<()>,
    {
        for shard in &self.shards {
            let file = std::fs::File::open(&shard.path).map_err(IoError::Io)?;
            let mmap = unsafe { Mmap::map(&file).map_err(IoError::Io)? };
            let data: &[u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
            let st = SafeTensors::deserialize(data)
                .map_err(|e| IoError::Safetensors(e.to_string()))?;
            for name in &shard.names {
                let tv = st.tensor(name)
                    .map_err(|e| IoError::Safetensors(e.to_string()))?;
                let src_dtype = st_dtype_to_synaptix(tv.dtype())
                    .ok_or_else(|| IoError::Safetensors(
                        format!("unsupported dtype {:?} for {name}", tv.dtype())
                    ))?;
                let shape = tv.shape().to_vec();
                let bytes = tv.data().to_vec();
                let tensor = Tensor::from_raw_bytes(bytes, shape, src_dtype, self.default_device)
                    .map_err(IoError::Core)?;
                f(name, tensor)?;
            }
        }
        Ok(())
    }

    fn load_internal(&self, name: &str, device: Device, want_dtype: Option<DType>) -> Result<Tensor> {
        let shard = self.shard_for(name)
            .ok_or_else(|| IoError::Safetensors(format!("tensor not found: {name}")))?;
        let file = std::fs::File::open(&shard.path).map_err(IoError::Io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(IoError::Io)? };
        let data: &[u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        let st = SafeTensors::deserialize(data)
            .map_err(|e| IoError::Safetensors(e.to_string()))?;
        let tv = st.tensor(name)
            .map_err(|e| IoError::Safetensors(e.to_string()))?;
        let src_dtype = st_dtype_to_synaptix(tv.dtype())
            .ok_or_else(|| IoError::Safetensors(
                format!("unsupported dtype {:?} for {name}", tv.dtype())
            ))?;
        let shape = tv.shape().to_vec();
        let bytes = tv.data().to_vec();
        let tensor = Tensor::from_raw_bytes(bytes, shape, src_dtype, device)
            .map_err(IoError::Core)?;
        match want_dtype {
            Some(d) if d != src_dtype => tensor.to_dtype(d).map_err(IoError::Core),
            _ => Ok(tensor),
        }
    }
}

impl WeightLoader for StreamingLoader {
    fn load(&self, name: &str) -> Result<Tensor> {
        self.load_internal(name, self.default_device, None)
    }

    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        self.load_internal(name, device, Some(dtype))
    }

    fn names(&self) -> Vec<&str> {
        self.name_to_shard.keys().map(|s| s.as_str()).collect()
    }
}
