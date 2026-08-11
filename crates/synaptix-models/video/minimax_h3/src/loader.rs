use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader, TensorInfo};
use synaptix_io::weights::WeightLoader;

use crate::config::{AudioVaeConfig, H3Config, H3Variant, VaeConfig};
use crate::H3Error;

#[derive(Debug, Clone)]
pub struct H3Paths {
    pub root: PathBuf,
    pub variant: H3Variant,
}

impl H3Paths {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, H3Error> {
        let root = root.as_ref();
        if root.join("transformer").is_dir() {
            let variant = H3Config::from_dir(root)?.variant();
            return Ok(Self { root: root.to_path_buf(), variant });
        }
        for v in [H3Variant::Fl2va, H3Variant::Ref2va] {
            let sub = root.join(v.dir_name());
            if sub.join("transformer").is_dir() {
                return Ok(Self { root: sub, variant: v });
            }
        }
        Err(H3Error::Load(format!(
            "не найден каталог варианта (FL2VA/Ref2VA) в {}",
            root.display()
        )))
    }

    pub fn open_variant(root: impl AsRef<Path>, variant: H3Variant) -> Result<Self, H3Error> {
        let root = root.as_ref();
        let sub = if root.join("transformer").is_dir() {
            root.to_path_buf()
        } else {
            root.join(variant.dir_name())
        };
        if !sub.join("transformer").is_dir() {
            return Err(H3Error::Load(format!("нет transformer/ в {}", sub.display())));
        }
        Ok(Self { root: sub, variant })
    }

    pub fn transformer_dir(&self) -> PathBuf {
        self.root.join("transformer")
    }

    pub fn video_vae_file(&self) -> PathBuf {
        self.root.join("video_vae").join("source").join("model.safetensors")
    }

    pub fn audio_vae_file(&self) -> PathBuf {
        self.root.join("audio_vae").join("model.safetensors")
    }

    pub fn text_encoder_dir(&self) -> PathBuf {
        self.root.join("text_encoder")
    }

    pub fn tokenizer_dir(&self) -> PathBuf {
        self.root.join("tokenizer")
    }

    pub fn processor_dir(&self) -> PathBuf {
        self.root.join("processor")
    }
}

pub struct LoraWeights {
    loader: SafetensorsLoader,
    strength: f32,
    device: Device,
    prefixes: Vec<String>,
}

impl LoraWeights {
    pub fn open(path: impl AsRef<Path>, device: Device, strength: f32) -> Result<Self, H3Error> {
        let loader = SafetensorsLoader::open(path.as_ref())
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        let mut prefixes = vec![String::new()];
        let names = loader.names();
        for p in ["transformer.", "diffusion_model.", "model.diffusion_model."] {
            if names.iter().any(|n| n.starts_with(p)) {
                prefixes.push(p.to_string());
            }
        }
        Ok(Self { loader, strength, device, prefixes })
    }

    pub fn strength(&self) -> f32 {
        self.strength
    }

    fn find_pair(&self, key: &str) -> Option<(String, String)> {
        let names = self.loader.names();
        for p in &self.prefixes {
            for (sa, sb) in [
                ("lora_A.weight", "lora_B.weight"),
                ("lora_down.weight", "lora_up.weight"),
                ("lora_A.default.weight", "lora_B.default.weight"),
            ] {
                let ka = format!("{p}{key}.{sa}");
                if names.iter().any(|n| *n == ka.as_str()) {
                    return Some((ka, format!("{p}{key}.{sb}")));
                }
            }
        }
        None
    }

    pub fn delta(&self, model_key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        let Some((ka, kb)) = self.find_pair(model_key) else {
            return Ok(None);
        };
        let a = self
            .loader
            .load_to(&ka, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("lora {ka}: {e}")))?;
        let b = self
            .loader
            .load_to(&kb, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("lora {kb}: {e}")))?;
        let bs = b.mul_scalar(self.strength)?;
        Ok(Some(bs.matmul(&a)?))
    }

    pub fn diff(&self, model_key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        let names = self.loader.names();
        for p in &self.prefixes {
            let k = format!("{p}{model_key}.diff");
            if names.iter().any(|n| *n == k.as_str()) {
                let t = self
                    .loader
                    .load_to(&k, self.device, dtype)
                    .map_err(|e| H3Error::Load(format!("lora {k}: {e}")))?;
                return Ok(Some(t.mul_scalar(self.strength)?));
            }
        }
        Ok(None)
    }
}

pub struct H3Checkpoint {
    loader: SafetensorsLoader,
    pub config: H3Config,
    pub paths: H3Paths,
    device: Device,
    dtype: DType,
    lora: Option<Arc<LoraWeights>>,
}

impl H3Checkpoint {
    pub fn open(paths: H3Paths, device: Device, dtype: DType) -> Result<Self, H3Error> {
        let config = H3Config::from_dir(&paths.root)?;
        let dir = paths.transformer_dir();
        let shards = scan_shards(&dir).map_err(|e| H3Error::Load(format!("{}: {e}", dir.display())))?;
        if shards.is_empty() {
            return Err(H3Error::Load(format!("нет safetensors в {}", dir.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, config, paths, device, dtype, lora: None })
    }

    pub fn open_root(root: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, H3Error> {
        Self::open(H3Paths::open(root)?, device, dtype)
    }

    pub fn with_lora(mut self, lora: Arc<LoraWeights>) -> Self {
        self.lora = Some(lora);
        self
    }

    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    pub fn lora_delta(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        match &self.lora {
            Some(l) => l.delta(key, dtype),
            None => Ok(None),
        }
    }

    pub fn lora_diff(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        match &self.lora {
            Some(l) => l.diff(key, dtype),
            None => Ok(None),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    pub fn get(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn get_raw(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load(name)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn get_as(&self, name: &str, dtype: DType) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.loader.names()
    }

    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.loader.tensor_info(name)
    }

    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.loader.infos()
    }

    pub fn view_on(&self, device: Device) -> Self {
        Self {
            loader: self.loader.clone_with_device(device),
            config: self.config.clone(),
            paths: self.paths.clone(),
            device,
            dtype: self.dtype,
            lora: self.lora.clone(),
        }
    }

    pub fn shard_bytes(&self) -> Vec<&[u8]> {
        self.loader.shard_bytes()
    }

    pub fn raw_bytes(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.loader.raw_bytes(name)
    }

    pub fn vae_config(&self) -> Result<VaeConfig, H3Error> {
        VaeConfig::from_dir(&self.paths.root)
    }

    pub fn audio_vae_config(&self) -> Result<AudioVaeConfig, H3Error> {
        AudioVaeConfig::from_dir(&self.paths.root)
    }
}

pub struct ComponentLoader {
    loader: SafetensorsLoader,
    device: Device,
}

impl ComponentLoader {
    pub fn open_file(path: impl AsRef<Path>, device: Device) -> Result<Self, H3Error> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(H3Error::Load(format!("не найден: {}", path.display())));
        }
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device })
    }

    pub fn open_dir(dir: impl AsRef<Path>, device: Device) -> Result<Self, H3Error> {
        let dir = dir.as_ref();
        let shards = scan_shards(dir).map_err(|e| H3Error::Load(format!("{}: {e}", dir.display())))?;
        if shards.is_empty() {
            return Err(H3Error::Load(format!("нет safetensors в {}", dir.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get(&self, name: &str) -> Result<Tensor, H3Error> {
        self.loader
            .load(name)
            .map_err(|e| H3Error::Load(format!("load '{name}': {e}")))
    }

    pub fn get_as(&self, name: &str, dtype: DType) -> Result<Tensor, H3Error> {
        self.loader
            .load_to(name, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("load '{name}': {e}")))
    }

    pub fn opt(&self, name: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        if self.contains(name) {
            Ok(Some(self.get_as(name, dtype)?))
        } else {
            Ok(None)
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.loader.names()
    }

    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.loader.infos()
    }

    pub fn total_bytes(&self) -> usize {
        self.loader
            .infos()
            .map(|(_, dt, shape)| dt.bytes_for_numel(shape.iter().product::<usize>()))
            .sum()
    }
}
