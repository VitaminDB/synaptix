//! Источник SDXL — каталог diffusers или `.syn`-бандл.
//!
//! Бандл (`syn-pack` сам распознаёт раскладку) зеркалит каталог:
//! tensors-чанки `tensors:unet`, `tensors:text_encoder`,
//! `tensors:text_encoder_2`, `tensors:vae`, а конфиги и токенайзеры лежат
//! вспомогательными файлами под теми же относительными путями.
//!
//! В каталоге у компонента бывает несколько точностей одной модели
//! (`vae/diffusion_pytorch_model.safetensors` и `….fp16.safetensors`):
//! берётся основной набор, а если есть только вариант (`unet/` в
//! fp16-репозитории) — он. Так же решает упаковщик, и бандл совпадает с
//! каталогом.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_bundle::{Bundle, ChunkType};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::WeightLoader;

use crate::SdxlError;

pub const UNET: &str = "unet";
pub const TEXT_ENCODER: &str = "text_encoder";
pub const TEXT_ENCODER_2: &str = "text_encoder_2";
pub const VAE: &str = "vae";
pub const COMPONENTS: [&str; 4] = [UNET, TEXT_ENCODER, TEXT_ENCODER_2, VAE];

const PRECISION_VARIANTS: &[&str] = &[".fp16", ".fp32", ".bf16", ".f16", ".f32"];

fn is_variant(path: &Path) -> bool {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    PRECISION_VARIANTS.iter().any(|v| stem.ends_with(v) || stem.contains(&format!("{v}-")))
}

/// Шарды компонента в каталоге: основной набор, иначе вариант точности.
fn component_shards(dir: &Path) -> Result<Vec<PathBuf>, SdxlError> {
    let all = scan_shards(dir).map_err(|e| SdxlError::Load(format!("{}: {e}", dir.display())))?;
    let main: Vec<PathBuf> = all.iter().filter(|p| !is_variant(p)).cloned().collect();
    if !main.is_empty() {
        return Ok(main);
    }
    // Только варианты: берём fp16 (если их несколько разных — первый по имени).
    let fp16: Vec<PathBuf> =
        all.iter().filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains(".fp16"))).cloned().collect();
    Ok(if fp16.is_empty() { all } else { fp16 })
}

fn bundle_has_component(bundle: &Bundle, name: &str) -> bool {
    bundle
        .cdir()
        .find_alive(&format!("tensors:{name}"))
        .is_some_and(|e| matches!(e.kind_typed(), ChunkType::Tensors))
}

#[derive(Clone)]
pub enum SdxlSource {
    Dir { root: PathBuf },
    Bundle { path: PathBuf, bundle: Arc<Bundle> },
}

impl std::fmt::Debug for SdxlSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SdxlSource({})", self.path().display())
    }
}

impl SdxlSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SdxlError> {
        let path = path.as_ref();
        if path.is_file() {
            let bundle = Bundle::open(path).map_err(|e| SdxlError::Load(format!("{}: {e}", path.display())))?;
            return Ok(SdxlSource::Bundle { path: path.to_path_buf(), bundle: Arc::new(bundle) });
        }
        if !path.is_dir() {
            return Err(SdxlError::Load(format!("нет такого файла или каталога: {}", path.display())));
        }
        Ok(SdxlSource::Dir { root: path.to_path_buf() })
    }

    pub fn path(&self) -> &Path {
        match self {
            SdxlSource::Dir { root } => root,
            SdxlSource::Bundle { path, .. } => path,
        }
    }

    pub fn is_bundle(&self) -> bool {
        matches!(self, SdxlSource::Bundle { .. })
    }

    pub fn read(&self, rel: &str) -> Result<Vec<u8>, SdxlError> {
        match self {
            SdxlSource::Dir { root } => {
                let p = root.join(rel);
                std::fs::read(&p).map_err(|e| SdxlError::Config(format!("{}: {e}", p.display())))
            }
            SdxlSource::Bundle { path, bundle } => bundle
                .read_file(rel)
                .map(|c| c.into_owned())
                .map_err(|e| SdxlError::Config(format!("{}:{rel}: {e}", path.display()))),
        }
    }

    pub fn read_opt(&self, rel: &str) -> Option<Vec<u8>> {
        self.read(rel).ok()
    }

    pub fn has_component(&self, name: &str) -> bool {
        match self {
            SdxlSource::Dir { root } => component_shards(&root.join(name)).map(|s| !s.is_empty()).unwrap_or(false),
            SdxlSource::Bundle { bundle, .. } => bundle_has_component(bundle, name),
        }
    }

    /// Веса компонента: mmap-шарды каталога или срез бандла без копии.
    pub fn weights(&self, name: &str) -> Result<Weights, SdxlError> {
        let loader = match self {
            SdxlSource::Dir { root } => {
                let dir = root.join(name);
                let shards = component_shards(&dir)?;
                if shards.is_empty() {
                    return Err(SdxlError::Load(format!("нет safetensors в {}", dir.display())));
                }
                SafetensorsLoader::open_sharded(&shards).map_err(|e| SdxlError::Load(format!("{}: {e}", dir.display())))?
            }
            SdxlSource::Bundle { path, bundle } => {
                if !bundle_has_component(bundle, name) {
                    return Err(SdxlError::Load(format!("в бандле {} нет компонента `{name}`", path.display())));
                }
                SafetensorsLoader::from_bundle(bundle.clone(), Some(name))
                    .map_err(|e| SdxlError::Load(format!("{}: {e}", path.display())))?
            }
        };
        Ok(Weights { loader: Arc::new(loader) })
    }
}

/// Веса одного компонента (общий mmap, дёшево клонируется).
#[derive(Clone)]
pub struct Weights {
    loader: Arc<SafetensorsLoader>,
}

impl Weights {
    /// Тензор в `dtype` на `device`.
    pub fn get(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, device, dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn names(&self) -> Vec<String> {
        self.loader.infos().map(|(n, _, _)| n.to_string()).collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.tensor_info(name).is_some()
    }

    pub fn raw(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.loader.raw_bytes(name)
    }

    pub fn total_bytes(&self) -> u64 {
        self.loader.shard_bytes().iter().map(|b| b.len() as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants() {
        assert!(is_variant(Path::new("vae/diffusion_pytorch_model.fp16.safetensors")));
        assert!(!is_variant(Path::new("vae/diffusion_pytorch_model.safetensors")));
        assert!(is_variant(Path::new("text_encoder/model.fp16.safetensors")));
        assert!(!is_variant(Path::new("unet/diffusion_pytorch_model-00001-of-00002.safetensors")));
    }
}
