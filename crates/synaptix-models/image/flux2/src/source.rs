//! Источник FLUX.2 — каталог diffusers или `.syn`-бандл.
//!
//! Бандл зеркалит каталог: tensors-чанки `tensors:transformer`,
//! `tensors:text_encoder`, `tensors:vae`, а конфиги и токенайзер лежат
//! вспомогательными файлами под теми же относительными путями. Корневые
//! `flux2-dev.safetensors` / `ae.safetensors` (формат BFL) не читаются — это
//! те же веса одним файлом под другими именами.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_bundle::{Bundle, ChunkType};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::WeightLoader;

use crate::Flux2Error;

pub const TRANSFORMER: &str = "transformer";
pub const TEXT_ENCODER: &str = "text_encoder";
pub const VAE: &str = "vae";
pub const COMPONENTS: [&str; 3] = [TRANSFORMER, TEXT_ENCODER, VAE];

fn bundle_has_component(bundle: &Bundle, name: &str) -> bool {
    bundle
        .cdir()
        .find_alive(&format!("tensors:{name}"))
        .is_some_and(|e| matches!(e.kind_typed(), ChunkType::Tensors))
}

#[derive(Clone)]
pub enum Flux2Source {
    Dir { root: PathBuf },
    Bundle { path: PathBuf, bundle: Arc<Bundle> },
}

impl std::fmt::Debug for Flux2Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Flux2Source({})", self.path().display())
    }
}

impl Flux2Source {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Flux2Error> {
        let path = path.as_ref();
        if path.is_file() {
            let bundle =
                Bundle::open(path).map_err(|e| Flux2Error::Load(format!("{}: {e}", path.display())))?;
            return Ok(Flux2Source::Bundle { path: path.to_path_buf(), bundle: Arc::new(bundle) });
        }
        if !path.is_dir() {
            return Err(Flux2Error::Load(format!("нет такого файла или каталога: {}", path.display())));
        }
        Ok(Flux2Source::Dir { root: path.to_path_buf() })
    }

    pub fn path(&self) -> &Path {
        match self {
            Flux2Source::Dir { root } => root,
            Flux2Source::Bundle { path, .. } => path,
        }
    }

    pub fn is_bundle(&self) -> bool {
        matches!(self, Flux2Source::Bundle { .. })
    }

    pub fn read(&self, rel: &str) -> Result<Vec<u8>, Flux2Error> {
        match self {
            Flux2Source::Dir { root } => {
                let p = root.join(rel);
                std::fs::read(&p).map_err(|e| Flux2Error::Config(format!("{}: {e}", p.display())))
            }
            Flux2Source::Bundle { path, bundle } => bundle
                .read_file(rel)
                .map(|c| c.into_owned())
                .map_err(|e| Flux2Error::Config(format!("{}:{rel}: {e}", path.display()))),
        }
    }

    pub fn read_opt(&self, rel: &str) -> Option<Vec<u8>> {
        self.read(rel).ok()
    }

    pub fn has_component(&self, name: &str) -> bool {
        match self {
            Flux2Source::Dir { root } => scan_shards(root.join(name)).map(|s| !s.is_empty()).unwrap_or(false),
            Flux2Source::Bundle { bundle, .. } => bundle_has_component(bundle, name),
        }
    }

    /// Веса компонента: mmap-шарды каталога или срез бандла без копии.
    pub fn weights(&self, name: &str) -> Result<Weights, Flux2Error> {
        let loader = match self {
            Flux2Source::Dir { root } => {
                let dir = root.join(name);
                let shards =
                    scan_shards(&dir).map_err(|e| Flux2Error::Load(format!("{}: {e}", dir.display())))?;
                if shards.is_empty() {
                    return Err(Flux2Error::Load(format!("нет safetensors в {}", dir.display())));
                }
                SafetensorsLoader::open_sharded(&shards)
                    .map_err(|e| Flux2Error::Load(format!("{}: {e}", dir.display())))?
            }
            Flux2Source::Bundle { path, bundle } => {
                // Без проверки `from_bundle` молча отдал бы основной чанк.
                if !bundle_has_component(bundle, name) {
                    return Err(Flux2Error::Load(format!(
                        "в бандле {} нет компонента `{name}`",
                        path.display()
                    )));
                }
                SafetensorsLoader::from_bundle(bundle.clone(), Some(name))
                    .map_err(|e| Flux2Error::Load(format!("{}: {e}", path.display())))?
            }
        };
        Ok(Weights { loader: Arc::new(loader) })
    }
}

/// Веса одного компонента. Дешёво клонируется (общий mmap) — потоки
/// префетча читают из него параллельно счёту.
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

    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.loader.tensor_info(name).map(|i| i.shape)
    }

    /// Сырые байты тензора из mmap (для выборки строк таблицы эмбеддингов
    /// без загрузки всей таблицы).
    pub fn raw(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.loader.raw_bytes(name)
    }

    /// Суммарный объём тензоров, чьё имя начинается с `prefix`.
    pub fn bytes_with_prefix(&self, prefix: &str) -> u64 {
        self.loader
            .infos()
            .filter(|(n, _, _)| n.starts_with(prefix))
            .map(|(_, dt, shape)| dt.bytes_for_numel(shape.iter().product()) as u64)
            .sum()
    }

    pub fn total_bytes(&self) -> u64 {
        self.loader.shard_bytes().iter().map(|b| b.len() as u64).sum()
    }
}
