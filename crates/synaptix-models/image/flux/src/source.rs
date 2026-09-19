//! Источник модели FLUX — HF-каталог в раскладке diffusers или однофайловый
//! `.syn`-бандл.
//!
//! Бандл зеркалит каталог: tensors-чанки `tensors:transformer`,
//! `tensors:text_encoder` (CLIP-L), `tensors:text_encoder_2` (T5-XXL),
//! `tensors:vae`, а конфиги и токенайзеры лежат вспомогательными файлами под
//! теми же относительными путями (`transformer/config.json`,
//! `tokenizer/vocab.json`, `tokenizer_2/tokenizer.json`, …). Поэтому чтение
//! одинаково для обоих источников — различается только backend.
//!
//! Корневые `flux1-dev.safetensors`/`ae.safetensors` (исходный формат BFL)
//! не читаются: это те же веса одним файлом и с другими именами тензоров.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_bundle::{Bundle, ChunkType};
use synaptix_core::device::Device;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};

use crate::FluxError;

/// Подмодель FLUX: имя tensors-чанка в бандле ≡ имя подкаталога на диске.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FluxComponent {
    Transformer,
    /// CLIP-L (pooled).
    ClipText,
    /// T5-XXL encoder.
    T5Text,
    Vae,
}

impl FluxComponent {
    pub const ALL: [FluxComponent; 4] = [
        FluxComponent::Transformer,
        FluxComponent::ClipText,
        FluxComponent::T5Text,
        FluxComponent::Vae,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FluxComponent::Transformer => "transformer",
            FluxComponent::ClipText => "text_encoder",
            FluxComponent::T5Text => "text_encoder_2",
            FluxComponent::Vae => "vae",
        }
    }
}

fn bundle_has_component(bundle: &Bundle, name: &str) -> bool {
    bundle
        .cdir()
        .find_alive(&format!("tensors:{name}"))
        .is_some_and(|e| matches!(e.kind_typed(), ChunkType::Tensors))
}

#[derive(Clone)]
pub enum FluxSource {
    /// Каталог diffusers (`…/FLUX.1-dev`).
    Dir { root: PathBuf },
    /// Однофайловый бандл.
    Bundle { path: PathBuf, bundle: Arc<Bundle> },
}

impl std::fmt::Debug for FluxSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FluxSource::Dir { root } => f.debug_struct("FluxSource::Dir").field("root", root).finish(),
            FluxSource::Bundle { path, .. } => {
                f.debug_struct("FluxSource::Bundle").field("path", path).finish()
            }
        }
    }
}

impl FluxSource {
    /// `.syn`-файл → бандл, каталог → HF-дерево. Открытие бандла — только
    /// mmap и разбор central directory, веса не читаются.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FluxError> {
        let path = path.as_ref();
        if path.is_file() {
            let bundle = Bundle::open(path)
                .map_err(|e| FluxError::Load(format!("{}: {e}", path.display())))?;
            return Ok(FluxSource::Bundle { path: path.to_path_buf(), bundle: Arc::new(bundle) });
        }
        if !path.is_dir() {
            return Err(FluxError::Load(format!("нет такого файла или каталога: {}", path.display())));
        }
        Ok(FluxSource::Dir { root: path.to_path_buf() })
    }

    pub fn is_bundle(&self) -> bool {
        matches!(self, FluxSource::Bundle { .. })
    }

    /// Путь источника — каталог или `.syn`-файл (для сообщений и ключей кэша).
    pub fn path(&self) -> &Path {
        match self {
            FluxSource::Dir { root } => root,
            FluxSource::Bundle { path, .. } => path,
        }
    }

    /// Вспомогательный файл по пути относительно корня модели.
    pub fn read(&self, rel: &str) -> Result<Vec<u8>, FluxError> {
        match self {
            FluxSource::Dir { root } => {
                let p = root.join(rel);
                std::fs::read(&p).map_err(|e| FluxError::Config(format!("{}: {e}", p.display())))
            }
            FluxSource::Bundle { path, bundle } => bundle
                .read_file(rel)
                .map(|c| c.into_owned())
                .map_err(|e| FluxError::Config(format!("{}:{rel}: {e}", path.display()))),
        }
    }

    pub fn read_opt(&self, rel: &str) -> Option<Vec<u8>> {
        self.read(rel).ok()
    }

    pub fn has_component(&self, c: FluxComponent) -> bool {
        match self {
            FluxSource::Dir { root } => {
                scan_shards(root.join(c.name())).map(|s| !s.is_empty()).unwrap_or(false)
            }
            FluxSource::Bundle { bundle, .. } => bundle_has_component(bundle, c.name()),
        }
    }

    /// Loader весов подмодели: mmap-шарды каталога либо срез бандла без копии.
    pub fn loader(&self, c: FluxComponent, device: Device) -> Result<SafetensorsLoader, FluxError> {
        let loader = match self {
            FluxSource::Dir { root } => {
                let dir = root.join(c.name());
                let shards = scan_shards(&dir)
                    .map_err(|e| FluxError::Load(format!("{}: {e}", dir.display())))?;
                if shards.is_empty() {
                    return Err(FluxError::Load(format!("нет safetensors в {}", dir.display())));
                }
                SafetensorsLoader::open_sharded(&shards)
                    .map_err(|e| FluxError::Load(format!("{}: {e}", dir.display())))?
            }
            FluxSource::Bundle { path, bundle } => {
                // Без этой проверки `from_bundle` молча отдал бы основной
                // чанк бандла вместо отсутствующего компонента.
                if !bundle_has_component(bundle, c.name()) {
                    return Err(FluxError::Load(format!(
                        "в бандле {} нет компонента `{}`",
                        path.display(),
                        c.name()
                    )));
                }
                SafetensorsLoader::from_bundle(bundle.clone(), Some(c.name()))
                    .map_err(|e| FluxError::Load(format!("{}: {e}", path.display())))?
            }
        };
        Ok(loader.with_device(device))
    }

    /// Объём весов подмодели в байтах хранения — для решения «влезет ли
    /// трансформер в VRAM целиком».
    pub fn component_bytes(&self, c: FluxComponent) -> Result<u64, FluxError> {
        let loader = self.loader(c, Device::Cpu)?;
        Ok(loader.shard_bytes().iter().map(|b| b.len() as u64).sum())
    }
}
