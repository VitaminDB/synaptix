//! Источник модели H3 — HF-каталог или однофайловый `.syn`-бандл.
//!
//! Бандл зеркалит раскладку HF-каталога варианта (`FL2VA/`):
//! - tensors-чанки `tensors:transformer` / `tensors:video_vae` /
//!   `tensors:audio_vae` (+ опц. `tensors:text_encoder`);
//! - вспомогательные файлы под теми же относительными путями, что и на диске
//!   (`transformer/config.json`, `video_vae/source/config.json`,
//!   `audio_vae/metadata.json`, `model_index.json`, `tokenizer/…`).
//!
//! Благодаря зеркальной раскладке чтение конфигов одинаково для обоих
//! источников — различается только backend (`fs::read` против `Bundle::read_file`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_bundle::{Bundle, ChunkType};
use synaptix_core::device::Device;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};

use crate::config::H3Variant;
use crate::H3Error;

/// Подмодель H3: имя tensors-чанка в бандле ≡ имя подкаталога на диске.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Component {
    Transformer,
    VideoVae,
    AudioVae,
    TextEncoder,
}

impl H3Component {
    pub fn name(self) -> &'static str {
        match self {
            H3Component::Transformer => "transformer",
            H3Component::VideoVae => "video_vae",
            H3Component::AudioVae => "audio_vae",
            H3Component::TextEncoder => "text_encoder",
        }
    }
}

/// `.syn`-бандл содержит `tensors:<name>`.
pub(crate) fn bundle_has_component(bundle: &Bundle, name: &str) -> bool {
    bundle
        .cdir()
        .find_alive(&format!("tensors:{name}"))
        .is_some_and(|e| matches!(e.kind_typed(), ChunkType::Tensors))
}

#[derive(Clone)]
pub enum H3Source {
    /// Каталог варианта (`…/MiniMax-H3/FL2VA`).
    Dir { root: PathBuf, variant: H3Variant },
    /// Однофайловый бандл.
    Bundle { path: PathBuf, bundle: Arc<Bundle>, variant: H3Variant },
}

impl std::fmt::Debug for H3Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H3Source::Dir { root, variant } => {
                f.debug_struct("H3Source::Dir").field("root", root).field("variant", variant).finish()
            }
            H3Source::Bundle { path, variant, .. } => {
                f.debug_struct("H3Source::Bundle").field("path", path).field("variant", variant).finish()
            }
        }
    }
}

impl H3Source {
    /// Открыть модель по пути: `.syn`-файл → бандл, каталог → HF-дерево.
    ///
    /// Для каталога `variant` выбирает подкаталог (`FL2VA`/`Ref2VA`), если путь
    /// указывает на корень MiniMax-H3. Для бандла вариант читается из
    /// `model_index.json` (`_minimax_h3.partition`), а `variant` — только
    /// fallback.
    pub fn open(path: impl AsRef<Path>, variant: H3Variant) -> Result<Self, H3Error> {
        let path = path.as_ref();
        if path.is_file() {
            return Self::open_bundle(path, variant);
        }
        let paths = crate::loader::H3Paths::open_variant(path, variant)?;
        Ok(H3Source::Dir { root: paths.root, variant: paths.variant })
    }

    pub fn open_bundle(path: impl AsRef<Path>, fallback: H3Variant) -> Result<Self, H3Error> {
        let path = path.as_ref().to_path_buf();
        let bundle = Bundle::open(&path)
            .map_err(|e| H3Error::Load(format!("{}: {e}", path.display())))?;
        let bundle = Arc::new(bundle);
        let variant = bundle_variant(&bundle).unwrap_or(fallback);
        Ok(H3Source::Bundle { path, bundle, variant })
    }

    /// Каталог без привязки к варианту — для точечного чтения конфигов.
    pub fn plain_dir(root: impl Into<PathBuf>) -> Self {
        H3Source::Dir { root: root.into(), variant: H3Variant::Fl2va }
    }

    pub fn variant(&self) -> H3Variant {
        match self {
            H3Source::Dir { variant, .. } | H3Source::Bundle { variant, .. } => *variant,
        }
    }

    pub fn is_bundle(&self) -> bool {
        matches!(self, H3Source::Bundle { .. })
    }

    /// Путь источника — каталог варианта или `.syn`-файл (для сообщений/кэша).
    pub fn path(&self) -> &Path {
        match self {
            H3Source::Dir { root, .. } => root,
            H3Source::Bundle { path, .. } => path,
        }
    }

    pub fn bundle(&self) -> Option<&Arc<Bundle>> {
        match self {
            H3Source::Bundle { bundle, .. } => Some(bundle),
            H3Source::Dir { .. } => None,
        }
    }

    /// Прочитать вспомогательный файл по пути относительно корня варианта.
    pub fn read(&self, rel: &str) -> Result<Vec<u8>, H3Error> {
        match self {
            H3Source::Dir { root, .. } => {
                let p = root.join(rel);
                std::fs::read(&p).map_err(|e| H3Error::Config(format!("{}: {e}", p.display())))
            }
            H3Source::Bundle { path, bundle, .. } => bundle
                .read_file(rel)
                .map(|c| c.into_owned())
                .map_err(|e| H3Error::Config(format!("{}:{rel}: {e}", path.display()))),
        }
    }

    /// То же, но `None` вместо ошибки, если файла нет.
    pub fn read_opt(&self, rel: &str) -> Option<Vec<u8>> {
        self.read(rel).ok()
    }

    /// Есть ли подмодель в источнике (в каталоге — по наличию весов).
    pub fn has_component(&self, c: H3Component) -> bool {
        match self {
            H3Source::Dir { root, .. } => match c {
                H3Component::Transformer | H3Component::TextEncoder => {
                    scan_shards(root.join(c.name())).map(|s| !s.is_empty()).unwrap_or(false)
                }
                H3Component::VideoVae => root.join("video_vae/source/model.safetensors").is_file(),
                H3Component::AudioVae => root.join("audio_vae/model.safetensors").is_file(),
            },
            H3Source::Bundle { bundle, .. } => bundle_has_component(bundle, c.name()),
        }
    }

    /// Loader весов подмодели: mmap-шарды каталога либо срез бандла (zero-copy).
    pub fn loader(&self, c: H3Component, device: Device) -> Result<SafetensorsLoader, H3Error> {
        match self {
            H3Source::Dir { root, .. } => {
                let loader = match c {
                    H3Component::Transformer | H3Component::TextEncoder => {
                        let dir = root.join(c.name());
                        let shards = scan_shards(&dir)
                            .map_err(|e| H3Error::Load(format!("{}: {e}", dir.display())))?;
                        if shards.is_empty() {
                            return Err(H3Error::Load(format!("нет safetensors в {}", dir.display())));
                        }
                        SafetensorsLoader::open_sharded(&shards)
                    }
                    H3Component::VideoVae => {
                        SafetensorsLoader::open(root.join("video_vae/source/model.safetensors"))
                    }
                    H3Component::AudioVae => {
                        SafetensorsLoader::open(root.join("audio_vae/model.safetensors"))
                    }
                };
                Ok(loader.map_err(|e| H3Error::Load(e.to_string()))?.with_device(device))
            }
            H3Source::Bundle { path, bundle, .. } => {
                if !bundle_has_component(bundle, c.name()) {
                    return Err(H3Error::Load(format!(
                        "в бандле {} нет компонента `{}`",
                        path.display(),
                        c.name()
                    )));
                }
                let loader = SafetensorsLoader::from_bundle(bundle.clone(), Some(c.name()))
                    .map_err(|e| H3Error::Load(format!("{}: {e}", path.display())))?;
                Ok(loader.with_device(device))
            }
        }
    }
}

/// Вариант партиции из `model_index.json` бандла.
fn bundle_variant(bundle: &Bundle) -> Option<H3Variant> {
    let bytes = bundle.read_file("model_index.json").ok()?;
    let root: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let p = root.get("_minimax_h3")?.get("partition")?.as_str()?;
    H3Variant::parse(p)
}

/// Источник текст-энкодера (Qwen3-VL): каталог, отдельный `.syn` или компонент
/// `text_encoder` внутри бандла модели.
#[derive(Clone)]
pub enum H3EncoderSource {
    Dir(PathBuf),
    Bundle {
        path: PathBuf,
        bundle: Arc<Bundle>,
        /// Имя tensors-чанка: `text_encoder` внутри модельного бандла либо
        /// `None` — канонический чанк отдельного бандла энкодера.
        component: Option<String>,
        /// Префикс вспомогательных файлов (`text_encoder/` или пусто).
        file_prefix: String,
    },
}

impl H3EncoderSource {
    /// Открыть по явному пути: каталог `text_encoder/` или `.syn`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, H3Error> {
        let path = path.as_ref();
        if path.is_dir() {
            return Ok(H3EncoderSource::Dir(path.to_path_buf()));
        }
        let bundle = Bundle::open(path)
            .map_err(|e| H3Error::Load(format!("{}: {e}", path.display())))?;
        Ok(Self::from_bundle(path.to_path_buf(), Arc::new(bundle)))
    }

    /// Энкодер из уже открытого бандла: выделенный компонент `text_encoder`
    /// (модельный бандл) либо канонический чанк (отдельный бандл энкодера).
    pub fn from_bundle(path: PathBuf, bundle: Arc<Bundle>) -> Self {
        if bundle_has_component(&bundle, H3Component::TextEncoder.name()) {
            H3EncoderSource::Bundle {
                path,
                bundle,
                component: Some(H3Component::TextEncoder.name().to_string()),
                file_prefix: "text_encoder/".to_string(),
            }
        } else {
            H3EncoderSource::Bundle { path, bundle, component: None, file_prefix: String::new() }
        }
    }

    /// Энкодер, идущий вместе с моделью: подкаталог `text_encoder/` или
    /// одноимённый компонент бандла. `None` — в источнике энкодера нет.
    pub fn from_model(source: &H3Source) -> Option<Self> {
        match source {
            H3Source::Dir { root, .. } => {
                let dir = root.join("text_encoder");
                dir.is_dir().then(|| H3EncoderSource::Dir(dir))
            }
            H3Source::Bundle { path, bundle, .. } => bundle_has_component(bundle, "text_encoder")
                .then(|| Self::from_bundle(path.clone(), bundle.clone())),
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            H3EncoderSource::Dir(p) => p,
            H3EncoderSource::Bundle { path, .. } => path,
        }
    }

    pub fn is_bundle(&self) -> bool {
        matches!(self, H3EncoderSource::Bundle { .. })
    }

    /// Прочитать файл энкодера (`config.json`, `tokenizer.json`).
    pub fn read(&self, name: &str) -> Result<Vec<u8>, H3Error> {
        match self {
            H3EncoderSource::Dir(dir) => {
                let p = dir.join(name);
                std::fs::read(&p).map_err(|e| H3Error::Config(format!("{}: {e}", p.display())))
            }
            H3EncoderSource::Bundle { path, bundle, file_prefix, .. } => {
                let rel = format!("{file_prefix}{name}");
                bundle
                    .read_file(&rel)
                    .map(|c| c.into_owned())
                    .map_err(|e| H3Error::Config(format!("{}:{rel}: {e}", path.display())))
            }
        }
    }

    pub fn loader(&self, device: Device) -> Result<SafetensorsLoader, H3Error> {
        match self {
            H3EncoderSource::Dir(dir) => {
                let shards = scan_shards(dir)
                    .map_err(|e| H3Error::Load(format!("{}: {e}", dir.display())))?;
                if shards.is_empty() {
                    return Err(H3Error::Load(format!("нет safetensors в {}", dir.display())));
                }
                Ok(SafetensorsLoader::open_sharded(&shards)
                    .map_err(|e| H3Error::Load(e.to_string()))?
                    .with_device(device))
            }
            H3EncoderSource::Bundle { path, bundle, component, .. } => {
                let loader = SafetensorsLoader::from_bundle(bundle.clone(), component.as_deref())
                    .map_err(|e| H3Error::Load(format!("{}: {e}", path.display())))?;
                Ok(loader.with_device(device))
            }
        }
    }
}
