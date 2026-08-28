use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
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

/// Владелец памяти, на которую указывает `Shard::data`: либо собственный
/// mmap одного `.safetensors`-файла, либо mmap `.syn`-бандла, внутри которого
/// лежит safetensors-поток компонента.
// Варианты только держат владение памятью — читается всегда `Shard::data`.
#[allow(dead_code)]
enum ShardOwner {
    Mmap(Mmap),
    Bundle(Arc<Bundle>),
}

struct Shard {
    _owner: ShardOwner,
    data: &'static [u8],
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(IoError::Io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(IoError::Io)? };
        let data: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        Ok(Self { _owner: ShardOwner::Mmap(mmap), data })
    }

    /// Слайс safetensors-потока компонента внутри `.syn`. Bundle держится
    /// `Arc`-ом в самом шарде, поэтому слайс живёт ровно столько же.
    fn from_bundle(bundle: Arc<Bundle>, ptr: *const u8, len: usize) -> Self {
        // SAFETY: `ptr/len` описывают слайс внутри mmap бандла; `Arc<Bundle>`
        // переезжает в шард, так что mmap переживёт слайс.
        let data: &'static [u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
        Self { _owner: ShardOwner::Bundle(bundle), data }
    }
}

/// Метаданные тензора без загрузки данных (форма + dtype в кодировке synaptix).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// Разобранная запись индекса: dtype, форма и zero-copy слайс в mmap-шарде.
#[derive(Clone)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    data: &'static [u8],
}

pub struct SafetensorsLoader {
    // SAFETY: `entries`/`metadata` держат `&'static`-слайсы в эти mmap'ы, поэтому
    // шарды обязаны жить не меньше loader'а.
    _shards: Vec<Arc<Shard>>,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
    default_device: Device,
    prefix: Option<String>,
}

impl SafetensorsLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let shard = Arc::new(Shard::open(path.as_ref())?);
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        index_shard(&shard, &mut entries, &mut metadata)?;
        Ok(Self {
            _shards: vec![shard],
            entries,
            metadata,
            default_device: Device::Cpu,
            prefix: None,
        })
    }

    pub fn open_sharded(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut shards = Vec::new();
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for path in paths.iter() {
            let shard = Arc::new(Shard::open(path.as_ref())?);
            index_shard(&shard, &mut entries, &mut metadata)?;
            shards.push(shard);
        }
        Ok(Self { _shards: shards, entries, metadata, default_device: Device::Cpu, prefix: None })
    }

    /// Открыть safetensors-поток компонента из `.syn`-бандла.
    ///
    /// `component=None` — канонический (единственный) Tensors-чанк бандла;
    /// `Some(name)` — либо выделенный чанк `tensors:<name>`, либо (legacy-layout)
    /// общий чанк с автоматически подставленным tensor-префиксом компонента.
    ///
    /// Зеркалит [`Self::open`]: тензоры отдаются zero-copy прямо из mmap
    /// бандла, без распаковки во временный файл.
    pub fn open_bundle(path: impl AsRef<Path>, component: Option<&str>) -> Result<Self> {
        let bundle = Bundle::open(path.as_ref()).map_err(|e| IoError::Bundle(e.to_string()))?;
        Self::from_bundle(Arc::new(bundle), component)
    }

    /// То же, что [`Self::open_bundle`], но поверх уже открытого бандла —
    /// несколько компонентов одного `.syn` делят один mmap.
    pub fn from_bundle(bundle: Arc<Bundle>, component: Option<&str>) -> Result<Self> {
        let (bytes, prefix) = match component {
            Some(c) => bundle
                .tensors_slice_for(c)
                .map_err(|e| IoError::Bundle(format!("component `{c}`: {e}")))?,
            None => (
                bundle.tensors_slice().map_err(|e| IoError::Bundle(e.to_string()))?,
                None,
            ),
        };
        // Указатель снимаем до move `bundle` в шард — сам слайс заимствует mmap,
        // который живёт внутри Arc и переезжает вместе с ним.
        let (ptr, len) = (bytes.as_ptr(), bytes.len());
        let shard = Arc::new(Shard::from_bundle(bundle, ptr, len));
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        index_shard(&shard, &mut entries, &mut metadata)?;
        Ok(Self {
            _shards: vec![shard],
            entries,
            metadata,
            default_device: Device::Cpu,
            prefix,
        })
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.default_device = device;
        self
    }

    /// Лёгкий клон с переопределённым `default_device`: разделяет mmap-шарды
    /// (`Arc`), дублирует только индекс (без повторного mmap/разбора заголовка).
    /// Нужно для streaming-offload — грузить веса напрямую mmap→GPU без
    /// резидентной host-копии.
    pub fn clone_with_device(&self, device: Device) -> Self {
        Self {
            _shards: self._shards.clone(),
            entries: self.entries.clone(),
            metadata: self.metadata.clone(),
            default_device: device,
            prefix: self.prefix.clone(),
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// `__metadata__`-секция safetensors (например `config`/`model_version` у LTX).
    /// При шардировании ключи поздних шардов перекрывают ранние.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Сырые байты mmap-шардов (целиком, с заголовками). Для host-register
    /// (pinned) при offload-стриминге: шарды живут пока жив loader (Arc).
    pub fn shard_bytes(&self) -> Vec<&[u8]> {
        self._shards.iter().map(|s| s.data).collect()
    }

    /// Форма + dtype тензора без загрузки данных (учитывает `with_prefix`).
    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.entries.get(&self.resolve_name(name)).map(|e| TensorInfo {
            dtype: e.dtype,
            shape: e.shape.clone(),
        })
    }

    /// Итератор `(имя, dtype, форма)` по всем тензорам (имена — как в файле, без
    /// снятия префикса).
    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.entries.iter().map(|(k, e)| (k.as_str(), e.dtype, e.shape.as_slice()))
    }

    /// Сырой mmap-слайс тензора + dtype файла + форма (учитывает `with_prefix`),
    /// без создания Tensor/H2D. Слайс жив пока жив loader (Arc-шарды).
    pub fn raw_bytes(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.entries
            .get(&self.resolve_name(name))
            .map(|e| (e.data, e.dtype, e.shape.as_slice()))
    }

    /// Отключить readahead для диапазона тензора (`MADV_RANDOM`): у таблиц с
    /// произвольным доступом (n-gram-эмбеддинги PLE) чтение одной строки в
    /// 320 байт иначе тянет за собой всё окно упреждающего чтения. `false` —
    /// тензор не найден или шард лежит не в собственном mmap (бандл).
    pub fn advise_random(&self, name: &str) -> bool {
        let Some(entry) = self.entries.get(&self.resolve_name(name)) else {
            return false;
        };
        advise_random_range(entry.data)
    }

    fn resolve_name(&self, name: &str) -> String {
        match &self.prefix {
            Some(p) if !name.starts_with(p.as_str()) => format!("{p}.{name}"),
            _ => name.to_string(),
        }
    }

    fn load_internal(&self, name: &str, device: Device, dtype: Option<DType>) -> Result<Tensor> {
        let key = self.resolve_name(name);
        let entry = self.entries.get(&key)
            .ok_or_else(|| IoError::Safetensors(format!("tensor not found: {key}")))?;
        // zero-copy: H2D напрямую из mmap-слайса. Индекс с dtype/shape/slice разобран
        // один раз в open(), без повторного SafeTensors::deserialize на каждый тензор
        // (для 5947-тензорного LTX это убирает ~5947 разборов 872KB-заголовка).
        let tensor = Tensor::from_raw_slice(entry.data, entry.shape.clone(), entry.dtype, device)
            .map_err(IoError::Core)?;
        match dtype {
            Some(d) if d != entry.dtype => tensor.to_dtype(d).map_err(IoError::Core),
            _ => Ok(tensor),
        }
    }
}

/// `MADV_RANDOM` на диапазон mmap: без него чтение одной строки тянет за
/// собой окно упреждающего чтения. Работает и для тензоров внутри `.syn` —
/// адрес берётся у самого слайса, чей это mmap, знать не нужно.
#[cfg(target_os = "linux")]
pub fn advise_random_range(data: &[u8]) -> bool {
    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const MADV_RANDOM: i32 = 1;
    let page = 4096usize;
    let start = data.as_ptr() as usize;
    let aligned = start & !(page - 1);
    let len = data.len() + (start - aligned);
    unsafe { madvise(aligned as *mut std::ffi::c_void, len, MADV_RANDOM) == 0 }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_random_range(_data: &[u8]) -> bool {
    false
}

fn index_shard(
    shard: &Arc<Shard>,
    entries: &mut HashMap<String, Entry>,
    metadata: &mut HashMap<String, String>,
) -> Result<()> {
    let st = SafeTensors::deserialize(shard.data)
        .map_err(|e| IoError::Safetensors(e.to_string()))?;
    for (name, view) in st.tensors() {
        let dtype = st_dtype_to_synaptix(view.dtype())
            .ok_or_else(|| IoError::Safetensors(format!("unsupported dtype {:?}", view.dtype())))?;
        // SAFETY: view.data() заимствует shard.data (он `&'static`), поэтому слайс
        // переживает локальный `st`; шард держится в loader'е, пока жив слайс.
        entries.insert(name, Entry { dtype, shape: view.shape().to_vec(), data: view.data() });
    }
    if let Ok((_, meta)) = SafeTensors::read_metadata(shard.data) {
        if let Some(m) = meta.metadata() {
            for (k, v) in m.iter() {
                metadata.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

impl WeightLoader for SafetensorsLoader {
    fn load(&self, name: &str) -> Result<Tensor> {
        self.load_internal(name, self.default_device, None)
    }

    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        self.load_internal(name, device, Some(dtype))
    }

    fn names(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }
}

pub fn load_file(path: impl AsRef<Path>, device: Device) -> Result<HashMap<String, Tensor>> {
    let loader = SafetensorsLoader::open(path)?.with_device(device);
    let mut out = HashMap::new();
    for name in loader.names() {
        let name = name.to_string();
        let t = loader.load(&name)?;
        out.insert(name, t);
    }
    Ok(out)
}

pub fn scan_shards(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(IoError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    paths.sort();
    Ok(paths)
}
