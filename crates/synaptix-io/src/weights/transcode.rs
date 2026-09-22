//! Перекодировка квантованных весов при загрузке и в файл (этап 4 плана
//! низкобитных квантов).
//!
//! Любой вес источника — блоки ggml из `.gguf`, NVFP4/MXFP8/SQ из `.syn`,
//! плотная матрица — деквантуется на карте в F16 и кодируется целевым
//! форматом (`Tensor::quantize_to`). Результат живёт **оверлеем** поверх
//! источника: [`SynBundleLoader`] при обращении к весу сперва смотрит в
//! оверлей и только потом в файл, поэтому модели, зеркала экспертов и
//! кэши ничего о перекодировке не знают.
//!
//! Где лежит оверлей, решает [`Placement`]: если целевой объём влезает в
//! свободную RAM за вычетом запаса — в памяти процесса (`Vec<u8>` на
//! тензор); иначе — в дисковом кэше: компактный `.syn` только с
//! перекодированными блобами и манифестом (`$XDG_CACHE_HOME/synaptix/
//! transcode/<ключ>.syn`), который открывается через mmap. Ключ кэша —
//! путь, размер и mtime источника плюс спецификация и версия энкодера.
//!
//! Заявка на перекодировку ставится на путь модели на время загрузки
//! ([`request`] → [`RequestGuard`]): все `SynBundleLoader::open` этого пути
//! (основной компонент, драфтер, башня зрения) получают один и тот же
//! оверлей. Отдельно [`write_bundle`] пишет полный самостоятельный бандл
//! (CLI `synaptix quantize`).

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use synaptix_bundle::inspect::{classify_in, LayerRole, QuantKind};
use synaptix_bundle::quant_layout::{format_key, MANIFEST_NAME};
use synaptix_bundle::{Bundle, BundleBuilder, FileTag, QuantEntry, QuantManifest, StDtype, StreamTensor, TensorStream, CAP_QUANT_WEIGHTS};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

use super::syn_bundle::SynBundleLoader;
use super::WeightLoader;
use crate::error::{IoError, Result};

/// Версия энкодера/раскладки в ключе кэша: меняется — старые кэши не годятся.
pub const ENCODER_VERSION: u32 = 1;

/// Запас RAM, который перекодировка в памяти обязана оставить системе.
const RAM_RESERVE_BYTES: u64 = 6 << 30;

/// Что во что перекодировать. `None` у роли — оставить как есть.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeSpec {
    pub attn: Option<DType>,
    pub mlp: Option<DType>,
    /// Стопки экспертов MoE (`[E, N, K]`).
    pub experts: Option<DType>,
    pub lm_head: Option<DType>,
    pub embed: Option<DType>,
    /// Перекодировать уже квантованные веса (двойной квант). Без этого
    /// перекодируются только плотные матрицы (см. `quantize_dense`).
    pub requant: bool,
    /// Квантовать плотные матрицы источника. При загрузке это не нужно —
    /// модель квантует их сама, — а писателю полного бандла нужно.
    pub quantize_dense: bool,
}

impl TranscodeSpec {
    /// Один формат на attn/mlp/экспертов/голову, эмбеддинг как есть.
    pub fn uniform(dt: DType) -> Self {
        Self {
            attn: Some(dt),
            mlp: Some(dt),
            experts: Some(dt),
            lm_head: Some(dt),
            embed: None,
            requant: true,
            quantize_dense: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.attn.is_none() && self.mlp.is_none() && self.experts.is_none() && self.lm_head.is_none() && self.embed.is_none()
    }

    /// Целевой формат для тензора по его роли (`None` — не трогать).
    pub fn target_for(&self, name: &str, slices: usize) -> Option<DType> {
        if slices > 1 || name.contains(".experts.") {
            return self.experts;
        }
        match classify_in(name, None) {
            LayerRole::Attention => self.attn,
            LayerRole::Mlp => self.mlp,
            LayerRole::LmHead => self.lm_head,
            LayerRole::Embedding => self.embed,
            _ => None,
        }
    }

    /// Строка для ключа кэша и логов: `attn=sq4,mlp=sq4,experts=sq4,lm_head=sq4,requant`.
    pub fn key(&self) -> String {
        let f = |d: Option<DType>| d.map(|d| synaptix_core::precision::dtype_name(d)).unwrap_or_else(|| "-".into());
        format!(
            "attn={},mlp={},experts={},lm_head={},embed={},requant={},dense={}",
            f(self.attn),
            f(self.mlp),
            f(self.experts),
            f(self.lm_head),
            f(self.embed),
            self.requant as u8,
            self.quantize_dense as u8
        )
    }
}

/// Где держать перекодированные веса.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// По свободной RAM: влезает — память, иначе дисковый кэш.
    Auto,
    Memory,
    Disk,
}

#[derive(Debug, Clone)]
pub struct TranscodeReport {
    pub tensors: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    /// `memory` | `disk` | `disk-cached` (кэш уже был).
    pub placement: &'static str,
    pub cache_path: Option<PathBuf>,
    pub seconds: f64,
}

/// Прогресс перекодировки: `(готово тензоров, всего, имя текущего)`.
pub type Progress = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

// ─────────────────────────────────────────────── оверлей

pub(crate) enum Blob {
    Owned(Vec<u8>),
    /// Срез tensors-чанка кэш-бандла (mmap).
    Mapped { bundle: Arc<Bundle>, off: usize, len: usize },
}

impl Blob {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Blob::Owned(v) => v,
            Blob::Mapped { bundle, off, len } => {
                let all = bundle.tensors_slice().expect("кэш перекодировки: tensors-чанк");
                &all[*off..*off + *len]
            }
        }
    }
}

pub(crate) struct OverlayEntry {
    pub entry: QuantEntry,
    pub packed: Blob,
    pub scales: Blob,
    pub from: String,
}

/// Перекодированные веса поверх источника.
pub struct Overlay {
    pub(crate) entries: BTreeMap<String, OverlayEntry>,
    pub report: TranscodeReport,
}

impl Overlay {
    pub(crate) fn get(&self, name: &str) -> Option<&OverlayEntry> {
        self.entries.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Формат, в котором вес лежит после перекодировки, и откуда пришёл.
    pub fn format_of(&self, name: &str) -> Option<(String, String)> {
        self.entries.get(name).map(|e| (e.entry.format.clone(), e.from.clone()))
    }
}

// ─────────────────────────────────────────────── заявки на время загрузки

#[derive(Clone)]
pub struct Request {
    pub spec: TranscodeSpec,
    pub device: Device,
    pub placement: Placement,
    pub progress: Option<Progress>,
}

struct Registry {
    requests: HashMap<PathBuf, Request>,
    overlays: HashMap<String, Arc<Overlay>>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry { requests: HashMap::new(), overlays: HashMap::new() }))
}

fn canon(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Держит заявку на перекодировку пути, пока жив; при сбросе снимает заявку
/// и отпускает кэш оверлеев этого пути.
pub struct RequestGuard {
    path: PathBuf,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let mut r = registry().lock().unwrap();
        r.requests.remove(&self.path);
        let prefix = format!("{}\u{1}", self.path.display());
        r.overlays.retain(|k, _| !k.starts_with(&prefix));
    }
}

/// Поставить заявку: все `SynBundleLoader::open(path)` до сброса guard'а
/// получат оверлей по `spec`.
pub fn request(path: &Path, req: Request) -> RequestGuard {
    let path = canon(path);
    registry().lock().unwrap().requests.insert(path.clone(), req);
    RequestGuard { path }
}

pub(crate) fn request_for(path: &Path) -> Option<Request> {
    let r = registry().lock().unwrap();
    if r.requests.is_empty() {
        return None;
    }
    r.requests.get(&canon(path)).cloned()
}

fn overlay_cache_key(path: &Path, spec: &TranscodeSpec) -> String {
    format!("{}\u{1}{}", canon(path).display(), spec.key())
}

// ─────────────────────────────────────────────── инвентарь

/// Тензор источника глазами перекодировщика.
#[derive(Clone)]
struct Item {
    name: String,
    /// `(число матриц, N, K)`.
    dims: (usize, usize, usize),
    shape: Vec<usize>,
    /// Текущий квант-формат (`None` — плотный).
    current: Option<QuantKind>,
    target: DType,
}

fn target_ok(target: DType, n: usize, k: usize) -> bool {
    match target {
        DType::NVFP4 => n % 64 == 0 && k % 64 == 0,
        DType::MXFP8 | DType::Sq { .. } => k % 32 == 0,
        _ => false,
    }
}

fn plan_items(loader: &SynBundleLoader, spec: &TranscodeSpec) -> Vec<Item> {
    let mut out = Vec::new();
    for name in loader.names() {
        if name.ends_with(".qpacked") || name.ends_with(".qscales") {
            continue;
        }
        if let (Some(kind), Some(dims)) = (loader.quant_kind(name), loader.quant_dims(name)) {
            if !spec.requant {
                continue;
            }
            let Some(target) = spec.target_for(name, dims.0) else { continue };
            if target == kind.dtype() || !target_ok(target, dims.1, dims.2) {
                continue;
            }
            let shape = if dims.0 == 1 { vec![dims.1, dims.2] } else { vec![dims.0, dims.1, dims.2] };
            out.push(Item { name: name.to_string(), dims, shape, current: Some(kind), target });
            continue;
        }
        if !spec.quantize_dense {
            continue;
        }
        let Some((shape, dtype)) = loader.dense_shape(name) else { continue };
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            continue;
        }
        let dims = match shape.as_slice() {
            [n, k] => (1usize, *n, *k),
            [e, n, k] => (*e, *n, *k),
            _ => continue,
        };
        let Some(target) = spec.target_for(name, dims.0) else { continue };
        if !target_ok(target, dims.1, dims.2) {
            continue;
        }
        out.push(Item { name: name.to_string(), dims, shape, current: None, target });
    }
    out
}

/// Байты перекодированного среза: `(packed, scales)`; масштабы пусты у
/// одноблобных форматов. Считается на `device`, возвращается на хосте.
fn transcode_slice(loader: &SynBundleLoader, item: &Item, slice: usize, device: Device) -> Result<(Vec<u8>, Vec<u8>)> {
    let (slices, n, k) = item.dims;
    let dense: Tensor = match item.current {
        Some(_) => {
            let qw = loader
                .load_quant_expert(&item.name, slice, device)
                .ok_or_else(|| IoError::Bundle(format!("{}: квант-вес пропал из источника", item.name)))??;
            qw.dequantize(DType::F16).map_err(|e| IoError::Bundle(format!("{}: деквант {:?}: {e}", item.name, qw.dtype())))?
        }
        None => {
            let host = loader.load_to(&item.name, Device::Cpu, DType::F16)?;
            let t = if slices == 1 { host } else { host.narrow(0, slice, 1).map_err(IoError::Core)? };
            t.to_device(device)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape((n, k)))
                .map_err(|e| IoError::Bundle(format!("{}: срез {slice} на карту: {e}", item.name)))?
        }
    };
    let q: QuantWeight = dense
        .quantize_to(item.target)
        .map_err(|e| IoError::Bundle(format!("{}: квант в {:?}: {e}", item.name, item.target)))?;
    drop(dense);
    let cpu = q.to_device(Device::Cpu).map_err(IoError::Core)?;
    let packed = cpu
        .packed_arc()
        .ok_or_else(|| IoError::Bundle(format!("{}: упакованные веса освобождены", item.name)))?;
    let packed = packed.as_cpu().ok_or_else(|| IoError::Bundle("квант не на хосте".into()))?.as_bytes().to_vec();
    let scales = match cpu.scales_opt() {
        Some(s) => s.as_cpu().ok_or_else(|| IoError::Bundle("масштабы не на хосте".into()))?.as_bytes().to_vec(),
        None => Vec::new(),
    };
    if let Device::Cuda(o) = device {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
    }
    Ok((packed, scales))
}

fn entry_for(item: &Item) -> QuantEntry {
    QuantEntry { format: format_key(QuantKind::from_dtype(item.target).expect("формат кванта")), shape: item.shape.clone() }
}

fn bytes_of_items(items: &[Item]) -> (u64, u64) {
    let mut before = 0u64;
    let mut after = 0u64;
    for it in items {
        after += synaptix_bundle::inspect::quantized_bytes(&it.shape, QuantKind::from_dtype(it.target).unwrap()).unwrap_or(0);
        before += match it.current {
            Some(kind) => synaptix_bundle::inspect::quantized_bytes(&it.shape, kind).unwrap_or(0),
            None => it.shape.iter().product::<usize>() as u64 * 2,
        };
    }
    (before, after)
}

/// `MemAvailable` из `/proc/meminfo`.
pub fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Каталог дискового кэша: `SYN_TRANSCODE_CACHE`, иначе
/// `$XDG_CACHE_HOME/synaptix/transcode`, иначе `~/.cache/synaptix/transcode`.
pub fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SYN_TRANSCODE_CACHE") {
        return PathBuf::from(p);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("synaptix").join("transcode")
}

fn cache_path_for(src: &Path, spec: &TranscodeSpec) -> PathBuf {
    let c = canon(src);
    let meta = std::fs::metadata(&c).ok();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    c.hash(&mut h);
    meta.as_ref().map(|m| m.len()).unwrap_or(0).hash(&mut h);
    meta.and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .hash(&mut h);
    spec.key().hash(&mut h);
    ENCODER_VERSION.hash(&mut h);
    let stem = c.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    cache_dir().join(format!("{stem}-{:016x}.syn", h.finish()))
}

/// Поток перекодированных блобов: тензоры считаются по одному в момент
/// записи — RAM на весь объём не нужна.
struct TranscodeStream {
    /// Свой загрузчик: `Box<dyn TensorStream>` обязан быть `'static`.
    loader: SynBundleLoader,
    device: Device,
    plan: Vec<StreamTensor>,
    /// Индекс элемента плана → (индекс item, это `.qscales`).
    items: Vec<(usize, bool)>,
    src: Vec<Item>,
    pending_scales: Option<(usize, Vec<u8>)>,
    progress: Option<Progress>,
    done: usize,
}

impl TranscodeStream {
    fn new(loader: SynBundleLoader, src: Vec<Item>, device: Device, manifest: &QuantManifest, progress: Option<Progress>) -> Self {
        let mut plan = Vec::new();
        let mut items = Vec::new();
        for (i, it) in src.iter().enumerate() {
            let entry = entry_for(it);
            let kind = entry.kind().unwrap();
            let packed_shape = packed_shape(kind, it.dims);
            plan.push(StreamTensor { name: manifest.packed_name(&it.name), dtype: StDtype::U8, shape: packed_shape });
            items.push((i, false));
            if kind.has_scales() {
                let sb = entry.scales_bytes().unwrap_or(0) as usize;
                plan.push(StreamTensor { name: manifest.scales_name(&it.name), dtype: StDtype::U8, shape: vec![sb] });
                items.push((i, true));
            }
        }
        Self { loader, device, plan, items, src, pending_scales: None, progress, done: 0 }
    }
}

/// Форма блоба `.qpacked` (как у упаковщика synthos): NVFP4 — `[.., N, K/2]`,
/// MXFP8 — `[.., N, K]`, одноблобные — `[.., N, row_bytes]`.
fn packed_shape(kind: QuantKind, dims: (usize, usize, usize)) -> Vec<usize> {
    let (slices, n, k) = dims;
    let last = match kind {
        QuantKind::Nvfp4 => k / 2,
        QuantKind::Mxfp8 => k,
        other => synaptix_core::quant::block_row_bytes(other.dtype(), k).unwrap_or(k),
    };
    if slices == 1 { vec![n, last] } else { vec![slices, n, last] }
}

impl TensorStream for TranscodeStream {
    fn plan(&self) -> &[StreamTensor] {
        &self.plan
    }

    fn write_tensor(&mut self, index: usize, w: &mut dyn std::io::Write) -> synaptix_bundle::Result<()> {
        let (ii, is_scales) = self.items[index];
        let item = &self.src[ii];
        if is_scales {
            let (i, bytes) = self.pending_scales.take().ok_or_else(|| synaptix_bundle::Error::Safetensors(format!("{}: масштабы не посчитаны", item.name)))?;
            if i != ii {
                return Err(synaptix_bundle::Error::Safetensors(format!("{}: масштабы от другого тензора", item.name)));
            }
            return w.write_all(&bytes).map_err(synaptix_bundle::Error::Io);
        }
        if let Some(p) = &self.progress {
            p(self.done, self.src.len(), &item.name);
        }
        let mut scales_all = Vec::new();
        for s in 0..item.dims.0 {
            let (packed, scales) = transcode_slice(&self.loader, item, s, self.device).map_err(|e| synaptix_bundle::Error::Safetensors(e.to_string()))?;
            w.write_all(&packed).map_err(synaptix_bundle::Error::Io)?;
            scales_all.extend_from_slice(&scales);
        }
        self.done += 1;
        if !scales_all.is_empty() {
            self.pending_scales = Some((ii, scales_all));
        }
        Ok(())
    }
}

/// Построить оверлей для `loader` (ещё без оверлея) по спецификации.
///
/// `shared` — оверлей строится по заявке и разделяется всеми открытиями
/// пути до её снятия; явный [`SynBundleLoader::transcode`] реестр не трогает.
pub(crate) fn build_overlay(loader: &SynBundleLoader, src_path: &Path, req: &Request, shared: bool) -> Result<Arc<Overlay>> {
    let key = overlay_cache_key(src_path, &req.spec);
    if shared {
        if let Some(ov) = registry().lock().unwrap().overlays.get(&key) {
            return Ok(ov.clone());
        }
    }
    let t0 = Instant::now();
    let items = plan_items(loader, &req.spec);
    let (bytes_before, bytes_after) = bytes_of_items(&items);
    let mut manifest = QuantManifest::new();
    for it in &items {
        manifest.tensors.insert(it.name.clone(), entry_for(it));
    }
    let placement = match req.placement {
        Placement::Memory => Placement::Memory,
        Placement::Disk => Placement::Disk,
        Placement::Auto => {
            if std::env::var("SYN_TRANSCODE_DISK").as_deref() == Ok("1") {
                Placement::Disk
            } else {
                let avail = mem_available_bytes().unwrap_or(0);
                if bytes_after + RAM_RESERVE_BYTES <= avail { Placement::Memory } else { Placement::Disk }
            }
        }
    };
    tracing::info!(
        "перекодировка {}: {} тензоров, {:.2} → {:.2} ГБ, {:?}, {}",
        src_path.display(),
        items.len(),
        bytes_before as f64 / 1e9,
        bytes_after as f64 / 1e9,
        placement,
        req.spec.key()
    );
    let mut entries = BTreeMap::new();
    let mut placement_str = "memory";
    let mut cache_path = None;
    if items.is_empty() {
        // Нечего перекодировать — пустой оверлей, модель идёт как есть.
    } else if placement == Placement::Memory {
        for (i, it) in items.iter().enumerate() {
            if let Some(p) = &req.progress {
                p(i, items.len(), &it.name);
            }
            let mut packed_all = Vec::new();
            let mut scales_all = Vec::new();
            for s in 0..it.dims.0 {
                let (p, sc) = transcode_slice(loader, it, s, req.device)?;
                packed_all.extend_from_slice(&p);
                scales_all.extend_from_slice(&sc);
            }
            entries.insert(
                it.name.clone(),
                OverlayEntry {
                    entry: entry_for(it),
                    packed: Blob::Owned(packed_all),
                    scales: Blob::Owned(scales_all),
                    from: it.current.map(|k| format_key(k)).unwrap_or_else(|| "dense".into()),
                },
            );
        }
    } else {
        let path = cache_path_for(src_path, &req.spec);
        if path.is_file() {
            placement_str = "disk-cached";
        } else {
            placement_str = "disk";
            std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| IoError::Bundle(format!("кэш перекодировки: {e}")))?;
            let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| IoError::Bundle(e.to_string()))?;
            let own = SynBundleLoader::open_plain(src_path)?;
            let stream = TranscodeStream::new(own, items.clone(), req.device, &manifest, req.progress.clone());
            let id = loader.bundle_id().unwrap_or_else(|| "model".into());
            BundleBuilder::new(format!("{id}-transcoded"), "1.0.0")
                .arch("transcode-cache")
                .purpose("transcode-cache")
                .require_capability(CAP_QUANT_WEIGHTS)
                .component("main", "")
                .add_tensor_stream("main", Box::new(stream))
                .add_file_bytes(MANIFEST_NAME, bytes, FileTag::Inference)
                .map_err(|e| IoError::Bundle(e.to_string()))?
                .write(&path)
                .map_err(|e| IoError::Bundle(format!("кэш перекодировки {}: {e}", path.display())))?;
        }
        let bundle = Arc::new(Bundle::open(&path).map_err(|e| IoError::Bundle(format!("кэш {}: {e}", path.display())))?);
        let all = bundle.tensors_slice().map_err(|e| IoError::Bundle(e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(all).map_err(|e| IoError::Safetensors(e.to_string()))?;
        let base = all.as_ptr() as usize;
        let locate = |name: &str| -> Result<(usize, usize)> {
            let tv = st.tensor(name).map_err(|e| IoError::Bundle(format!("кэш перекодировки: `{name}`: {e}")))?;
            let d = tv.data();
            Ok((d.as_ptr() as usize - base, d.len()))
        };
        for it in &items {
            let entry = entry_for(it);
            let kind = entry.kind().unwrap();
            let (po, pl) = locate(&manifest.packed_name(&it.name))?;
            let (so, sl) = if kind.has_scales() { locate(&manifest.scales_name(&it.name))? } else { (0, 0) };
            entries.insert(
                it.name.clone(),
                OverlayEntry {
                    entry,
                    packed: Blob::Mapped { bundle: bundle.clone(), off: po, len: pl },
                    scales: Blob::Mapped { bundle: bundle.clone(), off: so, len: sl },
                    from: it.current.map(|k| format_key(k)).unwrap_or_else(|| "dense".into()),
                },
            );
        }
        cache_path = Some(path);
    }
    let report = TranscodeReport {
        tensors: items.len(),
        bytes_before,
        bytes_after,
        placement: placement_str,
        cache_path,
        seconds: t0.elapsed().as_secs_f64(),
    };
    tracing::info!("перекодировка готова за {:.1} с ({})", report.seconds, report.placement);
    let ov = Arc::new(Overlay { entries, report });
    if shared {
        registry().lock().unwrap().overlays.insert(key, ov.clone());
    }
    Ok(ov)
}

// ─────────────────────────────────────────────── полный бандл

/// Тензоры одного компонента: перекодированные + всё остальное как есть.
struct FullStream {
    loader: SynBundleLoader,
    device: Device,
    plan: Vec<StreamTensor>,
    /// На элемент плана: `Pass(name)` — байты источника, `Quant(item, scales)`.
    kinds: Vec<Kind>,
    items: Vec<Item>,
    pending_scales: Option<(usize, Vec<u8>)>,
    progress: Option<Progress>,
    done: usize,
}

enum Kind {
    Pass(String),
    /// Квант источника без перекодировки: срезы блобов как есть.
    PassQuant(String, bool),
    Quant(usize, bool),
}

fn st_dtype_of(dt: DType) -> Option<StDtype> {
    Some(match dt {
        DType::F32 => StDtype::F32,
        DType::F16 => StDtype::F16,
        DType::BF16 => StDtype::BF16,
        DType::I64 => StDtype::I64,
        DType::I32 => StDtype::I32,
        DType::U8 => StDtype::U8,
        _ => return None,
    })
}

impl FullStream {
    fn new(loader: SynBundleLoader, spec: &TranscodeSpec, device: Device, manifest: &mut QuantManifest, progress: Option<Progress>) -> Result<Self> {
        let items = plan_items(&loader, spec);
        let by_name: HashMap<&str, usize> = items.iter().enumerate().map(|(i, it)| (it.name.as_str(), i)).collect();
        let mut plan = Vec::new();
        let mut kinds = Vec::new();
        for name in loader.names() {
            if name.ends_with(".qpacked") || name.ends_with(".qscales") {
                // Блобы источника идут через записи манифеста ниже.
                continue;
            }
            if let Some(&i) = by_name.get(name) {
                let it = &items[i];
                let entry = entry_for(it);
                let kind = entry.kind().unwrap();
                manifest.tensors.insert(it.name.clone(), entry.clone());
                plan.push(StreamTensor { name: manifest.packed_name(&it.name), dtype: StDtype::U8, shape: packed_shape(kind, it.dims) });
                kinds.push(Kind::Quant(i, false));
                if kind.has_scales() {
                    plan.push(StreamTensor { name: manifest.scales_name(&it.name), dtype: StDtype::U8, shape: vec![entry.scales_bytes().unwrap_or(0) as usize] });
                    kinds.push(Kind::Quant(i, true));
                }
                continue;
            }
            if let (Some(kind), Some(dims)) = (loader.quant_kind(name), loader.quant_dims(name)) {
                let shape = if dims.0 == 1 { vec![dims.1, dims.2] } else { vec![dims.0, dims.1, dims.2] };
                let entry = QuantEntry { format: format_key(kind), shape };
                manifest.tensors.insert(name.to_string(), entry.clone());
                plan.push(StreamTensor { name: manifest.packed_name(name), dtype: StDtype::U8, shape: packed_shape(kind, dims) });
                kinds.push(Kind::PassQuant(name.to_string(), false));
                if kind.has_scales() {
                    plan.push(StreamTensor { name: manifest.scales_name(name), dtype: StDtype::U8, shape: vec![entry.scales_bytes().unwrap_or(0) as usize] });
                    kinds.push(Kind::PassQuant(name.to_string(), true));
                }
                continue;
            }
            let Some((shape, dtype)) = loader.dense_shape(name) else { continue };
            let Some(st) = st_dtype_of(dtype) else {
                return Err(IoError::Bundle(format!("{name}: тип {dtype:?} писатель не копирует")));
            };
            plan.push(StreamTensor { name: name.to_string(), dtype: st, shape });
            kinds.push(Kind::Pass(name.to_string()));
        }
        Ok(Self { loader, device, plan, kinds, items, pending_scales: None, progress, done: 0 })
    }
}

impl TensorStream for FullStream {
    fn plan(&self) -> &[StreamTensor] {
        &self.plan
    }

    fn write_tensor(&mut self, index: usize, w: &mut dyn std::io::Write) -> synaptix_bundle::Result<()> {
        let err = |e: IoError| synaptix_bundle::Error::Safetensors(e.to_string());
        match &self.kinds[index] {
            Kind::Pass(name) => {
                let dt = self.plan[index].dtype;
                let want = match dt {
                    StDtype::F32 => DType::F32,
                    StDtype::F16 => DType::F16,
                    StDtype::BF16 => DType::BF16,
                    StDtype::I64 => DType::I64,
                    StDtype::I32 => DType::I32,
                    _ => DType::U8,
                };
                let bytes = self.loader.raw_bytes(name, want).map_err(err)?;
                w.write_all(&bytes).map_err(synaptix_bundle::Error::Io)
            }
            Kind::PassQuant(name, is_scales) => {
                let (packed, scales) = self.loader.quant_blob_slices(name).ok_or_else(|| synaptix_bundle::Error::Safetensors(format!("{name}: блоб кванта пропал")))?.map_err(err)?;
                w.write_all(if *is_scales { scales } else { packed }).map_err(synaptix_bundle::Error::Io)
            }
            Kind::Quant(ii, true) => {
                let item = &self.items[*ii];
                let (i, bytes) = self.pending_scales.take().ok_or_else(|| synaptix_bundle::Error::Safetensors(format!("{}: масштабы не посчитаны", item.name)))?;
                if i != *ii {
                    return Err(synaptix_bundle::Error::Safetensors(format!("{}: масштабы от другого тензора", item.name)));
                }
                w.write_all(&bytes).map_err(synaptix_bundle::Error::Io)
            }
            Kind::Quant(ii, false) => {
                let item = &self.items[*ii];
                if let Some(p) = &self.progress {
                    p(self.done, self.items.len(), &item.name);
                }
                let mut scales_all = Vec::new();
                for s in 0..item.dims.0 {
                    let (packed, scales) = transcode_slice(&self.loader, item, s, self.device).map_err(err)?;
                    w.write_all(&packed).map_err(synaptix_bundle::Error::Io)?;
                    scales_all.extend_from_slice(&scales);
                }
                self.done += 1;
                if !scales_all.is_empty() {
                    self.pending_scales = Some((*ii, scales_all));
                }
                Ok(())
            }
        }
    }
}

/// Полный самостоятельный бандл: веса источника (`.syn` или `.gguf`) с
/// перекодированными по `spec` матрицами, остальные тензоры и файлы — как
/// есть. Компоненты `.syn` (`tensors:<имя>`) переносятся все.
pub fn write_bundle(src: &Path, out: &Path, spec: &TranscodeSpec, device: Device, progress: Option<Progress>) -> Result<TranscodeReport> {
    let t0 = Instant::now();
    let base = SynBundleLoader::open(src)?;
    let mut manifest = QuantManifest::new();
    let id = base.bundle_id().unwrap_or_else(|| src.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string());
    let (version, arch, purpose, components): (String, String, String, Vec<(String, String)>) = match base.bundle_handle() {
        Some(b) => {
            let m = b.meta();
            let comps: Vec<(String, String)> = b
                .cdir()
                .entries
                .iter()
                .filter(|e| e.is_alive() && matches!(e.kind_typed(), synaptix_bundle::ChunkType::Tensors))
                .filter_map(|e| e.name.strip_prefix("tensors:").map(|c| (c.to_string(), m.components.get(c).cloned().unwrap_or_default())))
                .collect();
            let comps = if comps.is_empty() { vec![("main".to_string(), String::new())] } else { comps };
            (m.version.clone(), m.arch.clone(), m.purpose.clone(), comps)
        }
        None => ("1.0.0".into(), base.read_file("config.json").and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()).and_then(|v| v.get("model_type").and_then(|s| s.as_str()).map(String::from)).unwrap_or_default(), "text-generation".into(), vec![("main".to_string(), String::new())]),
    };
    let mut builder = BundleBuilder::new(id, version).arch(arch).purpose(purpose).require_capability(CAP_QUANT_WEIGHTS);
    // Каждый поток владеет своим загрузчиком компонента (mmap, дёшево):
    // `Box<dyn TensorStream>` обязан быть `'static`.
    let mut tensors = 0usize;
    let mut before = 0u64;
    let mut after = 0u64;
    let mut streams: Vec<(String, FullStream)> = Vec::new();
    for (c, prefix) in &components {
        let l = if base.bundle_handle().is_some() { SynBundleLoader::open_plain(src)?.with_component(c.clone()) } else { SynBundleLoader::open_plain(src)? };
        let s = FullStream::new(l, spec, device, &mut manifest, progress.clone())?;
        let (b, a) = bytes_of_items(&s.items);
        tensors += s.items.len();
        before += b;
        after += a;
        builder = builder.component(c.clone(), prefix.clone());
        streams.push((c.clone(), s));
    }
    for (c, s) in streams {
        builder = builder.add_tensor_stream(&c, Box::new(s));
    }
    for name in base.file_names() {
        if name == MANIFEST_NAME {
            continue;
        }
        if let Some(bytes) = base.read_file(&name) {
            builder = builder.add_file_bytes(&name, bytes, FileTag::Inference).map_err(|e| IoError::Bundle(e.to_string()))?;
        }
    }
    let mbytes = serde_json::to_vec_pretty(&manifest).map_err(|e| IoError::Bundle(e.to_string()))?;
    builder = builder.add_file_bytes(MANIFEST_NAME, mbytes, FileTag::Inference).map_err(|e| IoError::Bundle(e.to_string()))?;
    builder.write(out).map_err(|e| IoError::Bundle(format!("{}: {e}", out.display())))?;
    Ok(TranscodeReport { tensors, bytes_before: before, bytes_after: after, placement: "file", cache_path: Some(out.to_path_buf()), seconds: t0.elapsed().as_secs_f64() })
}
