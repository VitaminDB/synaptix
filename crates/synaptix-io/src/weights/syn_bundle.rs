use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use safetensors::SafeTensors;
use synaptix_bundle::quant_layout::QuantManifest;
use synaptix_bundle::Bundle;
use synaptix_gguf::{is_gguf_file, GgufSource};
use synaptix_core::tensor::quant::QuantWeight;
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

/// Загрузчик весов из файла-модели: `.syn`-бандла или `.gguf` (llama.cpp).
/// Для GGUF тензоры отдаются под HF-именами через план маппера
/// (`synaptix_gguf::GgufSource`), квантованные — блоками ggml как
/// [`QuantWeight`]; синтезированные `config.json`/`tokenizer.json` — через
/// [`Self::read_file`]. Потребителю всё равно, какой формат под капотом.
pub struct SynBundleLoader {
    bundle: Option<Arc<Bundle>>,
    gguf: Option<Arc<GgufSource>>,
    component: Option<String>,
    default_device: Device,
    index: OnceLock<TensorIndex>,
    /// Манифест квантованных весов. `None` внутри — бандл собран обычным
    /// образом; читается лениво, один раз.
    quant: OnceLock<Option<QuantManifest>>,
    /// Перекодированные веса поверх источника (см. [`super::transcode`]):
    /// все квант-запросы сперва смотрят сюда.
    overlay: Option<Arc<super::transcode::Overlay>>,
}

/// Файл GGUF (по расширению или сигнатуре).
pub fn is_gguf_model(path: &Path) -> bool {
    is_gguf_file(path)
}

/// Файл-модель, который умеет открыть [`SynBundleLoader`]: `.syn` или `.gguf`.
pub fn is_model_file(path: &Path) -> bool {
    if path.is_dir() {
        return false;
    }
    let ext = path.extension().and_then(|s| s.to_str()).map(|e| e.to_ascii_lowercase());
    matches!(ext.as_deref(), Some("syn") | Some("gguf")) || is_gguf_file(path)
}

/// Вспомогательный файл модели (`config.json`, `tokenizer.json`, …) из
/// HF-каталога, `.syn`-бандла или `.gguf` (у GGUF — синтезированный планом).
pub fn read_model_file(path: &Path, name: &str) -> Option<Vec<u8>> {
    if path.is_dir() {
        return std::fs::read(path.join(name)).ok();
    }
    if is_gguf_file(path) {
        let src = GgufSource::open(path).ok()?;
        return src.file(name).map(|b| b.to_vec());
    }
    let bundle = Bundle::open(path).ok()?;
    bundle.read_file(name).ok().map(|c| c.into_owned())
}

impl SynBundleLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut loader = Self::open_plain(path)?;
        // Заявка на перекодировку на время загрузки (см. `transcode::request`):
        // оверлей строится один раз на путь и разделяется всеми открытиями.
        if let Some(req) = super::transcode::request_for(path) {
            let ov = super::transcode::build_overlay(&loader, path, &req, true)?;
            loader.overlay = Some(ov);
        }
        Ok(loader)
    }

    pub(crate) fn open_plain(path: &Path) -> Result<Self> {
        if is_gguf_file(path) {
            let src = GgufSource::open(path).map_err(|e| IoError::Bundle(format!("gguf {}: {e}", path.display())))?;
            return Ok(Self {
                bundle: None,
                gguf: Some(Arc::new(src)),
                component: None,
                default_device: Device::Cpu,
                index: OnceLock::new(),
                quant: OnceLock::new(),
                overlay: None,
            });
        }
        let bundle = Bundle::open(path).map_err(|e| IoError::Bundle(e.to_string()))?;
        Ok(Self {
            bundle: Some(Arc::new(bundle)),
            gguf: None,
            component: None,
            default_device: Device::Cpu,
            index: OnceLock::new(),
            quant: OnceLock::new(),
            overlay: None,
        })
    }

    /// Перекодировать квант-веса по `spec` (см. [`super::transcode`]) и
    /// положить результат оверлеем поверх источника.
    pub fn transcode(
        &mut self,
        path: &Path,
        spec: super::transcode::TranscodeSpec,
        device: Device,
        placement: super::transcode::Placement,
        progress: Option<super::transcode::Progress>,
    ) -> Result<super::transcode::TranscodeReport> {
        let req = super::transcode::Request { spec, device, placement, progress };
        let ov = super::transcode::build_overlay(self, path, &req, false)?;
        let report = ov.report.clone();
        self.overlay = Some(ov);
        Ok(report)
    }

    /// Оверлей перекодировки, если он есть.
    pub fn overlay(&self) -> Option<&Arc<super::transcode::Overlay>> {
        self.overlay.as_ref()
    }

    /// Запись оверлея для `name` (по имени или по ключу компонента).
    fn overlay_entry(&self, name: &str) -> Option<&super::transcode::OverlayEntry> {
        let ov = self.overlay.as_ref()?;
        if let Some(e) = ov.get(name) {
            return Some(e);
        }
        let idx = self.index().ok()?;
        let key = Self::resolve_key(name, &idx.prefix);
        ov.get(&key)
    }

    /// Идентификатор бандла (`None` у GGUF без имени).
    pub fn bundle_id(&self) -> Option<String> {
        if let Some(b) = &self.bundle {
            return Some(b.id().to_string());
        }
        self.gguf.as_ref().map(|g| g.plan().bundle_id.clone())
    }

    /// Сам бандл `.syn` (нет у GGUF).
    pub fn bundle_handle(&self) -> Option<&Arc<Bundle>> {
        self.bundle.as_ref()
    }

    /// Имена вспомогательных файлов (`config.json`, …).
    pub fn file_names(&self) -> Vec<String> {
        if let Some(g) = &self.gguf {
            return g.file_names().into_iter().map(String::from).collect();
        }
        match &self.bundle {
            Some(b) => b.list_files().map(|e| e.name.clone()).collect(),
            None => Vec::new(),
        }
    }

    /// Форма и тип плотного тензора (`None` — нет или хранится квантом).
    pub fn dense_shape(&self, name: &str) -> Option<(Vec<usize>, DType)> {
        if let Some(g) = &self.gguf {
            if g.quant_dims(name).is_some() {
                return None;
            }
            let ty = g.ggml_type(name)?;
            let dt = match ty {
                synaptix_core::quant::GgmlType::F32 => DType::F32,
                synaptix_core::quant::GgmlType::BF16 => DType::BF16,
                _ => DType::F16,
            };
            return g.hf_shape(name).map(|s| (s, dt));
        }
        let idx = self.index().ok()?;
        let key = Self::resolve_key(name, &idx.prefix);
        let meta = idx.by_name.get(&key).or_else(|| idx.by_name.get(name))?;
        Some((meta.shape.clone(), meta.dtype?))
    }

    /// Байты плотного тензора в типе `want` (у бандла — как лежат, без
    /// копии в другой тип, если тип совпадает; у GGUF — материализация).
    pub fn raw_bytes(&self, name: &str, want: DType) -> Result<std::borrow::Cow<'_, [u8]>> {
        if self.gguf.is_none() {
            let (bytes, _) = self.st_bytes()?;
            let idx = self.index()?;
            let key = Self::resolve_key(name, &idx.prefix);
            if let Some(meta) = idx.by_name.get(&key).or_else(|| idx.by_name.get(name)) {
                if meta.dtype == Some(want) {
                    return Ok(std::borrow::Cow::Borrowed(&bytes[meta.off..meta.off + meta.len]));
                }
            }
        }
        let t = self.load_internal(name, Device::Cpu, Some(want))?;
        let t = t.contiguous().map_err(IoError::Core)?;
        let st = t.storage_arc();
        let cpu = st.as_cpu().ok_or_else(|| IoError::Bundle(format!("{name}: тензор не на хосте")))?;
        Ok(std::borrow::Cow::Owned(cpu.as_bytes().to_vec()))
    }

    /// Открыть уже разобранный GGUF-источник (переиспользовать план).
    pub fn from_gguf(src: Arc<GgufSource>) -> Self {
        Self {
            bundle: None,
            gguf: Some(src),
            component: None,
            default_device: Device::Cpu,
            index: OnceLock::new(),
            quant: OnceLock::new(),
            overlay: None,
        }
    }

    /// Источник GGUF, если загрузчик открыт над `.gguf`.
    pub fn gguf(&self) -> Option<&Arc<GgufSource>> {
        self.gguf.as_ref()
    }

    fn syn(&self) -> Result<&Arc<Bundle>> {
        self.bundle.as_ref().ok_or_else(|| IoError::Bundle("не .syn-бандл".into()))
    }

    /// Вспомогательный файл модели (`config.json`, `tokenizer.json`, …).
    pub fn read_file(&self, name: &str) -> Option<Vec<u8>> {
        if let Some(g) = &self.gguf {
            return g.file(name).map(|b| b.to_vec());
        }
        self.bundle.as_ref()?.read_file(name).ok().map(|c| c.into_owned())
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
        let bundle = self.syn()?;
        match &self.component {
            Some(comp) => {
                let (bytes, prefix) = bundle.tensors_slice_for(comp)
                    .map_err(|e| IoError::Bundle(e.to_string()))?;
                Ok((bytes, prefix))
            }
            None => {
                let bytes = bundle.tensors_slice()
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
        if self.overlay_entry(name).is_some() {
            if let Some(t) = self.dequant_block_entry(name, device, want_dtype)? {
                return Ok(t);
            }
            return Err(IoError::Bundle(format!("{name}: перекодированный вес без плотного пути (NVFP4/MXFP8 на хосте)")));
        }
        if let Some(g) = &self.gguf {
            let want = want_dtype.unwrap_or(match g.ggml_type(name) {
                Some(t) if !t.is_quantized() && t != synaptix_core::quant::GgmlType::F32 => DType::F16,
                _ => DType::F32,
            });
            return g.load_to(name, device, want).map_err(|e| IoError::Bundle(e.to_string()));
        }
        let (bytes, _) = self.st_bytes()?;
        let idx = self.index()?;

        let key = Self::resolve_key(name, &idx.prefix);
        let Some(meta) = idx.by_name.get(&key).or_else(|| idx.by_name.get(name)) else {
            // Плотной копии нет, но вес лежит блоками одноблобного формата
            // (`.qpacked` из GGUF-конверсии Keep или SQ): деквантуем на
            // хосте, как это делает прямое чтение `.gguf`.
            if let Some(t) = self.dequant_block_entry(name, device, want_dtype)? {
                return Ok(t);
            }
            return Err(IoError::Safetensors(format!("tensor not found: {name}")));
        };

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

/// Квантованные веса из бандла, собранного с `syn-quant-v1`.
impl SynBundleLoader {
    /// Плотный тензор из одноблобного квант-веса (`DType::Sq`/`DType::Ggml`):
    /// `None` — такого веса в манифесте нет или формат не одноблобный.
    fn dequant_block_entry(&self, name: &str, device: Device, want_dtype: Option<DType>) -> Result<Option<Tensor>> {
        let Some((key, entry)) = self.any_quant_entry(name) else { return Ok(None) };
        let Some(kind) = entry.kind() else { return Ok(None) };
        let dtype = kind.dtype();
        if !matches!(dtype, DType::Sq { .. } | DType::Ggml(_)) {
            return Ok(None);
        }
        let Some((slices, n, k)) = entry.dims() else { return Ok(None) };
        let (packed, _) = self
            .quant_blob_slices(&key)
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: нет блоба кванта")))??;
        let rb = synaptix_core::quant::block_row_bytes(dtype, k)
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: K={k} не кратен блоку формата")))?;
        if packed.len() < slices * n * rb {
            return Err(IoError::Bundle(format!("`{key}`: блоб короче {slices}×{n}×{rb}")));
        }
        let mut out = vec![0f32; slices * n * k];
        for r in 0..slices * n {
            synaptix_core::quant::dequant_row_f32(dtype, &packed[r * rb..(r + 1) * rb], k, &mut out[r * k..(r + 1) * k])
                .map_err(IoError::Core)?;
        }
        let t = Tensor::from_vec(out, entry.shape.clone(), Device::Cpu).map_err(IoError::Core)?;
        let t = match want_dtype {
            Some(d) if d != DType::F32 => t.to_dtype(d).map_err(IoError::Core)?,
            _ => t,
        };
        Ok(Some(if device == Device::Cpu { t } else { t.to_device(device).map_err(IoError::Core)? }))
    }

    /// Манифест квантования; `None` — бандл обычный.
    pub fn quant_manifest(&self) -> Option<&QuantManifest> {
        let bundle = self.bundle.as_ref()?;
        self.quant
            .get_or_init(|| QuantManifest::read_from(bundle))
            .as_ref()
    }

    /// Готовый квант-вес прямо из mmap: пара блобов `.qpacked`/`.qscales`
    /// поднимается на устройство как есть, без разжатия в F16 и повторного
    /// квантования.
    ///
    /// `None` — этот тензор в бандле не квантован (обычный путь загрузки).
    /// `Some(Err(_))` — квантован, но прочитать не вышло: молча свалиться на
    /// плотный путь нельзя, плотной копии в бандле уже нет.
    pub fn load_quant(&self, name: &str, device: Device) -> Option<Result<QuantWeight>> {
        if let Some(ov) = self.overlay_entry(name) {
            if ov.entry.slices() != Some(1) {
                return Some(Err(IoError::Bundle(format!("`{name}`: стопка — читайте её через load_quant_stack"))));
            }
            return Some(Self::quant_from_blobs(name, &ov.entry, 0, device, ov.packed.bytes(), ov.scales.bytes()));
        }
        if let Some(g) = &self.gguf {
            // `SYN_GGUF_DENSE=1` — отладочный обход: блоки ggml не отдаются,
            // модель читает веса плотно (`load_to`) и квантует на лету сама.
            if let Ok(v) = std::env::var("SYN_GGUF_DENSE") {
                if v == "1" || v.split(',').any(|pat| !pat.is_empty() && name.contains(pat)) {
                    return None;
                }
            }
            return g.load_quant(name, device).map(|r| r.map_err(|e| IoError::Bundle(e.to_string())));
        }
        let (key, entry) = self.quant_entry(name)?;
        let slices = match entry.dims() {
            Some((s, _, _)) => s,
            None => {
                return Some(Err(IoError::Bundle(format!(
                    "`{key}`: форма {:?} не матрица",
                    entry.shape
                ))))
            }
        };
        if slices != 1 {
            // Стопка экспертов — это `slices` независимых матриц, а
            // `QuantWeight` описывает одну. Отдать первую молча значило бы
            // подсунуть модели чужие веса, поэтому — явная ошибка с
            // указанием на stack-API.
            return Some(Err(IoError::Bundle(format!(
                "`{key}`: стопка из {slices} матриц — читайте её через load_quant_stack"
            ))));
        }
        Some(self.build_quant_slice(&key, 0, device))
    }

    /// Вся стопка `[E, N, K]` экспертов MoE: по одному [`QuantWeight`] на
    /// эксперта, в порядке ведущей оси. Обычная матрица `[N, K]` — стопка из
    /// одного элемента, поэтому вызывающему не нужно различать эти случаи.
    ///
    /// Каждый эксперт получает собственные буферы: писатель кладёт срезы
    /// подряд, а ядрам нужен непрерывный `packed`, начинающийся с нуля.
    pub fn load_quant_stack(&self, name: &str, device: Device) -> Option<Result<Vec<QuantWeight>>> {
        if let Some(ov) = self.overlay_entry(name) {
            let slices = ov.entry.slices()?;
            let r: Result<Vec<QuantWeight>> = (0..slices)
                .map(|i| Self::quant_from_blobs(name, &ov.entry, i, device, ov.packed.bytes(), ov.scales.bytes()))
                .collect();
            return Some(r);
        }
        if let Some(g) = &self.gguf {
            return g.load_quant_stack(name, device).map(|r| r.map_err(|e| IoError::Bundle(e.to_string())));
        }
        let (key, entry) = self.quant_entry(name)?;
        let slices = match entry.slices() {
            Some(s) => s,
            None => {
                return Some(Err(IoError::Bundle(format!(
                    "`{key}`: форма {:?} не матрица и не стопка матриц",
                    entry.shape
                ))))
            }
        };
        let mut out = Vec::with_capacity(slices);
        for i in 0..slices {
            match self.build_quant_slice(&key, i, device) {
                Ok(w) => out.push(w),
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(out))
    }

    /// Один эксперт стопки — когда вся стопка в память не нужна
    /// (expert-parallel, host-stream по требованию).
    pub fn load_quant_expert(
        &self,
        name: &str,
        expert: usize,
        device: Device,
    ) -> Option<Result<QuantWeight>> {
        if let Some(ov) = self.overlay_entry(name) {
            let slices = ov.entry.slices()?;
            if expert >= slices {
                return Some(Err(IoError::Bundle(format!("`{name}`: эксперт {expert} за границей стопки из {slices}"))));
            }
            return Some(Self::quant_from_blobs(name, &ov.entry, expert, device, ov.packed.bytes(), ov.scales.bytes()));
        }
        if let Some(g) = &self.gguf {
            return g.load_quant_expert(name, expert, device).map(|r| r.map_err(|e| IoError::Bundle(e.to_string())));
        }
        let (key, _) = self.quant_entry(name)?;
        Some(self.build_quant_slice(&key, expert, device))
    }

    /// Сырые блобы `.qpacked`/`.qscales` квантованного веса — срезы того же
    /// mmap, из которого `load_quant_expert` режет экспертов. Нужны, чтобы
    /// зеркалировать стопку в pinned-RAM по диапазону адресов: подкачка
    /// эксперта попадает в зеркало по указателю среза.
    pub fn quant_blob_slices(&self, name: &str) -> Option<Result<(&[u8], &[u8])>> {
        if let Some(ov) = self.overlay_entry(name) {
            return Some(Ok((ov.packed.bytes(), ov.scales.bytes())));
        }
        if let Some(g) = &self.gguf {
            // У одноблобных форматов масштабов нет; собранные копией веса
            // (перестановки, стопки из частей) зеркалом не адресуются.
            return g.quant_blob_slice(name).map(|r| r.map(|b| (b, &b[..0])).map_err(|e| IoError::Bundle(e.to_string())));
        }
        let (key, entry) = self.quant_entry(name)?;
        let manifest = self.quant_manifest()?;
        // У одноблобных форматов (`sq*`, `ggml:*`) блоба `.qscales` нет.
        let has_scales = entry.kind().is_none_or(|k| k.has_scales());
        let r = (|| -> Result<(&[u8], &[u8])> {
            let (bytes, _) = self.st_bytes()?;
            let idx = self.index()?;
            let take = |blob: &str| -> Result<&[u8]> {
                let meta = idx
                    .by_name
                    .get(blob)
                    .ok_or_else(|| IoError::Bundle(format!("`{blob}`: блоб не найден в бандле")))?;
                Ok(&bytes[meta.off..meta.off + meta.len])
            };
            let packed = take(&manifest.packed_name(&key))?;
            let scales = if has_scales { take(&manifest.scales_name(&key))? } else { &packed[..0] };
            Ok((packed, scales))
        })();
        Some(r)
    }

    /// Формат кванта веса (`None` — не квантован или формат неизвестен).
    pub fn quant_kind(&self, name: &str) -> Option<synaptix_bundle::inspect::QuantKind> {
        if let Some(ov) = self.overlay_entry(name) {
            return ov.entry.kind();
        }
        if let Some(g) = &self.gguf {
            return g.quant_kind(name).map(synaptix_bundle::inspect::QuantKind::Ggml);
        }
        let (_, entry) = self.quant_entry(name)?;
        entry.kind()
    }

    /// Форма квантованного веса: `(число матриц, N, K)`. `None` — вес в
    /// бандле не квантован.
    pub fn quant_dims(&self, name: &str) -> Option<(usize, usize, usize)> {
        if let Some(ov) = self.overlay_entry(name) {
            return ov.entry.dims();
        }
        if let Some(g) = &self.gguf {
            return g.quant_dims(name);
        }
        let (_, entry) = self.quant_entry(name)?;
        entry.dims()
    }

    /// Запись кванта из оверлея или манифеста (у GGUF — синтезированная).
    fn any_quant_entry(&self, name: &str) -> Option<(String, synaptix_bundle::QuantEntry)> {
        if let Some(ov) = self.overlay_entry(name) {
            return Some((name.to_string(), ov.entry.clone()));
        }
        if let Some(g) = &self.gguf {
            let kind = g.quant_kind(name)?;
            let (s, n, k) = g.quant_dims(name)?;
            let shape = if s == 1 { vec![n, k] } else { vec![s, n, k] };
            return Some((
                name.to_string(),
                synaptix_bundle::QuantEntry { format: synaptix_bundle::quant_layout::format_key(synaptix_bundle::inspect::QuantKind::Ggml(kind)), shape },
            ));
        }
        self.quant_entry(name)
    }

    /// Запись манифеста для `name` с учётом префикса компонента. Возвращает
    /// имя, под которым тензор лежит в бандле, — по нему же строятся имена
    /// блобов.
    fn quant_entry(&self, name: &str) -> Option<(String, synaptix_bundle::QuantEntry)> {
        let manifest = self.quant_manifest()?;
        if manifest.is_empty() {
            return None;
        }
        let idx = self.index().ok()?;
        let key = Self::resolve_key(name, &idx.prefix);
        if let Some(e) = manifest.entry(&key) {
            return Some((key, e.clone()));
        }
        manifest.entry(name).map(|e| (name.to_string(), e.clone()))
    }

    /// Собрать [`QuantWeight`] для среза `slice` стопки (для обычной матрицы
    /// — единственного среза 0).
    fn build_quant_slice(&self, key: &str, slice: usize, device: Device) -> Result<QuantWeight> {
        let manifest = self
            .quant_manifest()
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: манифест кванта пропал")))?;
        let entry = manifest
            .entry(key)
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: нет записи в манифесте кванта")))?;
        let kind = entry.kind().ok_or_else(|| {
            IoError::Bundle(format!("`{key}`: неизвестный квант-формат `{}`", entry.format))
        })?;
        let (bytes, _) = self.st_bytes()?;
        let idx = self.index()?;
        let take = |blob: &str| -> Result<&[u8]> {
            let meta = idx
                .by_name
                .get(blob)
                .ok_or_else(|| IoError::Bundle(format!("`{blob}`: блоб не найден в бандле")))?;
            Ok(&bytes[meta.off..meta.off + meta.len])
        };
        let packed_all = take(&manifest.packed_name(key))?;
        // У одноблобных форматов блоба масштабов нет.
        let scales_all: &[u8] = if kind.has_scales() { take(&manifest.scales_name(key))? } else { &[] };
        Self::quant_from_blobs(key, entry, slice, device, packed_all, scales_all)
    }

    /// [`QuantWeight`] среза `slice` из блобов `packed_all`/`scales_all`
    /// (все срезы подряд) по записи манифеста.
    fn quant_from_blobs(
        key: &str,
        entry: &synaptix_bundle::QuantEntry,
        slice: usize,
        device: Device,
        packed_all: &[u8],
        scales_all: &[u8],
    ) -> Result<QuantWeight> {
        let kind = entry.kind().ok_or_else(|| {
            IoError::Bundle(format!("`{key}`: неизвестный квант-формат `{}`", entry.format))
        })?;
        let (slices, n, k) = entry
            .dims()
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: форма {:?} не матрица", entry.shape)))?;
        if slice >= slices {
            return Err(IoError::Bundle(format!(
                "`{key}`: срез {slice} за границей стопки из {slices}"
            )));
        }

        // Размеры сверяем с манифестом: расхождение означает, что бандл
        // собран другой версией раскладки, и молча считать по нему нельзя.
        let want_packed = entry.packed_bytes().unwrap_or(0) as usize;
        let want_scales = entry.scales_bytes().unwrap_or(0) as usize;
        if packed_all.len() != want_packed || scales_all.len() != want_scales {
            return Err(IoError::Bundle(format!(
                "`{key}`: блобы {}/{} байт, а раскладка требует {want_packed}/{want_scales}",
                packed_all.len(),
                scales_all.len()
            )));
        }

        let packed_step = entry.packed_bytes_per_slice().unwrap_or(0) as usize;
        let scales_step = entry.scales_bytes_per_slice().unwrap_or(0) as usize;
        let packed_bytes = &packed_all[slice * packed_step..(slice + 1) * packed_step];
        let scales_bytes = &scales_all[slice * scales_step..(slice + 1) * scales_step];

        let dtype = kind.dtype();
        if !kind.has_scales() {
            let packed = Tensor::from_raw_slice(packed_bytes, vec![packed_bytes.len()], DType::U8, device)
                .map_err(IoError::Core)?;
            return QuantWeight::new_block(packed.storage_arc(), dtype, n, k).map_err(IoError::Core);
        }
        let scales = Tensor::from_raw_slice(scales_bytes, vec![scales_bytes.len()], DType::U8, device)
            .map_err(IoError::Core)?;
        // Pinned-зеркало стопки уже в перемешанной раскладке GEMV/GEMM: одна DMA
        // и вес готов, без перепаковки на карте (см. `MirrorRange::nvfp4_repack`).
        if dtype == DType::NVFP4 && matches!(device, Device::Cuda(_)) {
            if let Some((mirror, true)) =
                synaptix_core::device::cuda::offload_pin_resolve_tagged(packed_bytes)
            {
                let shuffled = synaptix_core::device::cuda::pinned_slice_to_device(device, mirror)
                    .map_err(IoError::Core)?;
                return QuantWeight::from_shuffled(
                    std::sync::Arc::new(shuffled),
                    scales.storage_arc(),
                    dtype,
                    n,
                    k,
                )
                .map_err(IoError::Core);
            }
        }
        let packed = Tensor::from_raw_slice(packed_bytes, vec![packed_bytes.len()], DType::U8, device)
            .map_err(IoError::Core)?;
        QuantWeight::new(packed.storage_arc(), scales.storage_arc(), dtype, n, k)
            .map_err(IoError::Core)
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
        if let Some(g) = &self.gguf {
            return g.names();
        }
        let Ok(idx) = self.index() else { return Vec::new(); };
        idx.names.iter().map(|s| s.as_str()).collect()
    }

    fn contains(&self, name: &str) -> bool {
        if let Some(g) = &self.gguf {
            return g.contains(name);
        }
        self.index().map(|idx| {
            let key = Self::resolve_key(name, &idx.prefix);
            idx.by_name.contains_key(&key) || idx.by_name.contains_key(name)
        }).unwrap_or(false)
    }
}
