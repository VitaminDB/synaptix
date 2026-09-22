//! Рантайм-источник весов прямо из `.gguf` (без конвертации в `.syn`).
//!
//! Открывает файл mmap'ом, строит план маппинга имён ggml → HF (тот же
//! [`crate::arch::build_plan`], что и у конвертера) и отдаёт тензоры под
//! HF-именами:
//! * квантованные веса (`Q4_0`, `Q4_K`, `IQ2_XXS`, …) — как
//!   [`QuantWeight`] в формате `DType::Ggml(тип)`, байт в байт из mmap (zero-copy
//!   для `Direct`; перестановки строк и склейка стопок экспертов — копией на
//!   уровне блоков, квант сохраняется);
//! * плавающие (`F32`/`F16`/`BF16`) — zero-copy тензором;
//! * всё, что нельзя отдать блоками (перестановка столбцов внутри блока,
//!   преобразования `SubOne`/`LogNeg`, запрос плотного веса) — деквант в f32
//!   на CPU (или на карте через `QuantWeight::dequantize` для прямых 2-D весов).
//!
//! Синтезированные файлы плана (`config.json`, `tokenizer.json`, …) доступны
//! через [`GgufSource::file`] — так `.gguf` выглядит для движка как бандл.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::GgmlType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

use crate::arch;
use crate::error::{GgufError, Result};
use crate::plan::{ConversionPlan, MappedTensor, Producer, Transform};
use crate::reader::{GgufFile, TensorInfo};
use crate::tensor_stream::{bytes_of, hf_shape_of, lookup, materialize_f32, quant_blob, quant_capable};

pub struct GgufSource {
    path: PathBuf,
    files: Vec<Arc<GgufFile>>,
    plan: ConversionPlan,
    /// HF-имя → (компонент, индекс в компоненте).
    index: HashMap<String, (usize, usize)>,
    files_by_name: HashMap<String, usize>,
}

/// Файл с сигнатурой GGUF?
pub fn is_gguf_file(path: &Path) -> bool {
    if path.is_dir() {
        return false;
    }
    if path.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        return true;
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
        .map(|_| &magic == b"GGUF")
        .unwrap_or(false)
}

impl GgufSource {
    /// Открыть текстовую модель. Файл `mmproj*.gguf` рядом не подхватывается
    /// автоматически — см. [`Self::open_with_mmproj`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_mmproj(path, None::<&Path>)
    }

    pub fn open_with_mmproj(path: impl AsRef<Path>, mmproj: Option<impl AsRef<Path>>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let model = GgufFile::open(&path)?;
        let mm = match mmproj {
            Some(p) => Some(GgufFile::open(p.as_ref())?),
            None => None,
        };
        let bundle_id = crate::convert::default_bundle_id(&path);
        let plan = arch::build_plan(&model, mm.as_ref(), &bundle_id)?;
        let mut files = vec![Arc::new(model)];
        if let Some(m) = mm {
            files.push(Arc::new(m));
        }
        let mut index = HashMap::new();
        for (ci, c) in plan.components.iter().enumerate() {
            for (ti, t) in c.tensors.iter().enumerate() {
                index.insert(t.hf_name.clone(), (ci, ti));
            }
        }
        let files_by_name = plan.files.iter().enumerate().map(|(i, f)| (f.path.clone(), i)).collect();
        Ok(Self { path, files, plan, index, files_by_name })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn plan(&self) -> &ConversionPlan {
        &self.plan
    }

    /// Архитектура в терминах HF `model_type` (как в синтезированном config.json).
    pub fn arch(&self) -> &str {
        &self.plan.arch
    }

    /// Синтезированный файл плана (`config.json`, `tokenizer.json`, …).
    pub fn file(&self, name: &str) -> Option<&[u8]> {
        self.files_by_name.get(name).map(|i| self.plan.files[*i].bytes.as_slice())
    }

    pub fn file_names(&self) -> Vec<&str> {
        self.plan.files.iter().map(|f| f.path.as_str()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.plan.components.iter().flat_map(|c| c.tensors.iter().map(|t| t.hf_name.as_str())).collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn item(&self, name: &str) -> Option<&MappedTensor> {
        let (c, t) = *self.index.get(name)?;
        Some(&self.plan.components[c].tensors[t])
    }

    fn first_info(&self, m: &MappedTensor) -> Result<&TensorInfo> {
        lookup(&self.files, &m.producer.sources()[0])
    }

    fn src_type(&self, m: &MappedTensor) -> Result<GgmlType> {
        let ty = self.first_info(m)?.ty;
        for s in m.producer.sources() {
            if lookup(&self.files, s)?.ty != ty {
                return Err(GgufError::BadTensor {
                    name: m.hf_name.clone(),
                    reason: "части имеют разные ggml-типы".into(),
                });
            }
        }
        Ok(ty)
    }

    /// Форма в HF-порядке.
    pub fn hf_shape(&self, name: &str) -> Option<Vec<usize>> {
        let m = self.item(name)?;
        hf_shape_of(&self.files, m)
    }

    /// Тип ggml источника тензора.
    pub fn ggml_type(&self, name: &str) -> Option<GgmlType> {
        self.item(name).and_then(|m| self.src_type(m).ok())
    }

    fn quant_capable(&self, m: &MappedTensor) -> Option<(GgmlType, usize, usize, usize)> {
        let ty = self.src_type(m).ok()?;
        quant_capable(&self.files, m, ty)
    }

    /// `(число матриц, N, K)` квантованного веса; `None` — тензор отдаётся
    /// только плотным.
    pub fn quant_dims(&self, name: &str) -> Option<(usize, usize, usize)> {
        let m = self.item(name)?;
        self.quant_capable(m).map(|(_, s, n, k)| (s, n, k))
    }

    pub fn quant_kind(&self, name: &str) -> Option<GgmlType> {
        let m = self.item(name)?;
        self.quant_capable(m).map(|(t, ..)| t)
    }

    fn quant_blob(&self, m: &MappedTensor, ty: GgmlType, slice: usize, n: usize, k: usize) -> Result<std::borrow::Cow<'_, [u8]>> {
        quant_blob(&self.files, m, ty, slice, n, k)
    }

    fn quant_weight(&self, m: &MappedTensor, ty: GgmlType, slice: usize, n: usize, k: usize, device: Device) -> Result<QuantWeight> {
        let blob = self.quant_blob(m, ty, slice, n, k)?;
        let packed = Tensor::from_raw_slice(&blob, vec![blob.len()], DType::U8, device)
            .map_err(|e| GgufError::BadTensor { name: m.hf_name.clone(), reason: format!("подъём блоба: {e}") })?;
        QuantWeight::new_block(packed.storage_arc(), DType::Ggml(ty), n, k)
            .map_err(|e| GgufError::BadTensor { name: m.hf_name.clone(), reason: format!("QuantWeight: {e}") })
    }

    /// Квантованный вес (одна матрица). `None` — не квантован / стопка.
    pub fn load_quant(&self, name: &str, device: Device) -> Option<Result<QuantWeight>> {
        let m = self.item(name)?;
        let (ty, slices, n, k) = self.quant_capable(m)?;
        if slices != 1 {
            return Some(Err(GgufError::BadTensor { name: name.into(), reason: format!("стопка из {slices} матриц — нужен load_quant_stack") }));
        }
        Some(self.quant_weight(m, ty, 0, n, k, device))
    }

    pub fn load_quant_stack(&self, name: &str, device: Device) -> Option<Result<Vec<QuantWeight>>> {
        let m = self.item(name)?;
        let (ty, slices, n, k) = self.quant_capable(m)?;
        let mut out = Vec::with_capacity(slices);
        for e in 0..slices {
            match self.quant_weight(m, ty, e, n, k, device) {
                Ok(w) => out.push(w),
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(out))
    }

    pub fn load_quant_expert(&self, name: &str, expert: usize, device: Device) -> Option<Result<QuantWeight>> {
        let m = self.item(name)?;
        let (ty, slices, n, k) = self.quant_capable(m)?;
        if expert >= slices {
            return Some(Err(GgufError::BadTensor { name: name.into(), reason: format!("эксперт {expert} за границей стопки из {slices}") }));
        }
        Some(self.quant_weight(m, ty, expert, n, k, device))
    }

    /// Сырой блоб прямого квантованного веса (mmap) — для pinned-зеркала.
    /// `None`, если вес собирается копией (перестановки, склейка стопки).
    pub fn quant_blob_slice(&self, name: &str) -> Option<Result<&[u8]>> {
        let m = self.item(name)?;
        let (ty, slices, n, k) = self.quant_capable(m)?;
        let Producer::Direct(src) = &m.producer else { return None };
        Some(bytes_of(&self.files, src).map(|all| &all[..slices * n * ty.bytes_for(k)]))
    }

    /// Плотный тензор под HF-именем на `device` в `dtype`.
    pub fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        let m = self.item(name).ok_or_else(|| GgufError::BadTensor { name: name.into(), reason: "нет в плане".into() })?;
        let ty = self.src_type(m)?;
        let shape = self.hf_shape(name).ok_or_else(|| GgufError::BadTensor { name: name.into(), reason: "форма не определена".into() })?;
        let err = |e: synaptix_core::error::SynaptixError| GgufError::BadTensor { name: name.into(), reason: e.to_string() };

        // Плавающий прямой тензор без преобразований — zero-copy.
        if let (Producer::Direct(src), Transform::None) = (&m.producer, m.transform) {
            let src_dt = match ty {
                GgmlType::F32 => Some(DType::F32),
                GgmlType::F16 => Some(DType::F16),
                GgmlType::BF16 => Some(DType::BF16),
                _ => None,
            };
            if let Some(sd) = src_dt {
                let bytes = bytes_of(&self.files, src)?;
                let t = Tensor::from_raw_slice(bytes, shape.clone(), sd, device).map_err(err)?;
                return if sd == dtype { Ok(t) } else { t.to_dtype(dtype).map_err(err) };
            }
        }
        // Квантованный 2-D вес на карте — деквант ядром.
        if device.is_cuda() && matches!(dtype, DType::F16 | DType::BF16) {
            if let Some((qty, 1, n, k)) = self.quant_capable(m) {
                let qw = self.quant_weight(m, qty, 0, n, k, device)?;
                return qw.dequantize(dtype).map_err(err);
            }
        }
        // Общий путь: f32 на CPU.
        let buf = materialize_f32(&self.files, &m.producer, m.transform, ty)?;
        if buf.len() != shape.iter().product::<usize>() {
            return Err(GgufError::BadTensor { name: name.into(), reason: format!("форма {shape:?} не совпадает с {} элементами", buf.len()) });
        }
        let t = Tensor::from_vec(buf, shape, Device::Cpu).map_err(err)?;
        let t = if dtype == DType::F32 { t } else { t.to_dtype(dtype).map_err(err)? };
        if device == Device::Cpu { Ok(t) } else { t.to_device(device).map_err(err) }
    }
}
