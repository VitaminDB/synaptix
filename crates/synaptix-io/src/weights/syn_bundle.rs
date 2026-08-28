use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use safetensors::SafeTensors;
use synaptix_bundle::quant_layout::QuantManifest;
use synaptix_bundle::Bundle;
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

pub struct SynBundleLoader {
    bundle: Arc<Bundle>,
    component: Option<String>,
    default_device: Device,
    index: OnceLock<TensorIndex>,
    /// Манифест квантованных весов. `None` внутри — бандл собран обычным
    /// образом; читается лениво, один раз.
    quant: OnceLock<Option<QuantManifest>>,
}

impl SynBundleLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bundle = Bundle::open(path).map_err(|e| IoError::Bundle(e.to_string()))?;
        Ok(Self {
            bundle: Arc::new(bundle),
            component: None,
            default_device: Device::Cpu,
            index: OnceLock::new(),
            quant: OnceLock::new(),
        })
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
        match &self.component {
            Some(comp) => {
                let (bytes, prefix) = self.bundle.tensors_slice_for(comp)
                    .map_err(|e| IoError::Bundle(e.to_string()))?;
                Ok((bytes, prefix))
            }
            None => {
                let bytes = self.bundle.tensors_slice()
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
        let (bytes, _) = self.st_bytes()?;
        let idx = self.index()?;

        let key = Self::resolve_key(name, &idx.prefix);
        let meta = idx.by_name.get(&key)
            .or_else(|| idx.by_name.get(name))
            .ok_or_else(|| IoError::Safetensors(format!("tensor not found: {name}")))?;

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
    /// Манифест квантования; `None` — бандл обычный.
    pub fn quant_manifest(&self) -> Option<&QuantManifest> {
        self.quant
            .get_or_init(|| QuantManifest::read_from(&self.bundle))
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
        let (key, _) = self.quant_entry(name)?;
        Some(self.build_quant_slice(&key, expert, device))
    }

    /// Форма квантованного веса: `(число матриц, N, K)`. `None` — вес в
    /// бандле не квантован.
    pub fn quant_dims(&self, name: &str) -> Option<(usize, usize, usize)> {
        let (_, entry) = self.quant_entry(name)?;
        entry.dims()
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
        let (slices, n, k) = entry
            .dims()
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: форма {:?} не матрица", entry.shape)))?;
        if slice >= slices {
            return Err(IoError::Bundle(format!(
                "`{key}`: срез {slice} за границей стопки из {slices}"
            )));
        }

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
        let scales_all = take(&manifest.scales_name(key))?;

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

        let dtype = match kind {
            synaptix_bundle::inspect::QuantKind::Nvfp4 => DType::NVFP4,
            synaptix_bundle::inspect::QuantKind::Mxfp8 => DType::MXFP8,
        };
        let packed = Tensor::from_raw_slice(packed_bytes, vec![packed_bytes.len()], DType::U8, device)
            .map_err(IoError::Core)?;
        let scales = Tensor::from_raw_slice(scales_bytes, vec![scales_bytes.len()], DType::U8, device)
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
        let Ok(idx) = self.index() else { return Vec::new(); };
        idx.names.iter().map(|s| s.as_str()).collect()
    }
}
