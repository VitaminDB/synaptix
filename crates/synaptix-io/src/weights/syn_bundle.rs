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
        let manifest = self.quant_manifest()?;
        if manifest.is_empty() {
            return None;
        }
        let idx = match self.index() {
            Ok(i) => i,
            Err(e) => return Some(Err(e)),
        };
        let key = Self::resolve_key(name, &idx.prefix);
        let entry = manifest
            .entry(&key)
            .or_else(|| manifest.entry(name))?;
        let name_in_bundle = if manifest.entry(&key).is_some() { key } else { name.to_string() };

        Some(self.build_quant(manifest, &name_in_bundle, entry, device))
    }

    fn build_quant(
        &self,
        manifest: &QuantManifest,
        key: &str,
        entry: &synaptix_bundle::QuantEntry,
        device: Device,
    ) -> Result<QuantWeight> {
        let kind = entry.kind().ok_or_else(|| {
            IoError::Bundle(format!("`{key}`: неизвестный квант-формат `{}`", entry.format))
        })?;
        let (slices, n, k) = entry
            .dims()
            .ok_or_else(|| IoError::Bundle(format!("`{key}`: форма {:?} не матрица", entry.shape)))?;
        if slices != 1 {
            // Стопка экспертов — это `slices` независимых матриц, а
            // `QuantWeight` описывает одну. Пока такие веса читать некому:
            // MoE-путь в движке не реализован. Ошибка честнее тихого
            // возврата плотного тензора, которого в бандле нет.
            return Err(IoError::Bundle(format!(
                "`{key}`: стопка из {slices} матриц — чтение квантованных MoE-весов пока не поддержано"
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
        let packed_bytes = take(&manifest.packed_name(key))?;
        let scales_bytes = take(&manifest.scales_name(key))?;

        // Размеры сверяем с манифестом: расхождение означает, что бандл
        // собран другой версией раскладки, и молча считать по нему нельзя.
        let want_packed = entry.packed_bytes().unwrap_or(0) as usize;
        let want_scales = entry.scales_bytes().unwrap_or(0) as usize;
        if packed_bytes.len() != want_packed || scales_bytes.len() != want_scales {
            return Err(IoError::Bundle(format!(
                "`{key}`: блобы {}/{} байт, а раскладка требует {want_packed}/{want_scales}",
                packed_bytes.len(),
                scales_bytes.len()
            )));
        }

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
