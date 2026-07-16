//! Загрузка чекпойнта LTX-2.3 из одного safetensors-файла.
//!
//! В отличие от FLUX (diffusers-раскладка по подкаталогам), LTX держит ВСЕ
//! подмодели в одном файле под top-level префиксами. [`LtxCheckpoint`] открывает
//! его zero-copy (mmap, без host-копии — см. урок о .syn-регрессии), разбирает
//! конфиг из `__metadata__` и отдаёт тензоры по имени. [`Component`] — лёгкое
//! представление подмодели с автодобавлением префикса (`model.diffusion_model.`,
//! `vae.`, …) в стиле `get`-замыкания FLUX.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::{SafetensorsLoader, TensorInfo};
use synaptix_io::weights::WeightLoader;

use crate::{Ltx23Config, LtxError};

/// Префиксы подмоделей внутри чекпойнта.
pub const DIT_PREFIX: &str = "model.diffusion_model";
pub const VAE_PREFIX: &str = "vae";
pub const AUDIO_VAE_PREFIX: &str = "audio_vae";
pub const VOCODER_PREFIX: &str = "vocoder";
pub const TEXT_PROJ_PREFIX: &str = "text_embedding_projection";

/// LoRA-адаптер (distilled-LoRA-384, IC-LoRA и т.п.): отдельный safetensors с
/// `{module}.lora_A.weight` `[rank,in]` + `{module}.lora_B.weight` `[out,rank]`.
/// При загрузке весов DiT мерджится: `W += (B·strength) @ A`.
pub struct LoraWeights {
    loader: SafetensorsLoader,
    strength: f32,
    device: Device,
}

impl LoraWeights {
    /// Открыть LoRA-файл (zero-copy mmap) со `strength` (официальный дефолт 1.0).
    pub fn open(path: impl AsRef<Path>, device: Device, strength: f32) -> Result<Self, LtxError> {
        let loader = SafetensorsLoader::open(path.as_ref())
            .map_err(|e| LtxError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, strength, device })
    }

    /// ΔW для линейки `model_key` (полный ключ модели, напр.
    /// `model.diffusion_model.transformer_blocks.0.attn1.to_q`). `None` если LoRA
    /// этот модуль не трогает. ΔW = `(B·strength) @ A` в `dtype`, форма `[out,in]`.
    pub fn delta(&self, model_key: &str, dtype: DType) -> Result<Option<Tensor>, LtxError> {
        let base = model_key.strip_prefix("model.").unwrap_or(model_key);
        let ka = format!("{base}.lora_A.weight");
        let kb = format!("{base}.lora_B.weight");
        if !self.loader.names().iter().any(|n| *n == ka.as_str()) {
            return Ok(None);
        }
        let a = self.loader.load_to(&ka, self.device, dtype)
            .map_err(|e| LtxError::Load(format!("lora {ka}: {e}")))?;
        let b = self.loader.load_to(&kb, self.device, dtype)
            .map_err(|e| LtxError::Load(format!("lora {kb}: {e}")))?;
        let bs = b.mul_scalar(self.strength).map_err(LtxError::from)?;
        let d = bs.matmul(&a).map_err(LtxError::from)?;
        Ok(Some(d))
    }
}

pub struct LtxCheckpoint {
    loader: SafetensorsLoader,
    pub config: Ltx23Config,
    device: Device,
    /// Compute-dtype для весов (bf16/f16). Параметр-таблицы adaLN (F32) грузятся
    /// сырыми через [`LtxCheckpoint::get_raw`].
    dtype: DType,
    /// LoRA для мерджа в веса DiT при загрузке (distilled-LoRA на refine-стадии).
    lora: Option<std::sync::Arc<LoraWeights>>,
}

impl LtxCheckpoint {
    /// Открыть `ltx-2.3-*.safetensors` (один файл). Разбирает `__metadata__["config"]`.
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, LtxError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(LtxError::Load(format!("not found: {}", path.display())));
        }
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| LtxError::Load(e.to_string()))?
            .with_device(device);
        let config = Ltx23Config::from_metadata(loader.metadata())?;
        Ok(Self { loader, config, device, dtype, lora: None })
    }

    /// Прикрепить LoRA для мерджа в веса DiT при загрузке (переживает [`view_on`]).
    pub fn with_lora(mut self, lora: std::sync::Arc<LoraWeights>) -> Self {
        self.lora = Some(lora);
        self
    }

    /// ΔW LoRA для линейки `key` (или `None`). Мерджится в [`get_raw`]-вес при сборке.
    pub fn lora_delta(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, LtxError> {
        match &self.lora {
            Some(l) => l.delta(key, dtype),
            None => Ok(None),
        }
    }

    /// Тензор по полному имени, приведённый к compute-dtype на целевом устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    /// Тензор по полному имени БЕЗ каста (сохраняет исходный dtype файла).
    /// Нужно для F32 adaLN scale_shift-таблиц и статистик VAE.
    pub fn get_raw(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load(name)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.loader.names()
    }

    /// `(dtype, форма)` тензора без загрузки данных.
    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.loader.tensor_info(name)
    }

    /// Итератор `(имя, dtype, форма)` по всем тензорам файла.
    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.loader.infos()
    }

    /// Подмодель с заданным префиксом (`model.diffusion_model`, `vae`, …).
    pub fn component<'a>(&'a self, prefix: &'a str) -> Component<'a> {
        Component { ckpt: self, prefix }
    }

    /// Вью того же чекпойнта с `device`. Разделяет mmap-шарды (`Arc`), дублирует
    /// только индекс. Нужно для streaming-offload: `get`/`get_raw` тогда грузят
    /// веса напрямую mmap→GPU (zero-copy H2D из слайса, без резидентной
    /// host-копии — см. [`Tensor::from_raw_slice`] на `Device::Cuda`).
    pub fn view_on(&self, device: Device) -> Self {
        Self {
            loader: self.loader.clone_with_device(device),
            config: self.config.clone(),
            device,
            dtype: self.dtype,
            lora: self.lora.clone(),
        }
    }

    /// Сырые mmap-шарды (диапазоны pinned-кэша при dense-offload стриминге).
    pub fn shard_bytes(&self) -> Vec<&[u8]> {
        self.loader.shard_bytes()
    }

    /// Сырой mmap-слайс тензора + dtype файла + форма, без Tensor/H2D
    /// (слот-стриминг блоков: прямой H2D в регион слота).
    pub fn raw_bytes(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.loader.raw_bytes(name)
    }

    /// LoRA прикреплена (слот-стриминг несовместим: веса слота — сырые байты
    /// файла, без мерджа ΔW).
    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }
}

/// Представление подмодели: добавляет префикс к именам перед загрузкой.
pub struct Component<'a> {
    ckpt: &'a LtxCheckpoint,
    prefix: &'a str,
}

impl<'a> Component<'a> {
    fn full(&self, name: &str) -> String {
        format!("{}.{}", self.prefix, name)
    }

    /// Тензор подмодели в compute-dtype.
    pub fn get(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.ckpt.get(&self.full(name))
    }

    /// Тензор подмодели без каста (исходный dtype).
    pub fn get_raw(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.ckpt.get_raw(&self.full(name))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.ckpt.contains(&self.full(name))
    }

    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.ckpt.tensor_info(&self.full(name))
    }
}
