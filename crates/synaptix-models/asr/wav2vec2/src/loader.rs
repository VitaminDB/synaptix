//! Загрузка весов Wav2Vec2 из `.syn`-бандла или распакованного HF-снапшота
//! (`model.safetensors` + `config.json` + `vocab.json`).
//!
//! Имена HF: `wav2vec2.*` (энкодер) и `lm_head.*`. Чекпойнты без обёртки
//! `Wav2Vec2ForCTC` (голый `Wav2Vec2Model`) хранят энкодер без префикса —
//! [`Wav2Vec2Weights::get`] принимает имя без префикса и пробует оба варианта.

use std::collections::HashSet;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::Wav2Vec2Config;
use crate::vocab::CtcVocab;
use crate::Wav2Vec2Error;

enum Source {
    Safetensors(SafetensorsLoader),
    Bundle(Box<Bundle>),
}

pub struct Wav2Vec2Weights {
    source: Source,
    names: HashSet<String>,
    pub config: Wav2Vec2Config,
    /// Исходный `config.json` целиком (для сохранения дообученной модели).
    pub config_json: serde_json::Value,
    pub vocab: CtcVocab,
    pub device: Device,
    pub dtype: DType,
}

impl Wav2Vec2Weights {
    pub fn from_unpacked(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, Wav2Vec2Error> {
        let dir = dir.as_ref();
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| Wav2Vec2Error::Load(format!("read {name}: {e}")))
        };
        let config_bytes = read("config.json")?;
        let config = Wav2Vec2Config::from_json_bytes(&config_bytes)?;
        let config_json = serde_json::from_slice(&config_bytes).unwrap_or_default();
        // Базовые (не CTC) чекпойнты словаря не имеют.
        let vocab = match read("vocab.json") {
            Ok(b) => CtcVocab::from_json_bytes(&b)?,
            Err(_) => CtcVocab::from_json_bytes(b"{}")?,
        };
        let weights_path = dir.join("model.safetensors");
        let loader = SafetensorsLoader::open(&weights_path)
            .map_err(|e| Wav2Vec2Error::Load(format!("open {}: {e}", weights_path.display())))?
            .with_device(device);
        let names = loader.names().into_iter().map(|s| s.to_string()).collect();
        Ok(Self { source: Source::Safetensors(loader), names, config, config_json, vocab, device, dtype })
    }

    pub fn from_syn(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, Wav2Vec2Error> {
        let bundle = Bundle::open(path).map_err(|e| Wav2Vec2Error::Bundle(e.to_string()))?;
        let read = |name: &str| {
            bundle.read_file(name).map(|c| c.into_owned()).map_err(|e| {
                Wav2Vec2Error::Bundle(format!(
                    "{name}: {e} — бандл `{}` не похож на модель Wav2Vec2-CTC",
                    bundle.id()
                ))
            })
        };
        let config_bytes = read("config.json")?;
        let config = Wav2Vec2Config::from_json_bytes(&config_bytes)?;
        let config_json = serde_json::from_slice(&config_bytes).unwrap_or_default();
        let vocab = match read("vocab.json") {
            Ok(b) => CtcVocab::from_json_bytes(&b)?,
            Err(_) => CtcVocab::from_json_bytes(b"{}")?,
        };
        let slice = bundle.tensors_slice().map_err(|e| Wav2Vec2Error::Bundle(e.to_string()))?;
        let st = SafeTensors::deserialize(slice)
            .map_err(|e| Wav2Vec2Error::Load(format!("safetensors header: {e}")))?;
        let names = st.names().into_iter().map(|s| s.to_string()).collect();
        Ok(Self { source: Source::Bundle(Box::new(bundle)), names, config, config_json, vocab, device, dtype })
    }

    /// Полное имя тензора: как есть, либо с префиксом энкодера
    /// (`wav2vec2.` / `wavlm.` / `hubert.`).
    fn resolve(&self, name: &str) -> Option<String> {
        if self.names.contains(name) {
            return Some(name.to_string());
        }
        ["wav2vec2.", "wavlm.", "hubert."]
            .iter()
            .map(|p| format!("{p}{name}"))
            .find(|n| self.names.contains(n))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.resolve(name).is_some()
    }

    /// Тензор в исходном dtype файла (без приведения), на целевом устройстве.
    pub fn get_raw(&self, name: &str) -> Result<Tensor, Wav2Vec2Error> {
        let full = self
            .resolve(name)
            .ok_or_else(|| Wav2Vec2Error::Load(format!("нет тензора '{name}'")))?;
        // Веса — параметры модели: на NPU они вшиваются в граф статикой
        // (`with_param_upload`), а не подаются входами на каждом запуске.
        synaptix_core::backend::with_param_upload(|| match &self.source {
            Source::Safetensors(loader) => loader
                .load(&full)
                .map_err(|e| Wav2Vec2Error::Load(format!("'{full}': {e}"))),
            Source::Bundle(bundle) => {
                let slice =
                    bundle.tensors_slice().map_err(|e| Wav2Vec2Error::Bundle(e.to_string()))?;
                let st = SafeTensors::deserialize(slice)
                    .map_err(|e| Wav2Vec2Error::Load(format!("safetensors header: {e}")))?;
                let view = st
                    .tensor(&full)
                    .map_err(|e| Wav2Vec2Error::Load(format!("st tensor '{full}': {e}")))?;
                let src_dtype = st_dtype(view.dtype()).ok_or_else(|| {
                    Wav2Vec2Error::Load(format!("st dtype {:?} unsupported", view.dtype()))
                })?;
                Tensor::from_raw_slice(view.data(), view.shape().to_vec(), src_dtype, self.device)
                    .map_err(|e| Wav2Vec2Error::Load(format!("from_raw_slice '{full}': {e}")))
            }
        })
    }

    /// Тензор, приведённый к compute-dtype.
    pub fn get(&self, name: &str) -> Result<Tensor, Wav2Vec2Error> {
        let t = self.get_raw(name)?;
        if t.dtype() == self.dtype {
            Ok(t)
        } else {
            t.to_dtype(self.dtype).map_err(|e| Wav2Vec2Error::Load(format!("cast '{name}': {e}")))
        }
    }
}

fn st_dtype(d: safetensors::Dtype) -> Option<DType> {
    match d {
        safetensors::Dtype::F32 => Some(DType::F32),
        safetensors::Dtype::F16 => Some(DType::F16),
        safetensors::Dtype::BF16 => Some(DType::BF16),
        safetensors::Dtype::I32 => Some(DType::I32),
        safetensors::Dtype::I64 => Some(DType::I64),
        safetensors::Dtype::U8 => Some(DType::U8),
        safetensors::Dtype::U32 => Some(DType::U32),
        _ => None,
    }
}
