//! Загрузка весов Whisper из `.syn`-бандла через [`SynBundleLoader`].
//!
//! Веса лежат под HF-именами (`model.encoder.layers.{i}.self_attn.q_proj.weight`
//! и т.п.) в чанке `tensors:main`. Вспомогательные файлы (config, tokenizer)
//! читаются как обычные file-чанки через [`Bundle::read_file`].

use std::collections::HashSet;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::{GenerationConfig, WhisperConfig};
use crate::WhisperError;

pub struct WhisperWeights {
    bundle: Bundle,
    loader: SynBundleLoader,
    names: HashSet<String>,
    pub config: WhisperConfig,
    pub gen_config: GenerationConfig,
    pub device: Device,
    pub dtype: DType,
    /// Квантовать линейные слои при загрузке (NVFP4/MXFP8/Sq) — на лету,
    /// независимо от формата весов; `None` — плотные в `dtype`.
    pub quant: Option<DType>,
    /// Энкодер внутри чужого бандла (аудио-башня WeMM-Video — `model.audio.*`):
    /// HF-префикс `model.encoder.` подменяется этим.
    encoder_prefix: Option<String>,
}

impl WhisperWeights {
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, WhisperError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| WhisperError::Bundle(e.to_string()))?;
        let config = WhisperConfig::from_bundle(&bundle)?;
        // generation_config.json опционален; пустой дефолт допустим.
        let gen_config = GenerationConfig::from_bundle(&bundle).unwrap_or_default();
        let loader = SynBundleLoader::open(path)
            .map_err(|e| WhisperError::Load(e.to_string()))?
            .with_device(device);

        // SynBundleLoader::names() отдаёт dangling-ссылки (st дропается) — собираем
        // владеющий набор имён сами из заголовка safetensors один раз при открытии.
        let slice = bundle
            .tensors_slice()
            .map_err(|e| WhisperError::Bundle(e.to_string()))?;
        let st = SafeTensors::deserialize(slice)
            .map_err(|e| WhisperError::Load(format!("safetensors header: {e}")))?;
        let names: HashSet<String> = st.names().into_iter().map(|s| s.to_string()).collect();

        Ok(Self { bundle, loader, names, config, gen_config, device, dtype, quant: None, encoder_prefix: None })
    }

    /// Только энкодер Whisper из чужого бандла: конфиг задаётся снаружи
    /// (у WeMM-Video — `audio_config` из его config.json), тензоры
    /// `model.encoder.*` ищутся под `encoder_prefix` (`model.audio.`).
    pub fn open_encoder(
        path: impl AsRef<Path>,
        config: WhisperConfig,
        encoder_prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self, WhisperError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| WhisperError::Bundle(e.to_string()))?;
        let loader = SynBundleLoader::open(path)
            .map_err(|e| WhisperError::Load(e.to_string()))?
            .with_device(device);
        let slice = bundle
            .tensors_slice()
            .map_err(|e| WhisperError::Bundle(e.to_string()))?;
        let st = SafeTensors::deserialize(slice)
            .map_err(|e| WhisperError::Load(format!("safetensors header: {e}")))?;
        let names: HashSet<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        Ok(Self {
            bundle,
            loader,
            names,
            config,
            gen_config: GenerationConfig::default(),
            device,
            dtype,
            quant: None,
            encoder_prefix: Some(encoder_prefix.to_string()),
        })
    }

    /// Имя тензора в бандле с учётом `encoder_prefix`.
    fn resolve<'a>(&self, name: &'a str) -> std::borrow::Cow<'a, str> {
        match (&self.encoder_prefix, name.strip_prefix("model.encoder.")) {
            (Some(p), Some(rest)) => std::borrow::Cow::Owned(format!("{p}{rest}")),
            _ => std::borrow::Cow::Borrowed(name),
        }
    }

    /// Тензор по HF-имени, приведённый к compute-dtype на целевом устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor, WhisperError> {
        let name = self.resolve(name);
        synaptix_core::prof::span("load.get", || self.loader.load_to(&name, self.device, self.dtype))
            .map_err(|e| WhisperError::Load(format!("'{name}': {e}")))
    }

    /// Тензор в типе файла (без приведения к compute-dtype): вес, который сразу
    /// квантуется, не материализуется в F32 целиком.
    pub fn get_native(&self, name: &str) -> Result<Tensor, WhisperError> {
        let name = self.resolve(name);
        synaptix_core::prof::span("load.get", || self.loader.load(&name))
            .map_err(|e| WhisperError::Load(format!("'{name}': {e}")))
    }

    pub fn get_opt(&self, name: &str) -> Option<Tensor> {
        if self.contains(name) {
            self.get(name).ok()
        } else {
            None
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(self.resolve(name).as_ref())
    }

    pub fn read_aux(&self, name: &str) -> Result<Vec<u8>, WhisperError> {
        self.bundle
            .read_file(name)
            .map(|c| c.into_owned())
            .map_err(|e| WhisperError::Bundle(format!("'{name}': {e}")))
    }
}

pub fn enc_layer(i: usize, suffix: &str) -> String {
    format!("model.encoder.layers.{i}.{suffix}")
}

pub fn dec_layer(i: usize, suffix: &str) -> String {
    format!("model.decoder.layers.{i}.{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn bundle_path() -> Option<PathBuf> {
        let p = PathBuf::from("models/whisper-large-v3-turbo.syn");
        p.exists().then_some(p)
    }

    #[test]
    fn loads_config_and_names() {
        synaptix_kernels_cpu::ensure_registered();
        let Some(path) = bundle_path() else { return };
        let w = WhisperWeights::open(&path, Device::Cpu, DType::F32).expect("open bundle");

        assert_eq!(w.config.d_model, 1280);
        assert_eq!(w.config.encoder_layers, 32);
        assert_eq!(w.config.decoder_layers, 4);
        assert_eq!(w.config.num_mel_bins, 128);
        assert_eq!(w.config.vocab_size, 51866);
        assert_eq!(w.config.encoder_head_dim(), 64);

        assert!(w.contains("model.encoder.conv1.weight"));
        assert!(w.contains("model.decoder.embed_tokens.weight"));
        // q/v/out имеют bias, k — нет (особенность Whisper).
        assert!(w.contains("model.encoder.layers.0.self_attn.q_proj.bias"));
        assert!(!w.contains("model.encoder.layers.0.self_attn.k_proj.bias"));
        // tied lm_head: отдельного proj_out нет.
        assert!(!w.contains("proj_out.weight"));

        let conv1 = w.get("model.encoder.conv1.weight").expect("conv1");
        assert_eq!(conv1.dims(), &[1280, 128, 3]);
        assert_eq!(conv1.dtype(), DType::F32);
    }
}
