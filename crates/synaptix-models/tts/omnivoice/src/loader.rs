//! Загрузка весов OmniVoice (`lm`-компонент: Qwen3-backbone + `audio_embeddings`
//! + `audio_heads`, embed tied) для backbone-forward.
//!
//! Гейт-путь читает распакованный HF-снапшот `model.safetensors` через
//! `synaptix-io` SafetensorsLoader (mmap). Для `.syn`-бандла позже — компонентное
//! чтение `tensors:lm` (см. SPEC.md «Веса»).

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::OmniVoiceError;

/// safetensors-dtype → synaptix DType.
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

/// Прочитать один тензор из распарсенного safetensors-блоба на `device` в
/// `want`-dtype (без mmap — байты приходят из `.syn`-бандла слайсом).
fn tensor_from_st(
    st: &SafeTensors,
    name: &str,
    device: Device,
    want: DType,
) -> Result<Tensor, OmniVoiceError> {
    let view = st
        .tensor(name)
        .map_err(|e| OmniVoiceError::Load(format!("st tensor '{name}': {e}")))?;
    let src_dtype = st_dtype(view.dtype())
        .ok_or_else(|| OmniVoiceError::Load(format!("st dtype {:?} unsupported", view.dtype())))?;
    let t = Tensor::from_raw_slice(view.data(), view.shape().to_vec(), src_dtype, device)
        .map_err(|e| OmniVoiceError::Load(format!("from_raw_slice '{name}': {e}")))?;
    if t.dtype() == want {
        Ok(t)
    } else {
        t.to_dtype(want)
            .map_err(|e| OmniVoiceError::Load(format!("cast '{name}': {e}")))
    }
}

/// Веса `lm`-компонента OmniVoice (F32, на одном устройстве).
pub struct OmniVoiceLmWeights {
    tensors: HashMap<String, Tensor>,
    pub device: Device,
    pub dtype: DType,
}

impl OmniVoiceLmWeights {
    /// Загрузить один `model.safetensors` (lm-компонент HF-снапшота) на `device`
    /// в `dtype` (F32 для bit-exact-гейта).
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, OmniVoiceError> {
        let path = path.as_ref();
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| OmniVoiceError::Load(format!("open {}: {e}", path.display())))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            // codebook_layer_offsets — I64-буфер, не вес; читаем как есть отдельно.
            let want = if name == "codebook_layer_offsets" {
                DType::I64
            } else {
                dtype
            };
            let t = loader
                .load_to(&name, device, want)
                .map_err(|e| OmniVoiceError::Load(format!("load '{name}': {e}")))?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    /// Загрузить lm-компонент из safetensors-байтов (`.syn` `tensors:lm`).
    pub fn load_safetensors_bytes(
        bytes: &[u8],
        device: Device,
        dtype: DType,
    ) -> Result<Self, OmniVoiceError> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| OmniVoiceError::Load(format!("deserialize lm st: {e}")))?;
        let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let want = if name == "codebook_layer_offsets" {
                DType::I64
            } else {
                dtype
            };
            let t = tensor_from_st(&st, &name, device, want)?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, OmniVoiceError> {
        self.tensors
            .get(name)
            .ok_or_else(|| OmniVoiceError::Load(format!("missing tensor '{name}'")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Какие тензоры codec'а грузим. Decode-путь: `quantizer.quantizers.{i}.*`,
/// `fc2.*`, `acoustic_decoder.*`. Encode-путь (voice-clone): `semantic_model.*`
/// (HuBERT), `encoder_semantic.*`, `acoustic_encoder.*`, `fc.*`,
/// `quantizer.quantizers.{i}.{project_in,codebook.embed}` (уже покрыт).
/// `decoder_semantic`/`fc1` не используются ни в одной стадии. Буферы codebook
/// (`inited`/`cluster_size`/`embed_avg`) пропускаем.
fn codec_tensor_keep(name: &str) -> bool {
    if name.ends_with(".inited")
        || name.ends_with(".cluster_size")
        || name.ends_with(".embed_avg")
    {
        return false;
    }
    name.starts_with("quantizer.quantizers.")
        || name.starts_with("fc.")
        || name.starts_with("fc2.")
        || name.starts_with("acoustic_decoder.")
        || name.starts_with("acoustic_encoder.")
        || name.starts_with("encoder_semantic.")
        || name.starts_with("semantic_model.")
}

/// Веса `codec`-компонента OmniVoice (HiggsAudioV2 neural codec, F32, на одном
/// устройстве). Decode-путь: `quantizer.quantizers.{i}.*` (RVQ codebook +
/// project_out), `fc2.*` (fusion → acoustic), `acoustic_decoder.*` (DAC decoder).
/// Encode-путь (voice-clone): `semantic_model.*` (HuBERT), `encoder_semantic.*`,
/// `acoustic_encoder.*`, `fc.*`, `quantizer.*.{project_in,codebook.embed}`.
/// Фильтр — `codec_tensor_keep`.
pub struct OmniVoiceCodecWeights {
    tensors: HashMap<String, Tensor>,
    pub device: Device,
    pub dtype: DType,
}

impl OmniVoiceCodecWeights {
    /// Загрузить `audio_tokenizer/model.safetensors` (codec-компонент) на
    /// `device` в `dtype` (F32 для гейта). Грузит только decode-path тензоры
    /// (RVQ/fc2/acoustic_decoder) — semantic/encoder для decode не нужны.
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, OmniVoiceError> {
        let path = path.as_ref();
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| OmniVoiceError::Load(format!("open {}: {e}", path.display())))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            if !codec_tensor_keep(&name) {
                continue;
            }
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| OmniVoiceError::Load(format!("load '{name}': {e}")))?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    /// Загрузить codec-компонент из safetensors-байтов (`.syn` `tensors:codec`).
    /// Тот же фильтр decode-path тензоров, что и `load_safetensors`.
    pub fn load_safetensors_bytes(
        bytes: &[u8],
        device: Device,
        dtype: DType,
    ) -> Result<Self, OmniVoiceError> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| OmniVoiceError::Load(format!("deserialize codec st: {e}")))?;
        let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            if !codec_tensor_keep(&name) {
                continue;
            }
            let t = tensor_from_st(&st, &name, device, dtype)?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, OmniVoiceError> {
        self.tensors
            .get(name)
            .ok_or_else(|| OmniVoiceError::Load(format!("missing codec tensor '{name}'")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
