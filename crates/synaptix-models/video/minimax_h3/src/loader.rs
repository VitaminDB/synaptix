use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader, TensorInfo};
use synaptix_io::weights::WeightLoader;

use crate::config::{AudioVaeConfig, H3Config, H3Variant, VaeConfig};
use crate::source::{H3Component, H3Source};
use crate::H3Error;

#[derive(Debug, Clone)]
pub struct H3Paths {
    pub root: PathBuf,
    pub variant: H3Variant,
}

impl H3Paths {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, H3Error> {
        let root = root.as_ref();
        if root.join("transformer").is_dir() {
            let variant = H3Config::from_dir(root)?.variant();
            return Ok(Self { root: root.to_path_buf(), variant });
        }
        for v in [H3Variant::Fl2va, H3Variant::Ref2va] {
            let sub = root.join(v.dir_name());
            if sub.join("transformer").is_dir() {
                return Ok(Self { root: sub, variant: v });
            }
        }
        Err(H3Error::Load(format!(
            "не найден каталог варианта (FL2VA/Ref2VA) в {}",
            root.display()
        )))
    }

    pub fn open_variant(root: impl AsRef<Path>, variant: H3Variant) -> Result<Self, H3Error> {
        let root = root.as_ref();
        let sub = if root.join("transformer").is_dir() {
            root.to_path_buf()
        } else {
            root.join(variant.dir_name())
        };
        if !sub.join("transformer").is_dir() {
            return Err(H3Error::Load(format!("нет transformer/ в {}", sub.display())));
        }
        Ok(Self { root: sub, variant })
    }

    pub fn transformer_dir(&self) -> PathBuf {
        self.root.join("transformer")
    }

    pub fn video_vae_file(&self) -> PathBuf {
        self.root.join("video_vae").join("source").join("model.safetensors")
    }

    pub fn audio_vae_file(&self) -> PathBuf {
        self.root.join("audio_vae").join("model.safetensors")
    }

    pub fn text_encoder_dir(&self) -> PathBuf {
        self.root.join("text_encoder")
    }

    pub fn tokenizer_dir(&self) -> PathBuf {
        self.root.join("tokenizer")
    }

    pub fn processor_dir(&self) -> PathBuf {
        self.root.join("processor")
    }
}

pub struct LoraWeights {
    loader: SafetensorsLoader,
    strength: f32,
    device: Device,
    prefixes: Vec<String>,
    /// Имена Diffusers (`transformer_blocks.N.attn.to_q`, `ff.net.0.proj`, …) —
    /// так выложены Turbo LoRA lightx2v; иначе — имена ComfyUI/H3 (`blocks.N.attn.qkv_proj`).
    diffusers: bool,
    /// alpha из метаданных (`alpha` / `training_alpha`): добавка ·alpha/rank.
    /// Turbo v1.3: alpha 8 при ранге 128 — масштаб 0.0625; без него LoRA
    /// прибавлялась в 16 раз сильнее и ролик рассыпался в мусор.
    alpha: Option<f32>,
}

/// Имя модуля H3 → имя Diffusers (`MiniMaxH3Transformer3DModel`), как в
/// convert.py InstantX; `fc1` в Diffusers — `[value; gate]` (у H3 `[gate; value]`).
fn diffusers_name(key: &str) -> String {
    let mut n = if let Some(r) = key.strip_prefix("token_refiner.blocks.") {
        format!("token_refiner.refiner_blocks.{r}")
    } else if let Some(r) = key.strip_prefix("blocks.") {
        format!("transformer_blocks.{r}")
    } else {
        key.to_string()
    };
    n = n.replace("final_layer.adaln_proj.linear", "norm_out.linear");
    n = n.replace(".attn.out_proj", ".attn.to_out.0");
    n = n.replace(".mlp.fc2", ".ff.net.2");
    n.replace(".mlp.fc1", ".ff.net.0.proj")
}

impl LoraWeights {
    pub fn open(path: impl AsRef<Path>, device: Device, strength: f32) -> Result<Self, H3Error> {
        let loader = SafetensorsLoader::open(path.as_ref())
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        let mut prefixes = vec![String::new()];
        let names = loader.names();
        for p in ["transformer.", "diffusion_model.", "model.diffusion_model."] {
            if names.iter().any(|n| n.starts_with(p)) {
                prefixes.push(p.to_string());
            }
        }
        let diffusers = names.iter().any(|n| n.contains("transformer_blocks.") || n.contains(".attn.to_q."));
        let md = loader.metadata();
        let alpha = ["alpha", "training_alpha", "lora_alpha"]
            .iter()
            .find_map(|k| md.get(*k).and_then(|v| v.trim().parse::<f32>().ok()))
            .filter(|a| *a > 0.0);
        // ComfyUI-сборка qkv (блочно-диагональная B, rank·3) хранит alpha·3 и
        // `training_scale` — масштаб берётся прямо из него.
        let alpha = match md.get("training_scale").and_then(|v| v.trim().parse::<f32>().ok()) {
            Some(sc) if sc > 0.0 && md.contains_key("qkv_fusion") => Some(-sc),
            _ => alpha,
        };
        // Turbo LoRA для 768p (lightx2v v1.x) учены со сдвигом сигмы 6/3, а не
        // 12/3 модели; без явного --sigma-shift-* — их сдвиг.
        let fname = path.as_ref().file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        if fname.contains("turbo") && fname.contains("768p") {
            let (v, a) = crate::runtime::sigma_shift();
            if v.is_none() {
                crate::runtime::set_sigma_shift(Some(6.0), a.or(Some(3.0)));
                eprintln!("[h3] Turbo 768p: сдвиг сигмы видео 6, звука {}", a.unwrap_or(3.0));
            }
        }
        eprintln!(
            "[h3] LoRA: {} имена, alpha {}",
            if diffusers { "Diffusers" } else { "ComfyUI/H3" },
            match alpha {
                Some(a) if a < 0.0 => format!("— training_scale {}", -a),
                Some(a) => format!("{a} (масштаб alpha/rank)"),
                None => "нет (масштаб 1)".into(),
            }
        );
        Ok(Self { loader, strength, device, prefixes, diffusers, alpha })
    }

    fn pair_delta(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        let Some((ka, kb)) = self.find_pair(key) else {
            return Ok(None);
        };
        let a = self
            .loader
            .load_to(&ka, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("lora {ka}: {e}")))?;
        let b = self
            .loader
            .load_to(&kb, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("lora {kb}: {e}")))?;
        let rank = a.dims()[0] as f32;
        let scale = match self.alpha {
            Some(a) if a < 0.0 => -a, // готовый training_scale
            Some(a) => a / rank,
            None => 1.0,
        };
        let bs = b.mul_scalar(self.strength * scale)?;
        Ok(Some(bs.matmul(&a)?))
    }

    /// Добавка к общему `qkv_proj` в раскладке весов H3 — по головам
    /// `[heads, (q,k,v), head_dim]`. LoRA хранят `[все q; все k; все v]`
    /// (ComfyUI переставляет qkv при загрузке) или три `to_q/to_k/to_v`
    /// (Diffusers); прежде добавка ComfyUI ложилась на веса без перестановки.
    pub fn delta_qkv(&self, model_key: &str, heads: usize, head_dim: usize, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        let fused = if self.diffusers {
            let base = diffusers_name(model_key.trim_end_matches(".qkv_proj"));
            let mut parts = Vec::with_capacity(3);
            for s in ["to_q", "to_k", "to_v"] {
                match self.pair_delta(&format!("{base}.{s}"), dtype)? {
                    Some(d) => parts.push(d),
                    None => return Ok(None),
                }
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 0)?
        } else {
            match self.pair_delta(model_key, dtype)? {
                Some(d) => d,
                None => return Ok(None),
            }
        };
        let k = fused.dims()[1];
        Ok(Some(
            fused
                .reshape(vec![3, heads, head_dim, k])?
                .permute([1, 0, 2, 3])?
                .contiguous()?
                .reshape(vec![3 * heads * head_dim, k])?,
        ))
    }

    pub fn strength(&self) -> f32 {
        self.strength
    }

    fn find_pair(&self, key: &str) -> Option<(String, String)> {
        let names = self.loader.names();
        for p in &self.prefixes {
            for (sa, sb) in [
                ("lora_A.weight", "lora_B.weight"),
                ("lora_down.weight", "lora_up.weight"),
                ("lora_A.default.weight", "lora_B.default.weight"),
            ] {
                let ka = format!("{p}{key}.{sa}");
                if names.iter().any(|n| *n == ka.as_str()) {
                    return Some((ka, format!("{p}{key}.{sb}")));
                }
            }
        }
        None
    }

    pub fn delta(&self, model_key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        if !self.diffusers {
            return self.pair_delta(model_key, dtype);
        }
        let Some(d) = self.pair_delta(&diffusers_name(model_key), dtype)? else {
            return Ok(None);
        };
        if model_key.ends_with(".mlp.fc1") {
            // Diffusers [value; gate] → H3 [gate; value]
            let half = d.dims()[0] / 2;
            let (v, g) = (d.narrow(0, 0, half)?.contiguous()?, d.narrow(0, half, half)?.contiguous()?);
            return Ok(Some(Tensor::cat(&[&g, &v], 0)?.contiguous()?));
        }
        Ok(Some(d))
    }

    /// Пара `(A [r,K], B·масштаб [N,r])` для прибавки при счёте: `y += (x·Aᵀ)·Bᵀ`.
    /// Вливать LoRA в квантованные веса нельзя: добавка с масштабом alpha/rank
    /// меньше шага MXFP8/NVFP4 и при повторном кванте почти вся пропадает.
    fn raw_pair(&self, key: &str, dtype: DType) -> Result<Option<(Tensor, Tensor)>, H3Error> {
        let Some((ka, kb)) = self.find_pair(key) else {
            return Ok(None);
        };
        let a = self.loader.load_to(&ka, self.device, dtype).map_err(|e| H3Error::Load(format!("lora {ka}: {e}")))?;
        let b = self.loader.load_to(&kb, self.device, dtype).map_err(|e| H3Error::Load(format!("lora {kb}: {e}")))?;
        let rank = a.dims()[0] as f32;
        let scale = match self.alpha {
            Some(al) if al < 0.0 => -al,
            Some(al) => al / rank,
            None => 1.0,
        };
        Ok(Some((a, b.mul_scalar(self.strength * scale)?)))
    }

    /// Пара для модуля H3 `model_key` (`blocks.N.attn.out_proj`, `…mlp.fc1`, …).
    pub fn pair(&self, model_key: &str, dtype: DType) -> Result<Option<(Tensor, Tensor)>, H3Error> {
        if !self.diffusers {
            return self.raw_pair(model_key, dtype);
        }
        let Some((a, b)) = self.raw_pair(&diffusers_name(model_key), dtype)? else {
            return Ok(None);
        };
        if model_key.ends_with(".mlp.fc1") {
            // Diffusers [value; gate] → H3 [gate; value]
            let half = b.dims()[0] / 2;
            let (v, g) = (b.narrow(0, 0, half)?.contiguous()?, b.narrow(0, half, half)?.contiguous()?);
            return Ok(Some((a, Tensor::cat(&[&g, &v], 0)?.contiguous()?)));
        }
        Ok(Some((a, b)))
    }

    /// Пары q, k, v для `…attn.qkv_proj` (строки B — `[голова, dim]`, как у
    /// проекций H3 после деления qkv по головам).
    pub fn qkv_pairs(&self, model_key: &str, dtype: DType) -> Result<Option<[(Tensor, Tensor); 3]>, H3Error> {
        if self.diffusers {
            let base = diffusers_name(model_key.trim_end_matches(".qkv_proj"));
            let q = self.raw_pair(&format!("{base}.to_q"), dtype)?;
            let k = self.raw_pair(&format!("{base}.to_k"), dtype)?;
            let v = self.raw_pair(&format!("{base}.to_v"), dtype)?;
            return Ok(match (q, k, v) {
                (Some(q), Some(k), Some(v)) => Some([q, k, v]),
                _ => None,
            });
        }
        // ComfyUI: B — [q; k; v] (блочно-диагональная), A общая.
        let Some((a, b)) = self.raw_pair(model_key, dtype)? else {
            return Ok(None);
        };
        let inner = b.dims()[0] / 3;
        let part = |i: usize| -> Result<(Tensor, Tensor), H3Error> { Ok((a.clone(), b.narrow(0, i * inner, inner)?.contiguous()?)) };
        Ok(Some([part(0)?, part(1)?, part(2)?]))
    }

    /// Сколько модулей H3 эта LoRA задевает (для журнала: 0 — имена не сошлись).
    pub fn modules(&self) -> usize {
        self.loader.names().iter().filter(|n| n.contains("lora_A") || n.contains("lora_down")).count()
    }

    pub fn diff(&self, model_key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        let names = self.loader.names();
        for p in &self.prefixes {
            let k = format!("{p}{model_key}.diff");
            if names.iter().any(|n| *n == k.as_str()) {
                let t = self
                    .loader
                    .load_to(&k, self.device, dtype)
                    .map_err(|e| H3Error::Load(format!("lora {k}: {e}")))?;
                return Ok(Some(t.mul_scalar(self.strength)?));
            }
        }
        Ok(None)
    }
}

pub struct H3Checkpoint {
    loader: SafetensorsLoader,
    pub config: H3Config,
    pub source: H3Source,
    device: Device,
    dtype: DType,
    lora: Option<Arc<LoraWeights>>,
}

impl H3Checkpoint {
    pub fn open(paths: H3Paths, device: Device, dtype: DType) -> Result<Self, H3Error> {
        Self::open_source(H3Source::Dir { root: paths.root, variant: paths.variant }, device, dtype)
    }

    /// Открыть DiT из каталога либо `.syn`-бандла — веса берутся из компонента
    /// `transformer` (в бандле — zero-copy срез mmap, без распаковки).
    pub fn open_source(source: H3Source, device: Device, dtype: DType) -> Result<Self, H3Error> {
        let config = H3Config::from_source(&source)?;
        let loader = source.loader(H3Component::Transformer, device)?;
        Ok(Self { loader, config, source, device, dtype, lora: None })
    }

    pub fn open_root(root: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, H3Error> {
        let root = root.as_ref();
        if root.is_file() {
            return Self::open_source(H3Source::open_bundle(root, H3Variant::Fl2va)?, device, dtype);
        }
        Self::open(H3Paths::open(root)?, device, dtype)
    }

    pub fn with_lora(mut self, lora: Arc<LoraWeights>) -> Self {
        self.lora = Some(lora);
        self
    }

    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    pub fn lora_pair(&self, key: &str, dtype: DType) -> Result<Option<(Tensor, Tensor)>, H3Error> {
        match &self.lora {
            Some(l) => l.pair(key, dtype),
            None => Ok(None),
        }
    }

    pub fn lora_qkv_pairs(&self, key: &str, dtype: DType) -> Result<Option<[(Tensor, Tensor); 3]>, H3Error> {
        match &self.lora {
            Some(l) => l.qkv_pairs(key, dtype),
            None => Ok(None),
        }
    }

    pub fn lora_delta_qkv(&self, key: &str, heads: usize, head_dim: usize, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        match &self.lora {
            Some(l) => l.delta_qkv(key, heads, head_dim, dtype),
            None => Ok(None),
        }
    }

    pub fn lora_delta(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        match &self.lora {
            Some(l) => l.delta(key, dtype),
            None => Ok(None),
        }
    }

    pub fn lora_diff(&self, key: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        match &self.lora {
            Some(l) => l.diff(key, dtype),
            None => Ok(None),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    pub fn get(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn get_raw(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load(name)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn get_as(&self, name: &str, dtype: DType) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.loader.names()
    }

    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.loader.tensor_info(name)
    }

    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.loader.infos()
    }

    pub fn view_on(&self, device: Device) -> Self {
        Self {
            loader: self.loader.clone_with_device(device),
            config: self.config.clone(),
            source: self.source.clone(),
            device,
            dtype: self.dtype,
            lora: self.lora.clone(),
        }
    }

    pub fn shard_bytes(&self) -> Vec<&[u8]> {
        self.loader.shard_bytes()
    }

    pub fn raw_bytes(&self, name: &str) -> Option<(&[u8], DType, &[usize])> {
        self.loader.raw_bytes(name)
    }

    /// Формат готового кванта веса в квантованном бандле; `None` — плотный.
    pub fn quant_dtype(&self, name: &str) -> Option<DType> {
        self.loader.quant_dtype(name)
    }

    /// Готовый квант-вес на устройство чекпойнта.
    pub fn load_quant(&self, name: &str) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, SynaptixError>> {
        self.loader.load_quant(name, self.device).map(|r| r.map_err(|e| SynaptixError::Other(format!("load '{name}': {e}"))))
    }

    pub fn source(&self) -> &H3Source {
        &self.source
    }

    pub fn vae_config(&self) -> Result<VaeConfig, H3Error> {
        VaeConfig::from_source(&self.source)
    }

    pub fn audio_vae_config(&self) -> Result<AudioVaeConfig, H3Error> {
        AudioVaeConfig::from_source(&self.source)
    }
}

pub struct ComponentLoader {
    loader: SafetensorsLoader,
    device: Device,
}

impl ComponentLoader {
    /// Тип тензора в файле (без загрузки).
    pub fn tensor_dtype(&self, name: &str) -> Option<DType> {
        self.loader.tensor_info(name).map(|i| i.dtype)
    }

    /// Копия с общим mmap (для стриминга весов из другого потока).
    pub fn duplicate(&self) -> Self {
        Self { loader: self.loader.clone_with_device(self.device), device: self.device }
    }

    pub fn open_file(path: impl AsRef<Path>, device: Device) -> Result<Self, H3Error> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(H3Error::Load(format!("не найден: {}", path.display())));
        }
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device })
    }

    /// Подмодель (VAE / audio-VAE / энкодер) из каталога или `.syn`-бандла.
    pub fn open_component(
        source: &H3Source,
        component: H3Component,
        device: Device,
    ) -> Result<Self, H3Error> {
        let loader = source.loader(component, device)?;
        Ok(Self { loader, device })
    }

    pub fn open_dir(dir: impl AsRef<Path>, device: Device) -> Result<Self, H3Error> {
        let dir = dir.as_ref();
        let shards = scan_shards(dir).map_err(|e| H3Error::Load(format!("{}: {e}", dir.display())))?;
        if shards.is_empty() {
            return Err(H3Error::Load(format!("нет safetensors в {}", dir.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| H3Error::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get(&self, name: &str) -> Result<Tensor, H3Error> {
        self.loader
            .load(name)
            .map_err(|e| H3Error::Load(format!("load '{name}': {e}")))
    }

    pub fn get_as(&self, name: &str, dtype: DType) -> Result<Tensor, H3Error> {
        self.loader
            .load_to(name, self.device, dtype)
            .map_err(|e| H3Error::Load(format!("load '{name}': {e}")))
    }

    pub fn opt(&self, name: &str, dtype: DType) -> Result<Option<Tensor>, H3Error> {
        if self.contains(name) {
            Ok(Some(self.get_as(name, dtype)?))
        } else {
            Ok(None)
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.loader.names()
    }

    pub fn infos(&self) -> impl Iterator<Item = (&str, DType, &[usize])> {
        self.loader.infos()
    }

    pub fn total_bytes(&self) -> usize {
        self.loader
            .infos()
            .map(|(_, dt, shape)| dt.bytes_for_numel(shape.iter().product::<usize>()))
            .sum()
    }
}
