//! Чтение весов Inkling-2: HF-каталог (modelopt NVFP4/MXFP8 эксперты, BF16/F32
//! остальное) или `.syn`-бандл той же раскладки имён.
//!
//! Эксперты собираются в [`QuantWeight`] движка без перекодировки значений:
//! полубайты NVFP4 и коды MXFP8 берутся как есть, масштабы NVFP4 переставляются
//! в тайловую раскладку. Строки w13 в чекпойнте чередуются (gate, up, gate, …) —
//! здесь они разводятся в `[gate; up]`, как ждёт движок. Общий множитель NVFP4
//! modelopt (`amax/denominator` на эксперта) отдаётся отдельно: в NVFP4 движка
//! его нет, а перекодировать его в блочные E4M3 без потерь нельзя.

use std::path::Path;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_kernels_cpu::quant::dequant::{nvfp4_scale_bytes, nvfp4_scale_index, nvfp4_sf_inner_dim};

use crate::config::Inkling2Config;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
    #[error("weights: {0}")]
    Weights(String),
}

pub const LLM: &str = "model.llm";

pub struct Inkling2Weights {
    loader: Arc<SafetensorsLoader>,
    /// `.syn`-бандл (упакован [`crate::pack`]); `loader` тогда смотрит в тот же mmap.
    bundle: Option<SynBundleLoader>,
    pub config: Inkling2Config,
    pub tokenizer_json: Vec<u8>,
    pub chat_template: Option<String>,
}

/// Стопка экспертов слоя: по [`QuantWeight`] на эксперта и общий множитель
/// (1 у MXFP8).
pub struct ExpertStack {
    pub gate_up: Vec<QuantWeight>,
    pub down: Vec<QuantWeight>,
    pub alpha_gate_up: Vec<f32>,
    pub alpha_down: Vec<f32>,
}

fn werr(e: impl std::fmt::Display) -> LoadError {
    LoadError::Weights(e.to_string())
}

impl Inkling2Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Self::open_bundle(path);
        }
        let cfg = std::fs::read(path.join("config.json")).map_err(|e| LoadError::Io(format!("config.json: {e}")))?;
        let config = Inkling2Config::from_bytes(&cfg).map_err(LoadError::Config)?;
        let tokenizer_json = std::fs::read(path.join("tokenizer.json")).unwrap_or_default();
        let chat_template = std::fs::read_to_string(path.join("chat_template.jinja")).ok();
        // mtp.safetensors — отдельная голова со своими именами; основная модель её не читает.
        let shards: Vec<_> = scan_shards(path)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .into_iter()
            .filter(|p| p.file_name().is_some_and(|n| n != "mtp.safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(LoadError::Io(format!("нет .safetensors в {}", path.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards).map_err(|e| LoadError::Io(e.to_string()))?;
        Ok(Self { loader: Arc::new(loader), bundle: None, config, tokenizer_json, chat_template })
    }

    fn open_bundle(path: &Path) -> Result<Self, LoadError> {
        let b = SynBundleLoader::open(path).map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
        let cfg = b.read_file("config.json").ok_or_else(|| LoadError::Io("в бандле нет config.json".into()))?;
        let config = Inkling2Config::from_bytes(&cfg).map_err(LoadError::Config)?;
        let tokenizer_json = b.read_file("tokenizer.json").unwrap_or_default();
        let chat_template = b.read_file("chat_template.jinja").and_then(|v| String::from_utf8(v).ok());
        let loader = SafetensorsLoader::open_bundle(path, None).map_err(|e| LoadError::Io(e.to_string()))?;
        Ok(Self { loader: Arc::new(loader), bundle: Some(b), config, tokenizer_json, chat_template })
    }

    pub fn is_bundle(&self) -> bool {
        self.bundle.is_some()
    }

    pub fn contains(&self, name: &str) -> bool {
        match &self.bundle {
            Some(b) => b.contains(name),
            None => self.loader.tensor_info(name).is_some(),
        }
    }

    pub fn tensor(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor, LoadError> {
        let r = match &self.bundle {
            Some(b) => b.load_to(name, device, dtype),
            None => self.loader.load_to(name, device, dtype),
        };
        r.map_err(|e| LoadError::Weights(format!("{name}: {e}")))
    }

    /// Квантованная в бандле матрица и её общий множитель; `None` — плотная.
    pub fn quant(&self, name: &str, device: Device) -> Result<Option<(QuantWeight, f32)>, LoadError> {
        let Some(b) = &self.bundle else { return Ok(None) };
        let Some(q) = b.load_quant(name, device) else { return Ok(None) };
        let q = q.map_err(|e| LoadError::Weights(format!("{name}: {e}")))?;
        let alpha = if self.contains(&format!("{name}_alpha")) { self.host_f32(&format!("{name}_alpha"))?[0] } else { 1.0 };
        Ok(Some((q, alpha)))
    }

    /// Тензор в F32 на хосте.
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let t = self.tensor(name, Device::Cpu, DType::F32)?;
        let n = t.numel();
        let t = t.reshape(vec![n]).map_err(werr)?;
        t.to_vec1::<f32>().map_err(werr)
    }

    fn raw(&self, name: &str) -> Result<(&[u8], DType, Vec<usize>), LoadError> {
        let (b, dt, sh) = self.loader.raw_bytes(name).ok_or_else(|| LoadError::Weights(format!("нет тензора {name}")))?;
        Ok((b, dt, sh.to_vec()))
    }

    /// Эксперты слоя `layer`: NVFP4 (modelopt) или MXFP8.
    pub fn experts(&self, layer: usize, device: Device) -> Result<ExpertStack, LoadError> {
        let p = format!("{LLM}.layers.{layer}.mlp.experts");
        if let Some(b) = &self.bundle {
            let stack = |short: &str| -> Result<(Vec<QuantWeight>, Vec<f32>), LoadError> {
                let name = format!("{p}.{short}");
                let w = b
                    .load_quant_stack(&name, device)
                    .ok_or_else(|| LoadError::Weights(format!("{name}: не квант-стопка")))?
                    .map_err(|e| LoadError::Weights(format!("{name}: {e}")))?;
                Ok((w, self.host_f32(&format!("{name}_alpha"))?))
            };
            let (gate_up, alpha_gate_up) = stack("gate_up")?;
            let (down, alpha_down) = stack("down")?;
            return Ok(ExpertStack { gate_up, down, alpha_gate_up, alpha_down });
        }
        let (gate_up, alpha_gate_up) = self.expert_matrix(&format!("{p}.w13_weight"), true, device)?;
        let (down, alpha_down) = self.expert_matrix(&format!("{p}.w2_weight"), false, device)?;
        Ok(ExpertStack { gate_up, down, alpha_gate_up, alpha_down })
    }

    /// Множители alpha экспертов слоя `(gate_up, down)` из бандла — без
    /// чтения самих весов (режим экспертов в RAM). `None` — не бандл.
    pub fn expert_alpha(&self, layer: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>, LoadError> {
        if self.bundle.is_none() {
            return Ok(None);
        }
        let p = format!("{LLM}.layers.{layer}.mlp.experts");
        Ok(Some((self.host_f32(&format!("{p}.gate_up_alpha"))?, self.host_f32(&format!("{p}.down_alpha"))?)))
    }

    /// Формат экспертов слоя в бандле (NVFP4 / MXFP8).
    pub fn expert_dtype(&self, layer: usize) -> Option<DType> {
        let b = self.bundle.as_ref()?;
        Some(b.quant_kind(&format!("{LLM}.layers.{layer}.mlp.experts.gate_up"))?.dtype())
    }

    /// Стопки экспертов бандла по слоям — под закреплённое зеркало и
    /// zero-copy (`synaptix_llm_common::expert_mirror`). `None` — не бандл.
    pub fn expert_stacks(&self) -> Option<synaptix_llm_common::expert_mirror::ExpertStacks> {
        let b = self.bundle.as_ref()?;
        let keys = (0..self.config.text.num_hidden_layers)
            .map(|l| {
                let p = format!("{LLM}.layers.{l}.mlp.experts");
                Some((format!("{p}.gate_up"), format!("{p}.down")))
            })
            .collect();
        Some(synaptix_llm_common::expert_mirror::ExpertStacks::new(b.clone(), keys))
    }

    fn expert_matrix(&self, name: &str, interleaved: bool, device: Device) -> Result<(Vec<QuantWeight>, Vec<f32>), LoadError> {
        let (w, _wdt, wsh) = self.raw(name)?;
        let (s, _sdt, ssh) = self.raw(&format!("{name}_scale"))?;
        if wsh.len() != 3 || ssh.len() != 3 || wsh[0] != ssh[0] || wsh[1] != ssh[1] {
            return Err(LoadError::Weights(format!("{name}: формы {wsh:?} / {ssh:?}")));
        }
        let (e, n) = (wsh[0], wsh[1]);
        let nvfp4 = self.contains(&format!("{name}_amax"));
        let k = if nvfp4 { wsh[2] * 2 } else { wsh[2] };
        let (block, row_bytes) = if nvfp4 { (16, k / 2) } else { (32, k) };
        if ssh[2] * block != k {
            return Err(LoadError::Weights(format!("{name}: блок масштаба {} не {block}", k / ssh[2])));
        }
        let alpha = if nvfp4 {
            let amax = self.host_f32(&format!("{name}_amax"))?;
            let den = self.host_f32(&format!("{name}_outer_scale_denominator"))?;
            let d = *den.first().ok_or_else(|| LoadError::Weights(format!("{name}: пустой denominator")))?;
            if amax.len() != e || !(d > 0.0) {
                return Err(LoadError::Weights(format!("{name}: amax {} / denominator {d}", amax.len())));
            }
            amax.iter().map(|a| a / d).collect()
        } else {
            vec![1.0; e]
        };
        let sb = k / block;
        // Порядок строк: у w13 — чётные (gate), затем нечётные (up).
        let order: Vec<usize> = if interleaved {
            (0..n / 2).map(|r| 2 * r).chain((0..n / 2).map(|r| 2 * r + 1)).collect()
        } else {
            (0..n).collect()
        };
        let mut out = Vec::with_capacity(e);
        let sf = nvfp4_sf_inner_dim(k);
        for ex in 0..e {
            let wb = &w[ex * n * row_bytes..(ex + 1) * n * row_bytes];
            let sbx = &s[ex * n * sb..(ex + 1) * n * sb];
            let mut packed = vec![0u8; n * row_bytes];
            for (dst, &src) in order.iter().enumerate() {
                packed[dst * row_bytes..(dst + 1) * row_bytes].copy_from_slice(&wb[src * row_bytes..(src + 1) * row_bytes]);
            }
            let scales = if nvfp4 {
                let mut sc = vec![0u8; nvfp4_scale_bytes(n, k)];
                for (dst, &src) in order.iter().enumerate() {
                    for b in 0..sb {
                        sc[nvfp4_scale_index(dst, b, sf)] = sbx[src * sb + b];
                    }
                }
                sc
            } else {
                let mut sc = vec![0u8; n * sb];
                for (dst, &src) in order.iter().enumerate() {
                    sc[dst * sb..(dst + 1) * sb].copy_from_slice(&sbx[src * sb..(src + 1) * sb]);
                }
                sc
            };
            let pl = packed.len();
            let sl = scales.len();
            let pt = Tensor::from_vec(packed, vec![pl], device).map_err(werr)?;
            let st = Tensor::from_vec(scales, vec![sl], device).map_err(werr)?;
            let dtype = if nvfp4 { DType::NVFP4 } else { DType::MXFP8 };
            let q = QuantWeight::new(pt.storage_arc(), st.storage_arc(), dtype, n, k).map_err(werr)?;
            out.push(q);
        }
        Ok((out, alpha))
    }
}
