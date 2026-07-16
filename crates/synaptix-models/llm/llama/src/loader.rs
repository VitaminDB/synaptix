use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::{LlamaConfig, QuantConfig};

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

/// Все веса в dense-виде под стандартными HF-именами
/// (`model.layers.{l}.self_attn.q_proj.weight`, ...). MLX affine int4-веса
/// дектвантятся при загрузке (см. [`dequant_mlx_affine`]) — модель не знает о
/// кванте источника и работает с обычным F16/BF16.
pub struct LlamaWeights {
    pub config: LlamaConfig,
    pub tensors: HashMap<String, Tensor>,
    pub device: Device,
    pub dtype: DType,
}

impl LlamaWeights {
    /// Загружает из HF/MLX-директории (`config.json` + `model.safetensors[.index.json]`).
    /// `dtype` — целевой dtype dense-весов (compute). При `config.quantization`
    /// заданном веса проекций хранятся как affine int4 и реконструируются в `dtype`.
    pub fn load(dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = LlamaConfig::from_hf_json(dir.join("config.json"))
            .map_err(|e| LoadError::Config(e.to_string()))?;

        let shards = resolve_shards(dir)?;
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);

        let all: HashSet<String> =
            loader.names().into_iter().map(|s| s.to_string()).collect();
        let quant = config.quantization;

        let mut tensors = HashMap::new();
        for name in &all {
            // Метаданные кванта потребляются вместе с их `.weight`.
            if name.ends_with(".scales") || name.ends_with(".biases") {
                continue;
            }
            if !name.ends_with(".weight") {
                // На всякий случай — прочие тензоры (bias и т.п.) грузим как есть.
                let t = loader
                    .load_to(name, device, dtype)
                    .map_err(|e| LoadError::Io(format!("load '{name}': {e}")))?;
                tensors.insert(name.clone(), t);
                continue;
            }
            let base = &name[..name.len() - ".weight".len()];
            let scales_key = format!("{base}.scales");
            let dense = if all.contains(&scales_key) {
                let qc = quant.ok_or_else(|| {
                    LoadError::Parse(format!(
                        "'{name}' has .scales but config.quantization missing"
                    ))
                })?;
                let packed = loader
                    .load(name)
                    .map_err(|e| LoadError::Io(format!("load '{name}': {e}")))?;
                let scales = loader
                    .load_to(&scales_key, device, DType::F32)
                    .map_err(|e| LoadError::Io(format!("load '{scales_key}': {e}")))?;
                let biases_key = format!("{base}.biases");
                let biases = loader
                    .load_to(&biases_key, device, DType::F32)
                    .map_err(|e| LoadError::Io(format!("load '{biases_key}': {e}")))?;
                dequant_mlx_affine(&packed, &scales, &biases, qc, dtype, device)
                    .map_err(|e| LoadError::Dequant(format!("{name}: {e}")))?
            } else {
                loader
                    .load_to(name, device, dtype)
                    .map_err(|e| LoadError::Io(format!("load '{name}': {e}")))?
            };
            tensors.insert(name.clone(), dense);
        }
        Ok(Self { config, tensors, device, dtype })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, LoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| LoadError::MissingKey(name.to_string()))
    }

    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

impl synaptix_llm_common::WeightSource for LlamaWeights {
    fn tensor(
        &self,
        key: &str,
        _device: Device,
        dtype: DType,
    ) -> Result<Tensor, synaptix_llm_common::ModelError> {
        let t = self
            .get(key)
            .map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))?;
        if t.dtype() == dtype {
            Ok(t.clone())
        } else {
            t.to_dtype(dtype)
                .map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }
}

/// Деквант MLX affine int`bits` → dense `out_dtype`.
///
/// Раскладка MLX: `packed` U32 `[N, K/(32/bits)]` — `32/bits` значений на слово,
/// младшие биты = младший индекс по K. `scales`/`biases` F32 `[N, K/group_size]`.
/// Формула: `w[n,k] = code(n,k) * scale[n, k/group_size] + bias[n, k/group_size]`.
///
/// Считается на CPU (rayon по строкам) в F32, затем кастуется в `out_dtype` и
/// переносится на `device`. По одному весу транзитом — пиковая память = один
/// dense-вес, а не вся модель.
pub fn dequant_mlx_affine(
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    qc: QuantConfig,
    out_dtype: DType,
    device: Device,
) -> Result<Tensor, String> {
    if qc.bits == 0 || 32 % qc.bits != 0 {
        return Err(format!("unsupported bits={} (must divide 32)", qc.bits));
    }
    if packed.dtype() != DType::U32 {
        return Err(format!("packed weight must be U32, got {:?}", packed.dtype()));
    }
    let dims = packed.dims();
    if dims.len() != 2 {
        return Err(format!("packed weight must be 2D [N, K/pack], got {dims:?}"));
    }
    let n = dims[0];
    let k_packed = dims[1];
    let vals_per_word = 32 / qc.bits;
    let k = k_packed * vals_per_word;
    let group_size = qc.group_size;
    if group_size == 0 || k % group_size != 0 {
        return Err(format!("K={k} not divisible by group_size={group_size}"));
    }
    let groups = k / group_size;
    if scales.dims() != [n, groups] {
        return Err(format!(
            "scales shape {:?} != [N={n}, groups={groups}]",
            scales.dims()
        ));
    }
    if biases.dims() != [n, groups] {
        return Err(format!(
            "biases shape {:?} != [N={n}, groups={groups}]",
            biases.dims()
        ));
    }

    let wv = packed
        .flatten_all()
        .and_then(|t| t.to_vec1::<u32>())
        .map_err(|e| format!("read packed: {e}"))?;
    let sc = scales
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| format!("read scales: {e}"))?;
    let bs = biases
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| format!("read biases: {e}"))?;

    let mask: u32 = (1u32 << qc.bits) - 1;
    let bits = qc.bits as u32;
    let mut out = vec![0.0_f32; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(row, orow)| {
        let w_base = row * k_packed;
        let g_base = row * groups;
        for kp in 0..k_packed {
            let word = wv[w_base + kp];
            let k0 = kp * vals_per_word;
            for p in 0..vals_per_word {
                let code = ((word >> (bits * p as u32)) & mask) as f32;
                let kidx = k0 + p;
                let g = kidx / group_size;
                orow[kidx] = code * sc[g_base + g] + bs[g_base + g];
            }
        }
    });

    let dense = Tensor::from_vec(out, (n, k), Device::Cpu)
        .map_err(|e| format!("from_vec: {e}"))?;
    let dense = if out_dtype != DType::F32 {
        dense.to_dtype(out_dtype).map_err(|e| format!("to_dtype: {e}"))?
    } else {
        dense
    };
    if device != Device::Cpu {
        dense.to_device(device).map_err(|e| format!("to_device: {e}"))
    } else {
        Ok(dense)
    }
}

fn resolve_shards(dir: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let single = dir.join("model.safetensors");
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let bytes = std::fs::read(&index_path)
            .map_err(|e| LoadError::Io(format!("read index: {e}")))?;
        let idx: ShardIndex = serde_json::from_slice(&bytes)
            .map_err(|e| LoadError::Parse(format!("parse index: {e}")))?;
        let mut shards: Vec<PathBuf> = idx
            .weight_map
            .values()
            .map(|s| dir.join(s))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(LoadError::Io("empty shard index".into()));
        }
        return Ok(shards);
    }
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(LoadError::Io(format!(
        "no safetensors / index in {}",
        dir.display()
    )))
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("config: {0}")]
    Config(String),
    #[error("dequant: {0}")]
    Dequant(String),
    #[error("missing tensor: {0}")]
    MissingKey(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn llama_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/mlx-community/Llama-3.2-1B-Instruct-4bit");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    #[test]
    fn dequant_known_pattern() {
        synaptix_kernels_cpu::ensure_registered();
        // N=1, K=8, group_size=8, bits=4 → один u32 = 8 nibbles 0..7, scale=2, bias=-1.
        // code p → значение в позиции p*4. word = 7<<28|6<<24|...|0 = 0x76543210.
        let packed = Tensor::from_vec(vec![0x76543210u32], (1usize, 1), Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![2.0f32], (1usize, 1), Device::Cpu).unwrap();
        let biases = Tensor::from_vec(vec![-1.0f32], (1usize, 1), Device::Cpu).unwrap();
        let qc = QuantConfig { group_size: 8, bits: 4 };
        let out = dequant_mlx_affine(&packed, &scales, &biases, qc, DType::F32, Device::Cpu).unwrap();
        assert_eq!(out.dims(), &[1, 8]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // code k → k*2 - 1
        for k in 0..8 {
            assert!((v[k] - (k as f32 * 2.0 - 1.0)).abs() < 1e-6, "k={k} got {}", v[k]);
        }
    }

    #[test]
    fn loads_mlx_llama_if_present() {
        let Some(dir) = llama_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = LlamaWeights::load(&dir, Device::Cpu, DType::F16).expect("load weights");
        assert_eq!(w.config.num_hidden_layers, 16);
        let emb = w.get("model.embed_tokens.weight").unwrap();
        assert_eq!(emb.dims(), &[w.config.vocab_size, w.config.hidden_size]);
        assert_eq!(emb.dtype(), DType::F16);

        let q0 = w.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q0.dims(), &[w.config.q_total_dim(), w.config.hidden_size]);
        let k0 = w.get("model.layers.0.self_attn.k_proj.weight").unwrap();
        assert_eq!(k0.dims(), &[w.config.kv_total_dim(), w.config.hidden_size]);
        // Норм-вес не квантован — остаётся dense.
        let n0 = w.get("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(n0.dims(), &[w.config.hidden_size]);
    }
}
