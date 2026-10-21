//! Адаптер на диске в формате PEFT: каталог с `adapter_model.safetensors`
//! (`base_model.model.<модуль>.lora_A.weight` `[r, K]` и `lora_B.weight`
//! `[N, r]`) и `adapter_config.json`. Такой адаптер читают transformers/PEFT,
//! llama.cpp (через convert_lora_to_gguf) и сам synaptix.

use std::collections::BTreeMap;
use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::error::other;
use crate::lora::LoraLayer;
use crate::model::TrainModel;
use crate::Result;

const WEIGHTS: &str = "adapter_model.safetensors";
const CONFIG: &str = "adapter_config.json";

fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

/// Сохранить адаптеры модели в каталог `dir` (создаётся). `base_name` —
/// откуда база (пишется в конфиг для справки).
pub fn save(model: &TrainModel<'_>, dir: &Path, base_name: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut blobs: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for l in model.layers() {
        for (suffix, t) in [("lora_A", &l.a), ("lora_B", &l.b)] {
            let bytes: Vec<u8> = host_f32(t)?.iter().flat_map(|x| x.to_le_bytes()).collect();
            blobs.insert(format!("base_model.model.{}.{suffix}.weight", l.module), (t.dims().to_vec(), bytes));
        }
    }
    let views = blobs
        .iter()
        .map(|(n, (s, b))| {
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| other(format!("{n}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let bytes = safetensors::serialize(views, None).map_err(|e| other(e.to_string()))?;
    let tmp = dir.join(format!("{WEIGHTS}.tmp"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, dir.join(WEIGHTS))?;

    let mut targets: Vec<&str> = model.cfg.targets.iter().map(|s| s.as_str()).collect();
    targets.sort();
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_name,
        "r": model.cfg.r,
        "lora_alpha": model.cfg.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": false,
        "target_modules": targets,
        "inference_mode": true,
    });
    std::fs::write(dir.join(CONFIG), serde_json::to_string_pretty(&cfg).map_err(|e| other(e.to_string()))?)?;
    Ok(())
}

/// Загрузить веса адаптера из `dir` в уже созданные слои модели (те же цели
/// и ранг). Моменты оптимизатора не восстанавливаются.
pub fn load_into(model: &mut TrainModel<'_>, dir: &Path) -> Result<()> {
    let bytes = std::fs::read(dir.join(WEIGHTS))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| other(e.to_string()))?;
    let dev = model.device();
    let read = |name: &str| -> Result<Tensor> {
        let v = st.tensor(name).map_err(|e| other(format!("{name}: {e}")))?;
        if v.dtype() != safetensors::Dtype::F32 {
            return Err(other(format!("{name}: ожидался F32")));
        }
        let data: Vec<f32> = v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Tensor::from_vec(data, v.shape().to_vec(), dev)?)
    };
    for l in model.layers_mut() {
        let a = read(&format!("base_model.model.{}.lora_A.weight", l.module))?;
        let b = read(&format!("base_model.model.{}.lora_B.weight", l.module))?;
        if a.dims() != l.a.dims() || b.dims() != l.b.dims() {
            return Err(other(format!("{}: форма адаптера не совпадает", l.module)));
        }
        let scale = l.scale;
        *l = LoraLayer::from_weights(l.module.clone(), a, b, scale)?;
    }
    Ok(())
}
