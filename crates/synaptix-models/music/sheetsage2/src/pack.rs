//! Упаковка релиза SheetSage2 в компонент `.syn`-бандла.
//!
//! Релиз — адаптер: `model.safetensors` SheetSage2 держит LoRA-адаптеры
//! внимания, декодер и смешивание слоёв, а энкодер — отдельный чекпойнт
//! MERT-v2-FullSong. Здесь адаптеры вливаются в веса MERT так же, как это
//! делает `merge_lora` релиза (`W += B·A · α/r` в F32), и получается
//! самостоятельный компонент — раскладка `save_pretrained` релиза
//! (`encoder.*`, `decoder.*`, `token_embedding.weight`, …).
//!
//! Хранение: матрицы linear и свёрток — в BF16 (эталонный режим релиза —
//! автокаст BF16, он всё равно округляет их до BF16 перед умножением), всё
//! остальное — в F32: эмбеддинги токенов и позиций, мел-фронтенд, нормы,
//! смещения, GRN и веса смешивания слоёв.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use synaptix_bundle::cdir::FileTag;
use synaptix_bundle::BundleEditor;

use crate::{SheetError, COMPONENT};

const PROJECTIONS: [&str; 4] = ["query_proj", "key_proj", "value_proj", "out_proj"];

#[derive(Debug, Clone, Default)]
pub struct PackStats {
    pub tensors: usize,
    pub merged: usize,
    pub bf16_tensors: usize,
    pub payload_bytes: u64,
}

struct Owned {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn map(path: &Path) -> Result<memmap2::Mmap, SheetError> {
    let f = std::fs::File::open(path).map_err(|e| SheetError::Load(format!("{}: {e}", path.display())))?;
    unsafe { memmap2::Mmap::map(&f) }.map_err(|e| SheetError::Load(format!("{}: {e}", path.display())))
}

fn to_f32(st: &SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>), SheetError> {
    let t = st.tensor(name).map_err(|e| SheetError::Load(format!("`{name}`: {e}")))?;
    let shape = t.shape().to_vec();
    let data = match t.dtype() {
        Dtype::F32 => t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        Dtype::BF16 => t
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::F16 => t.data().chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        other => return Err(SheetError::Load(format!("`{name}`: тип {other:?} не ожидался"))),
    };
    Ok((shape, data))
}

/// Хранить ли тензор в F32 (иначе — BF16).
fn keep_f32(name: &str, rank: usize) -> bool {
    rank <= 1
        || name == "token_embedding.weight"
        || name == "decoder.embed_positions.weight"
        || name.starts_with("encoder.feature_extractor.")
        || name.contains("pointwise_block.3.")
}

fn encode(name: &str, shape: Vec<usize>, values: &[f32]) -> Owned {
    if keep_f32(name, shape.len()) {
        Owned { dtype: Dtype::F32, shape, data: values.iter().flat_map(|v| v.to_le_bytes()).collect() }
    } else {
        Owned {
            dtype: Dtype::BF16,
            shape,
            data: values.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
        }
    }
}

/// `W += B·A · scale` (F32, построчно в несколько потоков).
fn merge_lora(w: &mut [f32], a: &[f32], b: &[f32], rows: usize, cols: usize, rank: usize, scale: f32) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (ci, chunk) in w.chunks_mut(per * cols).enumerate() {
            scope.spawn(move || {
                let mut acc = vec![0f32; cols];
                for (r, row) in chunk.chunks_mut(cols).enumerate() {
                    let gr = ci * per + r;
                    acc.fill(0.0);
                    for k in 0..rank {
                        let bv = b[gr * rank + k];
                        let arow = &a[k * cols..(k + 1) * cols];
                        for (dst, &av) in acc.iter_mut().zip(arow) {
                            *dst += bv * av;
                        }
                    }
                    for (dst, &u) in row.iter_mut().zip(&acc) {
                        *dst += u * scale;
                    }
                }
            });
        }
    });
}

fn sha256_file(path: &Path) -> Result<String, SheetError> {
    let mut f = std::fs::File::open(path).map_err(|e| SheetError::Load(format!("{}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| SheetError::Load(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Собрать safetensors-payload компонента. `mert_dir` не нужен, если релиз уже
/// в формате `merged`.
pub fn build_component(
    sheetsage_dir: &Path,
    mert_dir: Option<&Path>,
    progress: &mut dyn FnMut(&str),
) -> Result<(Vec<u8>, serde_json::Value, PackStats), SheetError> {
    let config_bytes = std::fs::read(sheetsage_dir.join("config.json"))
        .map_err(|e| SheetError::Load(format!("config.json SheetSage2: {e}")))?;
    let mut config: serde_json::Value =
        serde_json::from_slice(&config_bytes).map_err(|e| SheetError::Config(e.to_string()))?;
    crate::config::SheetSage2Config::from_json(&config_bytes)?;
    let format = config.get("weights_format").and_then(|v| v.as_str()).unwrap_or("adapter").to_string();
    let rank = config.get("lora_rank").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let alpha = config.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(128.0) as f32;
    let scale = alpha / rank as f32;

    let ss_map = map(&sheetsage_dir.join("model.safetensors"))?;
    let ss = SafeTensors::deserialize(&ss_map).map_err(|e| SheetError::Load(e.to_string()))?;
    let mut out: BTreeMap<String, Owned> = BTreeMap::new();
    let mut stats = PackStats::default();

    for name in ss.names() {
        if name.starts_with("adapter.") {
            continue;
        }
        let (shape, values) = to_f32(&ss, name)?;
        out.insert(name.to_string(), encode(name, shape, &values));
    }

    if format == "adapter" {
        let mert_dir = mert_dir.ok_or_else(|| {
            SheetError::Config("релиз SheetSage2 — адаптер: нужен каталог MERT-v2-FullSong".into())
        })?;
        let mert_path = mert_dir.join("model.safetensors");
        if let Some(expected) = config.get("base_model_sha256").and_then(|v| v.as_str()) {
            progress("сверка SHA-256 MERT-v2…");
            let got = sha256_file(&mert_path)?;
            if got != expected {
                return Err(SheetError::Load(format!(
                    "MERT-v2 не тот: SHA-256 {got}, релиз SheetSage2 ждёт {expected}"
                )));
            }
        }
        progress("слияние LoRA-адаптеров с MERT-v2…");
        let mert_map = map(&mert_path)?;
        let mert = SafeTensors::deserialize(&mert_map).map_err(|e| SheetError::Load(e.to_string()))?;
        for name in mert.names() {
            let (shape, mut values) = to_f32(&mert, name)?;
            if let Some(rest) = name.strip_prefix("layers.") {
                let mut parts = rest.split('.');
                let (layer, attn, proj, leaf) = (parts.next(), parts.next(), parts.next(), parts.next());
                if attn == Some("attn") && leaf == Some("weight") && PROJECTIONS.contains(&proj.unwrap_or("")) {
                    let (layer, proj) = (layer.unwrap(), proj.unwrap());
                    let adapter = format!("adapter.layers.{layer}.attn.{proj}");
                    let (_, a) = to_f32(&ss, &format!("{adapter}.lora_A.weight"))?;
                    let (_, b) = to_f32(&ss, &format!("{adapter}.lora_B.weight"))?;
                    let (rows, cols) = (shape[0], shape[1]);
                    if a.len() != rank * cols || b.len() != rows * rank {
                        return Err(SheetError::Load(format!("{adapter}: формы адаптера не сходятся с весом")));
                    }
                    merge_lora(&mut values, &a, &b, rows, cols, rank, scale);
                    stats.merged += 1;
                }
            }
            let full = format!("encoder.{name}");
            out.insert(full.clone(), encode(&full, shape, &values));
        }
        let expected = config
            .get("backbone_config")
            .and_then(|b| b.get("num_hidden_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(24) as usize
            * PROJECTIONS.len();
        if stats.merged != expected {
            return Err(SheetError::Load(format!("влито {} адаптеров из {expected}", stats.merged)));
        }
    }
    if !out.contains_key("encoder.layers.0.attn.query_proj.weight") {
        return Err(SheetError::Load("в компоненте нет весов энкодера".into()));
    }
    config["weights_format"] = serde_json::Value::String("merged".into());

    progress("сборка safetensors…");
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = out
        .iter()
        .map(|(name, t)| {
            safetensors::tensor::TensorView::new(t.dtype, t.shape.clone(), &t.data)
                .map(|v| (name.clone(), v))
                .map_err(|e| SheetError::Load(format!("`{name}`: {e}")))
        })
        .collect::<Result<_, _>>()?;
    stats.tensors = views.len();
    stats.bf16_tensors = out.values().filter(|t| t.dtype == Dtype::BF16).count();
    let payload = safetensors::serialize(views, None).map_err(|e| SheetError::Load(e.to_string()))?;
    stats.payload_bytes = payload.len() as u64;
    Ok((payload, config, stats))
}

/// Дописать компонент `sheetsage2` в существующий бандл (например,
/// `yue2-3b.syn`). Повторный запуск заменяет прежний компонент.
pub fn pack_into_bundle(
    bundle: &Path,
    sheetsage_dir: &Path,
    mert_dir: Option<&Path>,
    progress: &mut dyn FnMut(&str),
) -> Result<PackStats, SheetError> {
    let (payload, config, stats) = build_component(sheetsage_dir, mert_dir, progress)?;
    progress("запись в бандл…");
    let mut editor = BundleEditor::open(bundle).map_err(|e| SheetError::Load(e.to_string()))?;
    editor
        .add_tensors_component(COMPONENT, payload)
        .map_err(|e| SheetError::Load(e.to_string()))?;
    let config_bytes = serde_json::to_vec_pretty(&config).map_err(|e| SheetError::Config(e.to_string()))?;
    editor
        .replace_file(&format!("{COMPONENT}/config.json"), config_bytes, FileTag::Inference)
        .map_err(|e| SheetError::Load(e.to_string()))?;
    let mut extra: Vec<(std::path::PathBuf, String, FileTag)> = vec![
        (sheetsage_dir.join("processor_config.json"), "processor_config.json".into(), FileTag::Inference),
        (sheetsage_dir.join("LICENSE"), "LICENSE".into(), FileTag::Doc),
        (sheetsage_dir.join("README.md"), "README.md".into(), FileTag::Doc),
        (sheetsage_dir.join("THIRD_PARTY_NOTICES.md"), "THIRD_PARTY_NOTICES.md".into(), FileTag::Doc),
    ];
    if let Some(m) = mert_dir {
        extra.push((m.join("config.json"), "mert2_config.json".into(), FileTag::Inference));
        extra.push((m.join("README.md"), "mert2_README.md".into(), FileTag::Doc));
    }
    for (src, name, tag) in extra {
        if let Ok(bytes) = std::fs::read(&src) {
            editor
                .replace_file(&format!("{COMPONENT}/{name}"), bytes, tag)
                .map_err(|e| SheetError::Load(e.to_string()))?;
        }
    }
    editor.commit().map_err(|e| SheetError::Load(e.to_string()))?;
    Ok(stats)
}
