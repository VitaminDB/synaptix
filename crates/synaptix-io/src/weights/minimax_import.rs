//! Импорт чекпойнта MiniMax M3 (HF, compressed-tensors) в бандл `.syn`.
//!
//! Что куда:
//! - MXFP8-матрицы (внимание, индексатор, плотный MLP первых слоёв,
//!   shared-эксперт): `weight` (e4m3) и `weight_scale_inv` (E8M0, блок 32
//!   вдоль входа) совпадают с `DType::MXFP8` движка байт в байт — копируются;
//! - эксперты MoE (`block_sparse_moe.experts.<e>.w{1,2,3}`, NVFP4 с общим
//!   множителем) деквантуются точно и кодируются в одноблобный формат ggml
//!   стопками движка: `mlp.experts.gate_up_proj` `[E, 2I, H]` (строки w1 =
//!   gate, затем w3 = up) и `mlp.experts.down_proj` `[E, H, I]` (w2). Без
//!   формата экспертов они в бандл не пишутся (скелет: эксперты читаются из
//!   оригинала — эталонный прогон);
//! - роутер, поправка выбора, нормы, эмбеддинг и голова — как есть;
//! - башня зрения и проекторы пропускаются.
//!
//! Имена — в принятом у движка виде: `model.layers.N.mlp.gate.weight`,
//! `….mlp.shared_expert.{gate,up,down}_proj.weight`, `lm_head.weight`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::quant_layout::MANIFEST_NAME;
use synaptix_bundle::{BundleBuilder, FileTag, QuantEntry, QuantManifest, StDtype, StreamTensor, TensorStream, CAP_QUANT_WEIGHTS};
use synaptix_core::quant::ggml::GgmlType;

use super::hf_origin::HfOrigin;
use crate::error::{IoError, Result};

/// Кодировщик матрицы `[n, k]` f32 в блоб ggml; `imp` — важность столбцов.
pub type Encoder = Arc<dyn Fn(GgmlType, &[f32], usize, usize, Option<&[f32]>) -> Result<Vec<u8>> + Send + Sync>;
pub type Progress = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Кодировщик на CPU (эталон): IQ3_S-128, IQ2_XXS, IQ1_S.
pub fn cpu_encoder() -> Encoder {
    Arc::new(|ty, x, n, k, imp| match ty {
        GgmlType::Iq3S128 => Ok(synaptix_core::quant::ggml::quantize_iq3_s128(x, n, k, imp)),
        GgmlType::Iq2Xxs | GgmlType::Iq1S => Ok(synaptix_core::quant::ggml_lowbit::quantize_lowbit(ty, x, n, k, imp)),
        o => Err(IoError::Bundle(format!("эксперты в {}: кодировщика нет", o.name()))),
    })
}

enum Src {
    /// Тензор оригинала как есть.
    Pass(String),
    /// Стопка экспертов слоя: gate_up (w1 + w3) или down (w2).
    Experts { layer: usize, down: bool },
}

struct Stream {
    hf: Arc<HfOrigin>,
    plan: Vec<StreamTensor>,
    src: Vec<Src>,
    experts: usize,
    ty: Option<GgmlType>,
    encode: Encoder,
    progress: Option<Progress>,
    done: usize,
    total: usize,
}

const HF_LM: &str = "language_model.";

fn st_dtype(dt: &str) -> Result<StDtype> {
    Ok(match dt {
        "BF16" => StDtype::BF16,
        "F16" => StDtype::F16,
        "F32" => StDtype::F32,
        "U8" | "F8_E4M3" => StDtype::U8,
        o => return Err(IoError::Bundle(format!("тип {o}: писатель не копирует"))),
    })
}

/// Точная матрица эксперта из оригинала: `(f32, n, k)`.
pub fn expert_matrix(hf: &HfOrigin, layer: usize, expert: usize, w: &str) -> Result<(Vec<f32>, usize, usize)> {
    let base = format!("{HF_LM}model.layers.{layer}.block_sparse_moe.experts.{expert}.{w}");
    let (_, shape, packed) = hf.bytes(&format!("{base}.weight_packed"))?;
    let (_, _, scale) = hf.bytes(&format!("{base}.weight_scale"))?;
    let (_, _, g) = hf.bytes(&format!("{base}.weight_global_scale"))?;
    let [n, half] = shape.as_slice() else {
        return Err(IoError::Bundle(format!("{base}: форма {shape:?}")));
    };
    let g = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    Ok((synaptix_core::quant::nvfp4_ct::dequant(&packed, &scale, g, *n, half * 2), *n, half * 2))
}

/// Срез стопки движка: gate_up — строки w1, затем w3; down — w2.
pub fn expert_slice(hf: &HfOrigin, layer: usize, expert: usize, down: bool) -> Result<(Vec<f32>, usize, usize)> {
    if down {
        return expert_matrix(hf, layer, expert, "w2");
    }
    let (mut gate, n, k) = expert_matrix(hf, layer, expert, "w1")?;
    let (up, n2, k2) = expert_matrix(hf, layer, expert, "w3")?;
    if (n2, k2) != (n, k) {
        return Err(IoError::Bundle(format!("слой {layer}, эксперт {expert}: w1 {n}×{k}, w3 {n2}×{k2}")));
    }
    gate.extend_from_slice(&up);
    Ok((gate, 2 * n, k))
}

impl TensorStream for Stream {
    fn plan(&self) -> &[StreamTensor] {
        &self.plan
    }

    fn write_tensor(&mut self, index: usize, w: &mut dyn std::io::Write) -> synaptix_bundle::Result<()> {
        let err = |e: IoError| synaptix_bundle::Error::Safetensors(e.to_string());
        match &self.src[index] {
            Src::Pass(name) => {
                let (_, _, bytes) = self.hf.bytes(name).map_err(err)?;
                w.write_all(&bytes).map_err(synaptix_bundle::Error::Io)
            }
            Src::Experts { layer, down } => {
                let ty = self.ty.expect("формат экспертов");
                let stack = format!("model.layers.{layer}.mlp.experts.{}", if *down { "down_proj" } else { "gate_up_proj" });
                for e in 0..self.experts {
                    if let Some(p) = &self.progress {
                        p(self.done, self.total, &format!("{stack}#{e}"));
                    }
                    let (x, n, k) = expert_slice(&self.hf, *layer, e, *down).map_err(err)?;
                    let imp = synaptix_core::quant::calib::table_get(&format!("{stack}#{e}")).or_else(|| synaptix_core::quant::calib::table_get(&stack));
                    let blob = (self.encode)(ty, &x, n, k, imp.as_deref()).map_err(err)?;
                    w.write_all(&blob).map_err(synaptix_bundle::Error::Io)?;
                    self.done += 1;
                }
                Ok(())
            }
        }
    }
}

pub struct ImportReport {
    pub layers: usize,
    pub moe_layers: usize,
    pub experts: usize,
    pub seconds: f64,
}

/// Собрать бандл. `experts` — формат экспертов (`None` — скелет без них).
pub fn import(hf_dir: &Path, out: &Path, experts: Option<GgmlType>, encode: Encoder, progress: Option<Progress>) -> Result<ImportReport> {
    let t0 = std::time::Instant::now();
    let hf = HfOrigin::open(hf_dir)?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(hf_dir.join("config.json"))?).map_err(|e| IoError::Bundle(format!("config.json: {e}")))?;
    let tc = &config["text_config"];
    let layers = tc["num_hidden_layers"].as_u64().ok_or_else(|| IoError::Bundle("config: num_hidden_layers".into()))? as usize;
    let n_exp = tc["num_local_experts"].as_u64().unwrap_or(0) as usize;
    let hidden = tc["hidden_size"].as_u64().unwrap_or(0) as usize;
    let inter = tc["intermediate_size"].as_u64().unwrap_or(0) as usize;

    let mut manifest = QuantManifest::new();
    let mut plan: Vec<StreamTensor> = Vec::new();
    let mut src: Vec<Src> = Vec::new();
    let pass = |plan: &mut Vec<StreamTensor>, src: &mut Vec<Src>, hf_name: String, name: String| -> Result<()> {
        let (dt, shape) = hf.shape(&hf_name)?;
        plan.push(StreamTensor { name, dtype: st_dtype(&dt)?, shape });
        src.push(Src::Pass(hf_name));
        Ok(())
    };
    // MXFP8: пара weight + weight_scale_inv → .qpacked/.qscales.
    let mx = |plan: &mut Vec<StreamTensor>, src: &mut Vec<Src>, manifest: &mut QuantManifest, hf_name: String, name: String| -> Result<()> {
        let (dt, shape) = hf.shape(&hf_name)?;
        let sname = format!("{hf_name}_scale_inv");
        let (sdt, sshape) = hf.shape(&sname)?;
        let ok = dt == "F8_E4M3" && sdt == "U8" && shape.len() == 2 && sshape == [shape[0], shape[1] / 32];
        if !ok {
            return Err(IoError::Bundle(format!("{hf_name}: {dt} {shape:?} + масштабы {sdt} {sshape:?} — не MXFP8 с блоком 32")));
        }
        manifest.tensors.insert(name.clone(), QuantEntry { format: "mxfp8".into(), shape: shape.clone() });
        plan.push(StreamTensor { name: manifest.packed_name(&name), dtype: StDtype::U8, shape });
        src.push(Src::Pass(hf_name));
        plan.push(StreamTensor { name: manifest.scales_name(&name), dtype: StDtype::U8, shape: vec![sshape[0] * sshape[1]] });
        src.push(Src::Pass(sname));
        Ok(())
    };

    pass(&mut plan, &mut src, format!("{HF_LM}model.embed_tokens.weight"), "model.embed_tokens.weight".into())?;
    pass(&mut plan, &mut src, format!("{HF_LM}model.norm.weight"), "model.norm.weight".into())?;
    pass(&mut plan, &mut src, format!("{HF_LM}lm_head.weight"), "lm_head.weight".into())?;
    let mut moe_layers = 0usize;
    for l in 0..layers {
        let h = format!("{HF_LM}model.layers.{l}");
        let b = format!("model.layers.{l}");
        for n in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm", "self_attn.index_q_norm", "self_attn.index_k_norm"] {
            pass(&mut plan, &mut src, format!("{h}.{n}.weight"), format!("{b}.{n}.weight"))?;
        }
        for n in ["q_proj", "k_proj", "v_proj", "o_proj", "index_q_proj", "index_k_proj"] {
            mx(&mut plan, &mut src, &mut manifest, format!("{h}.self_attn.{n}.weight"), format!("{b}.self_attn.{n}.weight"))?;
        }
        if !hf.has(&format!("{h}.block_sparse_moe.gate.weight")) {
            for n in ["gate_proj", "up_proj", "down_proj"] {
                mx(&mut plan, &mut src, &mut manifest, format!("{h}.mlp.{n}.weight"), format!("{b}.mlp.{n}.weight"))?;
            }
            continue;
        }
        moe_layers += 1;
        pass(&mut plan, &mut src, format!("{h}.block_sparse_moe.gate.weight"), format!("{b}.mlp.gate.weight"))?;
        pass(&mut plan, &mut src, format!("{h}.block_sparse_moe.e_score_correction_bias"), format!("{b}.mlp.e_score_correction_bias"))?;
        for n in ["gate_proj", "up_proj", "down_proj"] {
            mx(&mut plan, &mut src, &mut manifest, format!("{h}.block_sparse_moe.shared_experts.{n}.weight"), format!("{b}.mlp.shared_expert.{n}.weight"))?;
        }
        if let Some(ty) = experts {
            for (down, n, k) in [(false, 2 * inter, hidden), (true, hidden, inter)] {
                let name = format!("{b}.mlp.experts.{}", if down { "down_proj" } else { "gate_up_proj" });
                if k % ty.block_elems() != 0 {
                    return Err(IoError::Bundle(format!("{name}: K={k} не кратно блоку {}", ty.name())));
                }
                manifest.tensors.insert(name.clone(), QuantEntry { format: format!("ggml:{}", ty.key()), shape: vec![n_exp, n, k] });
                plan.push(StreamTensor { name: manifest.packed_name(&name), dtype: StDtype::U8, shape: vec![n_exp, n, ty.bytes_for(k)] });
                src.push(Src::Experts { layer: l, down });
            }
        }
    }
    let total = if experts.is_some() { moe_layers * 2 * n_exp } else { 0 };
    let stream = Stream { hf: hf.clone(), plan, src, experts: n_exp, ty: experts, encode, progress, done: 0, total };

    let id = out.file_stem().and_then(|s| s.to_str()).unwrap_or("minimax-m3").to_string();
    let mut builder = BundleBuilder::new(id, "1.0.0").arch("minimax_m3").purpose("text-generation").require_capability(CAP_QUANT_WEIGHTS).component("main", "");
    builder = builder.add_tensor_stream("main", Box::new(stream));
    let mut meta: BTreeMap<String, String> = BTreeMap::new();
    meta.insert("source".into(), hf_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"] {
        let p = hf_dir.join(name);
        if p.exists() {
            builder = builder.add_file_bytes(name, std::fs::read(&p)?, FileTag::Inference).map_err(|e| IoError::Bundle(e.to_string()))?;
        }
    }
    let mbytes = serde_json::to_vec_pretty(&manifest).map_err(|e| IoError::Bundle(e.to_string()))?;
    builder = builder.add_file_bytes(MANIFEST_NAME, mbytes, FileTag::Inference).map_err(|e| IoError::Bundle(e.to_string()))?;
    builder.write(out).map_err(|e| IoError::Bundle(format!("{}: {e}", out.display())))?;
    let _ = meta;
    Ok(ImportReport { layers, moe_layers, experts: total, seconds: t0.elapsed().as_secs_f64() })
}
