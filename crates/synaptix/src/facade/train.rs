//! Дообучение LLM адаптером LoRA: на квантованной базе это QLoRA (база
//! заморожена в NVFP4/MXFP8/SQ, учатся только адаптеры в F32).
//!
//! Датасет — JSONL (или JSON-массив), строка — пример в одном из видов:
//! - `{"messages": [{"role": "user", "content": "…"}, …]}` (OpenAI);
//! - `{"conversations": [{"from": "human", "value": "…"}, …]}` (ShareGPT);
//! - `{"instruction": "…", "input": "…", "output": "…"}` (Alpaca);
//! - `{"prompt": "…", "completion": "…"}` (или `response`);
//! - `{"text": "…"}` — сырой текст, учится целиком.
//!
//! Диалоги рендерятся chat-шаблоном модели, лосс считается только по
//! ответам ассистента (вместе с токеном конца реплики).

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;

pub use synaptix_llm_train::linear::BaseMatmul;
pub use synaptix_llm_train::{LoraConfig, Sample, TrainConfig, TrainHooks, TrainProgress};

use super::arch::{detect_llm_arch, LlmArch};
use super::llm::{load_llm_tokenizer, LlmError, LlmTokenizer, Message};

/// Настройки дообучения.
#[derive(Debug, Clone)]
pub struct LoraTrainOptions {
    /// Формат весов базы на карте: NVFP4 / MXFP8 / `Sq{bits}` (QLoRA) или
    /// BF16 (обычная LoRA). Уже квантованный бандл грузится как есть.
    pub base_quant: DType,
    /// Как умножать на квантованную базу (по умолчанию — деквант в BF16).
    pub base_matmul: BaseMatmul,
    pub lora: LoraConfig,
    pub train: TrainConfig,
    /// Рендерить шаблон с размышлениями (`enable_thinking`).
    pub thinking: bool,
    /// Доля примеров на оценку (не больше 200 примеров).
    pub eval_fraction: f32,
    pub max_samples: Option<usize>,
    /// Продолжить с готового адаптера (те же цели и ранг).
    pub init_adapter: Option<PathBuf>,
    /// Только оценить лосс на всём датасете (без обучения и сохранения).
    pub eval_only: bool,
    pub device: Device,
}

impl Default for LoraTrainOptions {
    fn default() -> Self {
        Self {
            base_quant: DType::NVFP4,
            base_matmul: BaseMatmul::Dequant,
            lora: LoraConfig::default(),
            train: TrainConfig::default(),
            thinking: false,
            eval_fraction: 0.02,
            max_samples: None,
            init_adapter: None,
            eval_only: false,
            device: Device::Cuda(0),
        }
    }
}

/// Что получилось из датасета.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DatasetReport {
    pub samples: usize,
    pub skipped: usize,
    pub truncated: usize,
    pub tokens: usize,
    pub target_tokens: usize,
    /// Многоходовые диалоги, где шаблон не префиксно-устойчив (размышления
    /// прошлых ходов вырезаются): учится только последний ответ.
    pub last_turn_only: usize,
}

#[derive(Debug, Clone)]
pub struct TrainReport {
    pub dataset: DatasetReport,
    /// Лосс на всём датасете в режиме `eval_only`.
    pub eval_loss: Option<f32>,
    pub eval_samples: usize,
    pub trainable_params: usize,
    pub adapter_dir: PathBuf,
}

enum Example {
    Chat(Vec<Message>),
    Text(String),
}

fn str_field<'a>(v: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_str()))
}

fn parse_example(v: &serde_json::Value) -> Option<Example> {
    let role_of = |r: &str| -> &'static str {
        match r {
            "system" => "system",
            "assistant" | "gpt" | "bot" | "model" | "chatgpt" => "assistant",
            "tool" | "function" | "observation" => "tool",
            _ => "user",
        }
    };
    if let Some(arr) = v.get("messages").or_else(|| v.get("conversations")).and_then(|m| m.as_array()) {
        let mut msgs = Vec::new();
        for m in arr {
            let role = str_field(m, &["role", "from"]).unwrap_or("user");
            let content = str_field(m, &["content", "value", "text"]).unwrap_or("");
            let mut msg = match role_of(role) {
                "system" => Message::system(content),
                "assistant" => Message::assistant(content),
                "tool" => Message::tool(content),
                _ => Message::user(content),
            };
            if msg.role == "assistant" {
                msg.reasoning = str_field(m, &["reasoning_content", "reasoning", "thinking"]).map(str::to_string);
            }
            msgs.push(msg);
        }
        return msgs.iter().any(|m| m.role == "assistant").then_some(Example::Chat(msgs));
    }
    if let Some(out) = str_field(v, &["output", "completion", "response", "answer", "chosen"]) {
        let mut prompt = str_field(v, &["instruction", "prompt", "question", "query"]).unwrap_or("").to_string();
        if let Some(inp) = str_field(v, &["input", "context"]).filter(|s| !s.trim().is_empty()) {
            prompt = if prompt.is_empty() { inp.to_string() } else { format!("{prompt}\n\n{inp}") };
        }
        let mut msgs = Vec::new();
        if let Some(sys) = str_field(v, &["system"]).filter(|s| !s.is_empty()) {
            msgs.push(Message::system(sys));
        }
        msgs.push(Message::user(prompt));
        msgs.push(Message::assistant(out));
        return Some(Example::Chat(msgs));
    }
    str_field(v, &["text", "content"]).map(|t| Example::Text(t.to_string()))
}

fn read_examples(path: &Path) -> Result<Vec<serde_json::Value>, LlmError> {
    let text = std::fs::read_to_string(path).map_err(|e| LlmError(format!("{}: {e}", path.display())))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('[') {
        let v: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| LlmError(format!("{}: {e}", path.display())))?;
        return Ok(v.as_array().cloned().unwrap_or_default());
    }
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v = serde_json::from_str(line).map_err(|e| LlmError(format!("{}:{}: {e}", path.display(), i + 1)))?;
        out.push(v);
    }
    Ok(out)
}

/// Токены диалога и флаги «учить» — по ответам ассистента. `None` — пример
/// не размечается (шаблон не согласован с токенизацией).
fn tokenize_chat(tok: &LlmTokenizer, msgs: &[Message], thinking: bool, rep: &mut DatasetReport) -> Result<Option<Sample>, LlmError> {
    let full_text = tok.apply_chat_template_ex_tools(msgs, false, thinking, None)?;
    let full = tok.encode(&full_text)?;
    let mut target = vec![false; full.len()];
    let span = |upto: usize| -> Result<Option<(usize, usize)>, LlmError> {
        let pre = tok.encode(&tok.apply_chat_template_ex_tools(&msgs[..upto], true, thinking, None)?)?;
        let end = tok.encode(&tok.apply_chat_template_ex_tools(&msgs[..=upto], false, thinking, None)?)?;
        let ok = pre.len() <= end.len() && end.len() <= full.len() && full[..end.len()] == end[..] && end[..pre.len()] == pre[..];
        Ok(ok.then_some((pre.len(), end.len())))
    };
    let turns: Vec<usize> = msgs.iter().enumerate().filter(|(_, m)| m.role == "assistant").map(|(i, _)| i).collect();
    let mut spans = Vec::new();
    for &i in &turns {
        match span(i)? {
            Some(s) => spans.push(s),
            None => {
                spans.clear();
                break;
            }
        }
    }
    if spans.is_empty() {
        // Шаблон переписывает прошлые ходы — учим только последний ответ.
        let last = *turns.last().unwrap();
        let pre = tok.encode(&tok.apply_chat_template_ex_tools(&msgs[..last], true, thinking, None)?)?;
        if pre.len() >= full.len() || full[..pre.len()] != pre[..] {
            return Ok(None);
        }
        spans.push((pre.len(), full.len()));
        if turns.len() > 1 {
            rep.last_turn_only += 1;
        }
    }
    for (a, b) in spans {
        for t in target.iter_mut().take(b).skip(a) {
            *t = true;
        }
    }
    Ok(Some(Sample { tokens: full, target }))
}

/// Прочитать и разметить датасет.
pub fn load_samples(
    path: &Path,
    tok: &LlmTokenizer,
    thinking: bool,
    max_len: usize,
    max_samples: Option<usize>,
) -> Result<(Vec<Sample>, DatasetReport), LlmError> {
    let mut rep = DatasetReport::default();
    let mut out = Vec::new();
    for v in read_examples(path)? {
        if max_samples.is_some_and(|m| out.len() >= m) {
            break;
        }
        let sample = match parse_example(&v) {
            Some(Example::Chat(msgs)) => tokenize_chat(tok, &msgs, thinking, &mut rep)?,
            Some(Example::Text(t)) => {
                let tokens = tok.encode(&t)?;
                let target = vec![true; tokens.len()];
                Some(Sample { tokens, target })
            }
            None => None,
        };
        let Some(mut s) = sample else {
            rep.skipped += 1;
            continue;
        };
        if s.tokens.len() > max_len {
            s.tokens.truncate(max_len);
            s.target.truncate(max_len);
            rep.truncated += 1;
        }
        if s.n_targets() == 0 {
            rep.skipped += 1;
            continue;
        }
        rep.tokens += s.tokens.len();
        rep.target_tokens += s.n_targets();
        out.push(s);
    }
    rep.samples = out.len();
    Ok((out, rep))
}

/// Разделить на обучение и оценку: каждый k-й пример — в оценку.
fn split_eval(mut all: Vec<Sample>, fraction: f32) -> (Vec<Sample>, Vec<Sample>) {
    if fraction <= 0.0 || all.len() < 20 {
        return (all, Vec::new());
    }
    let n_eval = ((all.len() as f32 * fraction).round() as usize).clamp(1, 200);
    let every = all.len() / n_eval;
    let mut eval = Vec::with_capacity(n_eval);
    let mut train = Vec::with_capacity(all.len());
    for (i, s) in all.drain(..).enumerate() {
        if i % every == every - 1 && eval.len() < n_eval {
            eval.push(s);
        } else {
            train.push(s);
        }
    }
    (train, eval)
}

/// Дообучить модель `model` на датасете `data`; адаптер — в
/// `opts.train.out_dir`. `on_dataset` вызывается после разметки датасета,
/// `progress` — после каждого шага оптимизатора (`false` — остановиться).
pub fn train_lora(
    model: &Path,
    data: &Path,
    mut opts: LoraTrainOptions,
    hooks: TrainHooks,
    on_dataset: impl FnOnce(&DatasetReport, usize, usize),
    progress: impl FnMut(&TrainProgress) -> bool,
) -> Result<TrainReport, LlmError> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let tok = load_llm_tokenizer(model)?;
    let (samples, dataset) = load_samples(data, &tok, opts.thinking, opts.train.max_len, opts.max_samples)?;
    if samples.is_empty() {
        return Err(LlmError(format!("{}: ни одного пригодного примера", data.display())));
    }
    let (train, eval) = if opts.eval_only { (Vec::new(), samples) } else { split_eval(samples, opts.eval_fraction) };
    on_dataset(&dataset, train.len(), eval.len());

    let mut prec = PrecisionConfig::dense(DType::BF16);
    if opts.base_quant != DType::BF16 {
        prec.attn_w = opts.base_quant;
        prec.mlp_w = opts.base_quant;
    }
    let rope = opts.train.max_len.max(4096);
    let arch = detect_llm_arch(model).map_err(LlmError)?;
    if opts.train.base_name.is_empty() {
        opts.train.base_name = model.display().to_string();
    }
    // Имена модулей в адаптере — как у HF-модели: у Qwen3.5+ (общая модель с
    // башней зрения) слои лежат под `model.language_model`.
    let config = synaptix_io::weights::read_model_file(model, "config.json").and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    let multimodal = config.as_ref().is_some_and(|c| c.get("text_config").is_some());
    // MoE: эксперты — в RAM, на карте их кэш (модель с экспертами на карте
    // целиком не оставила бы места активациям).
    let moe = config
        .as_ref()
        .and_then(|c| c.get("text_config").unwrap_or(c).get("num_experts").and_then(|v| v.as_u64()))
        .is_some_and(|n| n > 0);
    opts.lora.module_prefix = if multimodal { "model.language_model.layers".into() } else { "model.layers".into() };
    let out_dir = opts.train.out_dir.clone();
    let mut eval_loss = None;
    let mut run = |base: &synaptix_llm_common::DecoderModel| -> Result<usize, LlmError> {
        let err = |e: synaptix_llm_train::TrainError| LlmError(e.to_string());
        let mut tm = synaptix_llm_train::TrainModel::new(base, opts.lora.clone()).map_err(err)?;
        tm.base_matmul = opts.base_matmul;
        if let Some(dir) = &opts.init_adapter {
            synaptix_llm_train::adapter::load_into(&mut tm, dir).map_err(err)?;
        }
        let params = tm.num_params();
        let mut trainer = synaptix_llm_train::Trainer::new(opts.train.clone());
        trainer.hooks = hooks.clone();
        if opts.eval_only {
            eval_loss = trainer.evaluate(&tm, &eval).map_err(err)?;
        } else {
            trainer.run(&mut tm, &train, &eval, progress).map_err(err)?;
        }
        Ok(params)
    };
    let load_guard = hooks.step_guard.as_ref().map(|g| g());
    let trainable_params = match arch {
        LlmArch::Qwen3 => {
            let _w = synaptix_core::device::cuda::WeightsAllocGuard::for_device(opts.device);
            let p = synaptix_llm_qwen3::pipeline::Qwen3Pipeline::load_with_precision(model, opts.device, prec, Some(rope))
                .map_err(|e| LlmError(format!("load qwen3: {e}")))?;
            drop(_w);
            drop(load_guard);
            run(&p.model)?
        }
        LlmArch::Hybrid => {
            let _w = synaptix_core::device::cuda::WeightsAllocGuard::for_device(opts.device);
            use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
            let p = if moe {
                HybridPipeline::load_with_experts_in_ram(model, opts.device, prec, Some(rope))
            } else {
                HybridPipeline::load_with_precision(model, opts.device, prec, Some(rope))
            }
            .map_err(|e| LlmError(format!("load hybrid: {e}")))?;
            drop(_w);
            drop(load_guard);
            run(&p.model)?
        }
        other => return Err(LlmError(format!("дообучение пока поддержано для Qwen3 и Qwen3.5/3.6 (плотных и MoE); у модели {other:?}"))),
    };
    Ok(TrainReport { dataset, eval_loss, eval_samples: eval.len(), trainable_params, adapter_dir: out_dir })
}

/// Влить адаптер PEFT (`adapter_dir`) в веса модели `base` и записать новый
/// бандл `out`. `format` — формат изменённых матриц: `None` — прежний,
/// иначе — этот. Добавка LoRA меньше шага 4-битного кванта, поэтому на
/// NVFP4-базе переквантование в тот же формат теряет бо́льшую часть
/// адаптера (Qwen3-1.7B, ru_turbo_alpaca: eval 1.118 с адаптером, 1.668
/// после слияния в NVFP4, 1.124 — в MXFP8, 1.118 — в BF16; база 1.948).
/// Остальные тензоры и файлы копируются как есть.
pub fn merge_lora(
    base: &Path,
    adapter_dir: &Path,
    out: &Path,
    format: Option<DType>,
    device: Device,
    progress: Option<synaptix_io::weights::transcode::Progress>,
) -> Result<usize, LlmError> {
    use std::collections::HashMap;
    use synaptix_io::weights::transcode::{TranscodeSpec, WeightDelta, WeightDeltas};

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let e = |m: String| LlmError(m);
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(adapter_dir.join("adapter_config.json")).map_err(|x| e(format!("adapter_config.json: {x}")))?,
    )
    .map_err(|x| e(format!("adapter_config.json: {x}")))?;
    let r = cfg.get("r").and_then(|v| v.as_u64()).ok_or_else(|| e("adapter_config.json: нет r".into()))? as usize;
    let alpha = cfg.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(r as f64) as f32;
    let scale = alpha / r as f32;
    let bytes = std::fs::read(adapter_dir.join("adapter_model.safetensors")).map_err(|x| e(format!("adapter_model.safetensors: {x}")))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|x| e(x.to_string()))?;
    let read = |name: &str| -> Result<(Vec<usize>, Vec<f32>), LlmError> {
        let v = st.tensor(name).map_err(|x| e(format!("{name}: {x}")))?;
        let data: Vec<f32> = match v.dtype() {
            safetensors::Dtype::F32 => v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            safetensors::Dtype::BF16 => v.data().chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            safetensors::Dtype::F16 => v.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => return Err(e(format!("{name}: тип {d:?}"))),
        };
        Ok((v.shape().to_vec(), data))
    };
    let mut map = HashMap::new();
    for name in st.names() {
        let Some(module) = name.strip_prefix("base_model.model.").and_then(|n| n.strip_suffix(".lora_A.weight")) else { continue };
        let (ash, a) = read(name)?;
        let (bsh, b) = read(&format!("base_model.model.{module}.lora_B.weight"))?;
        if ash.len() != 2 || bsh.len() != 2 || ash[0] != bsh[1] {
            return Err(e(format!("{module}: формы A {ash:?} и B {bsh:?} не согласованы")));
        }
        map.insert(format!("{module}.weight"), WeightDelta { a, b, r: ash[0], scale });
    }
    if map.is_empty() {
        return Err(e(format!("{}: в адаптере нет матриц LoRA", adapter_dir.display())));
    }
    // Каждая добавка должна найти свою матрицу.
    let loader = synaptix_io::weights::syn_bundle::SynBundleLoader::open(base).map_err(|x| e(x.to_string()))?;
    let have: std::collections::HashSet<String> = synaptix_io::weights::WeightLoader::names(&loader).into_iter().map(|s| s.to_string()).collect();
    let missing: Vec<&String> = map.keys().filter(|k| !have.contains(*k)).collect();
    if !missing.is_empty() {
        return Err(e(format!("в модели нет матриц адаптера: {} (первая: {})", missing.len(), missing[0])));
    }
    let n = map.len();
    let mut spec = TranscodeSpec::keep();
    spec.deltas = Some(WeightDeltas { map: std::sync::Arc::new(map), format });
    let _w = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
    synaptix_io::weights::transcode::write_bundle(base, out, &spec, device, progress).map_err(|x| e(x.to_string()))?;
    Ok(n)
}
