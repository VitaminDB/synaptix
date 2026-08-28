//! Разбор safetensors-заголовка без чтения весов и группировка тензоров по
//! ролям (embedding / attention / vision-башня / lm_head / …).
//!
//! Заголовок safetensors — `u64 LE` длины + JSON. Чтобы перечислить тензоры,
//! достаточно прочитать эти два блока: для 46-гигабайтного шарда это
//! килобайты вместо гигабайтов, поэтому скан каталога с моделями остаётся
//! мгновенным. Тот же парсер работает и над срезом `tensors:*`-чанка внутри
//! `.syn` ([`Bundle::tensors_slice_named`](crate::Bundle::tensors_slice_named)) —
//! payload чанка и есть safetensors-поток.
//!
//! [`group_tensors`] схлопывает числовые сегменты имён
//! (`model.layers.0.…` … `model.layers.47.…` → `model.layers.[0-47].…`), так
//! что вместо тысяч строк получается несколько десятков групп — по ним видно,
//! из чего состоит модель и сколько весит каждая часть.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};

/// Верхняя граница на длину заголовка. Реальные заголовки — десятки/сотни
/// килобайт (Qwen3-27B ≈ 300 КБ); 64 МБ отсекает мусорный `len` от битого
/// файла до того, как мы попробуем выделить под него память.
const MAX_HEADER_LEN: u64 = 64 << 20;

/// Один тензор из заголовка. `bytes` — длина payload'а по `data_offsets`,
/// а не произведение формы: так значение остаётся честным и для dtype,
/// которых нет в [`crate::StDtype`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

/// Роль тензора в модели. Порядок проверки в [`classify`] важен: префикс
/// башни (`visual.`) сильнее внутреннего `attn.`, а LoRA-суффикс сильнее
/// всего остального.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerRole {
    Embedding,
    LmHead,
    Attention,
    Mlp,
    Norm,
    /// Свёртки, не попавшие в башню: conv-модуль конформера у ASR, `conv_in`/
    /// `conv_out` автоэнкодеров. Отдельная роль нужна не для красоты —
    /// квант-ядра NVFP4/MXFP8 работают только с 2-D матрицами, и в UI такие
    /// группы обязаны показываться неквантуемыми.
    Conv,
    /// Модуляция и обусловливание диффузионных трансформеров: AdaLN-проекции,
    /// эмбеддер шага, проекции условия. У MiniMax-H3 это 40 % веса DiT, так
    /// что молча ссыпать их в [`LayerRole::Other`] нельзя.
    Conditioning,
    /// Выходная голова не-языковой модели (`final_layer.*`, `proj_out`).
    Head,
    /// Роутер MoE: матрица `[эксперты, hidden]`, по логитам которой выбираются
    /// эксперты, и её гейты. Весит крохи, а решает всё — квантовать нельзя:
    /// сдвиг логита на пол-процента меняет состав top-k, то есть модель
    /// начинает считать другими весами.
    Router,
    Vision,
    Audio,
    Vae,
    Lora,
    Other,
}

impl LayerRole {
    /// Стабильный машинный ключ. UI переводит его через свои каталоги строк,
    /// CLI печатает как есть.
    pub fn key(self) -> &'static str {
        match self {
            LayerRole::Router => "router",
            LayerRole::Embedding => "embedding",
            LayerRole::LmHead => "lm_head",
            LayerRole::Attention => "attention",
            LayerRole::Mlp => "mlp",
            LayerRole::Norm => "norm",
            LayerRole::Conv => "conv",
            LayerRole::Conditioning => "conditioning",
            LayerRole::Head => "head",
            LayerRole::Vision => "vision",
            LayerRole::Audio => "audio",
            LayerRole::Vae => "vae",
            LayerRole::Lora => "lora",
            LayerRole::Other => "other",
        }
    }
}

/// Квант-формат, для которого считается размер. Раскладка задана ядрами
/// synaptix (`quantize_nvfp4` / `quantize_mxfp8`) и от исходного dtype не
/// зависит — только от формы.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// 4 бита на вес + блочные E4M3-масштабы. Требует форму `[N, K]` с
    /// `N % 64 == 0` и `K % 64 == 0` (условие `QLinear::build`).
    Nvfp4,
    /// Байт на вес + масштаб на каждые 32 элемента строки. Требует
    /// `K % 32 == 0`.
    Mxfp8,
}

/// Сколько займёт тензор после квантования. `None` — форма не подходит, вес
/// останется плотным.
///
/// Формулы повторяют аллокации ядер:
/// * NVFP4 — `packed = N·K/2`, `scales = ceil(K/64)·4 · ceil(N/128)·128`;
/// * MXFP8 — `packed = N·K`, `scales = N·K/32`.
///
/// Расхождение с реальностью здесь означало бы враньё в UI («230 ГБ → 56 ГБ»),
/// поэтому округления взяты ровно такими же, как в `cuda_backend`.
pub fn quantized_bytes(shape: &[usize], kind: QuantKind) -> Option<u64> {
    // Ранг 3 — стопка матриц: так лежат веса экспертов MoE
    // (`[512, 1280, 2560]` у Qwen3.8-Flash). Каждый эксперт квантуется
    // отдельной матрицей, поэтому размер просто умножается на их число.
    // Ранг 4+ — это свёртки, их квант-ядра не берут.
    let (stack, n, k) = match shape {
        [n, k] => (1u64, *n as u64, *k as u64),
        [e, n, k] => (*e as u64, *n as u64, *k as u64),
        _ => return None,
    };
    if stack == 0 || n == 0 || k == 0 {
        return None;
    }
    match kind {
        QuantKind::Nvfp4 => {
            if n % 64 != 0 || k % 64 != 0 {
                return None;
            }
            let packed = n * k / 2;
            let s_rows = k.div_ceil(64) * 4;
            let s_cols = n.div_ceil(128) * 128;
            Some(stack * (packed + s_rows * s_cols))
        }
        QuantKind::Mxfp8 => {
            if k % 32 != 0 {
                return None;
            }
            Some(stack * (n * k + n * (k / 32)))
        }
    }
}

/// Вес набора тензоров в трёх вариантах: как есть и после каждого кванта.
/// Тензоры, чью форму квант не берёт, входят во все три числа своим
/// исходным размером — так «после» остаётся честным итогом, а не мечтой.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SizeEstimate {
    pub dense: u64,
    pub nvfp4: u64,
    pub mxfp8: u64,
}

impl SizeEstimate {
    pub fn add(&mut self, t: &TensorInfo) {
        self.dense = self.dense.saturating_add(t.bytes);
        self.nvfp4 = self
            .nvfp4
            .saturating_add(quantized_bytes(&t.shape, QuantKind::Nvfp4).unwrap_or(t.bytes));
        self.mxfp8 = self
            .mxfp8
            .saturating_add(quantized_bytes(&t.shape, QuantKind::Mxfp8).unwrap_or(t.bytes));
    }

    /// Размер при выбранном формате; `None` — без кванта.
    pub fn for_kind(&self, kind: Option<QuantKind>) -> u64 {
        match kind {
            None => self.dense,
            Some(QuantKind::Nvfp4) => self.nvfp4,
            Some(QuantKind::Mxfp8) => self.mxfp8,
        }
    }
}

/// Группа тензоров с одинаковым «узором» имени.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    /// Имя с схлопнутыми числами: `model.layers.[0-47].self_attn.q_proj.weight`.
    pub pattern: String,
    /// То же с плейсхолдером вместо диапазона (`model.layers.{}.…`) —
    /// стабильный ключ: по нему упаковщик сопоставляет тензор с выбором
    /// точности, сделанным в UI. В `pattern` диапазон уже подставлен и для
    /// сопоставления не годится.
    pub key: String,
    pub role: LayerRole,
    /// Сколько тензоров попало в группу.
    pub count: usize,
    /// Общий dtype или `"mixed"`, если внутри группы он разный.
    pub dtype: String,
    /// Форма одного тензора; пусто, если формы в группе разные.
    pub shape: Vec<usize>,
    pub bytes: u64,
    /// Вес группы как есть и после каждого из квантов.
    pub size: SizeEstimate,
}

/// Прочитать заголовок safetensors-файла: только `u64` длины и JSON за ним.
/// Возвращает тензоры в порядке возрастания `data_offsets` — так строки
/// инспектора идут в том же порядке, в каком веса лежат на диске.
pub fn read_header_file(path: &Path) -> Result<Vec<TensorInfo>> {
    let mut f = std::fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf);
    if len == 0 || len > MAX_HEADER_LEN {
        return Err(Error::Safetensors(format!(
            "{}: неправдоподобная длина заголовка ({len} байт)",
            path.display()
        )));
    }
    let mut json = vec![0u8; len as usize];
    f.read_exact(&mut json)?;
    parse_header_json(&json)
}

/// То же для уже отображённого в память потока (payload `tensors:*`-чанка).
pub fn read_header_slice(bytes: &[u8]) -> Result<Vec<TensorInfo>> {
    if bytes.len() < 8 {
        return Err(Error::Safetensors("safetensors-поток короче 8 байт".into()));
    }
    let len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let end = 8u64
        .checked_add(len)
        .filter(|e| *e <= bytes.len() as u64 && len <= MAX_HEADER_LEN)
        .ok_or_else(|| {
            Error::Safetensors(format!("заголовок ({len} байт) не помещается в поток"))
        })?;
    parse_header_json(&bytes[8..end as usize])
}

/// `__metadata__` из заголовка файла — например `config` у одиночных
/// чекпойнтов LTX, откуда берётся архитектура, когда `config.json` рядом нет.
pub fn read_metadata_file(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut f = std::fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf);
    if len == 0 || len > MAX_HEADER_LEN {
        return Ok(BTreeMap::new());
    }
    let mut json = vec![0u8; len as usize];
    f.read_exact(&mut json)?;
    let v: serde_json::Value = serde_json::from_slice(&json)
        .map_err(|e| Error::Safetensors(format!("{}: заголовок не JSON: {e}", path.display())))?;
    let Some(meta) = v.get("__metadata__").and_then(|m| m.as_object()) else {
        return Ok(BTreeMap::new());
    };
    Ok(meta
        .iter()
        .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
        .collect())
}

fn parse_header_json(json: &[u8]) -> Result<Vec<TensorInfo>> {
    let v: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| Error::Safetensors(format!("заголовок safetensors не JSON: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| Error::Safetensors("заголовок safetensors не объект".into()))?;
    let mut out: Vec<(u64, TensorInfo)> = Vec::with_capacity(obj.len());
    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        let Some(spec) = spec.as_object() else { continue };
        let dtype = spec
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("?")
            .to_string();
        let shape: Vec<usize> = spec
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .unwrap_or_default();
        let offsets = spec.get("data_offsets").and_then(|o| o.as_array());
        let (start, end) = match offsets {
            Some(a) if a.len() == 2 => (
                a[0].as_u64().unwrap_or(0),
                a[1].as_u64().unwrap_or(0),
            ),
            _ => (0, 0),
        };
        out.push((
            start,
            TensorInfo {
                name: name.clone(),
                dtype,
                shape,
                bytes: end.saturating_sub(start),
            },
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.name.cmp(&b.1.name)));
    Ok(out.into_iter().map(|(_, t)| t).collect())
}

/// Роль тензора по его имени. Чистая эвристика на префиксах/сегментах —
/// ошибиться она может только в подписи строки инспектора, на упаковку
/// это не влияет.
pub fn classify(name: &str) -> LayerRole {
    let n = name.to_ascii_lowercase();

    // LoRA — раньше всего: `…self_attn.q_proj.lora_A.weight` иначе уедет
    // в Attention, а пользователю важно видеть, что это адаптер.
    if n.contains("lora_a") || n.contains("lora_b") || n.contains(".lora.") || n.contains("lora_down")
    {
        return LayerRole::Lora;
    }
    // Башни: префикс сильнее внутренностей (`visual.blocks.0.attn.qkv.weight`).
    if starts_with_any(
        &n,
        &["visual.", "vision_tower.", "vision_model.", "image_encoder.", "vision."],
    ) || n.contains(".visual.")
        || n.contains(".vision_tower.")
    {
        return LayerRole::Vision;
    }
    if starts_with_any(
        &n,
        &["audio_tower.", "audio_encoder.", "audio_model.", "codec.", "audio_vae."],
    ) || n.contains(".audio_tower.")
    {
        return LayerRole::Audio;
    }
    if starts_with_any(&n, &["vae.", "video_vae.", "first_stage_model."])
        || n.contains("quant_conv")
        || n.contains("post_quant")
        || n.contains("decoder.up_blocks")
        || n.contains("encoder.down_blocks")
    {
        return LayerRole::Vae;
    }
    // Языковая голова.
    if n.starts_with("lm_head") || n.contains(".lm_head.") || n == "output.weight" {
        return LayerRole::LmHead;
    }
    // Нормировки — до attention/mlp: `self_attn.q_norm.weight` это норма, а не
    // проекция, и квантованию она не подлежит. Оговорка про adaln/modulation
    // нужна из-за diffusers, где `norm1.linear.weight` — не нормировка, а
    // модуляционная проекция на пару порядков крупнее.
    let is_modulation = n.contains("adaln")
        || n.contains("ada_ln")
        || n.contains("modulation")
        || n.contains("time_embed")
        || n.contains("t_embedder")
        || n.contains("timestep")
        || n.contains("cond_proj")
        || n.contains("condition_proj")
        || n.contains("guidance");
    if !is_modulation && (n.contains("norm") || n.contains("layernorm") || n.contains(".ln_")) {
        return LayerRole::Norm;
    }
    if is_modulation {
        return LayerRole::Conditioning;
    }
    if n.starts_with("final_layer")
        || n.contains(".final_layer")
        || n.contains("proj_out")
        || n.contains(".head.")
    {
        return LayerRole::Head;
    }
    if n.contains("embed_tokens")
        || n.contains("word_embeddings")
        || n.contains("patch_embed")
        || n.contains("patch_proj")
        || n.contains("pos_embed")
        || n.contains("proj_in")
        || n.contains("embedder")
        || n.contains("embedding")
        || n == "shared.weight"
        || n.ends_with(".wte.weight")
    {
        return LayerRole::Embedding;
    }
    if n.contains("self_attn.")
        || n.contains("linear_attn.")
        || n.contains("cross_attn.")
        || n.contains("encoder_attn.")
        || n.contains(".attn.")
        || n.contains("attention.")
    {
        return LayerRole::Attention;
    }
    // Роутер MoE — до общего правила MLP: `mlp.gate.weight` лежит внутри
    // `.mlp.`, но это не проекция FFN. `gate_proj` сюда не попадает — у него
    // другое имя.
    if n.ends_with(".gate.weight")
        || n.ends_with(".gate.bias")
        || n.contains("shared_expert_gate")
        || n.contains("router.")
        || n.contains("e_score_correction_bias")
    {
        return LayerRole::Router;
    }
    if n.contains(".mlp.")
        || n.contains(".ffn.")
        || n.contains(".ff.")
        || n.contains("feed_forward")
        || n.contains("experts.")
        || n.contains("gate_proj")
        || n.contains("up_proj")
        || n.contains("down_proj")
        || n.contains("linear_fc")
        || n.contains(".intermediate.dense")
        || n.contains(".output.dense")
        || n.contains(".fc1.")
        || n.contains(".fc2.")
    {
        return LayerRole::Mlp;
    }
    // Свёртки — последними: `patch_embed.proj` и `quant_conv` выше уже
    // разобраны более точными правилами.
    if n.contains("conv") {
        return LayerRole::Conv;
    }
    LayerRole::Other
}

/// Роль с подсказкой от имени компонента. Внутри чанка `tensors:video_vae`
/// тензоры зовутся просто `encoder.*`/`decoder.*` — без контекста они
/// неотличимы от чего угодно, и весь VAE ссыпался бы в [`LayerRole::Other`].
/// Подсказка применяется только к тому, что не опознано по имени.
pub fn classify_in(name: &str, component: Option<&str>) -> LayerRole {
    let role = classify(name);
    if role != LayerRole::Other {
        return role;
    }
    component_role(component).unwrap_or(LayerRole::Other)
}

/// Роль по имени компонента: `video_vae` → VAE, `text_encoder` → ничего
/// (внутри обычный трансформер, его имена опознаются сами).
fn component_role(component: Option<&str>) -> Option<LayerRole> {
    let c = component?.to_ascii_lowercase();
    if c.contains("vae") {
        return Some(LayerRole::Vae);
    }
    if c.contains("vision") || c.contains("visual") {
        return Some(LayerRole::Vision);
    }
    if c.contains("audio") || c.contains("codec") || c.contains("vocoder") {
        return Some(LayerRole::Audio);
    }
    None
}

fn starts_with_any(n: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| n.starts_with(p))
}

/// Сгруппировать тензоры по «узору» имени, схлопнув числовые сегменты.
/// Порядок групп — по первому вхождению, чтобы инспектор шёл сверху вниз
/// в порядке слоёв, а не по алфавиту. `component` — имя чанка без префикса
/// `tensors:`, подсказка для [`classify_in`].
pub fn group_tensors(tensors: &[TensorInfo], component: Option<&str>) -> Vec<LayerGroup> {
    struct Acc {
        order: usize,
        role: LayerRole,
        count: usize,
        dtype: String,
        shape: Vec<usize>,
        same_shape: bool,
        bytes: u64,
        size: SizeEstimate,
        indices: Vec<i64>,
    }
    let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
    for (i, t) in tensors.iter().enumerate() {
        let (pattern, index) = collapse_digits(&t.name);
        let e = acc.entry(pattern).or_insert_with(|| Acc {
            order: i,
            role: classify_in(&t.name, component),
            count: 0,
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            same_shape: true,
            bytes: 0,
            size: SizeEstimate::default(),
            indices: Vec::new(),
        });
        e.count += 1;
        e.bytes = e.bytes.saturating_add(t.bytes);
        e.size.add(t);
        if e.dtype != t.dtype {
            e.dtype = "mixed".to_string();
        }
        if e.shape != t.shape {
            e.same_shape = false;
        }
        if let Some(idx) = index {
            e.indices.push(idx);
        }
    }
    let mut groups: Vec<(usize, LayerGroup)> = acc
        .into_iter()
        .map(|(pattern, a)| {
            let filled = fill_range(&pattern, &a.indices);
            (
                a.order,
                LayerGroup {
                    pattern: filled,
                    key: pattern,
                    role: a.role,
                    count: a.count,
                    dtype: a.dtype,
                    shape: if a.same_shape { a.shape } else { Vec::new() },
                    bytes: a.bytes,
                    size: a.size,
                },
            )
        })
        .collect();
    groups.sort_by_key(|(order, _)| *order);
    groups.into_iter().map(|(_, g)| g).collect()
}

/// Суммарный вес по ролям — для полосы «из чего состоит модель».
pub fn bytes_by_role(
    tensors: &[TensorInfo],
    component: Option<&str>,
) -> BTreeMap<LayerRole, SizeEstimate> {
    let mut out: BTreeMap<LayerRole, SizeEstimate> = BTreeMap::new();
    for t in tensors {
        out.entry(classify_in(&t.name, component))
            .or_default()
            .add(t);
    }
    out
}

/// Заменить полностью числовые сегменты на `{}` и вернуть первый из них —
/// по нему потом собирается диапазон `[0-47]`. Схлопывается только первый
/// числовой сегмент: имена вроде `blocks.3.experts.7.w1` дают узор
/// `blocks.{}.experts.7.w1`, и эксперты остаются различимыми.
pub fn group_key(name: &str) -> String {
    collapse_digits(name).0
}

fn collapse_digits(name: &str) -> (String, Option<i64>) {
    let mut first: Option<i64> = None;
    let parts: Vec<String> = name
        .split('.')
        .map(|seg| {
            if first.is_none() && !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                first = seg.parse::<i64>().ok();
                "{}".to_string()
            } else {
                seg.to_string()
            }
        })
        .collect();
    (parts.join("."), first)
}

fn fill_range(pattern: &str, indices: &[i64]) -> String {
    if indices.is_empty() {
        return pattern.replace("{}", "*");
    }
    let min = indices.iter().copied().min().unwrap_or(0);
    let max = indices.iter().copied().max().unwrap_or(0);
    let range = if min == max {
        format!("{min}")
    } else {
        format!("[{min}-{max}]")
    };
    pattern.replace("{}", &range)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{safetensors_header, StDtype, StreamTensor};

    fn plan() -> Vec<StreamTensor> {
        vec![
            StreamTensor { name: "model.embed_tokens.weight".into(), dtype: StDtype::BF16, shape: vec![4, 8] },
            StreamTensor { name: "model.layers.0.self_attn.q_proj.weight".into(), dtype: StDtype::BF16, shape: vec![8, 8] },
            StreamTensor { name: "model.layers.1.self_attn.q_proj.weight".into(), dtype: StDtype::BF16, shape: vec![8, 8] },
            StreamTensor { name: "visual.blocks.0.attn.qkv.weight".into(), dtype: StDtype::F16, shape: vec![8, 8] },
            StreamTensor { name: "lm_head.weight".into(), dtype: StDtype::BF16, shape: vec![4, 8] },
        ]
    }

    fn header_bytes() -> Vec<u8> {
        let p = plan();
        let mut buf = safetensors_header(&p, 64).unwrap();
        let data: usize = p.iter().map(|t| t.nbytes() as usize).sum();
        buf.extend(std::iter::repeat(0u8).take(data));
        buf
    }

    #[test]
    fn reads_names_shapes_and_sizes_from_slice() {
        let buf = header_bytes();
        let t = read_header_slice(&buf).unwrap();
        assert_eq!(t.len(), 5);
        let embed = t.iter().find(|x| x.name == "model.embed_tokens.weight").unwrap();
        assert_eq!(embed.shape, vec![4, 8]);
        assert_eq!(embed.dtype, "BF16");
        assert_eq!(embed.bytes, 4 * 8 * 2);
    }

    #[test]
    fn reads_header_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, header_bytes()).unwrap();
        let t = read_header_file(&path).unwrap();
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn truncated_stream_errors_instead_of_panicking() {
        let mut buf = header_bytes();
        buf.truncate(12);
        assert!(read_header_slice(&buf).is_err());
        assert!(read_header_slice(&[0u8; 4]).is_err());
    }

    #[test]
    fn roles_separate_vision_tower_head_and_embedding() {
        assert_eq!(classify("visual.blocks.0.attn.qkv.weight"), LayerRole::Vision);
        assert_eq!(classify("model.embed_tokens.weight"), LayerRole::Embedding);
        assert_eq!(classify("lm_head.weight"), LayerRole::LmHead);
        assert_eq!(classify("model.layers.3.self_attn.q_proj.weight"), LayerRole::Attention);
        assert_eq!(classify("model.layers.3.mlp.gate_proj.weight"), LayerRole::Mlp);
        assert_eq!(classify("model.layers.3.input_layernorm.weight"), LayerRole::Norm);
        assert_eq!(classify("model.layers.3.self_attn.q_norm.weight"), LayerRole::Norm);
        assert_eq!(
            classify("model.layers.3.self_attn.q_proj.lora_A.weight"),
            LayerRole::Lora
        );
        assert_eq!(classify("decoder.up_blocks.0.conv.weight"), LayerRole::Vae);
    }

    /// Роутер MoE решает, какими весами считать токен, и весит крохи —
    /// квантовать его нельзя. Проекции FFN с похожими именами (`gate_proj`,
    /// `gate_up_proj`) при этом обязаны остаться MLP.
    #[test]
    fn moe_router_is_not_an_mlp_projection() {
        assert_eq!(classify("model.layers.0.mlp.gate.weight"), LayerRole::Router);
        assert_eq!(
            classify("model.language_model.layers.0.mlp.shared_expert_gate.weight"),
            LayerRole::Router
        );
        assert_eq!(
            classify("model.layers.0.block_sparse_moe.gate.weight"),
            LayerRole::Router
        );
        assert_eq!(
            classify("model.layers.0.mlp.gate.e_score_correction_bias"),
            LayerRole::Router
        );
        assert_eq!(
            classify("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            LayerRole::Mlp
        );
        assert_eq!(
            classify("model.language_model.layers.0.mlp.experts.down_proj"),
            LayerRole::Mlp
        );
        assert_eq!(
            classify("model.layers.0.mlp.shared_expert.gate_proj.weight"),
            LayerRole::Mlp
        );
    }

    /// Раскладка диффузионного трансформера (MiniMax-H3 / LTX): модуляция
    /// AdaLN весит десятки гигабайт и обязана иметь собственную строку, а
    /// `final_layer.norm` при этом должен остаться нормировкой.
    #[test]
    fn dit_modulation_and_heads_are_named() {
        assert_eq!(classify("blocks.0.adaln_proj.linear.weight"), LayerRole::Conditioning);
        assert_eq!(classify("time_embedder.proj_in.weight"), LayerRole::Conditioning);
        assert_eq!(classify("time_embedder.proj_out.weight"), LayerRole::Conditioning);
        assert_eq!(classify("condition_proj.weight"), LayerRole::Conditioning);
        assert_eq!(classify("final_layer.video_out.weight"), LayerRole::Head);
        assert_eq!(classify("final_layer.norm.weight"), LayerRole::Norm);
        assert_eq!(classify("video_patch_proj.weight"), LayerRole::Embedding);
        assert_eq!(classify("blocks.0.norm1.weight"), LayerRole::Norm);
        assert_eq!(classify("rope.inv_freq"), LayerRole::Other);
    }

    /// Энкодеры (Whisper, BERT/XLM-R, конформер GigaAM) называют FFN иначе,
    /// чем LLM. Без этих правил у `bge-m3` 37 % веса уходило в «прочее».
    #[test]
    fn encoder_families_are_recognised() {
        assert_eq!(classify("model.decoder.layers.0.encoder_attn.k_proj.weight"), LayerRole::Attention);
        assert_eq!(classify("model.encoder.layers.0.fc1.weight"), LayerRole::Mlp);
        assert_eq!(classify("encoder.layer.0.intermediate.dense.weight"), LayerRole::Mlp);
        assert_eq!(classify("encoder.layer.0.output.dense.weight"), LayerRole::Mlp);
        assert_eq!(classify("encoder.layers.0.feed_forward1.linear1.weight"), LayerRole::Mlp);
        assert_eq!(classify("embeddings.position_embeddings.weight"), LayerRole::Embedding);
        assert_eq!(classify("encoder.layers.0.conv.depthwise_conv.weight"), LayerRole::Conv);
    }

    /// Per-layer-эмбеддинги (PLE) у Qwen3.8-Flash — 29 % веса модели.
    /// Слово `embedding` в имени не обязано стоять в конце и не обязано
    /// быть во множественном числе.
    #[test]
    fn per_layer_embeddings_are_embeddings() {
        assert_eq!(
            classify("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_3.weight"),
            LayerRole::Embedding
        );
        // Но эмбеддер шага диффузии — по-прежнему модуляция.
        assert_eq!(classify("time_embedding.linear_1.weight"), LayerRole::Conditioning);
    }

    /// Подсказка по имени компонента раскрашивает только то, что не опознано
    /// по имени тензора, и не перебивает более точные правила.
    #[test]
    fn component_hint_only_fills_the_unknown() {
        assert_eq!(classify_in("decoder.mask_token", Some("video_vae")), LayerRole::Vae);
        assert_eq!(classify_in("decoder.mask_token", Some("main")), LayerRole::Other);
        assert_eq!(
            classify_in("decoder.transformer_blocks.0.ff.w1.weight", Some("video_vae")),
            LayerRole::Mlp
        );
        assert_eq!(classify_in("dec_in_proj.weight", Some("codec")), LayerRole::Audio);
    }

    #[test]
    fn groups_collapse_layer_indices() {
        let buf = header_bytes();
        let tensors = read_header_slice(&buf).unwrap();
        let groups = group_tensors(&tensors, None);
        let attn = groups
            .iter()
            .find(|g| g.pattern.contains("self_attn"))
            .unwrap();
        assert_eq!(attn.pattern, "model.layers.[0-1].self_attn.q_proj.weight");
        assert_eq!(attn.key, "model.layers.{}.self_attn.q_proj.weight");
        assert_eq!(group_key("model.layers.7.self_attn.q_proj.weight"), attn.key);
        assert_eq!(attn.count, 2);
        assert_eq!(attn.role, LayerRole::Attention);
        assert_eq!(attn.bytes, 2 * 8 * 8 * 2);
        // Одиночные имена без чисел остаются как есть.
        assert!(groups.iter().any(|g| g.pattern == "lm_head.weight"));
    }

    #[test]
    fn bytes_by_role_sums_each_part() {
        let buf = header_bytes();
        let tensors = read_header_slice(&buf).unwrap();
        let by_role = bytes_by_role(&tensors, None);
        assert_eq!(by_role[&LayerRole::Vision].dense, 8 * 8 * 2);
        assert_eq!(by_role[&LayerRole::Attention].dense, 2 * 8 * 8 * 2);
        assert_eq!(by_role[&LayerRole::LmHead].dense, 4 * 8 * 2);
    }

    /// Размер после кванта считается по раскладке ядер, а не «поделить на
    /// четыре»: у NVFP4 к упакованным весам добавляются блочные масштабы с
    /// выравниванием, и на реальной матрице MLP это ощутимые проценты.
    #[test]
    fn quantized_size_matches_kernel_layout() {
        // MLP Qwen3-27B: 17408×5120, BF16 = 170 МиБ.
        let shape = vec![17408usize, 5120];
        let dense = 17408u64 * 5120 * 2;
        let nvfp4 = quantized_bytes(&shape, QuantKind::Nvfp4).unwrap();
        // packed = N·K/2, scales = ceil(K/64)·4 · ceil(N/128)·128.
        assert_eq!(nvfp4, 17408 * 5120 / 2 + (5120 / 64 * 4) * (17408 / 128 * 128));
        assert!(nvfp4 * 4 > dense && nvfp4 * 3 < dense, "≈3.5× меньше, а не ровно 4");

        let mxfp8 = quantized_bytes(&shape, QuantKind::Mxfp8).unwrap();
        assert_eq!(mxfp8, 17408 * 5120 + 17408 * (5120 / 32));
        assert!(mxfp8 * 2 > dense);
    }

    #[test]
    fn shapes_the_kernels_reject_stay_dense() {
        // Вектор нормировки — не матрица.
        assert_eq!(quantized_bytes(&[5120], QuantKind::Nvfp4), None);
        // K не кратно 64 → NVFP4 не берёт, MXFP8 (кратно 32) берёт.
        assert_eq!(quantized_bytes(&[128, 96], QuantKind::Nvfp4), None);
        assert!(quantized_bytes(&[128, 96], QuantKind::Mxfp8).is_some());
        // N не кратно 64.
        assert_eq!(quantized_bytes(&[100, 128], QuantKind::Nvfp4), None);
        // Свёртка 5-D (`patch_embed.proj`) — не матрица и не стопка матриц.
        assert_eq!(quantized_bytes(&[1152, 3, 2, 16, 16], QuantKind::Nvfp4), None);
    }

    /// Веса экспертов MoE приходят стопкой `[E, N, K]` — это 69 % веса
    /// Qwen3.8-Flash. Пока их считали неквантуемыми, «после» отличалось от
    /// «до» на доли процента, и выбор точности выглядел бесполезным.
    #[test]
    fn stacked_expert_weights_are_quantized_per_slice() {
        let one = quantized_bytes(&[1280, 2560], QuantKind::Nvfp4).unwrap();
        let stack = quantized_bytes(&[512, 1280, 2560], QuantKind::Nvfp4).unwrap();
        assert_eq!(stack, 512 * one);

        let dense = 512u64 * 1280 * 2560 * 2;
        assert!(stack * 3 < dense, "квант должен давать больше трёх раз");

        // Ранг 3 с негодной формой среза остаётся плотным.
        assert_eq!(quantized_bytes(&[512, 100, 2560], QuantKind::Nvfp4), None);
    }

    #[test]
    fn estimate_counts_unquantizable_tensors_at_full_size() {
        let t = |name: &str, shape: Vec<usize>, bytes: u64| TensorInfo {
            name: name.into(),
            dtype: "BF16".into(),
            shape,
            bytes,
        };
        let mut est = SizeEstimate::default();
        est.add(&t("model.layers.0.mlp.up_proj.weight", vec![128, 128], 128 * 128 * 2));
        est.add(&t("model.layers.0.input_layernorm.weight", vec![128], 256));
        assert_eq!(est.dense, 128 * 128 * 2 + 256);
        // Нормировка входит в «после» своим полным размером.
        assert_eq!(est.nvfp4, quantized_bytes(&[128, 128], QuantKind::Nvfp4).unwrap() + 256);
        assert!(est.nvfp4 < est.dense);
    }
}
