//! Сэмплинг «из коробки»: пресеты, которые авторы модели рекомендуют в её
//! карточке, и уровни глубины размышлений, которые понимает её chat-шаблон.
//!
//! Как и [`super::llm::optimal_profile`], профиль считает движок: он знает,
//! какая архитектура чего ждёт. Приложению остаётся показать выбор и
//! применить его — пресет уходит в `GenerationOptions`, уровень — в
//! [`super::llm::LlmTokenizer::apply_chat_template_reasoning`].

use std::path::Path;

use serde_json::Value as Json;
use synaptix_tokenizer::templates::chat_template::RenderOptions;

use super::arch::{arch_key, detect_llm_arch, read_model_file, LlmArch};

/// Набор параметров сэмплинга под режим работы модели.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingPreset {
    /// Стабильный id для конфига и UI: `thinking`, `thinking_coding`,
    /// `instruct`, `recommended`, `generation_config`, `generic`.
    pub id: &'static str,
    /// Для какого режима пресет: `Some(true)` — с размышлениями,
    /// `Some(false)` — без них, `None` — для обоих.
    pub thinking: Option<bool>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
}

/// Через какую переменную шаблона задаётся глубина размышлений.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningVar {
    /// `reasoning_effort` — Qwen3.8 (`xhigh` / `medium` / `low`).
    Effort,
    /// `reasoning_strength` — Muse Glimmer (`low` … `xhigh`), строкой
    /// `Reasoning strength: …` в системном блоке.
    Strength,
}

/// Уровни глубины размышлений, которые принимает шаблон модели.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningLevels {
    pub var: ReasoningVar,
    /// По возрастанию глубины.
    pub levels: Vec<String>,
    /// Уровень, который шаблон берёт, если его не задать.
    pub default: String,
}

impl ReasoningLevels {
    /// Запрошенный уровень, если шаблон его знает. Иначе `None` — шаблон
    /// возьмёт свой уровень по умолчанию. Сырую строку в шаблон не отдаём:
    /// у Qwen3.8 неизвестный уровень — `raise_exception` и сорванный ход.
    pub fn resolve<'a>(&'a self, requested: Option<&str>) -> Option<&'a str> {
        let requested = requested?.trim();
        self.levels
            .iter()
            .find(|l| l.eq_ignore_ascii_case(requested))
            .map(String::as_str)
    }
}

/// Что модель умеет в сэмплинге: пресеты и уровни размышлений.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SamplingProfile {
    /// Никогда не пуст у профиля из [`sampling_profile`]: если модель ничего
    /// не рекомендует, в нём один `generic`.
    pub presets: Vec<SamplingPreset>,
    /// `None` — глубину размышлений шаблон не настраивает (Gemma-4,
    /// Qwen3.6): размышления только включаются и выключаются.
    pub reasoning: Option<ReasoningLevels>,
}

impl SamplingProfile {
    pub fn preset(&self, id: &str) -> Option<&SamplingPreset> {
        self.presets.iter().find(|p| p.id == id)
    }

    /// Пресет под режим размышлений: `preferred`, если он для этого режима
    /// годится, иначе первый пресет этого режима, иначе первый общий. Так
    /// переключатель размышлений сам меняет «Thinking» на «Instruct», а
    /// выбранный среди нескольких пресетов одного режима (у Qwen3.6 обычный
    /// и для кода) сохраняется.
    pub fn pick(&self, preferred: &str, thinking: bool) -> Option<&SamplingPreset> {
        let fits = |p: &&SamplingPreset| p.thinking.is_none_or(|t| t == thinking);
        self.preset(preferred)
            .filter(fits)
            .or_else(|| self.presets.iter().find(|p| p.thinking == Some(thinking)))
            .or_else(|| self.presets.iter().find(fits))
            .or_else(|| self.presets.first())
    }
}

/// Пресеты и уровни размышлений модели по пути бандла (или HF-каталога).
/// Читает только `config.json`, шаблон и `generation_config.json` — веса не
/// трогает, так что звать можно и до загрузки.
pub fn sampling_profile(path: &Path) -> SamplingProfile {
    let arch = detect_llm_arch(path).ok();
    let key = arch_key(path);
    let template = super::llm::load_template_source(path);
    let gen = read_model_file(path, "generation_config.json");
    profile_from_parts(arch, key.as_deref(), template.as_deref(), gen.as_deref())
}

const fn preset(
    id: &'static str,
    thinking: Option<bool>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
) -> SamplingPreset {
    SamplingPreset {
        id,
        thinking,
        temperature,
        top_p,
        top_k,
        min_p: 0.0,
        presence_penalty,
        repetition_penalty: 1.0,
    }
}

/// Qwen3.5 / 3.6 / 3.8 и Qwen3.8-Flash-Next (карточки моделей): размышления —
/// температура 1.0; без них — 0.7 / 0.8 и `presence_penalty` 1.5 против
/// повторов.
const QWEN35_THINKING: SamplingPreset = preset("thinking", Some(true), 1.0, 0.95, 20, 0.0);
/// Qwen3.5 / 3.6: размышления на точных задачах кода (WebDev).
const QWEN35_THINKING_CODING: SamplingPreset =
    preset("thinking_coding", Some(true), 0.6, 0.95, 20, 0.0);
const QWEN35_INSTRUCT: SamplingPreset = preset("instruct", Some(false), 0.7, 0.8, 20, 1.5);
/// Qwen3 и Qwen3-Next: размышления на 0.6, без штрафов.
const QWEN3_THINKING: SamplingPreset = preset("thinking", Some(true), 0.6, 0.95, 20, 0.0);
const QWEN3_INSTRUCT: SamplingPreset = preset("instruct", Some(false), 0.7, 0.8, 20, 0.0);
/// Gemma-4 и Muse Glimmer: один набор на все задачи.
const TOP_K64_RECOMMENDED: SamplingPreset = preset("recommended", None, 1.0, 0.95, 64, 0.0);
/// Модель ничего не рекомендует: умеренная температура и узкое ядро — то,
/// на чём агентный цикл synthos работал до пресетов.
const GENERIC: SamplingPreset = preset("generic", None, 0.6, 0.95, 20, 0.0);

pub(crate) fn profile_from_parts(
    arch: Option<LlmArch>,
    arch_key: Option<&str>,
    template: Option<&str>,
    generation_config: Option<&[u8]>,
) -> SamplingProfile {
    let reasoning = template.and_then(|t| reasoning_levels(arch, t));
    let key = arch_key.unwrap_or_default();
    let mut presets: Vec<SamplingPreset> = match arch {
        // Qwen3.8 отличается от 3.5/3.6 той же архитектуры только шаблоном:
        // в нём появился `reasoning_effort`, а пресет для кода из карточки ушёл.
        Some(LlmArch::Qwen4Exp) => vec![QWEN35_THINKING, QWEN35_INSTRUCT],
        Some(LlmArch::Hybrid) if key == "qwen3_next" => vec![QWEN3_THINKING, QWEN3_INSTRUCT],
        Some(LlmArch::Hybrid) if reasoning.is_some() => vec![QWEN35_THINKING, QWEN35_INSTRUCT],
        Some(LlmArch::Hybrid) => vec![QWEN35_THINKING, QWEN35_THINKING_CODING, QWEN35_INSTRUCT],
        // `Qwen3` у детектора — ещё и «всё неизвестное», поэтому по ключу.
        Some(LlmArch::Qwen3) if key.starts_with("qwen3") => vec![QWEN3_THINKING, QWEN3_INSTRUCT],
        Some(LlmArch::Gemma4 | LlmArch::MuseGlimmer) => vec![TOP_K64_RECOMMENDED],
        _ => Vec::new(),
    };
    // `generation_config.json` — то, что авторы положили рядом с весами.
    // Показываем его отдельным пресетом, только если он чем-то отличается
    // от уже известных, иначе в списке будут два одинаковых.
    if let Some(gen) = generation_config.and_then(generation_config_preset) {
        if !presets.iter().any(|p| same_values(p, &gen)) {
            presets.push(gen);
        }
    }
    if presets.is_empty() {
        presets.push(GENERIC);
    }
    SamplingProfile { presets, reasoning }
}

fn same_values(a: &SamplingPreset, b: &SamplingPreset) -> bool {
    a.temperature == b.temperature
        && a.top_p == b.top_p
        && a.top_k == b.top_k
        && a.min_p == b.min_p
        && a.presence_penalty == b.presence_penalty
        && a.repetition_penalty == b.repetition_penalty
}

/// Пресет из `generation_config.json`, если там есть хоть одно поле
/// сэмплинга. Недостающие поля — умолчания transformers (temperature 1.0,
/// top_p 1.0, top_k 50): именно так файл читает сам HF.
fn generation_config_preset(bytes: &[u8]) -> Option<SamplingPreset> {
    let v: Json = serde_json::from_slice(bytes).ok()?;
    let f = |k: &str| v.get(k).and_then(Json::as_f64).map(|x| x as f32);
    let has_any = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty"]
        .iter()
        .any(|k| v.get(*k).is_some_and(Json::is_number));
    if !has_any {
        return None;
    }
    Some(SamplingPreset {
        id: "generation_config",
        thinking: None,
        temperature: f("temperature").unwrap_or(1.0),
        top_p: f("top_p").unwrap_or(1.0),
        top_k: v.get("top_k").and_then(Json::as_u64).map_or(50, |x| x as usize),
        min_p: f("min_p").unwrap_or(0.0),
        presence_penalty: f("presence_penalty").unwrap_or(0.0),
        repetition_penalty: f("repetition_penalty").unwrap_or(1.0),
    })
}

/// Уровни размышлений, которые понимает шаблон.
///
/// `reasoning_effort` (Qwen3.8) шаблон перечисляет сам — в проверке
/// `resolved_reasoning_effort not in ('xhigh', 'medium', 'low')` и в
/// `reasoning_effort|default('xhigh')`; отсюда их и берём, чтобы следующая
/// модель с другим набором не требовала правки движка. `reasoning_strength`
/// (Muse Glimmer) шаблон не проверяет, уровни есть только в карточке модели.
pub fn reasoning_levels(arch: Option<LlmArch>, template: &str) -> Option<ReasoningLevels> {
    if template.contains("reasoning_effort") {
        let levels = quoted_list_after(template, "reasoning_effort not in (")?;
        let default = quoted_after(template, "reasoning_effort|default('")
            .filter(|d| levels.contains(d))
            .unwrap_or_else(|| levels.last().cloned().unwrap_or_default());
        return Some(ReasoningLevels { var: ReasoningVar::Effort, levels: by_depth(levels), default });
    }
    if template.contains("reasoning_strength") && matches!(arch, Some(LlmArch::MuseGlimmer)) {
        let levels: Vec<String> = ["low", "medium", "high", "xhigh"].map(String::from).into();
        let default = quoted_after(template, "reasoning_strength else '")
            .filter(|d| levels.contains(d))
            .unwrap_or_else(|| "high".to_string());
        return Some(ReasoningLevels { var: ReasoningVar::Strength, levels, default });
    }
    None
}

/// `'a', 'b', 'c'` сразу после `marker` до закрывающей скобки.
fn quoted_list_after(src: &str, marker: &str) -> Option<Vec<String>> {
    let start = src.find(marker)? + marker.len();
    let end = start + src[start..].find(')')?;
    let items: Vec<String> = src[start..end]
        .split(',')
        .map(|s| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (!items.is_empty()).then_some(items)
}

/// Строка до ближайшей `'` сразу после `marker`.
fn quoted_after(src: &str, marker: &str) -> Option<String> {
    let start = src.find(marker)? + marker.len();
    let end = start + src[start..].find('\'')?;
    Some(src[start..end].to_string())
}

fn by_depth(mut levels: Vec<String>) -> Vec<String> {
    const ORDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
    let rank = |s: &String| ORDER.iter().position(|o| o.eq_ignore_ascii_case(s)).unwrap_or(ORDER.len());
    levels.sort_by_key(rank);
    levels
}

/// Переменные размышлений для рендера шаблона.
///
/// `enable_thinking` отдаём всегда. `reasoning_strength` — тоже: канальный
/// шаблон Muse Glimmer не знает `enable_thinking`, и совсем отключить
/// размышления его протокол не позволяет, поэтому «без размышлений» там —
/// `low`. `reasoning_effort` — только уровнем, который шаблон знает, и
/// только при включённых размышлениях: без них Qwen3.8 инструкцию не рендерит.
pub(crate) fn with_thinking_vars(
    mut opts: RenderOptions,
    levels: Option<&ReasoningLevels>,
    enable_thinking: bool,
    effort: Option<&str>,
) -> RenderOptions {
    let chosen = if enable_thinking { levels.and_then(|l| l.resolve(effort)) } else { None };
    let strength = match levels {
        _ if !enable_thinking => "low",
        Some(l) if l.var == ReasoningVar::Strength => chosen.unwrap_or(&l.default),
        _ => "high",
    };
    opts = opts
        .with_var("enable_thinking", Json::Bool(enable_thinking))
        .with_var("reasoning_strength", Json::String(strength.into()));
    if let (Some(l), Some(level)) = (levels, chosen) {
        if l.var == ReasoningVar::Effort {
            opts = opts.with_var("reasoning_effort", Json::String(level.into()));
        }
    }
    opts
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message};

    const QWEN38: &str =
        include_str!("../../../synaptix-tokenizer/tests/fixtures/qwen3_8_chat_template.jinja");
    const MUSE: &str =
        include_str!("../../../synaptix-tokenizer/tests/fixtures/muse_glimmer_chat_template.jinja");

    fn render(template: &str, arch: LlmArch, thinking: bool, effort: Option<&str>) -> String {
        let levels = reasoning_levels(Some(arch), template);
        let opts = with_thinking_vars(
            RenderOptions::new().with_generation_prompt(true),
            levels.as_ref(),
            thinking,
            effort,
        );
        ChatTemplate::from_source(template)
            .render(&[Message::system("sys"), Message::user("hi")], &opts)
            .expect("render")
    }

    #[test]
    fn qwen38_levels_come_from_template() {
        let l = reasoning_levels(Some(LlmArch::Qwen4Exp), QWEN38).expect("levels");
        assert_eq!(l.var, ReasoningVar::Effort);
        assert_eq!(l.levels, ["low", "medium", "xhigh"]);
        assert_eq!(l.default, "xhigh");
    }

    #[test]
    fn muse_levels_and_default() {
        let l = reasoning_levels(Some(LlmArch::MuseGlimmer), MUSE).expect("levels");
        assert_eq!(l.var, ReasoningVar::Strength);
        assert_eq!(l.levels, ["low", "medium", "high", "xhigh"]);
        assert_eq!(l.default, "high");
    }

    #[test]
    fn qwen38_effort_reaches_prompt() {
        assert!(render(QWEN38, LlmArch::Qwen4Exp, true, Some("low"))
            .contains("Reasoning effort is set to low."));
        // medium у шаблона — без инструкции вовсе.
        assert!(!render(QWEN38, LlmArch::Qwen4Exp, true, Some("medium")).contains("Reasoning effort"));
        // Не задан — уровень шаблона, промпт тот же, что был до уровней:
        // префикс-KV старых чатов не сбрасывается.
        let default = render(QWEN38, LlmArch::Qwen4Exp, true, None);
        assert!(default.contains("Reasoning effort is set to xhigh."));
        assert_eq!(default, render(QWEN38, LlmArch::Qwen4Exp, true, Some("xhigh")));
    }

    #[test]
    fn unknown_effort_falls_back_instead_of_raising() {
        let out = render(QWEN38, LlmArch::Qwen4Exp, true, Some("turbo"));
        assert!(out.contains("Reasoning effort is set to xhigh."), "{out}");
    }

    #[test]
    fn effort_is_dropped_without_thinking() {
        let out = render(QWEN38, LlmArch::Qwen4Exp, false, Some("low"));
        assert!(!out.contains("Reasoning effort"), "{out}");
        assert!(out.ends_with("<think>\n\n</think>\n\n"), "{out}");
    }

    #[test]
    fn muse_strength_reaches_prompt() {
        assert!(render(MUSE, LlmArch::MuseGlimmer, true, Some("xhigh")).contains("Reasoning strength: xhigh."));
        assert!(render(MUSE, LlmArch::MuseGlimmer, true, None).contains("Reasoning strength: high."));
        assert!(render(MUSE, LlmArch::MuseGlimmer, false, Some("xhigh")).contains("Reasoning strength: low."));
    }

    #[test]
    fn flash_next_presets_match_model_card() {
        let p = profile_from_parts(Some(LlmArch::Qwen4Exp), Some("qwen4_exp"), Some(QWEN38), None);
        let ids: Vec<_> = p.presets.iter().map(|p| p.id).collect();
        assert_eq!(ids, ["thinking", "instruct"]);
        let t = p.preset("thinking").unwrap();
        assert_eq!((t.temperature, t.top_p, t.top_k, t.min_p), (1.0, 0.95, 20, 0.0));
        assert_eq!((t.presence_penalty, t.repetition_penalty), (0.0, 1.0));
        let i = p.preset("instruct").unwrap();
        assert_eq!((i.temperature, i.top_p, i.top_k, i.min_p), (0.7, 0.8, 20, 0.0));
        assert_eq!((i.presence_penalty, i.repetition_penalty), (1.5, 1.0));
    }

    #[test]
    fn qwen36_has_coding_preset_and_no_levels() {
        let tmpl = "{%- if enable_thinking is defined and enable_thinking is false %}{% endif %}";
        let p = profile_from_parts(Some(LlmArch::Hybrid), Some("qwen3_5"), Some(tmpl), None);
        assert!(p.reasoning.is_none());
        let ids: Vec<_> = p.presets.iter().map(|p| p.id).collect();
        assert_eq!(ids, ["thinking", "thinking_coding", "instruct"]);
    }

    #[test]
    fn pick_follows_thinking_and_keeps_same_mode_choice() {
        let tmpl = "";
        let p = profile_from_parts(Some(LlmArch::Hybrid), Some("qwen3_5"), Some(tmpl), None);
        assert_eq!(p.pick("thinking_coding", true).unwrap().id, "thinking_coding");
        assert_eq!(p.pick("thinking_coding", false).unwrap().id, "instruct");
        assert_eq!(p.pick("instruct", true).unwrap().id, "thinking");
        assert_eq!(p.pick("", false).unwrap().id, "instruct");
        let g = profile_from_parts(Some(LlmArch::Gemma4), Some("gemma4"), None, None);
        assert_eq!(g.pick("instruct", false).unwrap().id, "recommended");
    }

    #[test]
    fn generation_config_is_added_only_when_new() {
        let same = br#"{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "eos_token_id": [1]}"#;
        let p = profile_from_parts(Some(LlmArch::Gemma4), Some("gemma4"), None, Some(same));
        assert_eq!(p.presets.len(), 1);

        let llama = br#"{"temperature": 0.6, "top_p": 0.9, "eos_token_id": 128009}"#;
        let p = profile_from_parts(Some(LlmArch::Llama), Some("llama"), None, Some(llama));
        let g = p.preset("generation_config").expect("gen preset");
        assert_eq!((g.temperature, g.top_p, g.top_k), (0.6, 0.9, 50));

        let eos_only = br#"{"eos_token_id": [1, 2]}"#;
        let p = profile_from_parts(None, Some("mystery"), None, Some(eos_only));
        assert_eq!(p.presets.iter().map(|p| p.id).collect::<Vec<_>>(), ["generic"]);
    }
}
