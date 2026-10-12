use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Llm,
    Tts,
    Asr,
    Image,
    Video,
    Music,
    Ctc,
    Mfa,
    Embed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    Text { multiline: bool },
    Int { min: i64, max: i64, step: i64 },
    Float { min: f64, max: f64, step: f64, decimals: u8 },
    Bool,
    Choice { options: Vec<String> },
    /// Путь к файлу; `patterns` — MIME-типы или маски для диалога выбора.
    File { patterns: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    pub kind: ParamKind,
    pub default: Option<ParamValue>,
}

fn spec(key: &'static str, label: &'static str, hint: &'static str, kind: ParamKind, default: Option<ParamValue>) -> ParamSpec {
    ParamSpec { key, label, hint, kind, default }
}

fn float(min: f64, max: f64, step: f64, decimals: u8) -> ParamKind {
    ParamKind::Float { min, max, step, decimals }
}

fn int(min: i64, max: i64, step: i64) -> ParamKind {
    ParamKind::Int { min, max, step }
}

fn f(v: f64) -> Option<ParamValue> {
    Some(ParamValue::Float(v))
}

fn i(v: i64) -> Option<ParamValue> {
    Some(ParamValue::Int(v))
}

fn seed() -> ParamSpec {
    spec("seed", "Зерно (seed)", "−1 — случайное; одно зерно даёт повторяемый результат", int(-1, u32::MAX as i64, 1), i(-1))
}

fn steps(default: Option<i64>, hint: &'static str) -> ParamSpec {
    spec("steps", "Шаги", hint, int(1, 200, 1), default.and_then(i))
}

fn guidance(default: f64, label: &'static str) -> ParamSpec {
    spec("guidance", label, "Насколько строго следовать промпту", float(0.0, 30.0, 0.5, 2), f(default))
}

fn negative() -> ParamSpec {
    spec("negative_prompt", "Негативный промпт", "Чего не должно быть в результате", ParamKind::Text { multiline: true }, None)
}

fn size(w: i64, h: i64) -> [ParamSpec; 2] {
    [
        spec("width", "Ширина", "В пикселях", int(64, 4096, 64), i(w)),
        spec("height", "Высота", "В пикселях", int(64, 4096, 64), i(h)),
    ]
}

fn duration(default: f64, max: f64) -> ParamSpec {
    spec("duration", "Длительность, с", "Длина результата в секундах", float(0.5, max, 0.5, 1), f(default))
}

/// Образец голоса для клонирования — у всех движков синтеза речи.
fn ref_audio() -> ParamSpec {
    spec("ref_audio", "Образец голоса", "WAV-запись речи одного человека на 3–10 секунд: голос клонируется с неё", ParamKind::File { patterns: vec!["audio/wav".into(), "audio/x-wav".into(), "*.wav".into()] }, None)
}

fn ref_text() -> ParamSpec {
    spec("ref_text", "Текст образца", "Что именно сказано в записи, слово в слово", ParamKind::Text { multiline: true }, None)
}

fn language(hint: &'static str) -> ParamSpec {
    spec("language", "Язык", hint, ParamKind::Text { multiline: false }, None)
}

pub fn param_schema(task: Task, arch: &str, id: &str, path: Option<&Path>) -> Vec<ParamSpec> {
    let arch = arch.to_ascii_lowercase();
    let id = id.to_ascii_lowercase();
    match task {
        Task::Llm => llm(path),
        Task::Tts => match arch.as_str() {
            "vibevoice" => vec![ref_audio(), steps(Some(20), "Шаги диффузии"), guidance(1.3, "Guidance (CFG)"), seed()],
            "voxcpm" | "voxcpm2" => vec![
                spec(
                    "voice",
                    "Голос",
                    "Описание голоса, лучше по-английски: пол, возраст, тембр, эмоция (например, young woman, gentle and sweet voice). С образцом — манера поверх его тембра (slightly faster, cheerful tone); с текстом образца не действует",
                    ParamKind::Text { multiline: true },
                    None,
                ),
                ref_audio(),
                ref_text(),
                steps(Some(10), "Шаги диффузии"),
                guidance(2.0, "Guidance (CFG)"),
                seed(),
            ],
            _ => vec![
                language("Код языка (ru, en, kk); пусто — определять по тексту"),
                spec("voice", "Голос", "Описание голоса по-английски: пол, возраст, манера (например, calm female voice); цифры и русский текст модель может прочитать вслух. Не нужно, если задан образец", ParamKind::Text { multiline: true }, None),
                ref_audio(),
                ref_text(),
                spec("speed", "Скорость речи", "1 — обычная", float(0.25, 4.0, 0.05, 2), f(1.0)),
                steps(None, "Шаги маскированного декода; авто — 16 на NPU, 32 на остальных"),
                guidance(2.0, "Guidance (CFG)"),
            ],
        },
        Task::Asr => vec![language("Код языка (ru, en, kk); пусто — определять самой")],
        Task::Image => {
            let [w, h] = size(1024, 1024);
            match arch.as_str() {
                "sdxl" => vec![negative(), steps(Some(30), "Больше — качественнее и дольше"), guidance(5.0, "CFG"), w, h, seed()],
                "flux" => vec![steps(Some(28), "Больше — качественнее и дольше"), guidance(3.5, "Guidance"), w, h, seed()],
                "flux2" => vec![steps(None, "Авто — по варианту модели (dev 50, klein 4)"), guidance(4.0, "Guidance"), w, h, seed()],
                "qwen_image" => vec![negative(), steps(None, "Авто — по варианту модели"), guidance(4.0, "CFG"), w, h, seed()],
                "qwen_image21" => vec![steps(None, "Авто — 40"), guidance(1.0, "CFG"), w, h, seed()],
                _ => vec![negative(), steps(None, "Больше — качественнее и дольше"), w, h, seed()],
            }
        }
        Task::Video => match arch.as_str() {
            "minimax_h3" | "h3" => {
                let [w, h] = size(1344, 768);
                vec![negative(), steps(Some(20), "Больше — качественнее и дольше"), guidance(5.0, "CFG"), w, h, duration(5.0, 20.0), seed()]
            }
            _ => {
                let [w, h] = size(1024, 576);
                vec![
                    negative(),
                    w,
                    h,
                    duration(10.0, 20.0),
                    spec("fps", "Кадров в секунду", "Частота кадров видео", ParamKind::Choice { options: vec!["24".into(), "25".into(), "48".into(), "50".into()] }, Some(ParamValue::Text("24".into()))),
                    seed(),
                ]
            }
        },
        Task::Music => match arch.as_str() {
            "yue2" => vec![spec("lyrics", "Текст песни", "Куплеты и припевы", ParamKind::Text { multiline: true }, None), duration(60.0, 300.0), seed()],
            _ => {
                let (st, cfg) = if id.contains("turbo") {
                    (8, 1.0)
                } else if id.contains("sft") {
                    (50, 7.0)
                } else {
                    (32, 7.0)
                };
                vec![
                    spec("lyrics", "Текст песни", "Пусто — инструментал", ParamKind::Text { multiline: true }, None),
                    duration(30.0, 240.0),
                    steps(Some(st), "Больше — качественнее и дольше"),
                    guidance(cfg, "CFG"),
                    seed(),
                ]
            }
        },
        Task::Ctc | Task::Mfa | Task::Embed => Vec::new(),
    }
}

fn llm(path: Option<&Path>) -> Vec<ParamSpec> {
    let profile = path.map(super::sampling::sampling_profile).unwrap_or_default();
    let template = path.and_then(super::llm::load_template_source).unwrap_or_default();
    let preset = profile.pick("", false).or_else(|| profile.presets.first());
    let p = |get: fn(&super::sampling::SamplingPreset) -> f64, fallback: f64| f(preset.map(get).unwrap_or(fallback));
    let window = path.and_then(super::arch::config_max_seq).unwrap_or(131072).max(2048);
    let mut out = vec![
        spec("system_prompt", "Стартовый промпт", "Системное сообщение в начале диалога, если программа не задала своё", ParamKind::Text { multiline: true }, None),
        spec("context", "Размер контекста, токенов", "Сколько токенов модель держит в памяти: промпт, история и ответ. Кэш растёт по мере разговора до этого предела", int(2048, window as i64, 1024), i(window as i64)),
        spec("max_tokens", "Максимум генерируемых токенов", "Предел длины одного ответа вместе с размышлениями", int(16, window as i64, 64), i(16384.min(window as i64))),
        spec("temperature", "Температура", "Выше — разнообразнее, ниже — точнее", float(0.0, 2.0, 0.05, 2), p(|s| s.temperature as f64, 0.7)),
        spec("top_p", "Top-p", "Доля вероятности, из которой выбирается слово", float(0.0, 1.0, 0.05, 2), p(|s| s.top_p as f64, 0.95)),
        spec("top_k", "Top-k", "Из скольких самых вероятных слов выбирать; 0 — без ограничения", int(0, 1000, 1), i(preset.map(|s| s.top_k as i64).unwrap_or(20))),
        spec("min_p", "Min-p", "Отсекать слова вероятностью ниже этой доли от лучшего", float(0.0, 1.0, 0.01, 2), p(|s| s.min_p as f64, 0.0)),
        spec("repeat_penalty", "Штраф за повторы", "1 — без штрафа", float(0.5, 2.0, 0.05, 2), p(|s| s.repetition_penalty as f64, 1.0)),
        spec("presence_penalty", "Штраф за присутствие", "Против повторов уже сказанного; 0 — без штрафа", float(-2.0, 2.0, 0.1, 2), p(|s| s.presence_penalty as f64, 0.0)),
    ];
    if template.contains("enable_thinking") {
        out.push(spec("thinking", "Размышления", "Модель сначала рассуждает, потом отвечает — точнее и дольше", ParamKind::Bool, Some(ParamValue::Bool(false))));
    }
    if let Some(r) = &profile.reasoning {
        out.push(spec(
            "reasoning",
            "Глубина размышлений",
            "Сколько модель думает перед ответом",
            ParamKind::Choice { options: r.levels.clone() },
            Some(ParamValue::Text(r.default.clone())),
        ));
    }
    out.push(seed());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_differ() {
        let flux = param_schema(Task::Image, "flux", "flux-1-dev", None);
        let sdxl = param_schema(Task::Image, "sdxl", "sdxl-base", None);
        assert!(!flux.iter().any(|p| p.key == "negative_prompt"));
        assert!(sdxl.iter().any(|p| p.key == "negative_prompt"));
        let ltx = param_schema(Task::Video, "ltx23", "ltx-2.3", None);
        assert!(!ltx.iter().any(|p| p.key == "steps"));
        let turbo = param_schema(Task::Music, "acestep", "acestep-v15-turbo", None);
        assert_eq!(turbo.iter().find(|p| p.key == "steps").unwrap().default, Some(ParamValue::Int(8)));
        assert!(param_schema(Task::Ctc, "wav2vec2", "w2v2", None).is_empty());
    }

    #[test]
    fn llm_without_files() {
        let s = param_schema(Task::Llm, "gemma4", "g", None);
        assert!(s.iter().any(|p| p.key == "temperature"));
        assert!(!s.iter().any(|p| p.key == "reasoning"));
    }
}
