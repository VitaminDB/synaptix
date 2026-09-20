//! Грамматика промпта и дефолты генерации YuE2 (`yue2-native-v1`).
//!
//! Числа здесь — не наш выбор, а протокол чекпойнта: спец-токены сидят в
//! словаре на фиксированных местах, инструкции входят в префикс дословно.
//! Менять их нельзя, не сломав модель, поэтому они константы, а не настройки.

use crate::tokenizer::Yue2Tokenizer;
use crate::YueError;

/// `<|endoftext|>` — первый токен любого префикса.
pub const EOD: u32 = 151643;
/// Границы партитуры (ABC).
pub const ABC_START: u32 = 151847;
pub const ABC_END: u32 = 151848;
/// Границы музыки (семантические токены).
pub const MUSIC_START: u32 = 151851;
pub const MUSIC_END: u32 = 151852;
/// Семантический токен `c` лежит в словаре по адресу `CODEC_OFFSET + c`.
pub const CODEC_OFFSET: u32 = 151853;
pub const CODEC_SIZE: u32 = 32768;
/// Служебные позиции NAR-ветки (токенами не генерируются).
pub const LATENT_START: u32 = 184621;
pub const LATENT_END: u32 = 184622;
pub const LATENT_PAD: u32 = 184623;
pub const VOCAB_SIZE: usize = 184704;
/// Окно модели: AR-префикс + NAR-позиции обязаны уместиться целиком.
pub const CONTEXT: usize = 24576;
pub const PROTOCOL_VERSION: &str = "yue2-native-v1";

/// Один семантический токен = один латентный кадр = 1920 сэмплов при 48 кГц.
pub const FRAMES_PER_SECOND: f32 = 25.0;
pub const SAMPLE_RATE: u32 = 48000;

/// Режим «цепочки рассуждения»: сколько партитуры модель пишет перед музыкой.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cot {
    /// Без партитуры — сразу семантические токены.
    Off,
    /// Только мелодия, без аккордовых символов (рекомендуется для каверов).
    Melody,
    /// Мелодия с аккордами — дефолт для новых песен.
    Full,
}

impl Cot {
    pub fn as_str(self) -> &'static str {
        match self {
            Cot::Off => "off",
            Cot::Melody => "melody",
            Cot::Full => "full",
        }
    }

    pub fn parse(s: &str) -> Option<Cot> {
        match s {
            "off" => Some(Cot::Off),
            "melody" => Some(Cot::Melody),
            "full" => Some(Cot::Full),
            _ => None,
        }
    }

    /// Инструкция, с которой начинается текстовый префикс. Входит в веса —
    /// переводить и переписывать нельзя.
    pub fn instruction(self) -> &'static str {
        match self {
            Cot::Off => "Generate music with codec tokens from the given conditions.",
            Cot::Melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
            Cot::Full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        }
    }

    /// Guidance по умолчанию: у `off` ветка CFG чуть-чуть включена.
    pub fn default_guidance(self) -> f32 {
        if self == Cot::Off {
            1.01
        } else {
            1.0
        }
    }
}

/// Параметры сэмплинга одной фазы (партитура или музыка).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// Штраф считается по последним `penalty_window` выданным токенам.
    pub penalty_window: usize,
    /// До этого шага конец фазы запрещён.
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl Sampling {
    /// Дефолт фазы ABC (`yue2_generation_config.json`).
    pub fn abc() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }

    /// Дефолт фазы семантических токенов.
    pub fn semantic() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }

    pub fn validate(&self) -> Result<(), YueError> {
        if !(0.0..=5.0).contains(&self.temperature)
            || !self.temperature.is_finite()
            || !(0.0 < self.top_p && self.top_p <= 1.0)
            || self.top_k < 1
        {
            return Err(YueError::Config("sampling: temperature/top_p/top_k вне допустимого".into()));
        }
        if self.repetition_penalty <= 0.0
            || !self.repetition_penalty.is_finite()
            || !(1..=100).contains(&self.penalty_window)
        {
            return Err(YueError::Config("sampling: repetition_penalty/penalty_window вне допустимого".into()));
        }
        if self.min_tokens > self.max_tokens || self.max_tokens < 1 {
            return Err(YueError::Config("sampling: нужно 0 <= min_tokens <= max_tokens".into()));
        }
        Ok(())
    }
}

/// Настройки прогона целиком.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub abc: Sampling,
    pub semantic: Sampling,
    /// Шагов ODE (midpoint) на акустический чанк.
    pub ode_steps: usize,
    /// Окно модели для нарезки на чанки; меньше — меньше памяти, но чаще швы.
    pub context: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { abc: Sampling::abc(), semantic: Sampling::semantic(), ode_steps: 32, context: CONTEXT }
    }
}

/// Запрос на песню.
#[derive(Debug, Clone, PartialEq)]
pub struct SongRequest {
    /// Теги стиля: язык, жанр, инструменты, характер вокала.
    pub style: String,
    pub lyrics: String,
    pub cot: Cot,
    pub seed: u64,
    /// Готовая партитура вместо сгенерированной (`cot` ≠ `Off`).
    pub abc: Option<String>,
    /// `None` — дефолт режима ([`Cot::default_guidance`]).
    pub cfg_scale: Option<f32>,
}

impl Default for SongRequest {
    fn default() -> Self {
        Self {
            style: String::new(),
            lyrics: String::new(),
            cot: Cot::Full,
            seed: 831001,
            abc: None,
            cfg_scale: None,
        }
    }
}

impl SongRequest {
    pub fn guidance(&self) -> f32 {
        self.cfg_scale.unwrap_or_else(|| self.cot.default_guidance())
    }

    /// Текстовая часть префикса — ровно тот формат, на котором училась модель.
    pub fn text(&self) -> String {
        format!(
            "{}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            self.cot.instruction(),
            self.style,
            self.lyrics
        )
    }

    pub fn validate(&self) -> Result<(), YueError> {
        if let Some(scale) = self.cfg_scale {
            if !scale.is_finite() || !(0.0..=20.0).contains(&scale) {
                return Err(YueError::Config("cfg_scale вне [0, 20]".into()));
            }
        }
        if let Some(abc) = &self.abc {
            if self.cot == Cot::Off || abc.trim().is_empty() {
                return Err(YueError::Config(
                    "готовая партитура требует непустого текста и cot=melody/full".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Проверка, что ID партитуры остались в обычном текстовом словаре: в
/// префикс музыки нельзя протащить кодек-токен.
fn check_abc_ids(ids: &[u32]) -> Result<(), YueError> {
    if ids.iter().any(|&t| t >= EOD) {
        return Err(YueError::Config("ID партитуры вышли за текстовый словарь".into()));
    }
    Ok(())
}

/// Положительный префикс. `abc_ids = None` и `cot != Off` — префикс под
/// генерацию партитуры (кончается на `ABC_START`); иначе — под музыку.
pub fn token_prefixes(
    request: &SongRequest,
    tokenizer: &Yue2Tokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>, YueError> {
    let mut out = Vec::with_capacity(256);
    out.push(EOD);
    out.extend(tokenizer.encode(&request.text()));
    if request.cot == Cot::Off {
        out.extend([ABC_START, ABC_END, MUSIC_START]);
        return Ok(out);
    }
    let owned;
    let ids: &[u32] = match abc_ids {
        Some(ids) => ids,
        None => match &request.abc {
            None => {
                out.push(ABC_START);
                return Ok(out);
            }
            Some(abc) => {
                owned = tokenizer.encode(abc);
                &owned
            }
        },
    };
    check_abc_ids(ids)?;
    out.push(ABC_START);
    out.extend_from_slice(ids);
    out.extend([ABC_END, MUSIC_START]);
    Ok(out)
}

/// Отрицательная ветка CFG: та же инструкция и та же партитура, но без тегов
/// и лирики.
pub fn negative_prefix(
    request: &SongRequest,
    tokenizer: &Yue2Tokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>, YueError> {
    let mut out = Vec::with_capacity(128);
    out.push(EOD);
    out.extend(tokenizer.encode(request.cot.instruction()));
    if request.cot == Cot::Off {
        out.push(MUSIC_START);
        return Ok(out);
    }
    let Some(ids) = abc_ids else {
        return Err(YueError::Config(
            "символьный CFG требует тех же ID партитуры, что и положительная ветка".into(),
        ));
    };
    check_abc_ids(ids)?;
    out.push(ABC_START);
    out.extend_from_slice(ids);
    out.extend([ABC_END, MUSIC_START]);
    Ok(out)
}

/// Нарезка песни на акустические чанки. В каждом чанке AR-префикс повторяется,
/// а семантические токены и латентные кадры занимают по месту — отсюда деление
/// пополам.
pub fn chunk_ranges(
    frames: usize,
    prefix_tokens: usize,
    context: usize,
) -> Result<Vec<(usize, usize)>, YueError> {
    let size = context
        .saturating_sub(prefix_tokens + 3)
        / 2;
    let size = size.min(CONTEXT);
    if frames < 1 || size < 1 {
        return Err(YueError::Config(
            "пустая музыка или слишком длинный префикс — на акустику не осталось окна".into(),
        ));
    }
    let mut out = Vec::new();
    let mut a = 0usize;
    while a < frames {
        out.push((a, (a + size).min(frames)));
        a += size;
    }
    Ok(out)
}

/// Длительность песни по числу семантических токенов.
pub fn frames_to_seconds(frames: usize) -> f32 {
    frames as f32 / FRAMES_PER_SECOND
}

/// Сколько семантических токенов нужно на заданную длительность.
pub fn seconds_to_frames(seconds: f32) -> usize {
    (seconds * FRAMES_PER_SECOND).round().max(0.0) as usize
}
