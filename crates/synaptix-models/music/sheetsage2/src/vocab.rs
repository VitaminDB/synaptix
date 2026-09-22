//! Словарь событий SheetSage2 (схема `v1`) и разбор последовательности.
//!
//! Раскладка — протокол чекпойнта: блоки токенов идут подряд в фиксированном
//! порядке, и сдвинуть любой из них значит сломать модель. Поэтому всё здесь
//! константы, а соответствие релизу проверяет отпечаток
//! ([`Vocab::fingerprint`]) — тот же SHA-256 по JSON-описанию словаря, что
//! считает питоновский токенизатор.
//!
//! Последовательность: `<sos> <prompt>… <out>`, затем события. Событие — один
//! или несколько сдвигов по сетке долей (`subbeat_shift`), за ними поля в
//! порядке схемы: метка времени, ритм (размер и позиция восьмой), секция,
//! тональность, аккорд, ноты (высота и необязательная длительность).

use sha2::{Digest, Sha256};

use crate::SheetError;

pub const PAD: u32 = 0;
pub const SOS: u32 = 1;
pub const EOS: u32 = 2;
pub const OUT: u32 = 3;

/// Задачи (подсказки) схемы `v1` в каноническом порядке.
pub const PROMPT_NAMES: [&str; 8] = [
    "timestamp",
    "downbeat_meter",
    "structure",
    "key",
    "chord_majmin",
    "chord_full",
    "melody_vocal",
    "melody_full",
];

/// Группа выбора задачи: из одной группы подсказка может быть только одна.
const PROMPT_GROUPS: [Field; 8] = [
    Field::Timestamp,
    Field::Rhythm,
    Field::Structure,
    Field::Key,
    Field::Chord,
    Field::Chord,
    Field::Melody,
    Field::Melody,
];

/// Подсказки полной транскрипции (дефолт релиза).
pub const FULL_TASK_PROMPTS: [&str; 6] =
    ["timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full"];

pub const EVENT_FIELD_ORDER: [&str; 6] = ["timestamp", "rhythm", "structure", "key", "chord", "melody"];

pub const STRUCTURE_LABELS: [&str; 23] = [
    "silence",
    "intro",
    "outro",
    "verse",
    "chorus",
    "bridge",
    "pre-chorus",
    "post-chorus",
    "interlude",
    "fade-out",
    "loop",
    "rap",
    "preshot",
    "irregular",
    "instrumental",
    "intro and verse",
    "pre-chorus and chorus",
    "verse and pre-chorus",
    "solo",
    "theme",
    "development",
    "variation",
    "pre-outro",
];

pub const CHROMATIC_SHARPS: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const FULL_CHORD_QUALITIES: [&str; 15] = [
    "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)",
    "maj6", "min6",
];

fn chord_inversions(quality: &str) -> &'static [&'static str] {
    match quality {
        "maj" => &["/2", "/3", "/5"],
        "min" => &["/2", "/b3", "/5"],
        "maj7" => &["/3", "/5", "/7"],
        "min7" => &["/b3", "/5", "/b7"],
        "7" => &["/3", "/5", "/b7"],
        _ => &[],
    }
}

/// Шаблоны длительностей нот в шагах сетки (шаг — шестнадцатая доли).
pub const DURATION_TEMPLATES: [u32; 24] = [
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

const METER_DENOMINATORS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const PROMPT_CAPACITY: u32 = 256;
const MAX_SUBBEAT_SHIFT: u32 = 256;
const N_EIGHTH_POSITIONS: u32 = 256;
const N_KEY_TOKENS: u32 = 24;
const N_PITCH_TOKENS: u32 = 256;

/// Поле события (порядок = порядок схемы).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Timestamp = 0,
    Rhythm = 1,
    Structure = 2,
    Key = 3,
    Chord = 4,
    Melody = 5,
}

impl Field {
    pub const ALL: [Field; 6] =
        [Field::Timestamp, Field::Rhythm, Field::Structure, Field::Key, Field::Chord, Field::Melody];

    pub fn name(self) -> &'static str {
        EVENT_FIELD_ORDER[self as usize]
    }
}

/// Тип токена — по блоку словаря.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Pad,
    Sos,
    Eos,
    Out,
    Prompt,
    SubbeatShift,
    Time,
    Meter,
    EighthPosition,
    Structure,
    Key,
    ChordMajmin,
    ChordFull,
    Pitch,
    Duration,
}

impl TokenType {
    /// Поле события, в которое идёт токен (`None` — служебный токен).
    pub fn field(self) -> Option<Field> {
        Some(match self {
            TokenType::Time => Field::Timestamp,
            TokenType::Meter | TokenType::EighthPosition => Field::Rhythm,
            TokenType::Structure => Field::Structure,
            TokenType::Key => Field::Key,
            TokenType::ChordMajmin | TokenType::ChordFull => Field::Chord,
            TokenType::Pitch | TokenType::Duration => Field::Melody,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            TokenType::Pad => "pad",
            TokenType::Sos => "sos",
            TokenType::Eos => "eos",
            TokenType::Out => "out",
            TokenType::Prompt => "prompt",
            TokenType::SubbeatShift => "subbeat_shift",
            TokenType::Time => "time",
            TokenType::Meter => "meter",
            TokenType::EighthPosition => "eighth_position",
            TokenType::Structure => "structure",
            TokenType::Key => "key",
            TokenType::ChordMajmin => "chord_majmin",
            TokenType::ChordFull => "chord_full",
            TokenType::Pitch => "pitch",
            TokenType::Duration => "duration",
        }
    }
}

/// Ритм события: размер (если сменился) и позиция в восьмых от сильной доли.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rhythm {
    pub meter: Option<(u32, u32)>,
    pub eighth_position: Option<u32>,
}

/// Нота мелодии. `track` 0 — вокал, 1 — инструмент.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Note {
    pub pitch: u32,
    pub track: u32,
    pub duration_bin: usize,
    pub duration_steps: u32,
    /// Конец в секундах — проставляется при склейке окон.
    pub end_time: Option<f64>,
}

/// Расшифрованные поля события.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Values {
    pub timestamp: Option<f64>,
    pub rhythm: Option<Rhythm>,
    pub structure: Option<String>,
    pub key: Option<String>,
    pub chord: Option<String>,
    pub melody: Option<Vec<Note>>,
}

impl Values {
    pub fn has(&self, field: Field) -> bool {
        match field {
            Field::Timestamp => self.timestamp.is_some(),
            Field::Rhythm => self.rhythm.is_some(),
            Field::Structure => self.structure.is_some(),
            Field::Key => self.key.is_some(),
            Field::Chord => self.chord.is_some(),
            Field::Melody => self.melody.is_some(),
        }
    }
}

/// Событие: позиция на сетке долей, токены по полям и их значения. Поля склейки
/// (`time`, `window_*`, `*_subbeat`) заполняются, когда событие принято из окна.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Event {
    pub subbeat: i64,
    pub tokens: [Vec<u32>; 6],
    pub values: Values,
    pub time: Option<f64>,
    pub window_index: usize,
    pub window_start: f64,
    pub source_subbeat: i64,
    pub global_subbeat: i64,
}

impl Event {
    pub fn field_tokens(&self, field: Field) -> &[u32] {
        &self.tokens[field as usize]
    }
}

/// Разобранная последовательность.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// Индексы подсказок в [`PROMPT_NAMES`], в порядке последовательности.
    pub prompts: Vec<usize>,
    pub events: Vec<Event>,
    pub has_eos: bool,
}

/// Словарь схемы `v1` для окна `audio_length_seconds` при `time_hz` меток в секунду.
#[derive(Debug, Clone)]
pub struct Vocab {
    pub audio_length_seconds: f64,
    pub time_hz: u32,
    pub n_time_tokens: u32,
    pub subbeat_shift_start: u32,
    pub subbeat_shift_end: u32,
    pub time_start: u32,
    pub time_end: u32,
    pub meter_pairs: Vec<(u32, u32)>,
    pub meter_start: u32,
    pub meter_end: u32,
    pub eighth_start: u32,
    pub eighth_end: u32,
    pub structure_start: u32,
    pub structure_end: u32,
    pub key_start: u32,
    pub key_end: u32,
    pub majmin_labels: Vec<String>,
    pub majmin_start: u32,
    pub majmin_end: u32,
    pub full_chord_labels: Vec<String>,
    pub full_chord_start: u32,
    pub full_chord_end: u32,
    pub pitch_start: u32,
    pub pitch_end: u32,
    pub duration_start: u32,
    pub duration_end: u32,
    pub n_tokens: u32,
}

const PROMPT_START: u32 = 4;

pub fn full_chord_vocabulary() -> Vec<String> {
    let mut labels = vec!["N".to_string()];
    for quality in FULL_CHORD_QUALITIES {
        for root in CHROMATIC_SHARPS {
            for inversion in chord_inversions(quality).iter().copied().chain(std::iter::once("")) {
                labels.push(format!("{root}:{quality}{inversion}"));
            }
        }
    }
    labels
}

impl Vocab {
    pub fn new(audio_length_seconds: f64, time_hz: u32) -> Self {
        let n_time_tokens = (audio_length_seconds * time_hz as f64).round() as u32;
        let subbeat_shift_start = PROMPT_START + PROMPT_CAPACITY;
        let subbeat_shift_end = subbeat_shift_start + MAX_SUBBEAT_SHIFT + 1;
        let time_start = subbeat_shift_end;
        let time_end = time_start + n_time_tokens;
        let meter_pairs: Vec<(u32, u32)> = (1..=32u32)
            .flat_map(|n| METER_DENOMINATORS.iter().map(move |&d| (n, d)))
            .collect();
        let meter_start = time_end;
        let meter_end = meter_start + meter_pairs.len() as u32;
        let eighth_start = meter_end;
        let eighth_end = eighth_start + N_EIGHTH_POSITIONS;
        let structure_start = eighth_end;
        let structure_end = structure_start + STRUCTURE_LABELS.len() as u32;
        let key_start = structure_end;
        let key_end = key_start + N_KEY_TOKENS;
        let mut majmin_labels = vec!["N".to_string()];
        majmin_labels.extend(CHROMATIC_SHARPS.iter().map(|r| format!("{r}:maj")));
        majmin_labels.extend(CHROMATIC_SHARPS.iter().map(|r| format!("{r}:min")));
        let majmin_start = key_end;
        let majmin_end = majmin_start + majmin_labels.len() as u32;
        let full_chord_labels = full_chord_vocabulary();
        let full_chord_start = majmin_end;
        let full_chord_end = full_chord_start + full_chord_labels.len() as u32;
        let pitch_start = full_chord_end;
        let pitch_end = pitch_start + N_PITCH_TOKENS;
        let duration_start = pitch_end;
        let duration_end = duration_start + DURATION_TEMPLATES.len() as u32;
        Self {
            audio_length_seconds,
            time_hz,
            n_time_tokens,
            subbeat_shift_start,
            subbeat_shift_end,
            time_start,
            time_end,
            meter_pairs,
            meter_start,
            meter_end,
            eighth_start,
            eighth_end,
            structure_start,
            structure_end,
            key_start,
            key_end,
            majmin_labels,
            majmin_start,
            majmin_end,
            full_chord_labels,
            full_chord_start,
            full_chord_end,
            pitch_start,
            pitch_end,
            duration_start,
            duration_end,
            n_tokens: duration_end,
        }
    }

    /// SHA-256 (первые 16 hex-символов) по JSON-описанию словаря — ровно тот,
    /// что пишет релиз в `tokenizer_fingerprint`.
    pub fn fingerprint(&self) -> String {
        fn strings(items: &[impl AsRef<str>]) -> String {
            let parts: Vec<String> = items
                .iter()
                .map(|s| serde_json::to_string(s.as_ref()).unwrap_or_default())
                .collect();
            format!("[{}]", parts.join(","))
        }
        let meters: Vec<String> = self.meter_pairs.iter().map(|(n, d)| format!("[{n},{d}]")).collect();
        let durations: Vec<String> = DURATION_TEMPLATES.iter().map(|d| d.to_string()).collect();
        // Ключи — по алфавиту (json.dumps(sort_keys=True)), разделители без пробелов.
        let payload = format!(
            "{{\"appended_token_blocks\":[],\"audio_length_seconds\":{},\"duration_templates\":[{}],\
             \"event_field_order\":{},\"full_chord_labels\":{},\"majmin_chord_labels\":{},\
             \"meter_pairs\":[{}],\"n_tokens\":{},\"prompt_capacity\":{},\"prompt_names\":{},\
             \"schema_version\":\"v1\",\"structure_labels\":{},\"time_hz\":{}}}",
            py_float(self.audio_length_seconds),
            durations.join(","),
            strings(&EVENT_FIELD_ORDER),
            strings(&self.full_chord_labels),
            strings(&self.majmin_labels),
            meters.join(","),
            self.n_tokens,
            PROMPT_CAPACITY,
            strings(&PROMPT_NAMES),
            strings(&STRUCTURE_LABELS),
            self.time_hz,
        );
        let digest = Sha256::digest(payload.as_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    pub fn prompt_token(index: usize) -> u32 {
        PROMPT_START + index as u32
    }

    pub fn prompt_index(name: &str) -> Option<usize> {
        let name = name.trim();
        let name = name.strip_prefix("<|").and_then(|n| n.strip_suffix("|>")).unwrap_or(name);
        PROMPT_NAMES.iter().position(|p| *p == name)
    }

    /// Канонический порядок, без повторов, не больше одной подсказки на группу.
    pub fn normalize_prompts(names: &[&str]) -> Result<Vec<usize>, SheetError> {
        let mut out: Vec<usize> = Vec::new();
        for name in names {
            let idx = Self::prompt_index(name)
                .ok_or_else(|| SheetError::Config(format!("неизвестная подсказка `{name}`")))?;
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
        out.sort_unstable();
        for (i, &a) in out.iter().enumerate() {
            for &b in &out[i + 1..] {
                if PROMPT_GROUPS[a] == PROMPT_GROUPS[b] {
                    return Err(SheetError::Config(format!(
                        "подсказки `{}` и `{}` взаимоисключающие",
                        PROMPT_NAMES[a], PROMPT_NAMES[b]
                    )));
                }
            }
        }
        if out.is_empty() {
            return Err(SheetError::Config("нужна хотя бы одна подсказка".into()));
        }
        Ok(out)
    }

    /// `<sos> <prompt>… <out>`.
    pub fn prompt_prefix(prompts: &[usize]) -> Vec<u32> {
        let mut out = vec![SOS];
        out.extend(prompts.iter().map(|&p| Self::prompt_token(p)));
        out.push(OUT);
        out
    }

    pub fn token_type(&self, token: u32) -> Result<TokenType, SheetError> {
        let ranges = [
            (TokenType::SubbeatShift, self.subbeat_shift_start, self.subbeat_shift_end),
            (TokenType::Time, self.time_start, self.time_end),
            (TokenType::Meter, self.meter_start, self.meter_end),
            (TokenType::EighthPosition, self.eighth_start, self.eighth_end),
            (TokenType::Structure, self.structure_start, self.structure_end),
            (TokenType::Key, self.key_start, self.key_end),
            (TokenType::ChordMajmin, self.majmin_start, self.majmin_end),
            (TokenType::ChordFull, self.full_chord_start, self.full_chord_end),
            (TokenType::Pitch, self.pitch_start, self.pitch_end),
            (TokenType::Duration, self.duration_start, self.duration_end),
        ];
        for (kind, start, end) in ranges {
            if start <= token && token < end {
                return Ok(kind);
            }
        }
        if (PROMPT_START..PROMPT_START + PROMPT_NAMES.len() as u32).contains(&token) {
            return Ok(TokenType::Prompt);
        }
        match token {
            PAD => Ok(TokenType::Pad),
            SOS => Ok(TokenType::Sos),
            EOS => Ok(TokenType::Eos),
            OUT => Ok(TokenType::Out),
            _ => Err(SheetError::Sequence(format!(
                "токен {token} вне словаря из {} токенов",
                self.n_tokens
            ))),
        }
    }

    /// Сдвиг больше 256 шагов пишется несколькими токенами.
    pub fn subbeat_shift_tokens(&self, shift: i64) -> Result<Vec<u32>, SheetError> {
        if shift < 0 {
            return Err(SheetError::Sequence("сдвиг по сетке отрицательный".into()));
        }
        let mut shift = shift as u64;
        let mut out = Vec::new();
        while shift > MAX_SUBBEAT_SHIFT as u64 {
            out.push(self.subbeat_shift_start + MAX_SUBBEAT_SHIFT);
            shift -= MAX_SUBBEAT_SHIFT as u64;
        }
        out.push(self.subbeat_shift_start + shift as u32);
        Ok(out)
    }

    pub fn time_id_to_token(&self, time_id: u32) -> Result<u32, SheetError> {
        if time_id >= self.n_time_tokens {
            return Err(SheetError::Sequence(format!("метка времени {time_id} вне окна")));
        }
        Ok(self.time_start + time_id)
    }

    pub fn token_to_time_id(&self, token: u32) -> Option<u32> {
        (self.time_start..self.time_end).contains(&token).then(|| token - self.time_start)
    }

    pub fn is_time(&self, token: u32) -> bool {
        (self.time_start..self.time_end).contains(&token)
    }

    /// Значения поля по его токенам.
    pub fn decode_field(&self, field: Field, tokens: &[u32], values: &mut Values) -> Result<(), SheetError> {
        let first = tokens[0];
        match field {
            Field::Timestamp => {
                let id = self
                    .token_to_time_id(first)
                    .ok_or_else(|| SheetError::Sequence(format!("токен {first} — не метка времени")))?;
                values.timestamp = Some(id as f64 / self.time_hz as f64);
            }
            Field::Rhythm => {
                let mut rhythm = Rhythm::default();
                for &t in tokens {
                    match self.token_type(t)? {
                        TokenType::Meter => rhythm.meter = Some(self.meter_pairs[(t - self.meter_start) as usize]),
                        TokenType::EighthPosition => rhythm.eighth_position = Some(t - self.eighth_start),
                        _ => {}
                    }
                }
                values.rhythm = Some(rhythm);
            }
            Field::Structure => {
                values.structure = Some(STRUCTURE_LABELS[(first - self.structure_start) as usize].to_string());
            }
            Field::Key => {
                let id = first - self.key_start;
                let mode = if id >= 12 { "minor" } else { "major" };
                values.key = Some(format!("{}:{mode}", CHROMATIC_SHARPS[(id % 12) as usize]));
            }
            Field::Chord => {
                let label = if self.token_type(first)? == TokenType::ChordMajmin {
                    self.majmin_labels[(first - self.majmin_start) as usize].clone()
                } else {
                    self.full_chord_labels[(first - self.full_chord_start) as usize].clone()
                };
                values.chord = Some(label);
            }
            Field::Melody => {
                let mut notes = Vec::new();
                let mut i = 0usize;
                while i < tokens.len() {
                    let pitch_id = tokens[i].wrapping_sub(self.pitch_start);
                    let mut bin = 0usize;
                    if i + 1 < tokens.len() && self.token_type(tokens[i + 1])? == TokenType::Duration {
                        bin = (tokens[i + 1] - self.duration_start) as usize;
                        i += 2;
                    } else {
                        i += 1;
                    }
                    notes.push(Note {
                        pitch: pitch_id % 128,
                        track: u32::from(pitch_id >= 128),
                        duration_bin: bin,
                        duration_steps: DURATION_TEMPLATES[bin],
                        end_time: None,
                    });
                }
                values.melody = Some(notes);
            }
        }
        Ok(())
    }

    /// Пересчитать значения события по его токенам.
    pub fn refresh_values(&self, event: &mut Event) -> Result<(), SheetError> {
        let mut values = Values::default();
        for field in Field::ALL {
            let tokens = &event.tokens[field as usize];
            if !tokens.is_empty() {
                self.decode_field(field, tokens, &mut values)?;
            }
        }
        event.values = values;
        Ok(())
    }

    /// Разобрать последовательность в события. `strict` — как у релиза:
    /// канонический порядок подсказок, непустые события, поля только активных
    /// задач, ровно один токен в однозначных полях, обязательный `<eos>`.
    pub fn decode_sequence(&self, tokens: &[u32], strict: bool) -> Result<Decoded, SheetError> {
        let mut end = tokens.len();
        while end > 0 && tokens[end - 1] == PAD {
            end -= 1;
        }
        let tokens = &tokens[..end];
        if tokens.first() != Some(&SOS) {
            return Err(SheetError::Sequence("последовательность должна начинаться с <|sos|>".into()));
        }
        let out_index = tokens
            .iter()
            .skip(1)
            .position(|&t| t == OUT)
            .map(|p| p + 1)
            .ok_or_else(|| SheetError::Sequence("в последовательности нет <|out|>".into()))?;
        let mut prompts = Vec::new();
        for &t in &tokens[1..out_index] {
            if self.token_type(t)? != TokenType::Prompt {
                return Err(SheetError::Sequence(format!("токен {t} — не подсказка")));
            }
            prompts.push((t - PROMPT_START) as usize);
        }
        if strict {
            let names: Vec<&str> = prompts.iter().map(|&p| PROMPT_NAMES[p]).collect();
            if Self::normalize_prompts(&names)? != prompts {
                return Err(SheetError::Sequence("подсказки не в каноническом порядке".into()));
            }
        }
        let mut active = [false; 6];
        for &p in &prompts {
            active[PROMPT_GROUPS[p] as usize] = true;
        }

        let mut events = Vec::new();
        let mut position = out_index + 1;
        let mut current_step: i64 = 0;
        let mut saw_eos = false;
        while position < tokens.len() {
            let token = tokens[position];
            if token == EOS {
                saw_eos = true;
                position += 1;
                break;
            }
            if self.token_type(token)? != TokenType::SubbeatShift {
                return Err(SheetError::Sequence(format!(
                    "событие на позиции {position} без сдвига по сетке"
                )));
            }
            let mut shift: i64 = 0;
            while position < tokens.len() && self.token_type(tokens[position])? == TokenType::SubbeatShift {
                shift += (tokens[position] - self.subbeat_shift_start) as i64;
                position += 1;
            }
            current_step += shift;

            let mut by_field: [Vec<u32>; 6] = Default::default();
            while position < tokens.len() {
                let token = tokens[position];
                let kind = self.token_type(token)?;
                if kind == TokenType::SubbeatShift || token == EOS {
                    break;
                }
                let field = kind.field().ok_or_else(|| {
                    SheetError::Sequence(format!("токен {token} ({}) не относится ни к одному полю", kind.name()))
                })?;
                if strict && !active[field as usize] {
                    return Err(SheetError::Sequence(format!(
                        "token {token} belongs to inactive output field {:?}",
                        field.name()
                    )));
                }
                by_field[field as usize].push(token);
                position += 1;
            }
            if by_field.iter().all(|v| v.is_empty()) {
                if strict {
                    return Err(SheetError::Sequence(format!("empty event at subbeat {current_step}")));
                }
                continue;
            }
            if strict {
                for field in Field::ALL {
                    let values = &by_field[field as usize];
                    if values.is_empty() {
                        continue;
                    }
                    let types: Vec<TokenType> =
                        values.iter().map(|&v| self.token_type(v)).collect::<Result<_, _>>()?;
                    match field {
                        Field::Timestamp if types != [TokenType::Time] => {
                            return Err(SheetError::Sequence("в метке времени должен быть ровно один токен".into()));
                        }
                        Field::Rhythm
                            if types != [TokenType::EighthPosition]
                                && types != [TokenType::Meter, TokenType::EighthPosition] =>
                        {
                            return Err(SheetError::Sequence(format!("неверный ритм: {types:?}")));
                        }
                        Field::Structure | Field::Key | Field::Chord if values.len() != 1 => {
                            return Err(SheetError::Sequence(format!(
                                "в поле {} должен быть ровно один токен",
                                field.name()
                            )));
                        }
                        Field::Melody => {
                            let mut i = 0;
                            while i < types.len() {
                                if types[i] != TokenType::Pitch {
                                    return Err(SheetError::Sequence(
                                        "мелодия — высоты с необязательной длительностью".into(),
                                    ));
                                }
                                i += if i + 1 < types.len() && types[i + 1] == TokenType::Duration { 2 } else { 1 };
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut event = Event { subbeat: current_step, tokens: by_field, ..Default::default() };
            self.refresh_values(&mut event)?;
            events.push(event);
        }
        if strict && !saw_eos {
            return Err(SheetError::Sequence("в последовательности нет <|eos|>".into()));
        }
        if strict && position != tokens.len() {
            return Err(SheetError::Sequence("после <|eos|> идут токены".into()));
        }
        Ok(Decoded { prompts, events, has_eos: saw_eos })
    }

    /// Обратно в токены (без потерь относительно [`Self::decode_sequence`]).
    pub fn encode_decoded(&self, decoded: &Decoded) -> Result<Vec<u32>, SheetError> {
        let names: Vec<&str> = decoded.prompts.iter().map(|&p| PROMPT_NAMES[p]).collect();
        let prompts = Self::normalize_prompts(&names)?;
        let mut out = Self::prompt_prefix(&prompts);
        let mut previous: i64 = 0;
        for event in &decoded.events {
            if event.subbeat < previous {
                return Err(SheetError::Sequence("события должны идти по неубыванию позиции".into()));
            }
            out.extend(self.subbeat_shift_tokens(event.subbeat - previous)?);
            previous = event.subbeat;
            for field in Field::ALL {
                out.extend_from_slice(&event.tokens[field as usize]);
            }
        }
        if decoded.has_eos {
            out.push(EOS);
        }
        Ok(out)
    }

    /// Человекочитаемое имя токена (как `describe` релиза) — для логов и отладки.
    pub fn describe(&self, token: u32) -> String {
        let Ok(kind) = self.token_type(token) else {
            return format!("<?{token}>");
        };
        match kind {
            TokenType::Prompt => format!("<|{}|>", PROMPT_NAMES[(token - PROMPT_START) as usize]),
            TokenType::SubbeatShift => format!("<subbeat_shift_{}>", token - self.subbeat_shift_start),
            TokenType::Time => format!("<time_{:.2}s>", (token - self.time_start) as f64 / self.time_hz as f64),
            TokenType::Meter => {
                let (n, d) = self.meter_pairs[(token - self.meter_start) as usize];
                format!("<meter_{n}/{d}>")
            }
            TokenType::EighthPosition => format!("<eighth_pos_{}>", token - self.eighth_start),
            TokenType::Structure => format!("<structure_{}>", STRUCTURE_LABELS[(token - self.structure_start) as usize]),
            TokenType::Key => {
                let id = token - self.key_start;
                let mode = if id >= 12 { "minor" } else { "major" };
                format!("<key_{}:{mode}>", CHROMATIC_SHARPS[(id % 12) as usize])
            }
            TokenType::ChordMajmin => format!("<chord_majmin_{}>", self.majmin_labels[(token - self.majmin_start) as usize]),
            TokenType::ChordFull => format!("<chord_full_{}>", self.full_chord_labels[(token - self.full_chord_start) as usize]),
            TokenType::Pitch => {
                let p = token - self.pitch_start;
                format!("<pitch_{}_track_{}>", p % 128, u32::from(p >= 128))
            }
            TokenType::Duration => format!("<duration_{}>", token - self.duration_start),
            other => format!("<|{}|>", other.name()),
        }
    }
}

/// Число с плавающей точкой так, как его пишет питоновский `json.dumps`
/// (`300.0`, а не `300`).
fn py_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_release() {
        let v = Vocab::new(300.0, 100);
        assert_eq!(v.subbeat_shift_start, 260);
        assert_eq!(v.time_start, 517);
        assert_eq!(v.meter_start, 30517);
        assert_eq!(v.eighth_start, 30709);
        assert_eq!(v.structure_start, 30965);
        assert_eq!(v.key_start, 30988);
        assert_eq!(v.majmin_start, 31012);
        assert_eq!(v.full_chord_start, 31037);
        assert_eq!(v.pitch_start, 31398);
        assert_eq!(v.duration_start, 31654);
        assert_eq!(v.n_tokens, 31678);
        assert_eq!(v.full_chord_labels.len(), 361);
    }

    #[test]
    fn fingerprint_matches_release() {
        assert_eq!(Vocab::new(300.0, 100).fingerprint(), "5ba3325af0344c7f");
    }

    #[test]
    fn describe_matches_release_names() {
        let v = Vocab::new(300.0, 100);
        assert_eq!(v.describe(586), "<time_0.69s>");
        assert_eq!(v.describe(30525), "<meter_2/4>");
        assert_eq!(v.describe(30966), "<structure_intro>");
        assert_eq!(v.describe(31007), "<key_G:minor>");
        assert_eq!(v.describe(31037), "<chord_full_N>");
        assert_eq!(v.describe(31598), "<pitch_72_track_1>");
        assert_eq!(v.describe(11), "<|melody_full|>");
    }
}
