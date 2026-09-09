//! Разбор протокола хода Gemma-4 в потоке генерации.
//!
//! Gemma-4 размечает ход спецтокенами, которых в декодированном тексте не
//! видно (декод идёт со `skip_special_tokens`):
//!
//! ```text
//! <|channel>thought⏎размышления<channel|>
//! текст ответа
//! <|tool_call>call:notes{action:<|"|>read<|"|>,page:<|"|>all<|"|>}<tool_call|>
//! ```
//!
//! Без разбора по id токенов в ленту утекало `call:notes{action:read,page:all}`
//! как обычный текст: кавычки `<|"|>` и оба маркера вызова пропадали,
//! ChatML-парсер (`<tool_call>` буквами) ничего не находил, и модель дальше
//! «отвечала» выдуманным содержимым (чат MyLife, 09.09.2026).
//!
//! Поэтому парсер работает по id спецтокенов (их даёт колбэк стрима) и
//! раскладывает текст дельт по каналам: после `<|channel>` — размышления
//! (первая строка — имя канала, `thought`, она отбрасывается), между
//! `<|tool_call>` и `<tool_call|>` — тело вызова, остальное — ответ.
//!
//! Аргументы вызова — не JSON, а собственная нотация шаблона Gemma:
//! ключи без кавычек, строки в `<|"|>…<|"|>`, вложенные `{}` и `[]`,
//! числа/логические/`null` — как есть. [`parse_call`] переводит её в
//! JSON-объект. Внутри тела кавычка-токен хранится сентинелом
//! [`QUOTE_SENTINEL`]: в декодированном тексте её нет вовсе.

use serde_json::{Map, Number, Value};

use crate::tokenizer::Tokenizer;

/// Открытие вызова инструмента.
pub const TOK_TOOL_CALL_OPEN: &str = "<|tool_call>";
/// Закрытие вызова инструмента.
pub const TOK_TOOL_CALL_CLOSE: &str = "<tool_call|>";
/// Кавычка строкового значения в аргументах вызова.
pub const TOK_QUOTE: &str = "<|\"|>";
/// Открытие канала (размышлений).
pub const TOK_CHANNEL_OPEN: &str = "<|channel>";
/// Закрытие канала.
pub const TOK_CHANNEL_CLOSE: &str = "<channel|>";

/// Чем в накопленном теле вызова помечена кавычка `<|"|>`. Символ из
/// приватной области Unicode — в тексте модели его не бывает.
pub const QUOTE_SENTINEL: char = '\u{E000}';

/// Id спецтокенов протокола. `None` из [`Self::detect`] — модель не Gemma-4
/// (ни один из маркеров не является отдельным токеном словаря).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4Ids {
    pub tool_call_open: u32,
    pub tool_call_close: u32,
    pub quote: u32,
    pub channel_open: u32,
    pub channel_close: u32,
}

impl Gemma4Ids {
    /// Резолвит маркеры протокола через словарь токенайзера.
    pub fn detect(tokenizer: &dyn Tokenizer) -> Option<Self> {
        Self::detect_with(|s| tokenizer.token_to_id(s))
    }

    /// То же, но через произвольный резолвер «строка → один id»; удобно,
    /// когда доступен только `encode` (признак — строка кодируется ровно
    /// одним токеном).
    pub fn detect_with(mut lookup: impl FnMut(&str) -> Option<u32>) -> Option<Self> {
        Some(Self {
            tool_call_open: lookup(TOK_TOOL_CALL_OPEN)?,
            tool_call_close: lookup(TOK_TOOL_CALL_CLOSE)?,
            quote: lookup(TOK_QUOTE)?,
            channel_open: lookup(TOK_CHANNEL_OPEN)?,
            channel_close: lookup(TOK_CHANNEL_CLOSE)?,
        })
    }
}

/// Разобранный вызов инструмента.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4Call {
    pub name: String,
    /// Аргументы JSON-объектом (всегда `Value::Object`).
    pub arguments: Value,
}

impl Gemma4Call {
    /// Аргументы строкой JSON — в таком виде их ждут исполнители инструментов.
    pub fn arguments_json(&self) -> String {
        self.arguments.to_string()
    }
}

/// Результат разбора одной дельты: что дописать в тело ответа, что — в
/// размышления, что — в live-превью вызова. Любая часть может быть пустой.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Gemma4Split {
    pub body: String,
    pub thinking: String,
    /// Текст тела вызова, как его пишет модель (кавычки-токены показаны
    /// обычной `"`), — для превью команды, пока вызов не дописан.
    pub tool: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Обычный текст ответа.
    Body,
    /// Сразу после `<|channel>`: копится имя канала до конца строки.
    ChannelHeader,
    /// Внутри канала — размышления.
    Thinking,
    /// Между `<|tool_call>` и `<tool_call|>`.
    ToolCall,
}

/// Инкрементальный парсер хода Gemma-4. Один на ход генерации.
pub struct Gemma4StreamParser {
    ids: Gemma4Ids,
    state: State,
    /// Имя канала, копится в [`State::ChannelHeader`].
    header: String,
    /// Тело текущего вызова (кавычки — [`QUOTE_SENTINEL`]).
    inner: String,
    calls: Vec<Gemma4Call>,
    /// Хотя бы один вызов закрыт `<tool_call|>`.
    closed_call: bool,
}

impl Gemma4StreamParser {
    /// `thinking_open` — промпт уже кончается `<|channel>thought⏎` (так
    /// шаблон продолжает ход после результата инструмента при включённых
    /// размышлениях): модель начинает прямо с размышлений, заголовка канала
    /// в потоке не будет.
    pub fn new(ids: Gemma4Ids, thinking_open: bool) -> Self {
        Self {
            ids,
            state: if thinking_open { State::Thinking } else { State::Body },
            header: String::new(),
            inner: String::new(),
            calls: Vec::new(),
            closed_call: false,
        }
    }

    /// Подать очередной токен: `id` — чтобы поймать невидимые в тексте
    /// спецтокены, `delta` — декодированный текст (для спецтокенов пустой).
    pub fn feed(&mut self, id: u32, delta: &str) -> Gemma4Split {
        let ids = self.ids;
        if id == ids.channel_open {
            self.state = State::ChannelHeader;
            self.header.clear();
            return Gemma4Split::default();
        }
        if id == ids.channel_close {
            self.state = State::Body;
            return Gemma4Split::default();
        }
        if id == ids.tool_call_open {
            self.state = State::ToolCall;
            self.inner.clear();
            return Gemma4Split::default();
        }
        if id == ids.tool_call_close {
            if self.state == State::ToolCall {
                self.accept_call(false);
            }
            self.state = State::Body;
            return Gemma4Split::default();
        }
        if id == ids.quote {
            return match self.state {
                State::ToolCall => {
                    self.inner.push(QUOTE_SENTINEL);
                    Gemma4Split { tool: "\"".into(), ..Default::default() }
                }
                State::Thinking => Gemma4Split { thinking: "\"".into(), ..Default::default() },
                State::ChannelHeader => Gemma4Split::default(),
                State::Body => Gemma4Split { body: "\"".into(), ..Default::default() },
            };
        }
        if delta.is_empty() {
            return Gemma4Split::default();
        }
        match self.state {
            State::Body => Gemma4Split { body: delta.to_string(), ..Default::default() },
            State::ChannelHeader => {
                // Имя канала — до конца строки; остаток строки уже
                // размышления.
                self.header.push_str(delta);
                match self.header.find('\n') {
                    Some(pos) => {
                        let rest = self.header[pos + 1..].to_string();
                        self.header.clear();
                        self.state = State::Thinking;
                        Gemma4Split { thinking: rest, ..Default::default() }
                    }
                    None => Gemma4Split::default(),
                }
            }
            State::Thinking => Gemma4Split { thinking: delta.to_string(), ..Default::default() },
            State::ToolCall => {
                self.inner.push_str(delta);
                Gemma4Split { tool: delta.to_string(), ..Default::default() }
            }
        }
    }

    /// Есть ли уже собранный вызов с закрытым `<tool_call|>` — сигнал, что
    /// стрим можно рвать, не дожидаясь прозы после вызова.
    pub fn has_closed_call(&self) -> bool {
        self.closed_call && !self.calls.is_empty()
    }

    /// Собранные вызовы. Незакрытое тело (поток кончился без `<tool_call|>`)
    /// разбираем тоже: модель нередко ставит EOS сразу после `}`.
    pub fn finish(mut self) -> Vec<Gemma4Call> {
        if self.state == State::ToolCall {
            self.accept_call(true);
        }
        self.calls
    }

    fn accept_call(&mut self, unclosed: bool) {
        let body = std::mem::take(&mut self.inner);
        match parse_call(&body) {
            Some(call) => {
                if !unclosed {
                    self.closed_call = true;
                }
                self.calls.push(call);
            }
            None => {
                if !body.trim().is_empty() {
                    eprintln!(
                        "[gemma4] тело вызова не разбирается{}: {:?}",
                        if unclosed { " (поток оборван без <tool_call|>)" } else { "" },
                        body.replace(QUOTE_SENTINEL, TOK_QUOTE)
                    );
                }
            }
        }
    }
}

/// Разбирает тело вызова `call:name{key:value,…}` (кавычки —
/// [`QUOTE_SENTINEL`] либо обычные `"`) в имя и JSON-объект аргументов.
///
/// Терпимо к огрехам модели: префикс `call:` необязателен, пробелы вокруг
/// ключей и значений допустимы, незакрытая строка тянется до конца тела,
/// лишние запятые пропускаются. `None` — только когда нет имени.
pub fn parse_call(body: &str) -> Option<Gemma4Call> {
    let trimmed = body.trim();
    let trimmed = trimmed.strip_prefix("call:").map(str::trim_start).unwrap_or(trimmed);
    let (name, args_src) = match trimmed.find('{') {
        Some(pos) => (trimmed[..pos].trim(), &trimmed[pos..]),
        None => (trimmed, ""),
    };
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return None;
    }
    let arguments = if args_src.is_empty() {
        Value::Object(Map::new())
    } else {
        let mut p = ArgParser { chars: args_src.chars().collect(), pos: 0 };
        match p.value() {
            Value::Object(map) => Value::Object(map),
            _ => Value::Object(Map::new()),
        }
    };
    Some(Gemma4Call { name: name.to_string(), arguments })
}

/// Рекурсивный спуск по нотации аргументов Gemma.
struct ArgParser {
    chars: Vec<char>,
    pos: usize,
}

impl ArgParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Value {
        self.skip_ws();
        match self.peek() {
            Some('{') => {
                self.pos += 1;
                self.object()
            }
            Some('[') => {
                self.pos += 1;
                self.array()
            }
            Some(QUOTE_SENTINEL) => {
                self.pos += 1;
                Value::String(self.until(QUOTE_SENTINEL))
            }
            Some('"') => {
                self.pos += 1;
                Value::String(self.json_string())
            }
            _ => self.bare(),
        }
    }

    fn object(&mut self) -> Value {
        let mut map = Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                Some(',') => {
                    self.pos += 1;
                    continue;
                }
                _ => {}
            }
            let key = match self.peek() {
                Some(QUOTE_SENTINEL) => {
                    self.pos += 1;
                    self.until(QUOTE_SENTINEL)
                }
                Some('"') => {
                    self.pos += 1;
                    self.json_string()
                }
                _ => {
                    let raw = self.take_while(|c| c != ':' && c != '}' && c != ',');
                    raw.trim().to_string()
                }
            };
            self.skip_ws();
            if self.peek() != Some(':') {
                // Ключ без значения — пропускаем до разделителя.
                if key.is_empty() {
                    self.pos += 1;
                }
                continue;
            }
            self.pos += 1;
            let value = self.value();
            if !key.is_empty() {
                map.insert(key, value);
            }
        }
        Value::Object(map)
    }

    fn array(&mut self) -> Value {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some(',') => {
                    self.pos += 1;
                    continue;
                }
                _ => items.push(self.value()),
            }
        }
        Value::Array(items)
    }

    /// Строка до следующего `stop` (сам разделитель съедается); без него —
    /// до конца тела.
    fn until(&mut self, stop: char) -> String {
        let s = self.take_while(|c| c != stop);
        if self.peek() == Some(stop) {
            self.pos += 1;
        }
        s
    }

    /// JSON-строка после открывающей `"`: экранирование как в JSON.
    fn json_string(&mut self) -> String {
        let mut raw = String::from("\"");
        let mut escaped = false;
        while let Some(c) = self.peek() {
            self.pos += 1;
            raw.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                break;
            }
        }
        if !raw.ends_with('"') || raw.len() < 2 {
            raw.push('"');
        }
        serde_json::from_str::<String>(&raw).unwrap_or_else(|_| raw[1..raw.len() - 1].to_string())
    }

    /// Скаляр без кавычек — до `,`, `}` или `]`.
    fn bare(&mut self) -> Value {
        let raw = self.take_while(|c| c != ',' && c != '}' && c != ']');
        let s = raw.trim();
        match s {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "" => Value::Null,
            _ => {
                if let Ok(i) = s.parse::<i64>() {
                    return Value::Number(i.into());
                }
                if let Ok(f) = s.parse::<f64>() {
                    if let Some(n) = Number::from_f64(f) {
                        return Value::Number(n);
                    }
                }
                Value::String(s.to_string())
            }
        }
    }

    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> String {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if pred(c)) {
            self.pos += 1;
        }
        self.chars[start..self.pos].iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const IDS: Gemma4Ids = Gemma4Ids {
        tool_call_open: 48,
        tool_call_close: 49,
        quote: 52,
        channel_open: 100,
        channel_close: 101,
    };

    fn q(s: &str) -> String {
        s.replace('"', &QUOTE_SENTINEL.to_string())
    }

    #[test]
    fn parses_flat_call() {
        let call = parse_call(&q(r#"call:notes{action:"read",page:"all"}"#)).unwrap();
        assert_eq!(call.name, "notes");
        assert_eq!(call.arguments, json!({"action": "read", "page": "all"}));
    }

    #[test]
    fn parses_nested_and_scalars() {
        let body = q(r#"call:kanban{op:"add_card",cards:[{title:"a",priority:2,done:false},{title:"b, c"}],limit:null,ratio:0.5}"#);
        let call = parse_call(&body).unwrap();
        assert_eq!(
            call.arguments,
            json!({
                "op": "add_card",
                "cards": [{"title": "a", "priority": 2, "done": false}, {"title": "b, c"}],
                "limit": null,
                "ratio": 0.5
            })
        );
    }

    #[test]
    fn string_keeps_braces_and_newlines() {
        let body = q("call:bash{command:\"echo {x} | grep y\nls\"}");
        let call = parse_call(&body).unwrap();
        assert_eq!(call.arguments, json!({"command": "echo {x} | grep y\nls"}));
    }

    #[test]
    fn tolerates_json_style_quotes_and_missing_prefix() {
        let call = parse_call(r#"notes{"action": "read", "page": "Моя \"Жизнь\""}"#).unwrap();
        assert_eq!(call.name, "notes");
        assert_eq!(call.arguments, json!({"action": "read", "page": "Моя \"Жизнь\""}));
    }

    #[test]
    fn bare_value_without_quotes_is_string() {
        // Так тело выглядит, если кавычки-токены потеряны.
        let call = parse_call("call:notes{action:read,page:all}").unwrap();
        assert_eq!(call.arguments, json!({"action": "read", "page": "all"}));
    }

    #[test]
    fn unterminated_string_runs_to_end() {
        let body = q(r#"call:notes{action:"read",page:"all"#);
        let call = parse_call(&body).unwrap();
        assert_eq!(call.arguments, json!({"action": "read", "page": "all"}));
    }

    #[test]
    fn no_args_and_no_name() {
        assert_eq!(parse_call("call:notes").unwrap().arguments, json!({}));
        assert_eq!(parse_call("call:notes{}").unwrap().arguments, json!({}));
        assert!(parse_call("{action:x}").is_none());
        assert!(parse_call("").is_none());
    }

    /// Стрим, как его отдаёт фасад: спецтокены с пустой дельтой.
    fn run(p: &mut Gemma4StreamParser, stream: &[(u32, &str)]) -> Gemma4Split {
        let mut acc = Gemma4Split::default();
        for (id, delta) in stream {
            let s = p.feed(*id, delta);
            acc.body.push_str(&s.body);
            acc.thinking.push_str(&s.thinking);
            acc.tool.push_str(&s.tool);
        }
        acc
    }

    #[test]
    fn stream_splits_thinking_body_and_call() {
        let mut p = Gemma4StreamParser::new(IDS, false);
        let acc = run(
            &mut p,
            &[
                (100, ""),
                (1000, "thought"),
                (1001, "\nнадо "),
                (1002, "прочитать"),
                (101, ""),
                (1003, "Читаю."),
                (48, ""),
                (1004, "call:"),
                (1005, "notes{action:"),
                (52, ""),
                (1006, "read"),
                (52, ""),
                (1007, ",page:"),
                (52, ""),
                (1008, "all"),
                (52, ""),
                (1009, "}"),
            ],
        );
        assert_eq!(acc.thinking, "надо прочитать");
        assert_eq!(acc.body, "Читаю.");
        assert_eq!(acc.tool, "call:notes{action:\"read\",page:\"all\"}");
        assert!(!p.has_closed_call());
        let s = p.feed(49, "");
        assert_eq!(s, Gemma4Split::default());
        assert!(p.has_closed_call());
        let calls = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "notes");
        assert_eq!(calls[0].arguments, json!({"action": "read", "page": "all"}));
    }

    #[test]
    fn header_split_across_deltas() {
        let mut p = Gemma4StreamParser::new(IDS, false);
        let acc = run(&mut p, &[(100, ""), (1, "tho"), (2, "ught\n"), (3, "x"), (101, ""), (4, "y")]);
        assert_eq!(acc.thinking, "x");
        assert_eq!(acc.body, "y");
    }

    #[test]
    fn thinking_open_from_prompt() {
        let mut p = Gemma4StreamParser::new(IDS, true);
        let acc = run(&mut p, &[(1, "мысль"), (101, ""), (2, "ответ")]);
        assert_eq!(acc.thinking, "мысль");
        assert_eq!(acc.body, "ответ");
    }

    #[test]
    fn unclosed_call_is_still_parsed_on_finish() {
        let mut p = Gemma4StreamParser::new(IDS, false);
        run(&mut p, &[(48, ""), (1, "call:bash{command:"), (52, ""), (2, "ls"), (52, ""), (3, "}")]);
        assert!(!p.has_closed_call());
        let calls = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
    }

    #[test]
    fn detect_with_lookup() {
        let ids = Gemma4Ids::detect_with(|s| match s {
            TOK_TOOL_CALL_OPEN => Some(48),
            TOK_TOOL_CALL_CLOSE => Some(49),
            TOK_QUOTE => Some(52),
            TOK_CHANNEL_OPEN => Some(100),
            TOK_CHANNEL_CLOSE => Some(101),
            _ => None,
        });
        assert_eq!(ids, Some(IDS));
        assert!(Gemma4Ids::detect_with(|_| None).is_none());
    }
}
