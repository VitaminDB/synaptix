//! Разбор ответа Inkling-2: реплики `<|message_model|>` [имя] `<|content_…|>`
//! тело `<|end_message|>`; размышления — `<|content_thinking|>`, ответ —
//! `<|content_text|>`, вызов инструмента — `<|content_invoke_tool_json|>` с
//! телом `{"name":…,"args":{…}}`. Общий для службы и CLI.

/// Разметка реплик (для разбора ответа и для учёта и бюджета размышлений в
/// движке).
pub mod markup {
    /// Начало реплики модели.
    pub const MESSAGE: &str = "<|message_model|>";
    /// Часть с размышлениями.
    pub const THINKING: &str = "<|content_thinking|>";
    /// Часть с ответом.
    pub const TEXT: &str = "<|content_text|>";
    /// Часть с вызовом инструмента (JSON `{"name":…,"args":{…}}`).
    pub const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
    /// Конец части.
    pub const END: &str = "<|end_message|>";
    /// Принудительное закрытие размышлений по бюджету: конец части и начало
    /// следующей реплики — модель сама выбирает ответ или вызов инструмента.
    pub const FORCED_THINKING_CLOSE: &str = "<|end_message|><|message_model|>";
}

use crate::facade::llm::LlmTokenizer;

use super::tool_parser::RawToolCall;

#[derive(Debug, Clone, Copy)]
pub struct InklingIds {
    message: u32,
    thinking: u32,
    text: u32,
    tool: u32,
    end: u32,
}

impl InklingIds {
    pub fn detect(tokenizer: &LlmTokenizer) -> Option<Self> {
        let single = |s: &str| match tokenizer.encode(s) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        };
        Some(Self {
            message: single(markup::MESSAGE)?,
            thinking: single(markup::THINKING)?,
            text: single(markup::TEXT)?,
            tool: single(markup::TOOL_JSON)?,
            end: single(markup::END)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Thinking,
    Text,
    Tool,
}

/// Куски очередного токена (как у остальных разборщиков фасада).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InklingSplit {
    pub body: String,
    pub thinking: String,
    /// Текст вызова для показа: `[имя] {json}\n`.
    pub tool: String,
}

pub struct InklingParser {
    ids: InklingIds,
    /// `None` — заголовок реплики (имя автора копится в `header`).
    part: Option<Part>,
    header: String,
    /// Тело текущего вызова инструмента.
    call: String,
    calls: Vec<RawToolCall>,
    /// Вызов только что закрылся: если следом не начнётся новая реплика
    /// (ещё один вызов), ход готов к исполнению инструментов.
    after_call: bool,
    ready: bool,
    /// Была ли разметка реплики. Без неё (вывод под грамматикой с первого
    /// токена: JSON-схема) весь текст — ответ.
    marked: bool,
}

impl InklingParser {
    pub fn new(ids: InklingIds) -> Self {
        Self { ids, part: None, header: String::new(), call: String::new(), calls: Vec::new(), after_call: false, ready: false, marked: false }
    }

    pub fn feed(&mut self, id: u32, delta: &str) -> InklingSplit {
        let mut out = InklingSplit::default();
        if std::mem::take(&mut self.after_call) && id != self.ids.message {
            self.ready = true;
        }
        let ids = self.ids;
        let part = if id == ids.thinking {
            Some(Part::Thinking)
        } else if id == ids.text {
            Some(Part::Text)
        } else if id == ids.tool {
            Some(Part::Tool)
        } else {
            None
        };
        if id == ids.message || part.is_some() || id == ids.end {
            self.marked = true;
        }
        if id == ids.message {
            self.part = None;
            self.header.clear();
        } else if let Some(p) = part {
            if p == Part::Tool {
                self.call.clear();
                out.tool = format!("[{}] ", self.header.trim());
            }
            self.part = Some(p);
        } else if id == ids.end {
            if self.part == Some(Part::Tool) {
                self.close_call();
                out.tool.push('\n');
                self.after_call = true;
            }
            self.part = None;
            self.header.clear();
        } else {
            // Прочие спецтокены (`<|content_model_end_sampling|>`) декодируются
            // в пустую строку.
            match self.part {
                None if !self.marked => out.body.push_str(delta),
                None => self.header.push_str(delta),
                Some(Part::Thinking) => out.thinking.push_str(delta),
                Some(Part::Text) => out.body.push_str(delta),
                Some(Part::Tool) => {
                    self.call.push_str(delta);
                    out.tool.push_str(delta);
                }
            }
        }
        out
    }

    fn close_call(&mut self) {
        let raw = std::mem::take(&mut self.call);
        let header = self.header.trim();
        let value = serde_json::from_str::<serde_json::Value>(raw.trim())
            .ok()
            .or_else(|| super::json_repair::repair_bracket_tail(raw.trim()).and_then(|s| serde_json::from_str(&s).ok()));
        let (name, args) = match value {
            Some(serde_json::Value::Object(mut o)) => {
                let name = o.get("name").and_then(|v| v.as_str()).map(str::to_string);
                let args = o.remove("args").or_else(|| o.remove("arguments"));
                (name, args)
            }
            _ => (None, None),
        };
        let name = name.filter(|n| !n.is_empty()).unwrap_or_else(|| header.to_string());
        if name.is_empty() {
            return;
        }
        let args = match args {
            Some(serde_json::Value::String(s)) => s,
            Some(v) => v.to_string(),
            None => "{}".to_string(),
        };
        self.calls.push(RawToolCall { name, arguments_json: args });
    }

    /// Вызов закрыт, и следом модель не начала новый — пора исполнять.
    pub fn has_closed_call(&self) -> bool {
        self.ready
    }

    pub fn finish(mut self) -> Vec<RawToolCall> {
        // Ход оборван посреди вызова (стоп по готовности не успел) — берём,
        // что есть, если JSON удаётся починить.
        if self.part == Some(Part::Tool) && !self.call.is_empty() {
            self.close_call();
        }
        self.calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: InklingIds = InklingIds { message: 1, thinking: 2, text: 3, tool: 4, end: 5 };

    fn run(seq: &[(u32, &str)]) -> (String, String, String, Vec<RawToolCall>, bool) {
        let mut p = InklingParser::new(IDS);
        let (mut b, mut t, mut c) = (String::new(), String::new(), String::new());
        for &(id, s) in seq {
            let o = p.feed(id, s);
            b += &o.body;
            t += &o.thinking;
            c += &o.tool;
        }
        let ready = p.has_closed_call();
        (b, t, c, p.finish(), ready)
    }

    #[test]
    fn thinking_then_text() {
        let (b, t, c, calls, ready) =
            run(&[(1, ""), (2, ""), (0, "думаю"), (0, " ещё"), (5, ""), (1, ""), (3, ""), (0, "Ответ"), (5, ""), (9, "")]);
        assert_eq!((b.as_str(), t.as_str(), c.as_str()), ("Ответ", "думаю ещё", ""));
        assert!(calls.is_empty() && !ready);
    }

    #[test]
    fn tool_call() {
        let (b, t, _c, calls, ready) = run(&[
            (1, ""),
            (2, ""),
            (0, "нужен поиск"),
            (5, ""),
            (1, ""),
            (0, "search"),
            (4, ""),
            (0, r#"{"name":"search","args":{"q":"кот"}}"#),
            (5, ""),
            (9, ""),
        ]);
        assert_eq!((b.as_str(), t.as_str()), ("", "нужен поиск"));
        assert!(ready);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments_json, r#"{"q":"кот"}"#);
    }

    #[test]
    fn unmarked_output_is_answer() {
        let (b, t, c, calls, _) = run(&[(0, "{\"a\":"), (0, "1}"), (9, "")]);
        assert_eq!((b.as_str(), t.as_str(), c.as_str()), ("{\"a\":1}", "", ""));
        assert!(calls.is_empty());
    }

    #[test]
    fn two_calls_in_a_row() {
        let mut p = InklingParser::new(IDS);
        for &(id, s) in &[(1, ""), (0, "a"), (4, ""), (0, r#"{"name":"a","args":{}}"#), (5, "")] {
            p.feed(id, s);
        }
        assert!(!p.has_closed_call());
        p.feed(1, "");
        assert!(!p.has_closed_call());
        for &(id, s) in &[(0, "b"), (4, ""), (0, r#"{"name":"b","args":{"x":1}}"#), (5, ""), (9, "")] {
            p.feed(id, s);
        }
        assert!(p.has_closed_call());
        let calls = p.finish();
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }
}
