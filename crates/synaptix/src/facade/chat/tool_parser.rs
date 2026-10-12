const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

#[derive(Debug, Clone)]
pub struct RawToolCall {
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Default)]
pub struct ToolParseFeed {
    pub clean_delta: String,
    pub tool_delta: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Outside,
    Inside,
}

pub struct ToolCallParser {
    state: State,
    buf: String,
    inner: String,
    calls: Vec<RawToolCall>,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            state: State::Outside,
            buf: String::new(),
            inner: String::new(),
            calls: Vec::new(),
        }
    }

    pub fn calls_count(&self) -> usize {
        self.calls.len()
    }

    pub fn is_outside(&self) -> bool {
        matches!(self.state, State::Outside)
    }

    pub fn feed(&mut self, delta: &str) -> ToolParseFeed {
        let mut out = String::new();
        let mut tool_out = String::new();
        let mut work = std::mem::take(&mut self.buf);
        work.push_str(delta);

        loop {
            match self.state {
                State::Outside => {
                    if let Some(idx) = work.find(OPEN_TAG) {
                        out.push_str(&work[..idx]);
                        work.drain(..idx + OPEN_TAG.len());
                        self.state = State::Inside;
                        continue;
                    }
                    let hold = potential_tag_suffix_len(&work, OPEN_TAG);
                    let emit_end = work.len() - hold;
                    if emit_end > 0 {
                        out.push_str(&work[..emit_end]);
                        work.drain(..emit_end);
                    }
                    break;
                }
                State::Inside => {
                    if let Some(idx) = work.find(CLOSE_TAG) {
                        tool_out.push_str(&work[..idx]);
                        self.inner.push_str(&work[..idx]);
                        self.accept_body(false);
                        self.inner.clear();
                        work.drain(..idx + CLOSE_TAG.len());
                        self.state = State::Outside;
                        continue;
                    }
                    let hold = potential_tag_suffix_len(&work, CLOSE_TAG);
                    let emit_end = work.len() - hold;
                    if emit_end > 0 {
                        tool_out.push_str(&work[..emit_end]);
                        self.inner.push_str(&work[..emit_end]);
                        work.drain(..emit_end);
                    }
                    break;
                }
            }
        }

        self.buf = work;
        ToolParseFeed { clean_delta: out, tool_delta: tool_out }
    }

    pub fn finish(mut self) -> (Vec<RawToolCall>, String) {
        match self.state {
            State::Outside => {
                let tail = std::mem::take(&mut self.buf);
                (self.calls, tail)
            }
            State::Inside => {
                let buf = std::mem::take(&mut self.buf);
                self.inner.push_str(&buf);
                self.accept_body(true);
                (self.calls, String::new())
            }
        }
    }
}

impl ToolCallParser {
    fn accept_body(&mut self, unclosed: bool) {
        match parse_tool_call_body(&self.inner) {
            Some((call, format)) => {
                if unclosed {
                    tracing::debug!(
                        name = %call.name,
                        "tool_call без </tool_call> — тело целое, принимаем"
                    );
                }
                if call.arguments_json == "{}" {
                    tracing::warn!(
                        name = %call.name,
                        tool_call_body = %self.inner.trim(),
                        "tool_call без аргументов — модель не дописала параметры"
                    );
                } else if format == ToolCallFormat::AnthropicXml {
                    tracing::info!(
                        name = %call.name,
                        "tool_call в XML-стиле вместо родного JSON"
                    );
                    tracing::debug!(
                        name = %call.name,
                        tool_call_body = %self.inner.trim(),
                        "тело XML-вызова"
                    );
                }
                self.calls.push(call);
            }
            None if unclosed => tracing::warn!(
                truncated_body = %self.inner.trim(),
                "stream закончился внутри <tool_call> без </tool_call>, \
                 тело не разбирается — отбрасываем"
            ),
            None => tracing::warn!(
                tool_call_body = %self.inner.trim(),
                "failed to parse <tool_call> JSON — пропускаем"
            ),
        }
    }
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

fn potential_tag_suffix_len(text: &str, tag: &str) -> usize {
    let bytes = text.as_bytes();
    let tag_bytes = tag.as_bytes();
    let max_check = tag_bytes.len().saturating_sub(1).min(bytes.len());
    for hold in (1..=max_check).rev() {
        let suffix_start = bytes.len() - hold;
        if !text.is_char_boundary(suffix_start) {
            continue;
        }
        let suffix = &bytes[suffix_start..];
        if tag_bytes.starts_with(suffix) {
            return hold;
        }
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallFormat {
    Qwen3Json,
    AnthropicXml,
}

fn parse_tool_call_body(body: &str) -> Option<(RawToolCall, ToolCallFormat)> {
    if let Some(call) = parse_tool_call_json_qwen(body) {
        return Some((call, ToolCallFormat::Qwen3Json));
    }
    parse_tool_call_xml_anthropic(body).map(|c| (c, ToolCallFormat::AnthropicXml))
}

fn parse_tool_call_json_qwen(body: &str) -> Option<RawToolCall> {
    let (v, repaired) = super::json_repair::parse_with_repair(body.trim())?;
    if repaired {
        tracing::warn!(
            tool_call_body = %body.trim(),
            "скобки в <tool_call> закрыты не в том порядке — пересобрали хвост"
        );
    }
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments") {
        Some(args) => normalize_arguments(args.clone()),
        None => {
            let mut rest = v.as_object().cloned().unwrap_or_default();
            rest.remove("name");
            normalize_arguments(serde_json::Value::Object(rest))
        }
    };
    let arguments_json = serde_json::to_string(&arguments).ok()?;
    Some(RawToolCall { name, arguments_json })
}

fn normalize_arguments(v: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::String(s) => match lenient_json(&s) {
            Some(inner @ Value::Object(_)) => normalize_arguments(inner),
            _ => Value::String(s),
        },
        Value::Object(mut map) => {
            let blank = map
                .get("arguments")
                .and_then(Value::as_str)
                .is_some_and(|s| s.trim().is_empty());
            if blank {
                map.remove("arguments");
                return Value::Object(map);
            }
            if map.len() == 1 {
                if let Some(inner) = map.remove("arguments") {
                    let inner = normalize_arguments(inner);
                    if inner.is_object() {
                        return inner;
                    }
                    map.insert("arguments".to_string(), inner);
                }
            }
            Value::Object(map)
        }
        other => other,
    }
}

fn parse_tool_call_xml_anthropic(body: &str) -> Option<RawToolCall> {
    let body = body.trim();
    parse_xml_invoke_style(body).or_else(|| parse_xml_function_eq_style(body))
}

fn parse_xml_invoke_style(body: &str) -> Option<RawToolCall> {
    let start = body.find("<invoke")?;
    let after_tag = &body[start + "<invoke".len()..];
    let tag_end = after_tag.find('>')?;
    let name = xml_attr(&after_tag[..tag_end], "name")?;
    if name.is_empty() {
        return None;
    }
    let args = parse_xml_params_attr(&after_tag[tag_end + 1..]);
    let arguments_json =
        serde_json::to_string(&normalize_arguments(serde_json::Value::Object(args))).ok()?;
    Some(RawToolCall { name, arguments_json })
}

fn xml_attr(attrs: &str, key: &str) -> Option<String> {
    let pos = attrs.find(key)?;
    let rest = attrs[pos + key.len()..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quote = rest.chars().next().filter(|c| *c == '"' || *c == '\'');
    match quote {
        Some(q) => {
            let rest = &rest[q.len_utf8()..];
            let end = rest.find(q)?;
            Some(rest[..end].trim().to_string())
        }
        None => {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            Some(rest[..end].trim().to_string())
        }
    }
}

fn parse_xml_params_attr(mut cursor: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut args = serde_json::Map::new();
    while let Some(p_start) = cursor.find("<parameter") {
        let after_p = &cursor[p_start + "<parameter".len()..];
        let Some(tag_end) = after_p.find('>') else { break };
        let Some(key) = xml_attr(&after_p[..tag_end], "name") else { break };
        let value_start = tag_end + 1;
        let (raw_val, next) = match after_p[value_start..].find("</parameter>") {
            Some(rel) => (
                &after_p[value_start..value_start + rel],
                &after_p[value_start + rel + "</parameter>".len()..],
            ),
            None => (cut_at_invoke_close(&after_p[value_start..]), ""),
        };
        if !key.is_empty() {
            args.insert(key, xml_param_value(raw_val));
        }
        if next.is_empty() {
            break;
        }
        cursor = next;
    }
    args
}

fn cut_at_function_close(v: &str) -> &str {
    let end = v
        .find("</function")
        .or_else(|| v.find("</tool_call"))
        .unwrap_or(v.len());
    &v[..end]
}

fn cut_at_invoke_close(v: &str) -> &str {
    let end = v
        .find("</invoke")
        .or_else(|| v.find("</function_calls"))
        .unwrap_or(v.len());
    &v[..end]
}

fn xml_param_value(raw: &str) -> serde_json::Value {
    let v = raw.strip_prefix('\n').unwrap_or(raw);
    let v = v.strip_suffix('\n').unwrap_or(v);
    lenient_json(v).unwrap_or_else(|| serde_json::Value::String(v.to_string()))
}

fn lenient_json(text: &str) -> Option<serde_json::Value> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
        return Some(v);
    }
    if t.starts_with('{') || t.starts_with('[') {
        let mut stream = serde_json::Deserializer::from_str(t).into_iter::<serde_json::Value>();
        if let Some(Ok(v)) = stream.next() {
            return Some(v);
        }
    }
    super::json_repair::parse_with_repair(t).map(|(v, _)| v)
}

fn parse_xml_function_eq_style(body: &str) -> Option<RawToolCall> {
    let mut rest = body.strip_prefix('<')?;
    let mut stripped = false;
    while let Some(r) = rest.strip_prefix("function") {
        rest = r;
        stripped = true;
    }
    if !stripped {
        return None;
    }
    let after_eq = rest.strip_prefix('=')?;
    let name_end = after_eq.find('>')?;
    let name = after_eq[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut cursor = &after_eq[name_end + 1..];
    let mut args = serde_json::Map::new();
    loop {
        let Some(p_start) = cursor.find("<parameter=") else { break };
        let after_p = &cursor[p_start + "<parameter=".len()..];
        let Some(key_end) = after_p.find('>') else { break };
        let key = after_p[..key_end].trim().to_string();
        let value_start = key_end + 1;
        let (raw_val, next) = match after_p[value_start..].find("</parameter>") {
            Some(rel) => (
                &after_p[value_start..value_start + rel],
                &after_p[value_start + rel + "</parameter>".len()..],
            ),
            None => (cut_at_function_close(&after_p[value_start..]), ""),
        };
        if !key.is_empty() {
            args.insert(key, xml_param_value(raw_val));
        }
        if next.is_empty() {
            break;
        }
        cursor = next;
    }

    let arguments_json =
        serde_json::to_string(&normalize_arguments(serde_json::Value::Object(args))).ok()?;
    Some(RawToolCall { name, arguments_json })
}

pub struct LlamaJsonParser {
    python_tag: Option<u32>,
    state: LlamaState,
    held: String,
    depth: u32,
    in_str: bool,
    escaped: bool,
    calls: Vec<RawToolCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaState {
    Lead,
    Json,
    AfterCall,
    Text,
}

impl LlamaJsonParser {
    pub fn new(python_tag: Option<u32>) -> Self {
        Self {
            python_tag,
            state: LlamaState::Lead,
            held: String::new(),
            depth: 0,
            in_str: false,
            escaped: false,
            calls: Vec::new(),
        }
    }

    pub fn has_closed_call(&self) -> bool {
        !self.calls.is_empty() && self.state != LlamaState::Json
    }

    pub fn feed(&mut self, id: u32, delta: &str) -> (String, String) {
        if Some(id) == self.python_tag {
            if self.state == LlamaState::Lead {
                self.held.clear();
            }
            return (String::new(), String::new());
        }
        let mut body = String::new();
        let mut tool = String::new();
        let mut rest = delta;
        while !rest.is_empty() {
            match self.state {
                LlamaState::Text => {
                    body.push_str(rest);
                    break;
                }
                LlamaState::AfterCall => break,
                LlamaState::Lead => {
                    let trimmed = rest.trim_start();
                    self.held.push_str(&rest[..rest.len() - trimmed.len()]);
                    rest = trimmed;
                    if rest.is_empty() {
                        break;
                    }
                    if rest.starts_with('{') {
                        self.held.clear();
                        self.state = LlamaState::Json;
                    } else {
                        body.push_str(&std::mem::take(&mut self.held));
                        self.state = LlamaState::Text;
                    }
                }
                LlamaState::Json => {
                    let (consumed, closed) = self.scan(rest);
                    tool.push_str(&rest[..consumed]);
                    self.held.push_str(&rest[..consumed]);
                    rest = &rest[consumed..];
                    if closed {
                        match parse_llama_call(&self.held) {
                            Some(call) => {
                                self.calls.push(call);
                                self.held.clear();
                                self.state = LlamaState::AfterCall;
                            }
                            None => {
                                body.push_str(&std::mem::take(&mut self.held));
                                self.state = LlamaState::Text;
                            }
                        }
                    }
                }
            }
        }
        (body, tool)
    }

    fn scan(&mut self, s: &str) -> (usize, bool) {
        for (i, c) in s.char_indices() {
            if self.in_str {
                if self.escaped {
                    self.escaped = false;
                } else if c == '\\' {
                    self.escaped = true;
                } else if c == '"' {
                    self.in_str = false;
                }
                continue;
            }
            match c {
                '"' => self.in_str = true,
                '{' | '[' => self.depth += 1,
                '}' | ']' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        return (i + c.len_utf8(), true);
                    }
                }
                _ => {}
            }
        }
        (s.len(), false)
    }

    pub fn take_tail(&mut self) -> String {
        if self.state != LlamaState::Json {
            return String::new();
        }
        let held = std::mem::take(&mut self.held);
        match parse_llama_call(&held) {
            Some(call) => {
                self.calls.push(call);
                self.state = LlamaState::AfterCall;
                String::new()
            }
            None => {
                self.state = LlamaState::Text;
                held
            }
        }
    }

    pub fn finish(self) -> Vec<RawToolCall> {
        self.calls
    }
}

fn parse_llama_call(body: &str) -> Option<RawToolCall> {
    let (v, _) = super::json_repair::parse_with_repair(body.trim())?;
    let obj = v.as_object()?;
    let obj = match obj.get("function").and_then(|f| f.as_object()) {
        Some(f) if f.contains_key("name") => f,
        _ => obj,
    };
    let name = obj.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match obj.get("parameters").or_else(|| obj.get("arguments")) {
        Some(a) => normalize_arguments(a.clone()),
        None => serde_json::Value::Object(serde_json::Map::new()),
    };
    if !arguments.is_object() {
        return None;
    }
    let arguments_json = serde_json::to_string(&arguments).ok()?;
    Some(RawToolCall { name, arguments_json })
}

#[cfg(test)]
mod tests {
    fn llama_feed(p: &mut LlamaJsonParser, chunks: &[&str]) -> (String, String) {
        let (mut body, mut tool) = (String::new(), String::new());
        for c in chunks {
            let (b, t) = p.feed(0, c);
            body.push_str(&b);
            tool.push_str(&t);
        }
        (body, tool)
    }

    #[test]
    fn llama_bare_json_call() {
        let mut p = LlamaJsonParser::new(Some(7));
        let (body, tool) = llama_feed(
            &mut p,
            &["\n", "{\"name\": \"ba", "sh\", \"parameters\": {\"command\": \"echo }\"", "}}", "\n\nлишнее"],
        );
        assert!(body.is_empty(), "ответ пуст: {body:?}");
        assert!(tool.contains("\"bash\""));
        assert!(p.has_closed_call());
        let calls = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments_json, "{\"command\":\"echo }\"}");
    }

    #[test]
    fn llama_python_tag_and_arguments() {
        let mut p = LlamaJsonParser::new(Some(7));
        assert_eq!(p.feed(7, "<|python_tag|>"), (String::new(), String::new()));
        let (body, _) = llama_feed(&mut p, &["{\"name\": \"notes\", \"arguments\": \"{\\\"action\\\": \\\"read\\\"}\"}"]);
        assert!(body.is_empty());
        let calls = p.finish();
        assert_eq!(calls[0].name, "notes");
        assert_eq!(calls[0].arguments_json, "{\"action\":\"read\"}");
    }

    #[test]
    fn llama_plain_text_passes_through() {
        let mut p = LlamaJsonParser::new(None);
        let (body, tool) = llama_feed(&mut p, &["  Привет", "! {\"name\": \"x\", \"parameters\": {}}"]);
        assert_eq!(body, "  Привет! {\"name\": \"x\", \"parameters\": {}}");
        assert!(tool.is_empty());
        assert!(p.finish().is_empty());
    }

    #[test]
    fn llama_json_answer_is_text() {
        let mut p = LlamaJsonParser::new(None);
        let (body, _) = llama_feed(&mut p, &["{\"city\": \"Paris\"}", " — готово"]);
        assert_eq!(body, "{\"city\": \"Paris\"} — готово");
        assert!(p.finish().is_empty());
    }

    #[test]
    fn llama_unclosed_tail() {
        let mut p = LlamaJsonParser::new(None);
        llama_feed(&mut p, &["{\"name\": \"bash\", \"parameters\": {\"command\": \"ls\"}"]);
        assert!(!p.has_closed_call());
        assert_eq!(p.take_tail(), "");
        assert_eq!(p.finish()[0].name, "bash");

        let mut p = LlamaJsonParser::new(None);
        llama_feed(&mut p, &["{\"na"]);
        assert_eq!(p.take_tail(), "{\"na");
        assert!(p.finish().is_empty());
    }

    #[test]
    fn unclosed_tool_call_with_valid_body_is_accepted() {
        let mut p = ToolCallParser::new();
        p.feed(
            "<tool_call>{\"name\":\"pipelines\",\"arguments\":{\"action\":\"run\",\"free_vram\":true}}",
        );
        let (calls, tail) = p.finish();
        assert_eq!(calls.len(), 1, "вызов принят");
        assert_eq!(calls[0].name, "pipelines");
        assert!(calls[0].arguments_json.contains("\"action\":\"run\""));
        assert!(tail.is_empty());
    }

    #[test]
    fn unclosed_tool_call_with_broken_body_is_dropped() {
        let mut p = ToolCallParser::new();
        p.feed("<tool_call>{\"name\":\"pipel");
        let (calls, _) = p.finish();
        assert!(calls.is_empty());
    }

    use super::*;

    fn feed_all(parser: &mut ToolCallParser, chunks: &[&str]) -> String {
        let mut out = String::new();
        for c in chunks {
            out.push_str(&parser.feed(c).clean_delta);
        }
        out
    }

    #[test]
    fn stringified_arguments_are_unwrapped() {
        let mut p = ToolCallParser::new();
        p.feed(r#"<tool_call>{"name":"bash","arguments":"{\"command\": \"ls -la\"}"}</tool_call>"#);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments_json, r#"{"command":"ls -la"}"#);
    }

    #[test]
    fn nested_arguments_wrapper_is_unwrapped() {
        let mut p = ToolCallParser::new();
        p.feed(r#"<tool_call>{"name":"bash","arguments":{"arguments":{"command":"pwd"}}}</tool_call>"#);
        let (calls, _) = p.finish();
        assert_eq!(calls[0].arguments_json, r#"{"command":"pwd"}"#);
    }

    #[test]
    fn blank_arguments_stub_is_dropped() {
        let mut p = ToolCallParser::new();
        p.feed("<tool_call>{\"name\":\"bash\",\"arguments\":{\"arguments\":\"\\n\\n\",\"command\":\"id\"}}</tool_call>");
        let (calls, _) = p.finish();
        assert_eq!(calls[0].arguments_json, r#"{"command":"id"}"#);

        let mut p = ToolCallParser::new();
        p.feed("<tool_call>{\"name\":\"bash\",\"arguments\":{\"arguments\":\"\\n\"}}</tool_call>");
        let (calls, _) = p.finish();
        assert_eq!(calls[0].arguments_json, "{}");
    }

    #[test]
    fn stringified_arguments_with_extra_brace_are_unwrapped() {
        let raw = r#"{"arguments":"{\"command\":\"cd /tmp && echo \\\"---\\\" && wc -l a b\"}}"}"#;
        let v: serde_json::Value = serde_json::from_str(raw).unwrap();
        let out = normalize_arguments(v);
        assert_eq!(
            out,
            serde_json::json!({"command": "cd /tmp && echo \"---\" && wc -l a b"})
        );

        let mut p = ToolCallParser::new();
        p.feed("<tool_call><function=bash><parameter=arguments>{\"command\":\"ls\"}}</parameter></function></tool_call>");
        let (calls, _) = p.finish();
        assert_eq!(calls[0].arguments_json, r#"{"command":"ls"}"#);
    }

    #[test]
    fn function_style_without_closing_parameter_keeps_value() {
        let mut p = ToolCallParser::new();
        p.feed("<tool_call><function=web>\n<parameter=arguments>\n{\"action\":\"search\",\"query\":\"redmi\",\"max_results\":10}}\n");
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web");
        assert_eq!(
            calls[0].arguments_json,
            r#"{"action":"search","query":"redmi","max_results":10}"#
        );
    }

    #[test]
    fn parameters_next_to_name_become_arguments() {
        let mut p = ToolCallParser::new();
        p.feed(r#"<tool_call>{"name":"bash","command":"uname -a"}</tool_call>"#);
        let (calls, _) = p.finish();
        assert_eq!(calls[0].arguments_json, r#"{"command":"uname -a"}"#);
    }

    #[test]
    fn xml_arguments_parameter_is_unwrapped() {
        let mut p = ToolCallParser::new();
        p.feed(
            "<tool_call><invoke name=\"bash\"><parameter name=\"arguments\">\
             {\"command\": \"ls\"}</parameter></invoke></tool_call>",
        );
        let (calls, _) = p.finish();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments_json, r#"{"command":"ls"}"#);
    }

    #[test]
    fn feed_no_tool() {
        let mut p = ToolCallParser::new();
        let out = feed_all(&mut p, &["Hello", " world!"]);
        let (calls, tail) = p.finish();
        assert_eq!(out, "Hello world!");
        assert!(calls.is_empty());
        assert_eq!(tail, "");
    }

    #[test]
    fn feed_full_tool_call_one_chunk() {
        let mut p = ToolCallParser::new();
        let chunk = r#"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#;
        let out = feed_all(&mut p, &[chunk]);
        let (calls, tail) = p.finish();
        assert_eq!(out, "");
        assert_eq!(tail, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].arguments_json.contains("\"command\":\"ls\""));
    }

    #[test]
    fn feed_split_across_chunks() {
        let mut p = ToolCallParser::new();
        let out = feed_all(&mut p, &[
            "Сейчас вызову bash. ",
            "<to",
            "ol_call>",
            r#"{"name":"bash","#,
            r#""arguments":{"command":"pwd"}}"#,
            "</tool_call>",
            " готово.",
        ]);
        let (calls, _tail) = p.finish();
        assert_eq!(out, "Сейчас вызову bash.  готово.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn feed_partial_tag_held() {
        let mut p = ToolCallParser::new();
        let part1 = p.feed("hello <to");
        assert_eq!(part1.clean_delta, "hello ");
        let part2 = p.feed("o");
        assert_eq!(part2.clean_delta, "");
        let part3 = p.feed("ot");
        assert_eq!(part3.clean_delta, "<tooot");
    }

    #[test]
    fn out_of_order_brackets_are_repaired() {
        let mut p = ToolCallParser::new();
        p.feed(
            "<tool_call>\n{\"name\":\"pipelines\",\"arguments\":{\"action\":\"apply\",\
             \"set_state\":[{\"node\":9,\"state\":{\"kind\":\"TextView\",\
             \"data\":{\"output_text\":\"[verse]\"}}]}}\n</tool_call>",
        );
        let (calls, _tail) = p.finish();
        assert_eq!(calls.len(), 1, "вызов принят после починки скобок");
        assert_eq!(calls[0].name, "pipelines");
        assert!(calls[0].arguments_json.contains("\"node\":9"), "{}", calls[0].arguments_json);
    }

    #[test]
    fn multiple_calls() {
        let mut p = ToolCallParser::new();
        let chunk = format!(
            "{}{}",
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>"#,
        );
        let out = feed_all(&mut p, &[&chunk]);
        let (calls, _tail) = p.finish();
        assert_eq!(out, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert!(calls[1].arguments_json.contains("\"x\":1"));
    }

    #[test]
    fn mixed_prose_and_call() {
        let mut p = ToolCallParser::new();
        let chunk = r#"Сейчас<tool_call>{"name":"x","arguments":{}}</tool_call>готово"#;
        let out = feed_all(&mut p, &[chunk]);
        let (calls, _tail) = p.finish();
        assert_eq!(out, "Сейчасготово");
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn invalid_json_logged_not_panicking() {
        let mut p = ToolCallParser::new();
        let out = feed_all(&mut p, &["<tool_call>not a json</tool_call>"]);
        let (calls, _tail) = p.finish();
        assert!(out.is_empty());
        assert!(calls.is_empty(), "невалидный JSON — call пропущен");
    }

    #[test]
    fn arguments_missing_treated_as_empty_object() {
        let mut p = ToolCallParser::new();
        let _ = p.feed(r#"<tool_call>{"name":"foo"}</tool_call>"#);
        let (calls, _tail) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments_json, "{}");
    }

    #[test]
    fn anthropic_xml_inside_tool_call() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call>\n<function=subagent>\n\
            <parameter=system_prompt>You are helper</parameter>\n\
            <parameter=task>Do X</parameter>\n\
            </function>\n</tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "subagent");
        assert!(calls[0].arguments_json.contains("\"system_prompt\":\"You are helper\""));
        assert!(calls[0].arguments_json.contains("\"task\":\"Do X\""));
    }

    #[test]
    fn anthropic_xml_double_function_glitch() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><functionfunction=bash>\
            <parameter=command>ls -la</parameter></function></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].arguments_json.contains("\"command\":\"ls -la\""));
    }

    #[test]
    fn anthropic_xml_numeric_value_parsed_as_number() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><function=foo>\
            <parameter=n>42</parameter></function></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].arguments_json.contains("\"n\":42"));
    }

    #[test]
    fn anthropic_xml_without_closing_function_tag() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><function=foo>\
            <parameter=x>y</parameter></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "foo");
        assert!(calls[0].arguments_json.contains("\"x\":\"y\""));
    }

    #[test]
    fn anthropic_invoke_style_unclosed_stream() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call>\n<invoke name=\"bash\">\n\
            <parameter name=\"command\">cd /home/master/Projects/2027/quitsmoke \
            && find src styles -type f | sort</parameter>\n\
            </invoke>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1, "вызов не должен теряться");
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].arguments_json.contains("find src styles -type f"));
    }

    #[test]
    fn anthropic_invoke_style_with_function_calls_wrapper() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><function_calls>\
            <invoke name=\"subagent\">\
            <parameter name=\"task\">Do X</parameter>\
            <parameter name=\"depth\">2</parameter>\
            </invoke></function_calls></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "subagent");
        assert!(calls[0].arguments_json.contains("\"task\":\"Do X\""));
        assert!(calls[0].arguments_json.contains("\"depth\":2"));
    }

    #[test]
    fn anthropic_invoke_style_multiline_value_keeps_inner_layout() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><invoke name=\"bash\">\
            <parameter name=\"command\">\nls -la\n  nested\n</parameter>\
            </invoke></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0].arguments_json.contains("\"command\":\"ls -la\\n  nested\""),
            "было: {}",
            calls[0].arguments_json
        );
    }

    #[test]
    fn anthropic_invoke_style_truncated_inside_value() {
        let mut p = ToolCallParser::new();
        let _ = p.feed("<tool_call><invoke name=\"bash\">\
            <parameter name=\"command\">ls -la");
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].arguments_json.contains("\"command\":\"ls -la\""));
    }

    #[test]
    fn anthropic_invoke_style_single_quoted_name() {
        let mut p = ToolCallParser::new();
        let chunk = "<tool_call><invoke name='web'>\
            <parameter name='url'>https://example.com</parameter>\
            </invoke></tool_call>";
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web");
        assert!(calls[0].arguments_json.contains("https://example.com"));
    }

    #[test]
    fn qwen_json_still_works_after_fallback_added() {
        let mut p = ToolCallParser::new();
        let chunk = r#"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#;
        let _ = p.feed(chunk);
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].arguments_json.contains("\"command\":\"ls\""));
    }

    #[test]
    fn tool_delta_streams_incrementally() {
        let mut p = ToolCallParser::new();
        assert_eq!(p.feed("<tool_call>").tool_delta, "");
        assert_eq!(p.feed(r#"{"name":"bash","#).tool_delta, r#"{"name":"bash","#);
        assert_eq!(
            p.feed(r#""arguments":{"command":"ls"}}"#).tool_delta,
            r#""arguments":{"command":"ls"}}"#
        );
        assert_eq!(p.feed("</tool_call>").tool_delta, "");
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn tool_delta_holds_partial_close_tag() {
        let mut p = ToolCallParser::new();
        let _ = p.feed("<tool_call>");
        assert_eq!(p.feed(r#"{"name":"x"}</tool_c"#).tool_delta, r#"{"name":"x"}"#);
        assert_eq!(p.feed("all>").tool_delta, "");
        let (calls, _) = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "x");
    }

    #[test]
    fn tool_delta_empty_outside_block() {
        let mut p = ToolCallParser::new();
        let feed = p.feed("обычный текст");
        assert_eq!(feed.clean_delta, "обычный текст");
        assert_eq!(feed.tool_delta, "");
    }

    #[test]
    fn potential_tag_suffix_basic() {
        assert_eq!(potential_tag_suffix_len("abc<too", "<tool_call>"), 4);
        assert_eq!(potential_tag_suffix_len("abc<", "<tool_call>"), 1);
        assert_eq!(potential_tag_suffix_len("plain text", "<tool_call>"), 0);
        assert_eq!(potential_tag_suffix_len("", "<tool_call>"), 0);
    }
}
