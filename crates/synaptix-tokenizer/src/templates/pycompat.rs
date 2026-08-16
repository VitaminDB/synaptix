use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error as MjError, ErrorKind};

pub fn register_all(env: &mut Environment<'_>) {
    env.add_function("raise_exception", raise_exception);
    env.add_function("startswith", fn_startswith);
    env.add_function("endswith", fn_endswith);
    env.add_function("strip", fn_strip);
    env.add_function("lstrip", fn_lstrip);
    env.add_function("rstrip", fn_rstrip);
    env.add_function("lower", fn_lower);
    env.add_function("upper", fn_upper);
    env.add_function("split", fn_split);
    env.add_function("replace", fn_replace);
    env.add_function("items", fn_items);
    env.add_function("keys", fn_keys);
    env.add_function("values", fn_values);
    env.add_function("len", fn_len);
    env.add_function("get", fn_get);

    env.add_filter("startswith", filt_startswith);
    env.add_filter("endswith", filt_endswith);
    env.add_filter("strip", filt_strip);
    env.add_filter("lstrip", filt_lstrip);
    env.add_filter("rstrip", filt_rstrip);
    env.add_filter("split", filt_split);
    env.add_filter("replace", filt_replace);
    env.add_filter("tojson", filt_tojson);
}

fn raise_exception(msg: String) -> Result<Value, MjError> {
    Err(MjError::new(ErrorKind::InvalidOperation, format!("template raised: {msg}")))
}

fn as_str_required(v: &Value, what: &str) -> Result<String, MjError> {
    if let Some(s) = v.as_str() {
        Ok(s.to_owned())
    } else {
        Err(MjError::new(
            ErrorKind::InvalidOperation,
            format!("expected string for `{what}`, got {:?}", v.kind()),
        ))
    }
}

fn fn_startswith(s: Value, prefix: Value) -> Result<bool, MjError> {
    Ok(as_str_required(&s, "startswith")?
        .starts_with(as_str_required(&prefix, "startswith")?.as_str()))
}

fn filt_startswith(s: Value, prefix: Value) -> Result<bool, MjError> {
    fn_startswith(s, prefix)
}

fn fn_endswith(s: Value, suffix: Value) -> Result<bool, MjError> {
    Ok(as_str_required(&s, "endswith")?
        .ends_with(as_str_required(&suffix, "endswith")?.as_str()))
}

fn filt_endswith(s: Value, suffix: Value) -> Result<bool, MjError> {
    fn_endswith(s, suffix)
}

fn fn_strip(s: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "strip")?.trim().to_owned())
}

fn filt_strip(s: Value) -> Result<String, MjError> {
    fn_strip(s)
}

fn fn_lstrip(s: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "lstrip")?.trim_start().to_owned())
}

fn filt_lstrip(s: Value) -> Result<String, MjError> {
    fn_lstrip(s)
}

fn fn_rstrip(s: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "rstrip")?.trim_end().to_owned())
}

fn filt_rstrip(s: Value) -> Result<String, MjError> {
    fn_rstrip(s)
}

fn fn_lower(s: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "lower")?.to_lowercase())
}

fn fn_upper(s: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "upper")?.to_uppercase())
}

fn fn_split(s: Value, sep: Option<Value>) -> Result<Vec<String>, MjError> {
    let s = as_str_required(&s, "split")?;
    Ok(match sep {
        Some(v) => {
            let sep = as_str_required(&v, "split.sep")?;
            s.split(&sep).map(|x| x.to_owned()).collect()
        }
        None => s.split_whitespace().map(|x| x.to_owned()).collect(),
    })
}

fn filt_split(s: Value, sep: Option<Value>) -> Result<Vec<String>, MjError> {
    fn_split(s, sep)
}

fn fn_replace(s: Value, old: Value, new: Value) -> Result<String, MjError> {
    Ok(as_str_required(&s, "replace")?
        .replace(&as_str_required(&old, "replace.old")?, &as_str_required(&new, "replace.new")?))
}

fn filt_replace(s: Value, old: Value, new: Value) -> Result<String, MjError> {
    fn_replace(s, old, new)
}

fn fn_items(v: Value) -> Result<Vec<Value>, MjError> {
    if v.kind() != ValueKind::Map {
        return Err(MjError::new(
            ErrorKind::InvalidOperation,
            format!("items() expects a mapping, got {:?}", v.kind()),
        ));
    }
    let mut out = Vec::new();
    if let Ok(iter) = v.try_iter() {
        for k in iter {
            let val = v.get_item(&k).unwrap_or(Value::UNDEFINED);
            out.push(Value::from(vec![k, val]));
        }
    }
    Ok(out)
}

fn fn_keys(v: Value) -> Result<Vec<Value>, MjError> {
    if v.kind() != ValueKind::Map {
        return Err(MjError::new(
            ErrorKind::InvalidOperation,
            format!("keys() expects a mapping, got {:?}", v.kind()),
        ));
    }
    Ok(v.try_iter().map(|i| i.collect()).unwrap_or_default())
}

fn fn_values(v: Value) -> Result<Vec<Value>, MjError> {
    if v.kind() != ValueKind::Map {
        return Err(MjError::new(
            ErrorKind::InvalidOperation,
            format!("values() expects a mapping, got {:?}", v.kind()),
        ));
    }
    let mut out = Vec::new();
    if let Ok(iter) = v.try_iter() {
        for k in iter {
            let val = v.get_item(&k).unwrap_or(Value::UNDEFINED);
            out.push(val);
        }
    }
    Ok(out)
}

fn fn_get(v: Value, key: Value, default: Option<Value>) -> Result<Value, MjError> {
    let item = v.get_item(&key).unwrap_or(Value::UNDEFINED);
    if item.is_undefined() {
        Ok(default.unwrap_or_else(|| Value::from(())))
    } else {
        Ok(item)
    }
}

fn fn_len(v: Value) -> Result<usize, MjError> {
    v.len().ok_or_else(|| {
        MjError::new(
            ErrorKind::InvalidOperation,
            format!("len() not supported on {:?}", v.kind()),
        )
    })
}

fn filt_tojson(v: Value, indent: Option<usize>) -> Result<String, MjError> {
    let out = match indent {
        Some(n) if n > 0 => {
            let spaces = " ".repeat(n);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(spaces.as_bytes());
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            serde::Serialize::serialize(&v, &mut ser)
                .map_err(|e| MjError::new(ErrorKind::InvalidOperation, format!("tojson serialize failed: {e}")))?;
            String::from_utf8(buf).map_err(|e| {
                MjError::new(ErrorKind::InvalidOperation, format!("tojson utf8: {e}"))
            })?
        }
        _ => {
            let formatter = PySeparatorFormatter::default();
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            serde::Serialize::serialize(&v, &mut ser)
                .map_err(|e| MjError::new(ErrorKind::InvalidOperation, format!("tojson serialize failed: {e}")))?;
            String::from_utf8(buf).map_err(|e| {
                MjError::new(ErrorKind::InvalidOperation, format!("tojson utf8: {e}"))
            })?
        }
    };
    Ok(out)
}

#[derive(Default)]
struct PySeparatorFormatter;

impl serde_json::ser::Formatter for PySeparatorFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

pub fn preprocess(template: &str) -> String {
    const METHODS: &[&str] = &[
        "startswith",
        "endswith",
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "split",
        "replace",
        "items",
        "keys",
        "values",
        "get",
    ];
    let mut s = template.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        for m in METHODS {
            let needle = format!(".{}(", m);
            if let Some(pos) = s.find(&needle) {
                let expr_end = pos;
                let expr_start = find_expr_start(&s, expr_end);
                if expr_start == expr_end {
                    let stub = format!("__pyc_skip_{}_", m);
                    s = s.replacen(&needle, &stub, 1);
                    changed = true;
                    continue;
                }
                let args_start = pos + needle.len();
                let open_paren = pos + needle.len() - 1;
                let Some(args_end) = match_close_paren(&s, open_paren) else {
                    let stub = format!("__pyc_skip_{}_", m);
                    s = s.replacen(&needle, &stub, 1);
                    changed = true;
                    continue;
                };
                let expr = s[expr_start..expr_end].to_string();
                let args = s[args_start..args_end].to_string();
                let replacement = if args.trim().is_empty() {
                    format!("{}({})", m, expr)
                } else {
                    format!("{}({}, {})", m, expr, args)
                };
                s.replace_range(expr_start..=args_end, &replacement);
                changed = true;
                break;
            }
        }
    }
    s = s.replace("__pyc_skip_startswith_", ".startswith(");
    s = s.replace("__pyc_skip_endswith_", ".endswith(");
    s = s.replace("__pyc_skip_strip_", ".strip(");
    s = s.replace("__pyc_skip_lstrip_", ".lstrip(");
    s = s.replace("__pyc_skip_rstrip_", ".rstrip(");
    s = s.replace("__pyc_skip_lower_", ".lower(");
    s = s.replace("__pyc_skip_upper_", ".upper(");
    s = s.replace("__pyc_skip_split_", ".split(");
    s = s.replace("__pyc_skip_replace_", ".replace(");
    s = s.replace("__pyc_skip_items_", ".items(");
    s = s.replace("__pyc_skip_keys_", ".keys(");
    s = s.replace("__pyc_skip_values_", ".values(");
    s = s.replace("__pyc_skip_get_", ".get(");
    parenthesize_kwarg_ternaries(&s)
}

fn parenthesize_kwarg_ternaries(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(src.len() + 16);
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'=' && i > 0 && i + 1 < bytes.len() {
            let prev = bytes[i - 1];
            let next = bytes[i + 1];
            let is_kwarg = (prev.is_ascii_alphanumeric() || prev == b'_')
                && next != b'='
                && kwarg_name_position(bytes, i);
            if is_kwarg {
                let (seg_end, hit_close) = kwarg_segment_end(bytes, i + 1);
                let seg = &src[i + 1..seg_end];
                if hit_close
                    && segment_has_toplevel_ternary(seg)
                    && !seg.trim_start().starts_with('(')
                {
                    out.push(b'=');
                    out.push(b'(');
                    out.extend_from_slice(seg.trim_end().as_bytes());
                    out.push(b')');
                    let trimmed = seg.len() - seg.trim_end().len();
                    out.extend_from_slice(&bytes[seg_end - trimmed..seg_end]);
                    i = seg_end;
                    continue;
                }
            }
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

fn kwarg_name_position(bytes: &[u8], eq_pos: usize) -> bool {
    let mut i = eq_pos;
    while i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
        i -= 1;
    }
    if i == eq_pos {
        return false;
    }
    while i > 0 && (bytes[i - 1] == b' ' || bytes[i - 1] == b'\t') {
        i -= 1;
    }
    i > 0 && (bytes[i - 1] == b'(' || bytes[i - 1] == b',')
}

fn kwarg_segment_end(bytes: &[u8], start: usize) -> (usize, bool) {
    let mut depth: i32 = 0;
    let mut in_str: Option<u8> = None;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == q {
                in_str = None;
            }
        } else {
            match b {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    if depth == 0 {
                        return (i, b == b')');
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => return (i, true),
                b'%' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'}' => {
                    return (i, false)
                }
                b'"' | b'\'' => in_str = Some(b),
                _ => {}
            }
        }
        i += 1;
    }
    (bytes.len(), false)
}

fn segment_has_toplevel_ternary(seg: &str) -> bool {
    let bytes = seg.as_bytes();
    let mut depth: i32 = 0;
    let mut in_str: Option<u8> = None;
    let mut has_if = false;
    let mut has_else = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == q {
                in_str = None;
            }
        } else {
            match b {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b'"' | b'\'' => in_str = Some(b),
                b'i' if depth == 0
                    && seg[i..].starts_with("if")
                    && boundary(bytes, i, 2) =>
                {
                    has_if = true;
                }
                b'e' if depth == 0
                    && seg[i..].starts_with("else")
                    && boundary(bytes, i, 4) =>
                {
                    has_else = true;
                }
                _ => {}
            }
        }
        i += 1;
    }
    has_if && has_else
}

fn boundary(bytes: &[u8], start: usize, len: usize) -> bool {
    let before_ok = start == 0
        || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
    let after = start + len;
    let after_ok =
        after >= bytes.len() || !(bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_');
    before_ok && after_ok
}

fn find_expr_start(src: &str, end: usize) -> usize {
    let bytes = src.as_bytes();
    let mut i = end;
    let mut depth_paren: i32 = 0;
    let mut depth_brack: i32 = 0;
    while i > 0 {
        let b = bytes[i - 1];
        match b {
            b')' => {
                depth_paren += 1;
                i -= 1;
            }
            b'(' => {
                if depth_paren > 0 {
                    depth_paren -= 1;
                    i -= 1;
                } else {
                    return i;
                }
            }
            b']' => {
                depth_brack += 1;
                i -= 1;
            }
            b'[' => {
                if depth_brack > 0 {
                    depth_brack -= 1;
                    i -= 1;
                } else {
                    return i;
                }
            }
            b'"' | b'\'' if depth_paren == 0 && depth_brack == 0 => {
                let q = b;
                if i < 2 {
                    return i;
                }
                let mut j = i - 2;
                while j > 0 && bytes[j] != q {
                    j -= 1;
                }
                if j == 0 && bytes[0] != q {
                    return i;
                }
                i = j;
            }
            b' ' | b'\t' | b'\n' | b'\r' | b',' | b'{' | b'}' | b'+' | b'-' | b'*' | b'/'
            | b'%' | b'=' | b'<' | b'>' | b'!' | b'|' | b'?' | b':' | b';' | b'~'
                if depth_paren == 0 && depth_brack == 0 =>
            {
                return i;
            }
            _ => {
                i -= 1;
            }
        }
    }
    0
}

fn match_close_paren(src: &str, open_pos: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    if bytes.get(open_pos) != Some(&b'(') {
        return None;
    }
    let mut depth: i32 = 1;
    let mut i = open_pos + 1;
    let mut in_str: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == q {
                in_str = None;
            }
        } else {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                b'"' | b'\'' => in_str = Some(b),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_startswith_simple() {
        let r = preprocess("{{ message.content.startswith('Hello') }}");
        assert_eq!(r, "{{ startswith(message.content, 'Hello') }}");
    }

    #[test]
    fn rewrite_chained_split() {
        let r = preprocess("{{ name.split('-')[0] }}");
        assert_eq!(r, "{{ split(name, '-')[0] }}");
    }

    #[test]
    fn rewrite_with_inner_call() {
        let r = preprocess("{{ x.replace(other.lower(), '') }}");
        assert_eq!(r, "{{ replace(x, lower(other), '') }}");
    }

    #[test]
    fn no_change_when_no_method() {
        let r = preprocess("Hello {{ name }}!");
        assert_eq!(r, "Hello {{ name }}!");
    }
}
