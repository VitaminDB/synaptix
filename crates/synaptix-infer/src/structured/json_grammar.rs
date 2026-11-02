//! JSON-схема → грамматика [`Grammar`] для ограниченной генерации.
//!
//! Поддержано: type (в т.ч. списком), properties + required (поля в порядке
//! схемы, необязательные можно пропустить), additionalProperties (схема — как
//! словарь), items/prefixItems, minItems/maxItems, enum, const, anyOf/oneOf,
//! allOf (слияние объектов), $ref на `#/$defs/…` и `#/definitions/…` (с
//! рекурсией), nullable, minLength/maxLength у строк (до 256), integer с
//! небольшим диапазоном minimum..maximum (перечислением), pattern — нет
//! (строка как есть). Без схемы — любой JSON.

use std::collections::HashMap;

use serde_json::Value;

use super::cfg::{lit, ByteSet, Elem, Grammar};

/// Сколько пробельных символов подряд разрешено между лексемами (с
/// переводами строк): без предела модель может зациклиться на отступах.
const WS_MAX: usize = 40;
/// Длинные пределы длины строк и массивов — без счёта (только синтаксис).
const COUNT_MAX: usize = 256;
/// Диапазон целых, который перечисляется явно.
const INT_ENUM_MAX: i64 = 400;

pub struct JsonGrammarBuilder<'a> {
    g: Grammar,
    root: &'a Value,
    refs: HashMap<String, u32>,
    ws: u32,
    string: u32,
    number: u32,
    integer: u32,
    any: u32,
    /// Строки с ограничением длины: (min, max) → правило.
    sized: HashMap<(usize, usize), u32>,
    chr: u32,
}

/// Грамматика для JSON-схемы (`schema = null` или `{}` — любой JSON).
pub fn json_schema_grammar(schema: &Value) -> Result<Grammar, String> {
    let mut b = JsonGrammarBuilder::new(schema);
    let root = b.g.rule("root");
    let v = b.value(schema, 0)?;
    b.g.set(root, vec![vec![Elem::Rule(b.ws), Elem::Rule(v), Elem::Rule(b.ws)]]);
    b.g.start = root;
    b.g.validate()?;
    Ok(b.g)
}

fn cont() -> ByteSet {
    ByteSet::range(0x80, 0xBF)
}

impl<'a> JsonGrammarBuilder<'a> {
    fn new(root: &'a Value) -> Self {
        let mut g = Grammar::default();
        // ws: до WS_MAX пробельных подряд (цепочка правил).
        let wsset = ByteSet::of(b" \t\n\r");
        let mut next = g.rule("ws0");
        g.set(next, vec![vec![]]);
        for i in 1..=WS_MAX {
            let r = g.rule(&format!("ws{i}"));
            g.set(r, vec![vec![], vec![Elem::Bytes(wsset), Elem::Rule(next)]]);
            next = r;
        }
        let ws = next;

        // Символ строки: ASCII без кавычки, обратной косой и управляющих;
        // экранирование; корректный UTF-8.
        let chr = g.rule("char");
        let plain = ByteSet::range(0x20, 0x7F).minus(ByteSet::of(b"\"\\"));
        let hex = ByteSet::range(b'0', b'9').union(ByteSet::range(b'a', b'f')).union(ByteSet::range(b'A', b'F'));
        g.set(
            chr,
            vec![
                vec![Elem::Bytes(plain)],
                vec![Elem::Bytes(ByteSet::one(b'\\')), Elem::Bytes(ByteSet::of(b"\"\\/bfnrt"))],
                vec![
                    Elem::Bytes(ByteSet::one(b'\\')),
                    Elem::Bytes(ByteSet::one(b'u')),
                    Elem::Bytes(hex),
                    Elem::Bytes(hex),
                    Elem::Bytes(hex),
                    Elem::Bytes(hex),
                ],
                vec![Elem::Bytes(ByteSet::range(0xC2, 0xDF)), Elem::Bytes(cont())],
                vec![Elem::Bytes(ByteSet::range(0xE0, 0xEF)), Elem::Bytes(cont()), Elem::Bytes(cont())],
                vec![Elem::Bytes(ByteSet::range(0xF0, 0xF4)), Elem::Bytes(cont()), Elem::Bytes(cont()), Elem::Bytes(cont())],
            ],
        );
        let chars = g.rule("chars");
        g.set(chars, vec![vec![], vec![Elem::Rule(chr), Elem::Rule(chars)]]);
        let q = Elem::Bytes(ByteSet::one(b'"'));
        let string = g.rule("string");
        g.set(string, vec![vec![q, Elem::Rule(chars), q]]);

        // Числа.
        let digit = Elem::Bytes(ByteSet::range(b'0', b'9'));
        let digits = g.rule("digits");
        g.set(digits, vec![vec![], vec![digit, Elem::Rule(digits)]]);
        let uint = g.rule("uint");
        g.set(uint, vec![lit(b"0"), vec![Elem::Bytes(ByteSet::range(b'1', b'9')), Elem::Rule(digits)]]);
        let integer = g.rule("integer");
        g.set(integer, vec![vec![Elem::Rule(uint)], vec![Elem::Bytes(ByteSet::one(b'-')), Elem::Rule(uint)]]);
        let frac = g.rule("frac");
        g.set(frac, vec![vec![], vec![Elem::Bytes(ByteSet::one(b'.')), digit, Elem::Rule(digits)]]);
        let exp = g.rule("exp");
        g.set(
            exp,
            vec![
                vec![],
                vec![Elem::Bytes(ByteSet::of(b"eE")), Elem::Bytes(ByteSet::of(b"+-")), digit, Elem::Rule(digits)],
                vec![Elem::Bytes(ByteSet::of(b"eE")), digit, Elem::Rule(digits)],
            ],
        );
        let number = g.rule("number");
        g.set(number, vec![vec![Elem::Rule(integer), Elem::Rule(frac), Elem::Rule(exp)]]);

        // Любое значение.
        let any = g.rule("any");
        let members = g.rule("any_members");
        let member = g.rule("any_member");
        let elems = g.rule("any_elems");
        let comma = |ws: u32| vec![Elem::Rule(ws), Elem::Bytes(ByteSet::one(b',')), Elem::Rule(ws)];
        g.set(member, vec![vec![Elem::Rule(string), Elem::Rule(ws), Elem::Bytes(ByteSet::one(b':')), Elem::Rule(ws), Elem::Rule(any)]]);
        g.set(members, vec![vec![], [comma(ws), vec![Elem::Rule(member), Elem::Rule(members)]].concat()]);
        g.set(elems, vec![vec![], [comma(ws), vec![Elem::Rule(any), Elem::Rule(elems)]].concat()]);
        let lb = Elem::Bytes(ByteSet::one(b'{'));
        let rb = Elem::Bytes(ByteSet::one(b'}'));
        let ls = Elem::Bytes(ByteSet::one(b'['));
        let rs = Elem::Bytes(ByteSet::one(b']'));
        g.set(
            any,
            vec![
                vec![Elem::Rule(string)],
                vec![Elem::Rule(number)],
                lit(b"true"),
                lit(b"false"),
                lit(b"null"),
                vec![lb, Elem::Rule(ws), rb],
                vec![lb, Elem::Rule(ws), Elem::Rule(member), Elem::Rule(members), Elem::Rule(ws), rb],
                vec![ls, Elem::Rule(ws), rs],
                vec![ls, Elem::Rule(ws), Elem::Rule(any), Elem::Rule(elems), Elem::Rule(ws), rs],
            ],
        );
        Self { g, root, refs: HashMap::new(), ws, string, number, integer, any, sized: HashMap::new(), chr }
    }

    fn comma(&self) -> Vec<Elem> {
        vec![Elem::Rule(self.ws), Elem::Bytes(ByteSet::one(b',')), Elem::Rule(self.ws)]
    }

    fn resolve(&self, r: &str) -> Result<&'a Value, String> {
        let path = r.strip_prefix('#').ok_or_else(|| format!("$ref «{r}»: поддержаны только ссылки внутри схемы (#/…)"))?;
        self.root.pointer(path).ok_or_else(|| format!("$ref «{r}» не найден в схеме"))
    }

    /// Правило для значения по схеме.
    fn value(&mut self, s: &'a Value, depth: usize) -> Result<u32, String> {
        if depth > 64 {
            return Err("схема слишком глубокая".into());
        }
        let obj = match s {
            Value::Bool(true) | Value::Null => return Ok(self.any),
            Value::Bool(false) => return Err("схема false: значение невозможно".into()),
            Value::Object(o) => o,
            _ => return Err(format!("схема должна быть объектом: {s}")),
        };
        if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
            if let Some(&id) = self.refs.get(r) {
                return Ok(id);
            }
            let id = self.g.rule(r);
            self.refs.insert(r.to_string(), id);
            let target = self.resolve(r)?;
            let inner = self.value(target, depth + 1)?;
            self.g.set(id, vec![vec![Elem::Rule(inner)]]);
            return Ok(id);
        }
        let nullable = obj.get("nullable").and_then(Value::as_bool).unwrap_or(false);
        let id = self.value_inner(obj, depth)?;
        if nullable {
            let r = self.g.rule("nullable");
            self.g.set(r, vec![vec![Elem::Rule(id)], lit(b"null")]);
            return Ok(r);
        }
        Ok(id)
    }

    fn value_inner(&mut self, obj: &'a serde_json::Map<String, Value>, depth: usize) -> Result<u32, String> {
        if let Some(c) = obj.get("const") {
            return Ok(self.g.literal("const", &json_bytes(c)));
        }
        if let Some(Value::Array(vals)) = obj.get("enum") {
            if vals.is_empty() {
                return Err("пустой enum".into());
            }
            let r = self.g.rule("enum");
            self.g.set(r, vals.iter().map(|v| lit(&json_bytes(v))).collect());
            return Ok(r);
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(alts)) = obj.get(key) {
                let mut ids = Vec::new();
                for a in alts {
                    // Общие ключи (type, properties…) рядом с anyOf — внутрь каждой ветки.
                    let merged = merge_siblings(obj, key, a);
                    let v: &'a Value = Box::leak(Box::new(merged));
                    ids.push(self.value(v, depth + 1)?);
                }
                let r = self.g.rule(key);
                self.g.set(r, ids.into_iter().map(|i| vec![Elem::Rule(i)]).collect());
                return Ok(r);
            }
        }
        if let Some(Value::Array(parts)) = obj.get("allOf") {
            let mut merged = obj.clone();
            merged.remove("allOf");
            for p in parts {
                let p = match p.get("$ref").and_then(Value::as_str) {
                    Some(r) => self.resolve(r)?,
                    None => p,
                };
                merge_into(&mut merged, p);
            }
            let v: &'a Value = Box::leak(Box::new(Value::Object(merged)));
            return self.value(v, depth + 1);
        }
        let types: Vec<&str> = match obj.get("type") {
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            _ => {
                // Тип по ключам.
                if obj.contains_key("properties") || obj.contains_key("additionalProperties") || obj.contains_key("required") {
                    vec!["object"]
                } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
                    vec!["array"]
                } else {
                    return Ok(self.any);
                }
            }
        };
        let mut ids = Vec::new();
        for t in types {
            ids.push(match t {
                "string" => self.string_rule(obj),
                "integer" => self.integer_rule(obj),
                "number" => self.number,
                "boolean" => {
                    let r = self.g.rule("boolean");
                    self.g.set(r, vec![lit(b"true"), lit(b"false")]);
                    r
                }
                "null" => self.g.literal("null", b"null"),
                "object" => self.object_rule(obj, depth)?,
                "array" => self.array_rule(obj, depth)?,
                other => return Err(format!("неизвестный тип «{other}»")),
            });
        }
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
        let r = self.g.rule("types");
        self.g.set(r, ids.into_iter().map(|i| vec![Elem::Rule(i)]).collect());
        Ok(r)
    }

    fn string_rule(&mut self, obj: &serde_json::Map<String, Value>) -> u32 {
        let min = obj.get("minLength").and_then(Value::as_u64).unwrap_or(0) as usize;
        let max = obj.get("maxLength").and_then(Value::as_u64).map(|v| v as usize);
        if min == 0 && max.is_none() || min > COUNT_MAX || max.is_some_and(|m| m > COUNT_MAX) {
            return self.string;
        }
        let key = (min, max.unwrap_or(usize::MAX));
        if let Some(&r) = self.sized.get(&key) {
            return r;
        }
        let body = self.counted(self.chr, min, max, None);
        let q = Elem::Bytes(ByteSet::one(b'"'));
        let r = self.g.rule("string_sized");
        self.g.set(r, vec![vec![q, Elem::Rule(body), q]]);
        self.sized.insert(key, r);
        r
    }

    /// `item` от min до max раз (max = None — без верхнего предела), с
    /// разделителем `sep` между элементами.
    fn counted(&mut self, item: u32, min: usize, max: Option<usize>, sep: Option<&[Elem]>) -> u32 {
        let sep: Vec<Elem> = sep.map(<[Elem]>::to_vec).unwrap_or_default();
        let more = |next: u32| [sep.clone(), vec![Elem::Rule(item), Elem::Rule(next)]].concat();
        let start = self.g.rule("rep");
        match max {
            None => {
                // tail: «ещё элементы» без предела; сначала min обязательных.
                let tail = self.g.rule("rep_tail");
                self.g.set(tail, vec![vec![], more(tail)]);
                let mut after = tail;
                for _ in 1..min.max(1) {
                    let r = self.g.rule("rep_req");
                    self.g.set(r, vec![more(after)]);
                    after = r;
                }
                let mut alts = vec![vec![Elem::Rule(item), Elem::Rule(after)]];
                if min == 0 {
                    alts.push(vec![]);
                }
                self.g.set(start, alts);
            }
            Some(m) => {
                // after(k): уже k элементов (k = m..1).
                let mut after: Option<u32> = None;
                for k in (1..=m).rev() {
                    let r = self.g.rule("rep_k");
                    let mut alts = Vec::new();
                    if k >= min {
                        alts.push(vec![]);
                    }
                    if let Some(n) = after {
                        alts.push(more(n));
                    }
                    self.g.set(r, alts);
                    after = Some(r);
                }
                let mut alts = Vec::new();
                if min == 0 {
                    alts.push(vec![]);
                }
                if let Some(a1) = after {
                    alts.push(vec![Elem::Rule(item), Elem::Rule(a1)]);
                }
                self.g.set(start, alts);
            }
        }
        start
    }

    fn integer_rule(&mut self, obj: &serde_json::Map<String, Value>) -> u32 {
        let lo = obj.get("minimum").and_then(Value::as_i64);
        let hi = obj.get("maximum").and_then(Value::as_i64);
        let (Some(lo), Some(hi)) = (lo, hi) else { return self.integer };
        if hi < lo || hi - lo > INT_ENUM_MAX {
            return self.integer;
        }
        let r = self.g.rule("int_range");
        self.g.set(r, (lo..=hi).map(|v| lit(v.to_string().as_bytes())).collect());
        r
    }

    fn object_rule(&mut self, obj: &'a serde_json::Map<String, Value>, depth: usize) -> Result<u32, String> {
        let lb = Elem::Bytes(ByteSet::one(b'{'));
        let rb = Elem::Bytes(ByteSet::one(b'}'));
        let props = obj.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = obj.get("required").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let r = self.g.rule("object");
        let Some(props) = props.filter(|p| !p.is_empty()) else {
            // Без полей: словарь значений по additionalProperties.
            let val = match obj.get("additionalProperties") {
                Some(Value::Bool(false)) => {
                    self.g.set(r, vec![vec![lb, Elem::Rule(self.ws), rb]]);
                    return Ok(r);
                }
                Some(v @ Value::Object(_)) => self.value(v, depth + 1)?,
                _ => self.any,
            };
            let member = self.g.rule("map_member");
            self.g.set(
                member,
                vec![vec![Elem::Rule(self.string), Elem::Rule(self.ws), Elem::Bytes(ByteSet::one(b':')), Elem::Rule(self.ws), Elem::Rule(val)]],
            );
            let sep = self.comma();
            let body = self.counted(member, 0, None, Some(&sep));
            self.g.set(r, vec![vec![lb, Elem::Rule(self.ws), Elem::Rule(body), Elem::Rule(self.ws), rb]]);
            return Ok(r);
        };
        // Пары «ключ: значение» в порядке схемы.
        let mut kvs = Vec::new();
        for (name, ps) in props {
            let v = self.value(ps, depth + 1)?;
            let mut seq = lit(&json_bytes(&Value::String(name.clone())));
            seq.extend([Elem::Rule(self.ws), Elem::Bytes(ByteSet::one(b':')), Elem::Rule(self.ws), Elem::Rule(v)]);
            let kv = self.g.rule(&format!("kv_{name}"));
            self.g.set(kv, vec![seq]);
            kvs.push((kv, required.contains(&name.as_str())));
        }
        // rest(i, comma): поля с i-го; comma — нужна ли запятая перед следующим.
        let n = kvs.len();
        let mut rest = vec![[0u32; 2]; n + 1];
        for c in 0..2 {
            let e = self.g.rule("obj_end");
            self.g.set(e, vec![vec![]]);
            rest[n][c] = e;
        }
        for i in (0..n).rev() {
            for c in 0..2 {
                let (kv, req) = kvs[i];
                let mut take = if c == 1 { self.comma() } else { vec![] };
                take.extend([Elem::Rule(kv), Elem::Rule(rest[i + 1][1])]);
                let mut alts = vec![take];
                if !req {
                    alts.push(vec![Elem::Rule(rest[i + 1][c])]);
                }
                let id = self.g.rule("obj_rest");
                self.g.set(id, alts);
                rest[i][c] = id;
            }
        }
        self.g.set(r, vec![vec![lb, Elem::Rule(self.ws), Elem::Rule(rest[0][0]), Elem::Rule(self.ws), rb]]);
        Ok(r)
    }

    fn array_rule(&mut self, obj: &'a serde_json::Map<String, Value>, depth: usize) -> Result<u32, String> {
        let ls = Elem::Bytes(ByteSet::one(b'['));
        let rs = Elem::Bytes(ByteSet::one(b']'));
        let r = self.g.rule("array");
        let min = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let max = obj.get("maxItems").and_then(Value::as_u64).map(|v| v as usize);
        let (min, max) = if min > COUNT_MAX || max.is_some_and(|m| m > COUNT_MAX) { (0, None) } else { (min, max) };
        if let Some(Value::Array(prefix)) = obj.get("prefixItems") {
            // Кортеж фиксированной длины.
            let mut seq = vec![ls, Elem::Rule(self.ws)];
            for (i, p) in prefix.iter().enumerate() {
                if i > 0 {
                    seq.extend(self.comma());
                }
                seq.push(Elem::Rule(self.value(p, depth + 1)?));
            }
            seq.extend([Elem::Rule(self.ws), rs]);
            self.g.set(r, vec![seq]);
            return Ok(r);
        }
        let item = match obj.get("items") {
            Some(v @ Value::Object(_)) => self.value(v, depth + 1)?,
            _ => self.any,
        };
        let sep = self.comma();
        let body = self.counted(item, min, max, Some(&sep));
        self.g.set(r, vec![vec![ls, Elem::Rule(self.ws), Elem::Rule(body), Elem::Rule(self.ws), rs]]);
        Ok(r)
    }
}

/// Значение как JSON-текст (литерал enum/const/имени поля).
fn json_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

fn merge_siblings(obj: &serde_json::Map<String, Value>, skip: &str, branch: &Value) -> Value {
    let mut m: serde_json::Map<String, Value> = obj.iter().filter(|(k, _)| k.as_str() != skip).map(|(k, v)| (k.clone(), v.clone())).collect();
    if m.is_empty() {
        return branch.clone();
    }
    merge_into(&mut m, branch);
    Value::Object(m)
}

/// Слить схему `p` в `m`: поля и required объединяются, прочее — из `p`.
fn merge_into(m: &mut serde_json::Map<String, Value>, p: &Value) {
    let Some(po) = p.as_object() else { return };
    for (k, v) in po {
        match (k.as_str(), m.get_mut(k)) {
            ("properties", Some(Value::Object(dst))) => {
                if let Some(src) = v.as_object() {
                    for (pk, pv) in src {
                        dst.insert(pk.clone(), pv.clone());
                    }
                }
            }
            ("required", Some(Value::Array(dst))) => {
                if let Some(src) = v.as_array() {
                    for x in src {
                        if !dst.contains(x) {
                            dst.push(x.clone());
                        }
                    }
                }
            }
            _ => {
                m.insert(k.clone(), v.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cfg::ParseState;
    use super::*;
    use serde_json::json;

    fn accepts(schema: &Value, text: &str) -> bool {
        let g = json_schema_grammar(schema).unwrap();
        let st = ParseState::new(&g).advance_bytes(&g, text.as_bytes());
        st.accepting()
    }

    #[test]
    fn any_json() {
        let s = json!({});
        for ok in ["{}", "[]", "1", "-0.5e3", "\"a\\nb\"", "{\"a\": [1, {\"b\": null}], \"c\": true}", " \n[ 1 ,2 ]", "\"Сәлем, әлем\""] {
            assert!(accepts(&s, ok), "{ok}");
        }
        for bad in ["{", "[1,]", "01", "\"a", "{\"a\" 1}", "tru", "{'a':1}"] {
            assert!(!accepts(&s, bad), "{bad}");
        }
    }

    #[test]
    fn object_required_optional_enum() {
        let s = json!({
            "type": "object",
            "properties": {
                "word": {"type": "string", "minLength": 1},
                "level": {"enum": ["A1", "A2", "B1", "B2", "C1", "C2"]},
                "ipa": {"type": "string"},
                "examples": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                "score": {"type": "integer", "minimum": 0, "maximum": 9}
            },
            "required": ["word", "level", "examples"]
        });
        assert!(accepts(&s, r#"{"word":"cat","level":"A1","examples":["a cat"]}"#));
        assert!(accepts(&s, r#"{"word": "cat", "level": "A1", "ipa": "kæt", "examples": ["a", "b", "c"], "score": 7}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"D1","examples":["a"]}"#));
        assert!(!accepts(&s, r#"{"word":"","level":"A1","examples":["a"]}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"A1","examples":[]}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"A1","examples":["a","b","c","d"]}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"A1"}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"A1","examples":["a"],"score":10}"#));
        assert!(!accepts(&s, r#"{"word":"cat","level":"A1","examples":["a"],"extra":1}"#));
    }

    #[test]
    fn refs_anyof_nullable() {
        let s = json!({
            "$defs": {"node": {"type": "object", "properties": {"v": {"type": "number"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}}, "required": ["v"]}},
            "anyOf": [{"$ref": "#/$defs/node"}, {"type": ["string", "null"]}]
        });
        assert!(accepts(&s, r#"{"v":1,"kids":[{"v":2,"kids":[]},{"v":3}]}"#));
        assert!(accepts(&s, "null"));
        assert!(accepts(&s, "\"x\""));
        assert!(!accepts(&s, "true"));
    }
}
