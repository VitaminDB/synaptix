//! Чтение ONNX (`ModelProto`, protobuf) без сторонних библиотек: граф (узлы,
//! атрибуты, входы/выходы) и веса (инициализаторы и тензоры `Constant`).
//!
//! Нужен, чтобы переносить модели, опубликованные только в ONNX, на свои ядра:
//! веса уходят в `.syn` ([`write_safetensors`]), граф читается дампом
//! ([`Model::dump`], `synaptix inspect model.onnx`). Внешние данные
//! (`external_data`) читаются из файла рядом с моделью.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::{IoError, Result};

fn err(msg: impl Into<String>) -> IoError {
    IoError::Other(msg.into())
}

/// Поле protobuf: номер и значение.
enum Val<'a> {
    Varint(u64),
    I64(u64),
    Bytes(&'a [u8]),
    I32(u32),
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self.b.get(self.p).ok_or_else(|| err("onnx: обрыв varint"))?;
            self.p += 1;
            v |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err(err("onnx: длинный varint"));
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| err("onnx: обрыв поля"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn next(&mut self) -> Result<Option<(u32, Val<'a>)>> {
        if self.p >= self.b.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        let v = match key & 7 {
            0 => Val::Varint(self.varint()?),
            1 => Val::I64(u64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            2 => {
                let n = self.varint()? as usize;
                Val::Bytes(self.take(n)?)
            }
            5 => Val::I32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            w => return Err(err(format!("onnx: тип поля {w}"))),
        };
        Ok(Some((field, v)))
    }
}

fn s(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// repeated int64 (упакованные или по одному).
fn push_ints(v: &Val, out: &mut Vec<i64>) -> Result<()> {
    match v {
        Val::Varint(x) => out.push(*x as i64),
        Val::Bytes(b) => {
            let mut r = Reader::new(b);
            while r.p < b.len() {
                out.push(r.varint()? as i64);
            }
        }
        _ => {}
    }
    Ok(())
}

fn push_floats(v: &Val, out: &mut Vec<f32>) {
    match v {
        Val::I32(x) => out.push(f32::from_bits(*x)),
        Val::Bytes(b) => out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        _ => {}
    }
}

/// Тип элемента ONNX (`TensorProto.DataType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    U8,
    I8,
    U16,
    I16,
    I32,
    I64,
    Bool,
    F16,
    F64,
    U32,
    U64,
    BF16,
    Other(i32),
}

impl DataType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::F32,
            2 => Self::U8,
            3 => Self::I8,
            4 => Self::U16,
            5 => Self::I16,
            6 => Self::I32,
            7 => Self::I64,
            9 => Self::Bool,
            10 => Self::F16,
            11 => Self::F64,
            12 => Self::U32,
            13 => Self::U64,
            16 => Self::BF16,
            o => Self::Other(o),
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I64 | Self::F64 | Self::U64 => 8,
            Self::Other(_) => 0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
            Self::F16 => "f16",
            Self::F64 => "f64",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::BF16 => "bf16",
            Self::Other(_) => "?",
        }
    }
}

/// Тензор ONNX: форма, тип и сырые байты (little-endian).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DataType,
    pub data: Vec<u8>,
}

impl OnnxTensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Значения как f32 (для F32/F16/BF16/I64/I32 — с приведением).
    pub fn to_f32(&self) -> Vec<f32> {
        let d = &self.data;
        match self.dtype {
            DataType::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DataType::F16 => d.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DataType::BF16 => d.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            DataType::I64 => d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            DataType::I32 => d.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            DataType::F64 => d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            _ => Vec::new(),
        }
    }

    pub fn to_i64(&self) -> Vec<i64> {
        let d = &self.data;
        match self.dtype {
            DataType::I64 => d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            DataType::I32 => d.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            _ => self.to_f32().into_iter().map(|v| v as i64).collect(),
        }
    }

    fn parse(b: &[u8], dir: Option<&Path>) -> Result<Self> {
        let mut r = Reader::new(b);
        let mut t = OnnxTensor { name: String::new(), dims: Vec::new(), dtype: DataType::F32, data: Vec::new() };
        let mut dims = Vec::new();
        let (mut floats, mut ints64, mut ints32, mut doubles) = (Vec::new(), Vec::new(), Vec::new(), Vec::<f64>::new());
        let mut ext: BTreeMap<String, String> = BTreeMap::new();
        while let Some((f, v)) = r.next()? {
            match (f, &v) {
                (1, _) => push_ints(&v, &mut dims)?,
                (2, Val::Varint(x)) => t.dtype = DataType::from_i32(*x as i32),
                (4, _) => push_floats(&v, &mut floats),
                (5, _) => push_ints(&v, &mut ints32)?,
                (7, _) => push_ints(&v, &mut ints64)?,
                (8, Val::Bytes(n)) => t.name = s(n),
                (9, Val::Bytes(raw)) => t.data = raw.to_vec(),
                (10, Val::Bytes(bb)) => doubles.extend(bb.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))),
                (10, Val::I64(x)) => doubles.push(f64::from_bits(*x)),
                (13, Val::Bytes(kv)) => {
                    let (mut k, mut val) = (String::new(), String::new());
                    let mut rr = Reader::new(kv);
                    while let Some((ff, vv)) = rr.next()? {
                        if let Val::Bytes(x) = vv {
                            if ff == 1 {
                                k = s(x)
                            } else if ff == 2 {
                                val = s(x)
                            }
                        }
                    }
                    ext.insert(k, val);
                }
                _ => {}
            }
        }
        t.dims = dims.into_iter().map(|d| d.max(0) as usize).collect();
        if t.data.is_empty() {
            if let Some(loc) = ext.get("location") {
                let dir = dir.ok_or_else(|| err("onnx: внешние данные без каталога"))?;
                let bytes = std::fs::read(dir.join(loc)).map_err(|e| err(format!("onnx: {loc}: {e}")))?;
                let off: usize = ext.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
                let len: usize = ext.get("length").and_then(|v| v.parse().ok()).unwrap_or(bytes.len() - off);
                t.data = bytes[off..off + len].to_vec();
            } else {
                t.data = match t.dtype {
                    DataType::F32 => floats.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    DataType::I64 => ints64.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    DataType::F64 => doubles.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    DataType::I32 => ints32.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect(),
                    DataType::U8 | DataType::I8 | DataType::Bool => ints32.iter().map(|v| *v as u8).collect(),
                    DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => ints32.iter().flat_map(|v| (*v as u16).to_le_bytes()).collect(),
                    _ => Vec::new(),
                };
            }
        }
        Ok(t)
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xFF << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Значение атрибута узла.
#[derive(Debug, Clone)]
pub enum Attr {
    Float(f32),
    Int(i64),
    Str(String),
    Tensor(OnnxTensor),
    Graph(Box<Graph>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strs(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: BTreeMap<String, Attr>,
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub elem: Option<DataType>,
    /// Измерение: число или имя (`dim_param`).
    pub shape: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: Vec<OnnxTensor>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

fn parse_value_info(b: &[u8]) -> Result<ValueInfo> {
    let mut vi = ValueInfo { name: String::new(), elem: None, shape: Vec::new() };
    let mut r = Reader::new(b);
    while let Some((f, v)) = r.next()? {
        match (f, v) {
            (1, Val::Bytes(n)) => vi.name = s(n),
            (2, Val::Bytes(tp)) => {
                // TypeProto.tensor_type (1) → elem_type (1), shape (2) → dim (1) → value (1) | param (2)
                let mut r2 = Reader::new(tp);
                while let Some((f2, v2)) = r2.next()? {
                    if let (1, Val::Bytes(tt)) = (f2, v2) {
                        let mut r3 = Reader::new(tt);
                        while let Some((f3, v3)) = r3.next()? {
                            match (f3, v3) {
                                (1, Val::Varint(e)) => vi.elem = Some(DataType::from_i32(e as i32)),
                                (2, Val::Bytes(sh)) => {
                                    let mut r4 = Reader::new(sh);
                                    while let Some((f4, v4)) = r4.next()? {
                                        if let (1, Val::Bytes(dim)) = (f4, v4) {
                                            let mut r5 = Reader::new(dim);
                                            let mut d = "?".to_string();
                                            while let Some((f5, v5)) = r5.next()? {
                                                match (f5, v5) {
                                                    (1, Val::Varint(x)) => d = x.to_string(),
                                                    (2, Val::Bytes(p)) => d = s(p),
                                                    _ => {}
                                                }
                                            }
                                            vi.shape.push(d);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(vi)
}

fn parse_attr(b: &[u8], dir: Option<&Path>) -> Result<(String, Attr)> {
    let mut r = Reader::new(b);
    let mut name = String::new();
    let mut ty = 0u64;
    let (mut f, mut i, mut st, mut t, mut g) = (None, None, None, None, None);
    let (mut fs, mut is, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    while let Some((fl, v)) = r.next()? {
        match (fl, &v) {
            (1, Val::Bytes(n)) => name = s(n),
            (20, Val::Varint(x)) => ty = *x,
            (2, Val::I32(x)) => f = Some(f32::from_bits(*x)),
            (3, Val::Varint(x)) => i = Some(*x as i64),
            (4, Val::Bytes(x)) => st = Some(s(x)),
            (5, Val::Bytes(x)) => t = Some(OnnxTensor::parse(x, dir)?),
            (6, Val::Bytes(x)) => g = Some(parse_graph(x, dir)?),
            (7, _) => push_floats(&v, &mut fs),
            (8, _) => push_ints(&v, &mut is)?,
            (9, Val::Bytes(x)) => ss.push(s(x)),
            _ => {}
        }
    }
    let a = match ty {
        1 => Attr::Float(f.unwrap_or(0.0)),
        2 => Attr::Int(i.unwrap_or(0)),
        3 => Attr::Str(st.unwrap_or_default()),
        4 => Attr::Tensor(t.ok_or_else(|| err("onnx: атрибут-тензор пуст"))?),
        5 => Attr::Graph(Box::new(g.unwrap_or_default())),
        6 => Attr::Floats(fs),
        7 => Attr::Ints(is),
        8 => Attr::Strs(ss),
        _ => match (f, i, st, t) {
            (Some(x), ..) => Attr::Float(x),
            (_, Some(x), ..) => Attr::Int(x),
            (_, _, Some(x), _) => Attr::Str(x),
            (_, _, _, Some(x)) => Attr::Tensor(x),
            _ => Attr::Ints(is),
        },
    };
    Ok((name, a))
}

fn parse_node(b: &[u8], dir: Option<&Path>) -> Result<Node> {
    let mut n = Node { name: String::new(), op: String::new(), domain: String::new(), inputs: Vec::new(), outputs: Vec::new(), attrs: BTreeMap::new() };
    let mut r = Reader::new(b);
    while let Some((f, v)) = r.next()? {
        match (f, v) {
            (1, Val::Bytes(x)) => n.inputs.push(s(x)),
            (2, Val::Bytes(x)) => n.outputs.push(s(x)),
            (3, Val::Bytes(x)) => n.name = s(x),
            (4, Val::Bytes(x)) => n.op = s(x),
            (7, Val::Bytes(x)) => n.domain = s(x),
            (5, Val::Bytes(x)) => {
                let (k, a) = parse_attr(x, dir)?;
                n.attrs.insert(k, a);
            }
            _ => {}
        }
    }
    Ok(n)
}

fn parse_graph(b: &[u8], dir: Option<&Path>) -> Result<Graph> {
    let mut g = Graph::default();
    let mut r = Reader::new(b);
    while let Some((f, v)) = r.next()? {
        match (f, v) {
            (1, Val::Bytes(x)) => g.nodes.push(parse_node(x, dir)?),
            (2, Val::Bytes(x)) => g.name = s(x),
            (5, Val::Bytes(x)) => g.initializers.push(OnnxTensor::parse(x, dir)?),
            (11, Val::Bytes(x)) => g.inputs.push(parse_value_info(x)?),
            (12, Val::Bytes(x)) => g.outputs.push(parse_value_info(x)?),
            _ => {}
        }
    }
    Ok(g)
}

/// Модель ONNX.
#[derive(Debug, Clone)]
pub struct Model {
    pub ir_version: i64,
    pub producer: String,
    pub opsets: Vec<(String, i64)>,
    pub graph: Graph,
    pub path: PathBuf,
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        let dir = path.parent();
        let mut m = Model { ir_version: 0, producer: String::new(), opsets: Vec::new(), graph: Graph::default(), path: path.to_path_buf() };
        let mut r = Reader::new(&bytes);
        while let Some((f, v)) = r.next()? {
            match (f, v) {
                (1, Val::Varint(x)) => m.ir_version = x as i64,
                (2, Val::Bytes(x)) => m.producer = s(x),
                (7, Val::Bytes(x)) => m.graph = parse_graph(x, dir)?,
                (8, Val::Bytes(x)) => {
                    let (mut d, mut ver) = (String::new(), 0i64);
                    let mut rr = Reader::new(x);
                    while let Some((ff, vv)) = rr.next()? {
                        match (ff, vv) {
                            (1, Val::Bytes(dd)) => d = s(dd),
                            (2, Val::Varint(vv)) => ver = vv as i64,
                            _ => {}
                        }
                    }
                    m.opsets.push((d, ver));
                }
                _ => {}
            }
        }
        Ok(m)
    }

    /// Все веса: инициализаторы и тензоры узлов `Constant` (имя — выход узла).
    /// Вещественные — все (и скаляры: PReLU, масштабы), целочисленные — от
    /// `min_numel` элементов (мелкие — служебные константы форм графа).
    pub fn weights(&self, min_numel: usize) -> Vec<OnnxTensor> {
        let keep = |t: &OnnxTensor| match t.dtype {
            DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64 => true,
            _ => t.numel() >= min_numel,
        };
        let mut out: Vec<OnnxTensor> = self.graph.initializers.iter().filter(|t| keep(t)).cloned().collect();
        for n in &self.graph.nodes {
            if n.op == "Constant" {
                if let Some(Attr::Tensor(t)) = n.attrs.get("value") {
                    if keep(t) && !n.outputs.is_empty() && t.numel() > 1 {
                        let mut t = t.clone();
                        t.name = n.outputs.first().cloned().unwrap_or_default();
                        out.push(t);
                    }
                }
            }
        }
        out
    }

    /// Текстовый дамп графа: входы/выходы, узлы с атрибутами, формы весов.
    pub fn dump(&self, small_values: usize) -> String {
        self.dump_filtered(small_values, None)
    }

    /// Как [`Self::dump`], только узлы с именем, содержащим `filter`.
    pub fn dump_filtered(&self, small_values: usize, filter: Option<&str>) -> String {
        use std::fmt::Write;
        let g = &self.graph;
        let mut o = String::new();
        let _ = writeln!(o, "# {} (ir {}, {}; opset {:?})", self.path.display(), self.ir_version, self.producer, self.opsets);
        for vi in &g.inputs {
            let _ = writeln!(o, "input  {} {:?} [{}]", vi.name, vi.elem.map(|e| e.name()), vi.shape.join(","));
        }
        for vi in &g.outputs {
            let _ = writeln!(o, "output {} {:?} [{}]", vi.name, vi.elem.map(|e| e.name()), vi.shape.join(","));
        }
        let mut shapes: BTreeMap<&str, (Vec<usize>, DataType, Option<String>)> = BTreeMap::new();
        let fmt_small = |t: &OnnxTensor| -> Option<String> {
            (t.numel() <= small_values).then(|| {
                if matches!(t.dtype, DataType::I64 | DataType::I32) {
                    format!("{:?}", t.to_i64())
                } else {
                    format!("{:?}", t.to_f32())
                }
            })
        };
        for t in &g.initializers {
            shapes.insert(&t.name, (t.dims.clone(), t.dtype, fmt_small(t)));
        }
        // Константы подставляются в места использования, узлы Constant не печатаются.
        for n in &g.nodes {
            if n.op == "Constant" {
                if let (Some(Attr::Tensor(t)), Some(o)) = (n.attrs.get("value"), n.outputs.first()) {
                    shapes.insert(o.as_str(), (t.dims.clone(), t.dtype, fmt_small(t)));
                }
            }
        }
        let short = |x: &str| -> String {
            // `/a/b/c/Op_output_0` → `c/Op`
            let x = x.strip_suffix("_output_0").unwrap_or(x);
            let parts: Vec<&str> = x.trim_start_matches('/').split('/').collect();
            let k = parts.len().saturating_sub(2);
            parts[k..].join("/")
        };
        for n in &g.nodes {
            if n.op == "Constant" || filter.is_some_and(|f| !n.name.contains(f)) {
                continue;
            }
            let outs: Vec<String> = n.outputs.iter().map(|o| short(o)).collect();
            let mut line = format!("{} = {}(", outs.join(", "), n.op);
            let ins: Vec<String> = n
                .inputs
                .iter()
                .map(|i| match shapes.get(i.as_str()) {
                    Some((d, t, Some(v))) if i.starts_with('/') || i.starts_with("onnx::") => format!("{}{:?}={v}", t.name(), d),
                    Some((d, t, Some(v))) => format!("{i}<{}{:?}={v}>", t.name(), d),
                    Some((d, t, None)) => format!("{i}<{}{:?}>", t.name(), d),
                    None => short(i),
                })
                .collect();
            line.push_str(&ins.join(", "));
            line.push(')');
            for (k, a) in &n.attrs {
                let v = match a {
                    Attr::Float(x) => format!("{x}"),
                    Attr::Int(x) => format!("{x}"),
                    Attr::Str(x) => format!("{x:?}"),
                    Attr::Tensor(t) => match fmt_small(t) {
                        Some(v) => format!("{}{:?}={v}", t.dtype.name(), t.dims),
                        None => format!("{}{:?}", t.dtype.name(), t.dims),
                    },
                    Attr::Graph(g) => format!("<граф {} узлов>", g.nodes.len()),
                    Attr::Floats(x) => format!("{x:?}"),
                    Attr::Ints(x) => format!("{x:?}"),
                    Attr::Strs(x) => format!("{x:?}"),
                };
                let _ = write!(line, " {k}={v}");
            }
            let _ = writeln!(o, "{line}");
        }
        o
    }
}

impl Model {
    /// Веса с понятными именами: безымянные инициализаторы `onnx::*` (так
    /// PyTorch-экспорт называет транспонированные/свёрнутые веса) получают
    /// имя узла-потребителя: `/decoder/embed/Conv` + вход 1 → `decoder.embed.Conv.w1`.
    pub fn named_weights(&self, min_numel: usize) -> Vec<OnnxTensor> {
        let mut user: BTreeMap<&str, (String, usize)> = BTreeMap::new();
        for n in &self.graph.nodes {
            for (i, inp) in n.inputs.iter().enumerate() {
                user.entry(inp.as_str()).or_insert_with(|| (n.name.trim_start_matches('/').replace('/', "."), i));
            }
        }
        self.weights(min_numel)
            .into_iter()
            .map(|mut t| {
                if t.name.starts_with("onnx::") || t.name.starts_with('/') {
                    if let Some((node, slot)) = user.get(t.name.as_str()) {
                        t.name = format!("{node}.w{slot}");
                    }
                }
                t
            })
            .collect()
    }
}

/// Веса в safetensors (F32/F16/BF16/I64/I32 как есть). Имена — как в ONNX.
pub fn write_safetensors(tensors: &[OnnxTensor], out: &Path) -> Result<()> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let mut views = Vec::with_capacity(tensors.len());
    for t in tensors {
        let dt = match t.dtype {
            DataType::F32 => Dtype::F32,
            DataType::F16 => Dtype::F16,
            DataType::BF16 => Dtype::BF16,
            DataType::I64 => Dtype::I64,
            DataType::I32 => Dtype::I32,
            DataType::U8 => Dtype::U8,
            DataType::I8 => Dtype::I8,
            DataType::Bool => Dtype::BOOL,
            DataType::F64 => Dtype::F64,
            o => return Err(err(format!("onnx: {} типа {} в safetensors не пишется", t.name, o.name()))),
        };
        let v = TensorView::new(dt, t.dims.clone(), &t.data).map_err(|e| err(format!("{}: {e}", t.name)))?;
        views.push((t.name.clone(), v));
    }
    serialize_to_file(views, None, out).map_err(|e| err(format!("{}: {e}", out.display())))
}
