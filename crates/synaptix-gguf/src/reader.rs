use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{GgufError, Result};
use crate::ggml::GgmlType;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml: не больше 4 измерений у тензора.
const MAX_DIMS: usize = 4;
/// Потолок `general.alignment` — защита от мусора в битом файле.
const MAX_ALIGNMENT: u64 = 1 << 16;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Array),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::I8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::I16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::U64(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::F64(v) => v.len(),
            Array::Bool(v) => v.len(),
            Array::String(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        Some(match self {
            Array::U8(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I8(v) => v.iter().map(|x| *x as i64).collect(),
            Array::U16(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I16(v) => v.iter().map(|x| *x as i64).collect(),
            Array::U32(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I32(v) => v.iter().map(|x| *x as i64).collect(),
            Array::U64(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I64(v) => v.clone(),
            _ => return None,
        })
    }
    pub fn as_str_slice(&self) -> Option<&[String]> {
        match self {
            Array::String(v) => Some(v),
            _ => None,
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64,
            Value::I8(v) => *v as u64,
            Value::U16(v) => *v as u64,
            Value::I16(v) => *v as u64,
            Value::U32(v) => *v as u64,
            Value::I32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I64(v) => *v as u64,
            Value::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            Value::F32(v) => *v,
            Value::F64(v) => *v as f32,
            other => other.as_u64()? as f32,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            other => other.as_u64().map(|v| v != 0),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,

    pub dims: Vec<u64>,
    pub ty: GgmlType,

    pub offset: u64,
}

impl TensorInfo {
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn byte_len(&self) -> usize {
        self.ty.bytes_for(self.elem_count())
    }

    pub fn hf_shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|d| *d as usize).collect()
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        // Длины берутся из файла: у битого `o + n` переполнялось, проверка
        // проходила, и срез ниже паниковал.
        if self.o.checked_add(n).is_none_or(|end| end > self.b.len()) {
            return Err(GgufError::Truncated {
                at: self.o,
                need: n,
                have: self.b.len().saturating_sub(self.o),
            });
        }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.o)
    }
    fn len_u64(&mut self) -> Result<usize> {
        let at = self.o;
        let n = self.u64()?;
        usize::try_from(n).map_err(|_| GgufError::Truncated {
            at,
            need: usize::MAX,
            have: self.remaining(),
        })
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len_u64()?;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => Value::Array(self.array()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(GgufError::BadValueType(other)),
        })
    }
    fn array(&mut self) -> Result<Array> {
        let et = self.u32()?;
        let n = self.len_u64()?;
        let at = self.o;
        let remaining = self.remaining();
        let too_long = move || GgufError::Truncated { at, need: usize::MAX, have: remaining };
        macro_rules! fixed {
            ($v:ident, $w:expr, $conv:expr) => {{
                let raw = self.take(n.checked_mul($w).ok_or_else(too_long)?)?;
                Array::$v(raw.chunks_exact($w).map($conv).collect())
            }};
        }
        Ok(match et {
            0 => Array::U8(self.take(n)?.to_vec()),
            1 => Array::I8(self.take(n)?.iter().map(|b| *b as i8).collect()),
            2 => fixed!(U16, 2, |c| u16::from_le_bytes(c.try_into().unwrap())),
            3 => fixed!(I16, 2, |c| i16::from_le_bytes(c.try_into().unwrap())),
            4 => fixed!(U32, 4, |c| u32::from_le_bytes(c.try_into().unwrap())),
            5 => fixed!(I32, 4, |c| i32::from_le_bytes(c.try_into().unwrap())),
            6 => fixed!(F32, 4, |c| f32::from_le_bytes(c.try_into().unwrap())),
            7 => Array::Bool(self.take(n)?.iter().map(|b| *b != 0).collect()),
            8 => {
                // Ёмкость — не больше, чем строк может поместиться в остаток
                // файла (у каждой 8 байт длины): `n` из битого файла иначе
                // выделял бы терабайты и ронял процесс.
                let mut v = Vec::with_capacity(n.min(self.remaining() / 8));
                for _ in 0..n {
                    v.push(self.string()?);
                }
                Array::String(v)
            }
            10 => fixed!(U64, 8, |c| u64::from_le_bytes(c.try_into().unwrap())),
            11 => fixed!(I64, 8, |c| i64::from_le_bytes(c.try_into().unwrap())),
            12 => fixed!(F64, 8, |c| f64::from_le_bytes(c.try_into().unwrap())),
            other => return Err(GgufError::BadValueType(other)),
        })
    }
}

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mmap: memmap2::Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        Self::from_mmap(path, mmap)
    }

    fn from_mmap(path: PathBuf, mmap: memmap2::Mmap) -> Result<Self> {
        let mut c = Cursor { b: &mmap, o: 0 };
        let magic: [u8; 4] = c.take(4)?.try_into().unwrap();
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic { path, got: magic });
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::BadVersion(version));
        }
        let n_tensors = c.len_u64()?;
        let n_kv = c.len_u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty)?;
            metadata.insert(key, val);
        }

        // Описание тензора — минимум 24 байта (имя, nd, тип, оффсет).
        let mut tensors = Vec::with_capacity(n_tensors.min(c.remaining() / 24));
        let mut by_name = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            // ggml: от 1 до 4 измерений. Ноль ронял `hf_shape()[0]` в
            // конвертере, огромное число — аллокацию.
            if nd == 0 || nd > MAX_DIMS {
                return Err(GgufError::BadTensor {
                    name,
                    reason: format!("{nd} измерений (ожидалось 1..={MAX_DIMS})"),
                });
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let ty_id = c.u32()?;
            let ty = GgmlType::from_u32(ty_id).ok_or(GgufError::BadTensorType(ty_id))?;
            let offset = c.u64()?;
            by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);
        if !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
            return Err(GgufError::WrongKeyType {
                key: "general.alignment".into(),
                expected: "степень двойки ≤ 65536",
                actual: "другое число",
            });
        }
        let data_start = (c.o as u64).div_ceil(alignment) * alignment;
        // Размеры и оффсеты тензоров — из файла: проверяем здесь с
        // переполнением, чтобы `elem_count`/`byte_len`/`tensor_bytes` дальше
        // считали по заведомо разумным числам.
        let data_len = (mmap.len() as u64).saturating_sub(data_start);
        for t in &tensors {
            let bad = |reason: String| GgufError::BadTensor { name: t.name.clone(), reason };
            let elems = t
                .dims
                .iter()
                .try_fold(1u64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(|| bad("число элементов переполняет u64".into()))?;
            // Самый плотный тип — меньше 2 бит на элемент.
            if elems > data_len.saturating_mul(8) {
                return Err(bad(format!("{elems} элементов не помещаются в файл")));
            }
            let len = t.ty.bytes_for(elems as usize) as u64;
            if t.offset.checked_add(len).is_none_or(|end| end > data_len) {
                return Err(GgufError::Truncated {
                    at: data_start.saturating_add(t.offset) as usize,
                    need: len as usize,
                    have: data_len.saturating_sub(t.offset) as usize,
                });
            }
        }

        Ok(Self {
            path,
            mmap,
            version,
            metadata,
            tensors,
            by_name,
            data_start: data_start as usize,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|i| &self.tensors[*i])
    }

    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let start = self.data_start.saturating_add(t.offset as usize);
        let len = t.byte_len();
        if start.checked_add(len).is_none_or(|end| end > self.mmap.len()) {
            return Err(GgufError::Truncated {
                at: start,
                need: len,
                have: self.mmap.len().saturating_sub(start),
            });
        }
        Ok(&self.mmap[start..start + len])
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn require(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.into()))
    }

    pub fn u64_of(&self, key: &str) -> Result<u64> {
        let v = self.require(key)?;
        v.as_u64().ok_or_else(|| GgufError::WrongKeyType {
            key: key.into(),
            expected: "integer",
            actual: v.type_name(),
        })
    }

    pub fn usize_of(&self, key: &str) -> Result<usize> {
        Ok(self.u64_of(key)? as usize)
    }

    pub fn f32_of(&self, key: &str) -> Result<f32> {
        let v = self.require(key)?;
        v.as_f32().ok_or_else(|| GgufError::WrongKeyType {
            key: key.into(),
            expected: "float",
            actual: v.type_name(),
        })
    }

    pub fn str_of(&self, key: &str) -> Result<&str> {
        let v = self.require(key)?;
        v.as_str().ok_or_else(|| GgufError::WrongKeyType {
            key: key.into(),
            expected: "string",
            actual: v.type_name(),
        })
    }

    pub fn opt_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    }

    pub fn opt_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn opt_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn architecture(&self) -> Result<&str> {
        self.str_of("general.architecture")
    }

    pub fn arch_key(&self, suffix: &str) -> Result<String> {
        Ok(format!("{}.{suffix}", self.architecture()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn synth_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());

        let kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&8u32.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        kv_str(&mut b, "general.architecture", "testarch");

        let k = "testarch.block_count";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());

        let k = "testarch.sections";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&5u32.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        for v in [11i32, 11, 10] {
            b.extend_from_slice(&v.to_le_bytes());
        }

        let n = "blk.0.attn_q.weight";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());

        while b.len() % 32 != 0 {
            b.push(0);
        }
        for i in 0..8 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        b
    }

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parses_header_metadata_and_tensor() {
        let f = write_tmp(&synth_gguf());
        let g = GgufFile::open(f.path()).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture().unwrap(), "testarch");
        assert_eq!(g.usize_of("testarch.block_count").unwrap(), 7);
        let sections = g
            .get("testarch.sections")
            .unwrap()
            .as_array()
            .unwrap()
            .as_i64_vec()
            .unwrap();
        assert_eq!(sections, vec![11, 11, 10]);

        let t = g.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(t.dims, vec![4, 2]);

        assert_eq!(t.hf_shape(), vec![2, 4]);
        assert_eq!(t.byte_len(), 32);
        let raw = g.tensor_bytes(t).unwrap();
        assert_eq!(f32::from_le_bytes(raw[4..8].try_into().unwrap()), 1.0);
    }

    #[test]
    fn rejects_non_gguf() {
        let f = write_tmp(b"NOTGGUF___");
        let err = GgufFile::open(f.path()).err().unwrap();
        assert!(matches!(err, GgufError::BadMagic { .. }));
    }

    #[test]
    fn rejects_truncated_tail() {
        let mut bytes = synth_gguf();
        bytes.truncate(bytes.len() - 16);
        let f = write_tmp(&bytes);
        // Обрезанные данные тензора видны уже при открытии.
        assert!(matches!(
            GgufFile::open(f.path()),
            Err(GgufError::Truncated { .. })
        ));
    }

    /// Битые длины и размеры из файла дают ошибку, а не панику/аллокацию.
    #[test]
    fn rejects_garbage_sizes() {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&u64::MAX.to_le_bytes()); // n_tensors
        head.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        head.extend_from_slice(&u64::MAX.to_le_bytes()); // длина ключа
        let f = write_tmp(&head);
        assert!(GgufFile::open(f.path()).is_err());

        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&0u64.to_le_bytes());
        head.extend_from_slice(&1u64.to_le_bytes());
        let key = b"tokenizer.ggml.tokens";
        head.extend_from_slice(&(key.len() as u64).to_le_bytes());
        head.extend_from_slice(key);
        head.extend_from_slice(&9u32.to_le_bytes()); // массив
        head.extend_from_slice(&6u32.to_le_bytes()); // f32
        head.extend_from_slice(&(1u64 << 62).to_le_bytes()); // n * 4 переполняется
        let f = write_tmp(&head);
        assert!(GgufFile::open(f.path()).is_err());
    }
}
