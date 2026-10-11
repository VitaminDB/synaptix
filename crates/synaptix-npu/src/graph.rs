use crate::{NpuError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I32,
    U8,
    U16,
    I8,
    I4,
}

impl DType {
    pub fn bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::F16 | DType::U16 => 16,
            DType::U8 | DType::I8 => 8,
            DType::I4 => 4,
        }
    }

    pub fn bytes(self, elems: usize) -> usize {
        (elems * self.bits()).div_ceil(8)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Quant {
    None,
    Tensor { scale: f32, offset: i32 },
    Axis { axis: i32, scales: Vec<(f32, i32)> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub enum TensorKind {
    Input,
    Output,
    Weight { offset: usize },
    Internal,
}

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub id: TensorId,
    pub kind: TensorKind,
    pub dtype: DType,
    pub shape: Vec<u32>,
    pub quant: Quant,
}

impl TensorDesc {
    pub fn elems(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn bytes(&self) -> usize {
        self.dtype.bytes(self.elems())
    }
}

#[derive(Debug, Clone)]
pub enum Op {
    Linear { x: TensorId, w: TensorId, bias: Option<TensorId>, out: TensorId },
    MatMul { a: TensorId, b: TensorId, transpose_a: bool, transpose_b: bool, out: TensorId },
    Add { a: TensorId, b: TensorId, out: TensorId },
    Mul { a: TensorId, b: TensorId, out: TensorId },
    Softmax { x: TensorId, out: TensorId },
    Gelu { x: TensorId, out: TensorId },
    Sigmoid { x: TensorId, out: TensorId },
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub tensors: Vec<TensorDesc>,
    pub ops: Vec<Op>,
    pub weights: Vec<u8>,
}

fn broadcast(a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(NpuError::Graph(format!("формы {a:?} и {b:?} не транслируются"))),
        };
    }
    Ok(out)
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tensor(&self, id: TensorId) -> &TensorDesc {
        &self.tensors[id.0 as usize]
    }

    fn push(&mut self, kind: TensorKind, dtype: DType, shape: &[u32], quant: Quant) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorDesc { id, kind, dtype, shape: shape.to_vec(), quant });
        id
    }

    pub fn input(&mut self, dtype: DType, shape: &[u32]) -> TensorId {
        self.push(TensorKind::Input, dtype, shape, Quant::None)
    }

    pub fn weight(&mut self, dtype: DType, shape: &[u32], data: &[u8]) -> Result<TensorId> {
        self.weight_quant(dtype, shape, Quant::None, data)
    }

    pub fn weight_quant(&mut self, dtype: DType, shape: &[u32], quant: Quant, data: &[u8]) -> Result<TensorId> {
        let need = dtype.bytes(shape.iter().map(|&d| d as usize).product());
        if data.len() != need {
            return Err(NpuError::Graph(format!("вес {shape:?} {dtype:?}: {} байт вместо {need}", data.len())));
        }
        let offset = self.weights.len().next_multiple_of(64);
        self.weights.resize(offset, 0);
        self.weights.extend_from_slice(data);
        Ok(self.push(TensorKind::Weight { offset }, dtype, shape, quant))
    }

    fn internal(&mut self, dtype: DType, shape: &[u32]) -> TensorId {
        self.push(TensorKind::Internal, dtype, shape, Quant::None)
    }

    pub fn output(&mut self, t: TensorId) -> Result<()> {
        let d = &mut self.tensors[t.0 as usize];
        match d.kind {
            TensorKind::Internal => {
                d.kind = TensorKind::Output;
                Ok(())
            }
            _ => Err(NpuError::Graph(format!("тензор {} не может быть выходом", t.0))),
        }
    }

    pub fn linear(&mut self, x: TensorId, w: TensorId, bias: Option<TensorId>) -> Result<TensorId> {
        let xs = self.tensor(x).shape.clone();
        let ws = self.tensor(w).shape.clone();
        if ws.len() != 2 || xs.last() != Some(&ws[1]) {
            return Err(NpuError::Graph(format!("linear: x {xs:?}, w {ws:?} (ждём w[N,K])")));
        }
        if let Some(b) = bias {
            if self.tensor(b).shape != [ws[0]] {
                return Err(NpuError::Graph("linear: bias[N]".into()));
            }
        }
        let mut os = xs.clone();
        *os.last_mut().unwrap() = ws[0];
        let out = self.internal(self.tensor(x).dtype, &os);
        self.ops.push(Op::Linear { x, w, bias, out });
        Ok(out)
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId, transpose_a: bool, transpose_b: bool) -> Result<TensorId> {
        let (sa, sb) = (self.tensor(a).shape.clone(), self.tensor(b).shape.clone());
        if sa.len() < 2 || sb.len() < 2 {
            return Err(NpuError::Graph("matmul: ранг ≥ 2".into()));
        }
        let (m, ka) = if transpose_a { (sa[sa.len() - 1], sa[sa.len() - 2]) } else { (sa[sa.len() - 2], sa[sa.len() - 1]) };
        let (kb, n) = if transpose_b { (sb[sb.len() - 1], sb[sb.len() - 2]) } else { (sb[sb.len() - 2], sb[sb.len() - 1]) };
        if ka != kb {
            return Err(NpuError::Graph(format!("matmul: K {ka} ≠ {kb}")));
        }
        let mut os = broadcast(&sa[..sa.len() - 2], &sb[..sb.len() - 2])?;
        os.extend([m, n]);
        let out = self.internal(self.tensor(a).dtype, &os);
        self.ops.push(Op::MatMul { a, b, transpose_a, transpose_b, out });
        Ok(out)
    }

    fn binary(&mut self, a: TensorId, b: TensorId, f: fn(TensorId, TensorId, TensorId) -> Op) -> Result<TensorId> {
        let s = broadcast(&self.tensor(a).shape.clone(), &self.tensor(b).shape.clone())?;
        let out = self.internal(self.tensor(a).dtype, &s);
        self.ops.push(f(a, b, out));
        Ok(out)
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        self.binary(a, b, |a, b, out| Op::Add { a, b, out })
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        self.binary(a, b, |a, b, out| Op::Mul { a, b, out })
    }

    fn unary(&mut self, x: TensorId, f: fn(TensorId, TensorId) -> Op) -> TensorId {
        let d = self.tensor(x);
        let (dt, s) = (d.dtype, d.shape.clone());
        let out = self.internal(dt, &s);
        self.ops.push(f(x, out));
        out
    }

    pub fn softmax(&mut self, x: TensorId) -> TensorId {
        self.unary(x, |x, out| Op::Softmax { x, out })
    }

    pub fn gelu(&mut self, x: TensorId) -> TensorId {
        self.unary(x, |x, out| Op::Gelu { x, out })
    }

    pub fn sigmoid(&mut self, x: TensorId) -> TensorId {
        self.unary(x, |x, out| Op::Sigmoid { x, out })
    }

    pub fn structure_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |s: &str| {
            for b in s.bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        };
        for t in &self.tensors {
            eat(&format!("{:?}{:?}{:?}{:?};", t.kind, t.dtype, t.shape, t.quant));
        }
        for o in &self.ops {
            eat(&format!("{o:?};"));
        }
        eat(&self.weights.len().to_string());
        h
    }
}
