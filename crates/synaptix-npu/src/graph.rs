use crate::buffer::NpuBuffer;
use crate::{NpuError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I32,
    U32,
    U8,
    U16,
    I8,
    I4,
}

impl DType {
    pub fn bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSrc {
    Inline { offset: usize },
    Buffer { uid: u64, offset: usize, fingerprint: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorKind {
    Input,
    Output,
    Weight(WeightSrc),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryKind {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Recip,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    Gelu,
    Round,
    Floor,
    Ceil,
    Sign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone)]
pub enum Op {
    Linear { x: TensorId, w: TensorId, bias: Option<TensorId>, out: TensorId },
    MatMul { a: TensorId, b: TensorId, transpose_a: bool, transpose_b: bool, out: TensorId },
    Unary { kind: UnaryKind, x: TensorId, out: TensorId },
    Binary { kind: BinaryKind, a: TensorId, b: TensorId, out: TensorId },
    Clamp { x: TensorId, min: f32, max: f32, out: TensorId },
    Softmax { x: TensorId, axis: u32, out: TensorId },
    Reduce { kind: ReduceKind, x: TensorId, axes: Vec<u32>, keep_dims: bool, out: TensorId },
    ArgMax { x: TensorId, axis: u32, keep_dims: bool, out: TensorId },
    Reshape { x: TensorId, out: TensorId },
    Transpose { x: TensorId, perm: Vec<u32>, out: TensorId },
    Slice { x: TensorId, begin: Vec<i32>, end: Vec<i32>, stride: Vec<i32>, out: TensorId },
    Tile { x: TensorId, multiples: Vec<u32>, out: TensorId },
    Concat { xs: Vec<TensorId>, axis: u32, out: TensorId },
    Gather { x: TensorId, indices: TensorId, axis: i32, out: TensorId },
    Cast { x: TensorId, out: TensorId },
    Dequantize { x: TensorId, out: TensorId },
    LayerNorm { x: TensorId, gamma: TensorId, beta: Option<TensorId>, eps: f32, axis: u32, out: TensorId },
    RmsNorm { x: TensorId, gamma: TensorId, eps: f32, axis: u32, out: TensorId },
    Conv2d {
        x: TensorId,
        w: TensorId,
        bias: Option<TensorId>,
        stride: [u32; 2],
        pad: [[u32; 2]; 2],
        dilation: [u32; 2],
        groups: u32,
        out: TensorId,
    },
}

impl Op {
    pub fn out(&self) -> TensorId {
        match self {
            Op::Linear { out, .. }
            | Op::MatMul { out, .. }
            | Op::Unary { out, .. }
            | Op::Binary { out, .. }
            | Op::Clamp { out, .. }
            | Op::Softmax { out, .. }
            | Op::Reduce { out, .. }
            | Op::ArgMax { out, .. }
            | Op::Reshape { out, .. }
            | Op::Transpose { out, .. }
            | Op::Slice { out, .. }
            | Op::Tile { out, .. }
            | Op::Concat { out, .. }
            | Op::Gather { out, .. }
            | Op::Cast { out, .. }
            | Op::Dequantize { out, .. }
            | Op::LayerNorm { out, .. }
            | Op::RmsNorm { out, .. }
            | Op::Conv2d { out, .. } => *out,
        }
    }

    pub fn inputs(&self) -> Vec<TensorId> {
        match self {
            Op::Linear { x, w, bias, .. } => [Some(*x), Some(*w), *bias].into_iter().flatten().collect(),
            Op::MatMul { a, b, .. } | Op::Binary { a, b, .. } => vec![*a, *b],
            Op::Unary { x, .. }
            | Op::Clamp { x, .. }
            | Op::Softmax { x, .. }
            | Op::Reduce { x, .. }
            | Op::ArgMax { x, .. }
            | Op::Reshape { x, .. }
            | Op::Transpose { x, .. }
            | Op::Slice { x, .. }
            | Op::Tile { x, .. }
            | Op::Cast { x, .. }
            | Op::Dequantize { x, .. } => vec![*x],
            Op::Concat { xs, .. } => xs.clone(),
            Op::Gather { x, indices, .. } => vec![*x, *indices],
            Op::LayerNorm { x, gamma, beta, .. } => [Some(*x), Some(*gamma), *beta].into_iter().flatten().collect(),
            Op::RmsNorm { x, gamma, .. } => vec![*x, *gamma],
            Op::Conv2d { x, w, bias, .. } => [Some(*x), Some(*w), *bias].into_iter().flatten().collect(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub tensors: Vec<TensorDesc>,
    pub ops: Vec<Op>,
    pub weights: Vec<u8>,
    pub buffers: std::collections::HashMap<u64, std::sync::Arc<NpuBuffer>>,
    reshape_src: std::collections::HashMap<TensorId, TensorId>,
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
        Ok(self.push(TensorKind::Weight(WeightSrc::Inline { offset }), dtype, shape, quant))
    }

    pub fn weight_buffer(
        &mut self,
        dtype: DType,
        shape: &[u32],
        buffer: &std::sync::Arc<NpuBuffer>,
        offset: usize,
        fingerprint: u64,
    ) -> Result<TensorId> {
        let need = dtype.bytes(shape.iter().map(|&d| d as usize).product());
        if offset + need > buffer.len() {
            return Err(NpuError::Graph(format!("вес {shape:?} {dtype:?} вне буфера NPU")));
        }
        self.buffers.insert(buffer.uid(), buffer.clone());
        let src = WeightSrc::Buffer { uid: buffer.uid(), offset, fingerprint };
        Ok(self.push(TensorKind::Weight(src), dtype, shape, Quant::None))
    }

    pub fn scalar_f32(&mut self, v: f32, dtype: DType) -> Result<TensorId> {
        match dtype {
            DType::F16 => self.weight(dtype, &[1], &half::f16::from_f32(v).to_le_bytes()),
            _ => self.weight(DType::F32, &[1], &v.to_le_bytes()),
        }
    }

    pub fn node(&mut self, dtype: DType, shape: &[u32], f: impl FnOnce(TensorId) -> Op) -> TensorId {
        let out = self.push(TensorKind::Internal, dtype, shape, Quant::None);
        let op = f(out);
        if let Op::Reshape { x, .. } = op {
            let root = self.reshape_root(x);
            self.reshape_src.insert(out, root);
        }
        self.ops.push(op);
        out
    }

    pub fn reshape_root(&self, x: TensorId) -> TensorId {
        self.reshape_src.get(&x).copied().unwrap_or(x)
    }

    pub fn output(&mut self, t: TensorId) -> Result<()> {
        let d = &mut self.tensors[t.0 as usize];
        match d.kind {
            TensorKind::Internal => {
                d.kind = TensorKind::Output;
                Ok(())
            }
            TensorKind::Output => Ok(()),
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
        let dt = self.tensor(x).dtype;
        Ok(self.node(dt, &os, |out| Op::Linear { x, w, bias, out }))
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
        let dt = self.tensor(a).dtype;
        Ok(self.node(dt, &os, |out| Op::MatMul { a, b, transpose_a, transpose_b, out }))
    }

    pub fn binary(&mut self, kind: BinaryKind, a: TensorId, b: TensorId) -> Result<TensorId> {
        let s = broadcast(&self.tensor(a).shape.clone(), &self.tensor(b).shape.clone())?;
        let dt = self.tensor(a).dtype;
        Ok(self.node(dt, &s, |out| Op::Binary { kind, a, b, out }))
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        self.binary(BinaryKind::Add, a, b)
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        self.binary(BinaryKind::Mul, a, b)
    }

    pub fn unary(&mut self, kind: UnaryKind, x: TensorId) -> TensorId {
        let d = self.tensor(x);
        let (dt, s) = (d.dtype, d.shape.clone());
        self.node(dt, &s, |out| Op::Unary { kind, x, out })
    }

    pub fn softmax(&mut self, x: TensorId) -> TensorId {
        let d = self.tensor(x);
        let (dt, s) = (d.dtype, d.shape.clone());
        let axis = s.len() as u32 - 1;
        self.node(dt, &s, |out| Op::Softmax { x, axis, out })
    }

    pub fn gelu(&mut self, x: TensorId) -> TensorId {
        self.unary(UnaryKind::Gelu, x)
    }

    pub fn sigmoid(&mut self, x: TensorId) -> TensorId {
        self.unary(UnaryKind::Sigmoid, x)
    }

    pub fn reshape(&mut self, x: TensorId, shape: &[u32]) -> Result<TensorId> {
        let d = self.tensor(x);
        let n: usize = shape.iter().map(|&v| v as usize).product();
        if n != d.elems() {
            return Err(NpuError::Graph(format!("reshape {:?} → {shape:?}", d.shape)));
        }
        if d.shape == shape {
            return Ok(x);
        }
        let root = self.reshape_root(x);
        if self.tensor(root).shape == shape {
            return Ok(root);
        }
        let dt = self.tensor(root).dtype;
        Ok(self.node(dt, shape, |out| Op::Reshape { x: root, out }))
    }

    pub fn transpose(&mut self, x: TensorId, perm: &[u32]) -> TensorId {
        if perm.iter().enumerate().all(|(i, &p)| p as usize == i) {
            return x;
        }
        let d = self.tensor(x);
        let s: Vec<u32> = perm.iter().map(|&p| d.shape[p as usize]).collect();
        let dt = d.dtype;
        let perm = perm.to_vec();
        self.node(dt, &s, |out| Op::Transpose { x, perm, out })
    }

    pub fn live(&self) -> (Vec<bool>, Vec<bool>) {
        let mut t = vec![false; self.tensors.len()];
        for d in &self.tensors {
            if matches!(d.kind, TensorKind::Output | TensorKind::Input) {
                t[d.id.0 as usize] = true;
            }
        }
        let mut o = vec![false; self.ops.len()];
        for (i, op) in self.ops.iter().enumerate().rev() {
            if t[op.out().0 as usize] {
                o[i] = true;
                for x in op.inputs() {
                    t[x.0 as usize] = true;
                }
            }
        }
        (t, o)
    }

    pub fn structure_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |s: &str| {
            for b in s.bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        };
        for t in &self.tensors {
            let k = match &t.kind {
                TensorKind::Weight(WeightSrc::Buffer { fingerprint, .. }) => format!("WB{fingerprint:x}"),
                k => format!("{k:?}"),
            };
            eat(&format!("{k}{:?}{:?}{:?};", t.dtype, t.shape, t.quant));
        }
        for o in &self.ops {
            eat(&format!("{o:?};"));
        }
        let mut w = 0xcbf2_9ce4_8422_2325u64;
        for c in self.weights.chunks_exact(8) {
            w = (w ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x100_0000_01b3);
        }
        for &b in self.weights.chunks_exact(8).remainder() {
            w = (w ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
        eat(&format!("{}:{w:x}", self.weights.len()));
        h
    }
}
