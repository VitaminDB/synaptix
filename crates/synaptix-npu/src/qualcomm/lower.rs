use super::*;
use crate::graph::{BinaryKind, Op, ReduceKind, TensorDesc, UnaryKind, WeightSrc};

const KIND_INPUT: u8 = 0;
const KIND_OUTPUT: u8 = 1;
const KIND_STATIC: u8 = 2;
const KIND_NATIVE: u8 = 3;
const PARAM_SCALAR: u8 = 0;
const PARAM_TENSOR: u8 = 1;
const BUILD_FP16: u32 = 1;

pub(super) enum Param {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
    Tensor(u32),
}

struct WireTensor {
    id: u32,
    kind: u8,
    dtype: u32,
    shape: Vec<u32>,
    quant: Quant,
    data: Option<(Option<u64>, usize, usize)>,
}

pub(super) fn fused_norms(model: &crate::NpuModel) -> bool {
    match std::env::var("SYNNPU_FUSED_NORMS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => model.fused_norms,
    }
}

pub(super) fn norm_prescale() -> f32 {
    std::env::var("SYNNPU_NORM_PRESCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0)
}

pub(super) fn opt_level() -> u32 {
    std::env::var("SYNNPU_OPT").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(15)
}

pub(super) fn variant(model: &crate::NpuModel) -> String {
    format!("{}{}+o{}+ns{}", model.id, if fused_norms(model) { "+ln" } else { "" }, opt_level(), norm_prescale())
}

pub(super) struct Wire<'a> {
    g: &'a Graph,
    live_t: Vec<bool>,
    live_o: Vec<bool>,
    fused_norms: bool,
    tensors: Vec<WireTensor>,
    nodes: Wr,
    n_nodes: u32,
    pub blob: Vec<u8>,
    pub trace: Vec<String>,
}

impl<'a> Wire<'a> {
    pub fn new(g: &'a Graph, model: &crate::NpuModel) -> Self {
        let mut tensors = Vec::with_capacity(g.tensors.len());
        for t in &g.tensors {
            tensors.push(Self::from_desc(t));
        }
        let (live_t, live_o) = g.live();
        let fused_norms = fused_norms(model);
        Self { g, live_t, live_o, fused_norms, tensors, nodes: Wr::default(), n_nodes: 0, blob: g.weights.clone(), trace: Vec::new() }
    }

    fn from_desc(t: &TensorDesc) -> WireTensor {
        let (kind, data) = match t.kind {
            TensorKind::Input => (KIND_INPUT, None),
            TensorKind::Output => (KIND_OUTPUT, None),
            TensorKind::Internal => (KIND_NATIVE, None),
            TensorKind::Weight(WeightSrc::Inline { offset }) => (KIND_STATIC, Some((None, offset, t.bytes()))),
            TensorKind::Weight(WeightSrc::Buffer { uid, offset, .. }) => {
                (KIND_STATIC, Some((Some(uid), offset, t.bytes())))
            }
        };
        WireTensor { id: t.id.0, kind, dtype: qnn_dtype(t.dtype, &t.quant), shape: t.shape.clone(), quant: t.quant.clone(), data }
    }

    fn shape(&self, id: TensorId) -> &[u32] {
        &self.tensors[id.0 as usize].shape
    }

    fn dtype(&self, id: TensorId) -> u32 {
        self.tensors[id.0 as usize].dtype
    }

    fn native(&mut self, dtype: u32, shape: Vec<u32>) -> TensorId {
        let id = self.tensors.len() as u32;
        self.tensors.push(WireTensor { id, kind: KIND_NATIVE, dtype, shape, quant: Quant::None, data: None });
        TensorId(id)
    }

    fn param_tensor(&mut self, dtype: u32, shape: Vec<u32>, bytes: &[u8]) -> Param {
        let offset = self.blob.len().next_multiple_of(64);
        self.blob.resize(offset, 0);
        self.blob.extend_from_slice(bytes);
        let id = self.tensors.len() as u32;
        self.tensors.push(WireTensor {
            id,
            kind: KIND_STATIC,
            dtype,
            shape,
            quant: Quant::None,
            data: Some((None, offset, bytes.len())),
        });
        Param::Tensor(id)
    }

    fn u32s(&mut self, v: &[u32]) -> Param {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.param_tensor(QNN_U32, vec![v.len() as u32], &b)
    }

    fn u32s_2d(&mut self, v: &[u32], rows: u32, cols: u32) -> Param {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.param_tensor(QNN_U32, vec![rows, cols], &b)
    }

    fn i32s_2d(&mut self, v: &[i32], rows: u32, cols: u32) -> Param {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.param_tensor(QNN_I32, vec![rows, cols], &b)
    }

    fn scalar_tensor(&mut self, v: f32, like: u32) -> TensorId {
        let (dt, b) = if like == QNN_F16 {
            (QNN_F16, half::f16::from_f32(v).to_le_bytes().to_vec())
        } else {
            (QNN_F32, v.to_le_bytes().to_vec())
        };
        match self.param_tensor(dt, vec![1], &b) {
            Param::Tensor(id) => TensorId(id),
            _ => unreachable!(),
        }
    }

    fn node(&mut self, ty: &str, ins: &[TensorId], out: TensorId, params: &[(&str, Param)]) {
        let desc = |w: &Self, t: &TensorId| {
            let x = &w.tensors[t.0 as usize];
            format!("t{}{:?}:{:#x}{}", x.id, x.shape, x.dtype, if x.kind == KIND_STATIC { "s" } else { "" })
        };
        let line = format!(
            "n{} {ty}({}) -> {}",
            self.n_nodes,
            ins.iter().map(|t| desc(self, t)).collect::<Vec<_>>().join(", "),
            desc(self, &out)
        );
        self.trace.push(line);
        let w = &mut self.nodes;
        w.str(&format!("n{}", self.n_nodes));
        self.n_nodes += 1;
        w.str("qti.aisw");
        w.str(ty);
        w.u32(ins.len() as u32);
        for t in ins {
            w.u32(t.0);
        }
        w.u32(1);
        w.u32(out.0);
        w.u32(params.len() as u32);
        for (n, p) in params {
            w.str(n);
            match *p {
                Param::Bool(v) => {
                    w.u8(PARAM_SCALAR);
                    w.u32(QNN_BOOL);
                    w.u64(v as u64);
                }
                Param::U32(v) => {
                    w.u8(PARAM_SCALAR);
                    w.u32(QNN_U32);
                    w.u64(v as u64);
                }
                Param::I32(v) => {
                    w.u8(PARAM_SCALAR);
                    w.u32(QNN_I32);
                    w.u64(v as u32 as u64);
                }
                Param::F32(v) => {
                    w.u8(PARAM_SCALAR);
                    w.u32(QNN_F32);
                    w.u64(v.to_bits() as u64);
                }
                Param::Tensor(id) => {
                    w.u8(PARAM_TENSOR);
                    w.u32(id);
                }
            }
        }
    }

    fn transpose(&mut self, x: TensorId, perm: &[u32]) -> TensorId {
        let s = self.shape(x).to_vec();
        let os: Vec<u32> = perm.iter().map(|&p| s[p as usize]).collect();
        let out = self.native(self.dtype(x), os);
        let p = self.u32s(perm);
        self.node("Transpose", &[x], out, &[("perm", p)]);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_decomposed(&mut self, x: TensorId, gamma: TensorId, beta: Option<TensorId>, eps: f32, axis: u32, out: TensorId, center: bool) {
        let dt = self.dtype(x);
        let shape = self.shape(x).to_vec();
        let mut ks = shape.clone();
        ks[axis as usize] = 1;
        let c = if center {
            let m = self.native(dt, ks.clone());
            let a = self.u32s(&[axis]);
            self.node("ReduceMean", &[x], m, &[("axes", a), ("keep_dims", Param::Bool(true))]);
            let c = self.native(dt, shape.clone());
            self.node("ElementWiseSubtract", &[x, m], c, &[]);
            c
        } else {
            x
        };
        let sq = self.native(dt, shape.clone());
        self.node("ElementWiseMultiply", &[c, c], sq, &[]);
        let var = self.native(dt, ks.clone());
        let a = self.u32s(&[axis]);
        self.node("ReduceMean", &[sq], var, &[("axes", a), ("keep_dims", Param::Bool(true))]);
        let e = self.scalar_tensor(eps, dt);
        let ve = self.native(dt, ks.clone());
        self.node("ElementWiseAdd", &[var, e], ve, &[]);
        let inv = self.native(dt, ks);
        self.node("ElementWiseRsqrt", &[ve], inv, &[]);
        let n = self.native(dt, shape.clone());
        self.node("ElementWiseMultiply", &[c, inv], n, &[]);
        match beta {
            Some(b) => {
                let s = self.native(dt, shape);
                self.node("ElementWiseMultiply", &[n, gamma], s, &[]);
                self.node("ElementWiseAdd", &[s, b], out, &[]);
            }
            None => self.node("ElementWiseMultiply", &[n, gamma], out, &[]),
        }
    }

    pub fn lower_ops(&mut self) {
        for (i, op) in self.g.ops.clone().iter().enumerate() {
            if self.live_o[i] {
                self.lower_op(op);
            }
        }
    }

    fn prescale(&mut self, x: TensorId, eps: f32) -> (TensorId, f32) {
        let dt = self.dtype(x);
        let s = norm_prescale();
        if dt != QNN_F16 || s == 1.0 {
            return (x, eps);
        }
        let k = self.scalar_tensor(1.0 / s, dt);
        let y = self.native(dt, self.shape(x).to_vec());
        self.node("ElementWiseMultiply", &[x, k], y, &[]);
        (y, eps / (s * s))
    }

    fn lower_op(&mut self, op: &Op) {
        let op = &match op.clone() {
            Op::LayerNorm { x, gamma, beta, eps, axis, out } => {
                let (x, eps) = self.prescale(x, eps);
                Op::LayerNorm { x, gamma, beta, eps, axis, out }
            }
            Op::RmsNorm { x, gamma, eps, axis, out } => {
                let (x, eps) = self.prescale(x, eps);
                Op::RmsNorm { x, gamma, eps, axis, out }
            }
            o => o,
        };
        match op {
            &Op::Linear { x, w, bias, out } => {
                let mut ins = vec![x, w];
                ins.extend(bias);
                let keep = self.shape(x).len() > 2;
                self.node("FullyConnected", &ins, out, &[("keep_dims", Param::Bool(keep))]);
            }
            &Op::MatMul { a, b, transpose_a, transpose_b, out } => self.node(
                "MatMul",
                &[a, b],
                out,
                &[("transpose_in0", Param::Bool(transpose_a)), ("transpose_in1", Param::Bool(transpose_b))],
            ),
            &Op::Unary { kind, x, out } => {
                let ty = match kind {
                    UnaryKind::Neg => "ElementWiseNeg",
                    UnaryKind::Abs => "ElementWiseAbs",
                    UnaryKind::Sqrt => "ElementWiseSquareRoot",
                    UnaryKind::Rsqrt => "ElementWiseRsqrt",
                    UnaryKind::Exp => "ElementWiseExp",
                    UnaryKind::Log => "ElementWiseLog",
                    UnaryKind::Sin => "ElementWiseSin",
                    UnaryKind::Cos => "ElementWiseCos",
                    UnaryKind::Tanh => "Tanh",
                    UnaryKind::Sigmoid => "Sigmoid",
                    UnaryKind::Relu => "Relu",
                    UnaryKind::Gelu => "Gelu",
                    UnaryKind::Round => "ElementWiseRound",
                    UnaryKind::Floor => "ElementWiseFloor",
                    UnaryKind::Ceil => "ElementWiseCeil",
                    UnaryKind::Sign => "ElementWiseSign",
                    UnaryKind::Recip => {
                        let one = self.scalar_tensor(1.0, self.dtype(x));
                        self.node("ElementWiseDivide", &[one, x], out, &[]);
                        return;
                    }
                };
                self.node(ty, &[x], out, &[]);
            }
            &Op::Binary { kind, a, b, out } => {
                let ty = match kind {
                    BinaryKind::Add => "ElementWiseAdd",
                    BinaryKind::Sub => "ElementWiseSubtract",
                    BinaryKind::Mul => "ElementWiseMultiply",
                    BinaryKind::Div => "ElementWiseDivide",
                    BinaryKind::Max => "ElementWiseMaximum",
                    BinaryKind::Min => "ElementWiseMinimum",
                    BinaryKind::Pow => "ElementWisePower",
                };
                self.node(ty, &[a, b], out, &[]);
            }
            &Op::Clamp { x, min, max, out } => {
                self.node("ReluMinMax", &[x], out, &[("min_value", Param::F32(min)), ("max_value", Param::F32(max))])
            }
            &Op::Softmax { x, axis, out } => {
                self.node("Softmax", &[x], out, &[("axis", Param::U32(axis)), ("beta", Param::F32(1.0))])
            }
            Op::Reduce { kind, x, axes, keep_dims, out } => {
                let ty = match kind {
                    ReduceKind::Sum => "ReduceSum",
                    ReduceKind::Mean => "ReduceMean",
                    ReduceKind::Max => "ReduceMax",
                    ReduceKind::Min => "ReduceMin",
                };
                let a = self.u32s(axes);
                self.node(ty, &[*x], *out, &[("axes", a), ("keep_dims", Param::Bool(*keep_dims))]);
            }
            &Op::ArgMax { x, axis, keep_dims, out } => {
                self.node("Argmax", &[x], out, &[("axis", Param::U32(axis)), ("keep_dims", Param::Bool(keep_dims))])
            }
            &Op::Reshape { x, out } => self.node("Reshape", &[x], out, &[]),
            Op::Transpose { x, perm, out } => {
                let p = self.u32s(perm);
                self.node("Transpose", &[*x], *out, &[("perm", p)]);
            }
            Op::Slice { x, begin, end, stride, out } => {
                let r: Vec<i32> = (0..begin.len()).flat_map(|i| [begin[i], end[i], stride[i]]).collect();
                let p = self.i32s_2d(&r, begin.len() as u32, 3);
                self.node("StridedSlice", &[*x], *out, &[("ranges", p)]);
            }
            Op::Tile { x, multiples, out } => {
                let p = self.u32s(multiples);
                self.node("Tile", &[*x], *out, &[("multiples", p)]);
            }
            Op::Concat { xs, axis, out } => self.node("Concat", xs, *out, &[("axis", Param::U32(*axis))]),
            &Op::Cast { x, out } => self.node("Cast", &[x], out, &[]),
            &Op::Gather { x, indices, axis, out } => {
                self.node("Gather", &[x, indices], out, &[("axis", Param::I32(axis))])
            }
            &Op::LayerNorm { x, gamma, beta, eps, axis, out } if !self.fused_norms => {
                self.norm_decomposed(x, gamma, beta, eps, axis, out, true)
            }
            &Op::RmsNorm { x, gamma, eps, axis, out } if !self.fused_norms => {
                self.norm_decomposed(x, gamma, None, eps, axis, out, false)
            }
            &Op::LayerNorm { x, gamma, beta, eps, axis, out } => {
                let mut ins = vec![x, gamma];
                ins.extend(beta);
                let a = self.u32s(&[axis]);
                self.node("LayerNorm", &ins, out, &[("epsilon", Param::F32(eps)), ("axes", a)]);
            }
            &Op::RmsNorm { x, gamma, eps, axis, out } => {
                let a = self.u32s(&[axis]);
                self.node("RmsNorm", &[x, gamma], out, &[("epsilon", Param::F32(eps)), ("axes", a)]);
            }
            &Op::Conv2d { x, w, bias, stride, pad, dilation, groups, out } => {
                let xn = self.transpose(x, &[0, 2, 3, 1]);
                let wn = self.transpose(w, &[2, 3, 1, 0]);
                let os = self.shape(out).to_vec();
                let on = self.native(self.dtype(out), vec![os[0], os[2], os[3], os[1]]);
                let mut ins = vec![xn, wn];
                ins.extend(bias);
                let st = self.u32s(&stride);
                let pd = self.u32s_2d(&[pad[0][0], pad[0][1], pad[1][0], pad[1][1]], 2, 2);
                let dl = self.u32s(&dilation);
                self.node(
                    "Conv2d",
                    &ins,
                    on,
                    &[("stride", st), ("pad_amount", pd), ("dilation", dl), ("group", Param::U32(groups))],
                );
                let p = self.u32s(&[0, 3, 1, 2]);
                self.node("Transpose", &[on], out, &[("perm", p)]);
            }
        }
    }

    pub fn buffer_uids(&self) -> Vec<u64> {
        let n_graph = self.live_t.len();
        let mut v: Vec<u64> = self
            .tensors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= n_graph || self.live_t[*i])
            .filter_map(|(_, t)| t.data.and_then(|d| d.0))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn encode(&self, key: &str, blob_handle: u32, handles: &std::collections::HashMap<u64, u32>) -> Vec<u8> {
        let mut w = Wr::default();
        w.str(key);
        w.u32(BUILD_FP16 | (opt_level() << 8));
        let n_graph = self.live_t.len();
        let emit: Vec<&WireTensor> =
            self.tensors.iter().enumerate().filter(|(i, _)| *i >= n_graph || self.live_t[*i]).map(|(_, t)| t).collect();
        w.u32(emit.len() as u32);
        for t in emit {
            w.u32(t.id);
            w.u8(t.kind);
            w.u32(t.dtype);
            w.u8(t.shape.len() as u8);
            for &d in &t.shape {
                w.u32(d);
            }
            match &t.quant {
                Quant::None => w.u8(0),
                Quant::Tensor { scale, offset } => {
                    w.u8(1);
                    w.f32(*scale);
                    w.i32(*offset);
                }
                Quant::Axis { axis, scales } => {
                    w.u8(2);
                    w.i32(*axis);
                    w.u32(scales.len() as u32);
                    for &(s, o) in scales {
                        w.f32(s);
                        w.i32(o);
                    }
                }
            }
            if let Some((h, off, bytes)) = t.data {
                w.u32(h.map(|u| handles[&u]).unwrap_or(blob_handle));
                w.u64(off as u64);
                w.u64(bytes as u64);
            }
        }
        w.u32(self.n_nodes);
        w.0.extend_from_slice(&self.nodes.0);
        w.0
    }
}
