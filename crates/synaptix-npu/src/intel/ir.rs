//! Граф synaptix → OpenVINO IR v11: XML с узлами и рёбрами и блоб констант.
//!
//! Семантика операций — как у перевода в QNN (`qualcomm/lower.rs`): Linear —
//! `x·wᵀ (+ bias)`, вес `[N, K]`; int8-веса с поканальными шкалами (`Quant::Axis`)
//! — константа i8 → Convert → Multiply на шкалы (шаблон сжатых весов, который
//! компилятор NPU исполняет без распаковки в F16). Нормы раскладываются; RMSNorm
//! в F16 — с предмасштабом 1/16, чтобы x² не переполнял F16.

use super::io_of;
use crate::graph::{BinaryKind, DType, Graph, Op, Quant, ReduceKind, TensorId, TensorKind, UnaryKind, WeightSrc};
use crate::{IoTensor, NpuError, Result};
use std::collections::HashMap;
use std::fmt::Write as _;

pub(super) struct Lowered {
    pub xml: String,
    pub bin: Vec<u8>,
    pub inputs: Vec<IoTensor>,
    pub outputs: Vec<IoTensor>,
}

/// Тип значения в IR: типы графа и служебный i64 (оси, формы).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Et {
    G(DType),
    I64,
}

impl Et {
    fn name(self) -> &'static str {
        match self {
            Et::G(DType::F32) => "f32",
            Et::G(DType::F16) => "f16",
            Et::G(DType::I32) => "i32",
            // NPU 37xx не умеет Convert u32 → i32, а индексы Gather нужны целые со
            // знаком: u32 графа объявляется i32 (значения < 2³¹, байты те же).
            Et::G(DType::U32) => "i32",
            Et::G(DType::U8) => "u8",
            Et::G(DType::U16) => "u16",
            Et::G(DType::I8) => "i8",
            Et::G(DType::I4) => "i4",
            Et::I64 => "i64",
        }
    }

    fn precision(self) -> &'static str {
        match self {
            Et::G(DType::F32) => "FP32",
            Et::G(DType::F16) => "FP16",
            Et::G(DType::I32) => "I32",
            Et::G(DType::U32) => "I32",
            Et::G(DType::U8) => "U8",
            Et::G(DType::U16) => "U16",
            Et::G(DType::I8) => "I8",
            Et::G(DType::I4) => "I4",
            Et::I64 => "I64",
        }
    }

}

#[derive(Clone, Debug)]
struct Val {
    layer: u32,
    port: u32,
    et: Et,
    shape: Vec<u32>,
}

struct B<'a> {
    g: &'a Graph,
    xml: String,
    edges: Vec<(u32, u32, u32, u32)>,
    next: u32,
    bin: Vec<u8>,
    vals: HashMap<u32, Val>,
}

fn dims(shape: &[u32]) -> String {
    shape.iter().map(|d| format!("<dim>{d}</dim>")).collect()
}

fn csv<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

impl<'a> B<'a> {
    /// Узел: входы — порты 0..n, выходы — n.. . Возвращает значения выходов.
    fn layer(&mut self, ty: &str, version: &str, data: &str, ins: &[&Val], outs: &[(Et, Vec<u32>)]) -> Vec<Val> {
        let id = self.next;
        self.next += 1;
        let _ = write!(self.xml, "<layer id=\"{id}\" name=\"n{id}\" type=\"{ty}\" version=\"{version}\">");
        if !data.is_empty() {
            let _ = write!(self.xml, "<data {data}/>");
        }
        if !ins.is_empty() {
            self.xml.push_str("<input>");
            for (i, v) in ins.iter().enumerate() {
                let _ = write!(self.xml, "<port id=\"{i}\" precision=\"{}\">{}</port>", v.et.precision(), dims(&v.shape));
                self.edges.push((v.layer, v.port, id, i as u32));
            }
            self.xml.push_str("</input>");
        }
        let mut res = Vec::with_capacity(outs.len());
        if !outs.is_empty() {
            self.xml.push_str("<output>");
            for (j, (et, shape)) in outs.iter().enumerate() {
                let port = (ins.len() + j) as u32;
                let _ = write!(self.xml, "<port id=\"{port}\" precision=\"{}\">{}</port>", et.precision(), dims(shape));
                res.push(Val { layer: id, port, et: *et, shape: shape.clone() });
            }
            self.xml.push_str("</output>");
        }
        self.xml.push_str("</layer>");
        res
    }

    fn one(&mut self, ty: &str, version: &str, data: &str, ins: &[&Val], et: Et, shape: Vec<u32>) -> Val {
        self.layer(ty, version, data, ins, &[(et, shape)]).remove(0)
    }

    fn constant(&mut self, et: Et, shape: &[u32], bytes: &[u8]) -> Val {
        let off = self.bin.len().next_multiple_of(64);
        self.bin.resize(off, 0);
        self.bin.extend_from_slice(bytes);
        let data = format!("element_type=\"{}\" shape=\"{}\" offset=\"{off}\" size=\"{}\"", et.name(), csv(shape), bytes.len());
        self.one("Const", "opset1", &data, &[], et, shape.to_vec())
    }

    fn i64s(&mut self, v: &[i64]) -> Val {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.constant(Et::I64, &[v.len() as u32], &b)
    }

    fn i64_scalar(&mut self, v: i64) -> Val {
        self.constant(Et::I64, &[], &v.to_le_bytes())
    }

    fn scalar(&mut self, v: f32, et: Et) -> Val {
        match et {
            Et::G(DType::F16) => self.constant(et, &[], &half::f16::from_f32(v).to_le_bytes()),
            _ => self.constant(Et::G(DType::F32), &[], &v.to_le_bytes()),
        }
    }

    fn convert(&mut self, v: &Val, et: Et) -> Val {
        if v.et == et {
            return v.clone();
        }
        let data = format!("destination_type=\"{}\"", et.name());
        self.one("Convert", "opset1", &data, &[v], et, v.shape.clone())
    }

    fn bin_op(&mut self, ty: &str, a: &Val, b: &Val, shape: Vec<u32>) -> Val {
        let et = a.et;
        let b = self.convert(b, et);
        self.one(ty, "opset1", "auto_broadcast=\"numpy\"", &[a, &b], et, shape)
    }

    fn reshape(&mut self, v: &Val, shape: &[u32]) -> Val {
        let s = self.i64s(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
        self.one("Reshape", "opset1", "special_zero=\"false\"", &[v, &s], v.et, shape.to_vec())
    }

    /// Значение тензора графа; веса превращаются в константы при первом
    /// использовании, квантованные — сразу с деквантованием в F16.
    fn get(&mut self, id: TensorId) -> Result<Val> {
        if let Some(v) = self.vals.get(&id.0) {
            return Ok(v.clone());
        }
        let t = self.g.tensor(id).clone();
        let TensorKind::Weight(src) = t.kind else {
            return Err(NpuError::Graph(format!("тензор {} не вычислен к моменту использования", id.0)));
        };
        let bytes: Vec<u8> = match src {
            WeightSrc::Inline { offset } => self.g.weights[offset..offset + t.bytes()].to_vec(),
            WeightSrc::Buffer { uid, offset, .. } => {
                let b = self.g.buffers.get(&uid).ok_or_else(|| NpuError::Graph(format!("буфер веса {uid} не найден")))?;
                b.as_slice()[offset..offset + t.bytes()].to_vec()
            }
        };
        let raw = self.constant(Et::G(t.dtype), &t.shape, &bytes);
        let v = match &t.quant {
            Quant::None => raw,
            q => {
                let f = Et::G(DType::F16);
                let x = self.convert(&raw, f);
                let (scales, offsets, sshape): (Vec<f32>, Vec<i32>, Vec<u32>) = match q {
                    Quant::Tensor { scale, offset } => (vec![*scale], vec![*offset], vec![]),
                    Quant::Axis { axis, scales } => {
                        let ax = if *axis < 0 { (t.shape.len() as i32 + axis) as usize } else { *axis as usize };
                        let mut s = vec![1u32; t.shape.len()];
                        s[ax] = scales.len() as u32;
                        (scales.iter().map(|p| p.0).collect(), scales.iter().map(|p| p.1).collect(), s)
                    }
                    Quant::None => unreachable!(),
                };
                let x = if offsets.iter().any(|&o| o != 0) {
                    let ob: Vec<u8> = offsets.iter().flat_map(|&o| half::f16::from_f32(o as f32).to_le_bytes()).collect();
                    let oc = self.constant(f, &sshape, &ob);
                    self.bin_op("Subtract", &x, &oc, t.shape.clone())
                } else {
                    x
                };
                let sb: Vec<u8> = scales.iter().flat_map(|&s| half::f16::from_f32(s).to_le_bytes()).collect();
                let sc = self.constant(f, &sshape, &sb);
                self.bin_op("Multiply", &x, &sc, t.shape.clone())
            }
        };
        self.vals.insert(id.0, v.clone());
        Ok(v)
    }

    fn out_desc(&self, out: TensorId) -> (Et, Vec<u32>) {
        let t = self.g.tensor(out);
        (Et::G(t.dtype), t.shape.clone())
    }

    fn set(&mut self, out: TensorId, v: Val) {
        let (et, _) = self.out_desc(out);
        let v = self.convert(&v, et);
        self.vals.insert(out.0, v);
    }

    fn lower_op(&mut self, op: &Op) -> Result<()> {
        match op {
            &Op::Linear { x, w, bias, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let xv = self.convert(&xv, et);
                let wv = self.get(w)?;
                let wv = self.convert(&wv, et);
                let mut y = self.one("MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"true\"", &[&xv, &wv], et, shape.clone());
                if let Some(b) = bias {
                    let bv = self.get(b)?;
                    y = self.bin_op("Add", &y, &bv, shape);
                }
                self.set(out, y);
            }
            &Op::MatMul { a, b, transpose_a, transpose_b, out } => {
                let (et, shape) = self.out_desc(out);
                let av = self.get(a)?;
                let av = self.convert(&av, et);
                let bv = self.get(b)?;
                let bv = self.convert(&bv, et);
                let data = format!("transpose_a=\"{transpose_a}\" transpose_b=\"{transpose_b}\"");
                let y = self.one("MatMul", "opset1", &data, &[&av, &bv], et, shape);
                self.set(out, y);
            }
            &Op::Unary { kind, x, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let xv = self.convert(&xv, et);
                let simple = |k: UnaryKind| -> Option<(&'static str, &'static str, &'static str)> {
                    Some(match k {
                        UnaryKind::Neg => ("Negative", "opset1", ""),
                        UnaryKind::Abs => ("Abs", "opset1", ""),
                        UnaryKind::Sqrt => ("Sqrt", "opset1", ""),
                        UnaryKind::Exp => ("Exp", "opset1", ""),
                        UnaryKind::Log => ("Log", "opset1", ""),
                        UnaryKind::Sin => ("Sin", "opset1", ""),
                        UnaryKind::Cos => ("Cos", "opset1", ""),
                        UnaryKind::Tanh => ("Tanh", "opset1", ""),
                        UnaryKind::Sigmoid => ("Sigmoid", "opset1", ""),
                        UnaryKind::Relu => ("Relu", "opset1", ""),
                        UnaryKind::Gelu => ("Gelu", "opset7", "approximation_mode=\"ERF\""),
                        UnaryKind::Round => ("Round", "opset5", "mode=\"half_away_from_zero\""),
                        UnaryKind::Floor => ("Floor", "opset1", ""),
                        UnaryKind::Ceil => ("Ceiling", "opset1", ""),
                        UnaryKind::Sign => ("Sign", "opset1", ""),
                        UnaryKind::Rsqrt | UnaryKind::Recip => return None,
                    })
                };
                let y = match simple(kind) {
                    Some((ty, ver, data)) => self.one(ty, ver, data, &[&xv], et, shape.clone()),
                    None => {
                        let one = self.scalar(1.0, et);
                        let d = if kind == UnaryKind::Rsqrt { self.one("Sqrt", "opset1", "", &[&xv], et, shape.clone()) } else { xv };
                        self.bin_op("Divide", &one, &d, shape)
                    }
                };
                self.set(out, y);
            }
            &Op::Binary { kind, a, b, out } => {
                let (et, shape) = self.out_desc(out);
                let ty = match kind {
                    BinaryKind::Add => "Add",
                    BinaryKind::Sub => "Subtract",
                    BinaryKind::Mul => "Multiply",
                    BinaryKind::Div => "Divide",
                    BinaryKind::Max => "Maximum",
                    BinaryKind::Min => "Minimum",
                    BinaryKind::Pow => "Power",
                };
                let av = self.get(a)?;
                let av = self.convert(&av, et);
                let bv = self.get(b)?;
                let y = self.bin_op(ty, &av, &bv, shape);
                self.set(out, y);
            }
            &Op::Clamp { x, min, max, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let y = self.one("Clamp", "opset1", &format!("min=\"{min}\" max=\"{max}\""), &[&xv], et, shape);
                self.set(out, y);
            }
            &Op::Softmax { x, axis, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let y = self.one("SoftMax", "opset8", &format!("axis=\"{axis}\""), &[&xv], et, shape);
                self.set(out, y);
            }
            Op::Reduce { kind, x, axes, keep_dims, out } => {
                let (et, shape) = self.out_desc(*out);
                let ty = match kind {
                    ReduceKind::Sum => "ReduceSum",
                    ReduceKind::Mean => "ReduceMean",
                    ReduceKind::Max => "ReduceMax",
                    ReduceKind::Min => "ReduceMin",
                };
                let xv = self.get(*x)?;
                let ax = self.i64s(&axes.iter().map(|&a| a as i64).collect::<Vec<_>>());
                let y = self.one(ty, "opset1", &format!("keep_dims=\"{keep_dims}\""), &[&xv, &ax], et, shape);
                self.set(*out, y);
            }
            &Op::ArgMax { x, axis, keep_dims, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let k = self.i64_scalar(1);
                let mut ks = xv.shape.clone();
                ks[axis as usize] = 1;
                let outs = self.layer(
                    "TopK",
                    "opset11",
                    &format!("axis=\"{axis}\" mode=\"max\" sort=\"none\" index_element_type=\"i32\" stable=\"false\""),
                    &[&xv, &k],
                    &[(xv.et, ks.clone()), (Et::G(DType::I32), ks.clone())],
                );
                let idx = outs[1].clone();
                let idx = if keep_dims && ks == shape { idx } else { self.reshape(&idx, &shape) };
                let y = self.convert(&idx, et);
                self.set(out, y);
            }
            &Op::Reshape { x, out } => {
                let (_, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let y = self.reshape(&xv, &shape);
                self.set(out, y);
            }
            Op::Transpose { x, perm, out } => {
                let (_, shape) = self.out_desc(*out);
                let xv = self.get(*x)?;
                let p = self.i64s(&perm.iter().map(|&a| a as i64).collect::<Vec<_>>());
                let y = self.one("Transpose", "opset1", "", &[&xv, &p], xv.et, shape);
                self.set(*out, y);
            }
            Op::Slice { x, begin, end, stride, out } => {
                let (_, shape) = self.out_desc(*out);
                let xv = self.get(*x)?;
                let to64 = |v: &[i32]| v.iter().map(|&a| a as i64).collect::<Vec<_>>();
                let (b, e, s) = (self.i64s(&to64(begin)), self.i64s(&to64(end)), self.i64s(&to64(stride)));
                let axes = self.i64s(&(0..begin.len() as i64).collect::<Vec<_>>());
                let y = self.one("Slice", "opset8", "", &[&xv, &b, &e, &s, &axes], xv.et, shape);
                self.set(*out, y);
            }
            Op::Tile { x, multiples, out } => {
                let (_, shape) = self.out_desc(*out);
                let xv = self.get(*x)?;
                let r = self.i64s(&multiples.iter().map(|&a| a as i64).collect::<Vec<_>>());
                let y = self.one("Tile", "opset1", "", &[&xv, &r], xv.et, shape);
                self.set(*out, y);
            }
            Op::Concat { xs, axis, out } => {
                let (et, shape) = self.out_desc(*out);
                let mut vs = Vec::with_capacity(xs.len());
                for &x in xs {
                    let v = self.get(x)?;
                    vs.push(self.convert(&v, et));
                }
                let refs: Vec<&Val> = vs.iter().collect();
                let y = self.one("Concat", "opset1", &format!("axis=\"{axis}\""), &refs, et, shape);
                self.set(*out, y);
            }
            &Op::Gather { x, indices, axis, out } => {
                let (_, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let iv = self.get(indices)?;
                let a = self.i64_scalar(axis as i64);
                let y = self.one("Gather", "opset8", "batch_dims=\"0\"", &[&xv, &iv, &a], xv.et, shape);
                self.set(out, y);
            }
            &Op::Cast { x, out } | &Op::Dequantize { x, out } => {
                let xv = self.get(x)?;
                self.set(out, xv);
            }
            &Op::LayerNorm { x, gamma, beta, eps, axis, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let xv = self.convert(&xv, et);
                let ax = self.i64s(&[axis as i64]);
                let n = self.one(
                    "MVN",
                    "opset6",
                    &format!("eps=\"{eps}\" normalize_variance=\"true\" eps_mode=\"inside_sqrt\""),
                    &[&xv, &ax],
                    et,
                    shape.clone(),
                );
                let gv = self.get(gamma)?;
                let mut y = self.bin_op("Multiply", &n, &gv, shape.clone());
                if let Some(b) = beta {
                    let bv = self.get(b)?;
                    y = self.bin_op("Add", &y, &bv, shape);
                }
                self.set(out, y);
            }
            &Op::RmsNorm { x, gamma, eps, axis, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let xv = self.convert(&xv, et);
                // F16: x² переполняется уже при |x| > 256 — считаем по x/16, eps/256.
                let (xs, eps) = if et == Et::G(DType::F16) {
                    let k = self.scalar(1.0 / 16.0, et);
                    (self.bin_op("Multiply", &xv, &k, shape.clone()), eps / 256.0)
                } else {
                    (xv, eps)
                };
                let sq = self.bin_op("Multiply", &xs, &xs, shape.clone());
                let mut ks = shape.clone();
                ks[axis as usize] = 1;
                let ax = self.i64s(&[axis as i64]);
                let m = self.one("ReduceMean", "opset1", "keep_dims=\"true\"", &[&sq, &ax], et, ks.clone());
                let e = self.scalar(eps, et);
                let me = self.bin_op("Add", &m, &e, ks.clone());
                let r = self.one("Sqrt", "opset1", "", &[&me], et, ks);
                let n = self.bin_op("Divide", &xs, &r, shape.clone());
                let gv = self.get(gamma)?;
                let y = self.bin_op("Multiply", &n, &gv, shape);
                self.set(out, y);
            }
            &Op::Conv2d { x, w, bias, stride, pad, dilation, groups, out } => {
                let (et, shape) = self.out_desc(out);
                let xv = self.get(x)?;
                let xv = self.convert(&xv, et);
                let wv = self.get(w)?;
                let wv = self.convert(&wv, et);
                let data = format!(
                    "strides=\"{}\" pads_begin=\"{},{}\" pads_end=\"{},{}\" dilations=\"{}\" auto_pad=\"explicit\"",
                    csv(&stride),
                    pad[0][0],
                    pad[1][0],
                    pad[0][1],
                    pad[1][1],
                    csv(&dilation)
                );
                let y = if groups <= 1 {
                    self.one("Convolution", "opset1", &data, &[&xv, &wv], et, shape.clone())
                } else {
                    let ws = &wv.shape;
                    let gs = [groups, ws[0] / groups, ws[1], ws[2], ws[3]];
                    let wg = self.reshape(&wv, &gs);
                    self.one("GroupConvolution", "opset1", &data, &[&xv, &wg], et, shape.clone())
                };
                let y = match bias {
                    Some(b) => {
                        let bv = self.get(b)?;
                        let bv = self.reshape(&bv, &[1, shape[1], 1, 1]);
                        self.bin_op("Add", &y, &bv, shape)
                    }
                    None => y,
                };
                self.set(out, y);
            }
        }
        Ok(())
    }
}

pub(super) fn lower(g: &Graph) -> Result<Lowered> {
    let (live_t, live_o) = g.live();
    let inputs = io_of(g, TensorKind::Input, &live_t);
    let outputs = io_of(g, TensorKind::Output, &live_t);
    if outputs.is_empty() {
        return Err(NpuError::Graph("граф без выходов".into()));
    }
    let mut b = B { g, xml: String::with_capacity(1 << 16), edges: Vec::new(), next: 0, bin: Vec::new(), vals: HashMap::new() };
    b.xml.push_str("<?xml version=\"1.0\"?><net name=\"synaptix\" version=\"11\"><layers>");
    // Скаляры на границе графа — `[1]` (так их передаёт исполнение), внутри — `[]`.
    for io in &inputs {
        let et = Et::G(io.dtype);
        let shape = if io.shape.is_empty() { vec![1] } else { io.shape.clone() };
        let data = format!("shape=\"{}\" element_type=\"{}\"", csv(&shape), et.name());
        let mut v = b.one("Parameter", "opset1", &data, &[], et, shape);
        if io.shape.is_empty() {
            v = b.reshape(&v, &[]);
        }
        b.vals.insert(io.id.0, v);
    }
    for (i, op) in g.ops.iter().enumerate() {
        if live_o[i] {
            b.lower_op(op)?;
        }
    }
    for io in &outputs {
        let mut v = b.get(io.id)?;
        if v.shape.is_empty() {
            v = b.reshape(&v, &[1]);
        }
        b.layer("Result", "opset1", "", &[&v], &[]);
    }
    b.xml.push_str("</layers><edges>");
    for (fl, fp, tl, tp) in std::mem::take(&mut b.edges) {
        let _ = write!(b.xml, "<edge from-layer=\"{fl}\" from-port=\"{fp}\" to-layer=\"{tl}\" to-port=\"{tp}\"/>");
    }
    b.xml.push_str("</edges></net>");
    Ok(Lowered { xml: b.xml, bin: b.bin, inputs, outputs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_linear_rms_softmax() {
        let mut g = Graph::new();
        let x = g.input(DType::F16, &[1, 4, 8]);
        let w = g.weight(DType::F16, &[6, 8], &vec![0u8; 6 * 8 * 2]).unwrap();
        let y = g.linear(x, w, None).unwrap();
        let s = g.softmax(y);
        g.output(s).unwrap();
        let l = lower(&g).unwrap();
        assert!(l.xml.contains("type=\"MatMul\"") && l.xml.contains("type=\"SoftMax\""));
        assert_eq!(l.inputs.len(), 1);
        assert_eq!(l.outputs.len(), 1);
    }
}
