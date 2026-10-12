use crate::value::{Origin, Ready, State, Value, ValueRef};
use crate::view;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use synaptix_core::backend::{BinaryOp, ReduceOp, UnaryOp};
use synaptix_core::dtype::DType as CoreDType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_npu::graph::{BinaryKind, Op, Quant, ReduceKind, UnaryKind};
use synaptix_npu::{Binding, CompiledGraph, DType, Graph, NpuBuffer, TensorId};

pub struct Input {
    pub v: ValueRef,
    pub lo: Layout,
}

pub enum NodeOp {
    Copy,
    Cast,
    Unary(UnaryOp),
    Binary(BinaryOp),
    MatMul,
    Reduce(ReduceOp, Vec<usize>),
    Linear { bias: bool, residual: bool },
    LayerNorm { eps: f32, bias: bool },
    RmsNorm { eps: f32, qwen: bool },
    Conv2d { stride: (usize, usize), padding: (usize, usize), bias: bool },
    Attention { scale: f32, causal: bool },
    EmbedGather,
    KvAppend,
    AttentionDev { scale: f32, window: Option<i32>, causal: bool },
    RopeDev { rotary_dim: usize },
    RopeSplit,
    Concat(usize),
    LinearQ { n: usize, k: usize, scales: Arc<Vec<f32>> },
}

struct Node {
    serial: u64,
    op: NodeOp,
    ins: Vec<Input>,
    out: Weak<Value>,
    out_lo: Layout,
}

static REC: Lazy<Mutex<Vec<Node>>> = Lazy::new(|| Mutex::new(Vec::new()));
static SERIAL: AtomicU64 = AtomicU64::new(1);
/// Кэш скомпилированных графов процесса: по хэшу структуры, с вытеснением
/// давно не использованных по бюджету резидентной статики (веса графов
/// занимают память PD на DSP и процесса-исполнителя демона; модель с
/// несколькими формами входа — корзинами длины — иначе не помещается).
/// Вытеснение снимает только ссылку кэша; контекст освобождается, когда
/// граф никем не держится (`Drop`). Бюджет — `SYNAPTIX_NPU_RESIDENT_MB`,
/// по умолчанию половина `session_bytes` модели NPU (контексты несут ещё
/// spill-fill активаций, которого клиент не видит).
#[derive(Default)]
struct GraphCache {
    map: HashMap<u64, (Arc<dyn CompiledGraph>, usize, u64)>,
    total: usize,
    tick: u64,
}

impl GraphCache {
    fn budget() -> usize {
        if let Some(mb) = std::env::var("SYNAPTIX_NPU_RESIDENT_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            return mb << 20;
        }
        // Intel NPU работает в общей памяти: графы держатся в ОЗУ процесса,
        // бюджет — весь `session_bytes` (VoxCPM2: префилл и шаг — ~7 ГБ весов).
        crate::info()
            .map(|i| if i.model.vendor == synaptix_npu::models::Vendor::Intel { i.model.session_bytes as usize * 2 } else { i.model.session_bytes as usize / 2 })
            .unwrap_or(1 << 30)
    }

    fn get(&mut self, hash: u64) -> Option<Arc<dyn CompiledGraph>> {
        self.tick += 1;
        let tick = self.tick;
        self.map.get_mut(&hash).map(|(g, _, last)| {
            *last = tick;
            g.clone()
        })
    }

    fn insert(&mut self, hash: u64, g: Arc<dyn CompiledGraph>, bytes: usize) {
        let budget = Self::budget();
        while self.total + bytes > budget && !self.map.is_empty() {
            let Some((&victim, _)) = self.map.iter().min_by_key(|(_, (_, _, last))| *last) else { break };
            if let Some((_, b, _)) = self.map.remove(&victim) {
                self.total -= b;
                if std::env::var_os("SYNAPTIX_NPU_LOG").is_some() {
                    eprintln!("npu: граф {victim:016x} вытеснен из памяти ({} МБ статики)", b >> 20);
                }
            }
        }
        self.tick += 1;
        self.total += bytes;
        self.map.insert(hash, (g, bytes, self.tick));
    }
}

static GRAPHS: Lazy<Mutex<GraphCache>> = Lazy::new(|| Mutex::new(GraphCache::default()));
static STATS: Lazy<Mutex<Stats>> = Lazy::new(|| Mutex::new(Stats::default()));

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub flushes: u64,
    pub nodes: u64,
    pub compiles: u64,
    pub cache_hits: u64,
    pub compile_ms: f64,
    pub exec_ms: f64,
    pub cpu_fallbacks: u64,
}

pub fn stats() -> Stats {
    STATS.lock().clone()
}

pub fn note_cpu_fallback() {
    STATS.lock().cpu_fallbacks += 1;
}

fn err(e: synaptix_npu::NpuError) -> SynaptixError {
    SynaptixError::Other(format!("NPU: {e}"))
}

pub fn npu_dtype(d: CoreDType) -> Option<DType> {
    Some(match d {
        CoreDType::F32 => DType::F32,
        CoreDType::F16 => DType::F16,
        CoreDType::I32 => DType::I32,
        CoreDType::U32 => DType::U32,
        CoreDType::U8 => DType::U8,
        _ => return None,
    })
}

pub fn snapshot(v: &ValueRef) -> ValueRef {
    let st = std::mem::replace(&mut *v.state.lock(), State::Unset);
    let pending = if let State::Pending(s) = st { Some(s) } else { None };
    let snap = Value::new(v.bytes, st);
    // Записанные раньше потребители читают прежнее содержимое — теперь оно в
    // снимке (иначе внимание шага N увидит KV-буфер, уже переписанный шагом N+1
    // того же сброса).
    for n in REC.lock().iter_mut() {
        if pending == Some(n.serial) {
            n.out = Arc::downgrade(&snap);
        }
        for i in n.ins.iter_mut() {
            if Arc::ptr_eq(&i.v, v) {
                i.v = snap.clone();
            }
        }
    }
    snap
}

pub fn record(op: NodeOp, ins: Vec<Input>, dst: &ValueRef, out_lo: Layout) -> Result<()> {
    for i in &ins {
        if matches!(*i.v.state.lock(), State::Unset) {
            return Err(SynaptixError::Other("NPU: вход операции не записан".into()));
        }
    }
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    *dst.state.lock() = State::Pending(serial);
    REC.lock().push(Node { serial, op, ins, out: Arc::downgrade(dst), out_lo });
    Ok(())
}

struct Builder {
    g: Graph,
    static_bytes: usize,
    act_elems: usize,
    by_serial: HashMap<u64, TensorId>,
    ready: HashMap<(u64, DType, Vec<u32>), TensorId>,
    bind_in: Vec<(TensorId, Arc<NpuBuffer>)>,
    used: Vec<ValueRef>,
}

fn u32s(v: &[usize]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

impl Builder {
    fn input(&mut self, inp: &Input, materialize: bool) -> Result<TensorId> {
        let dt = npu_dtype(inp.lo.dtype()).ok_or(SynaptixError::Unsupported("тип тензора на NPU"))?;
        let base = {
            let mut st = inp.v.state.lock();
            match &mut *st {
                State::Unset => return Err(SynaptixError::Other("NPU: вход не записан".into())),
                State::Pending(s) => *self
                    .by_serial
                    .get(s)
                    .ok_or_else(|| SynaptixError::Other(format!("NPU: значение из другого графа не готово (узел {s}, вход {:?})", inp.lo.dims())))?,
                State::Ready(r) => {
                    let elem = dt.bits() / 8;
                    let numel = inp.lo.numel();
                    let full = inp.lo.is_contiguous() && inp.lo.offset() == 0 && numel * elem == inp.v.bytes;
                    let shape = if full { u32s(inp.lo.dims()) } else { vec![(inp.v.bytes / elem) as u32] };
                    let key = (inp.v.id, dt, shape.clone());
                    if let Some(&t) = self.ready.get(&key) {
                        t
                    } else {
                        let is_static = r.origin == Origin::Param;
                        let t = if is_static {
                            let fp = r.fingerprint(inp.v.bytes);
                            self.static_bytes += inp.v.bytes;
                            self.g.weight_buffer(dt, &shape, &r.buf, 0, fp).map_err(err)?
                        } else {
                            let t = self.g.input(dt, &shape);
                            self.bind_in.push((t, r.buf.clone()));
                            t
                        };
                        self.used.push(inp.v.clone());
                        self.ready.insert(key, t);
                        t
                    }
                }
            }
        };
        view::view(&mut self.g, base, &inp.lo, materialize).map_err(err)
    }

    fn qweight(&mut self, inp: &Input, n: usize, k: usize, scales: &[f32]) -> Result<TensorId> {
        let shape = vec![n as u32, k as u32];
        let key = (inp.v.id, DType::I8, shape.clone());
        if let Some(&t) = self.ready.get(&key) {
            return Ok(t);
        }
        let mut st = inp.v.state.lock();
        let State::Ready(r) = &mut *st else {
            return Err(SynaptixError::Other("NPU: int8-вес не загружен".into()));
        };
        if r.origin != Origin::Param || inp.v.bytes < n * k {
            return Err(SynaptixError::Unsupported("NPU: int8-вес должен быть параметром модели"));
        }
        let fp = r.fingerprint(inp.v.bytes);
        let quant = Quant::Axis { axis: 0, scales: scales.iter().map(|&s| (s, 0)).collect() };
        let t = self.g.weight_buffer_quant(DType::I8, &shape, quant, &r.buf, 0, fp).map_err(err)?;
        drop(st);
        self.static_bytes += n * k;
        self.used.push(inp.v.clone());
        self.ready.insert(key, t);
        Ok(t)
    }

    fn scalar(&mut self, v: f32, like: TensorId) -> Result<TensorId> {
        let dt = self.g.tensor(like).dtype;
        self.g.scalar_f32(v, dt).map_err(err)
    }

    fn bin(&mut self, kind: BinaryKind, a: TensorId, b: TensorId) -> Result<TensorId> {
        self.g.binary(kind, a, b).map_err(err)
    }

    fn unary(&mut self, op: UnaryOp, x: TensorId) -> Result<TensorId> {
        let u = |b: &mut Self, k| Ok(b.g.unary(k, x));
        match op {
            UnaryOp::Identity => Ok(view::fresh(&mut self.g, x)),
            UnaryOp::Neg => u(self, UnaryKind::Neg),
            UnaryOp::Abs => u(self, UnaryKind::Abs),
            UnaryOp::Sqrt => u(self, UnaryKind::Sqrt),
            UnaryOp::Recip => u(self, UnaryKind::Recip),
            UnaryOp::Exp => u(self, UnaryKind::Exp),
            UnaryOp::Log => u(self, UnaryKind::Log),
            UnaryOp::Sin => u(self, UnaryKind::Sin),
            UnaryOp::Cos => u(self, UnaryKind::Cos),
            UnaryOp::Tanh => u(self, UnaryKind::Tanh),
            UnaryOp::Sigmoid => u(self, UnaryKind::Sigmoid),
            UnaryOp::Relu => u(self, UnaryKind::Relu),
            UnaryOp::GeluExact => u(self, UnaryKind::Gelu),
            UnaryOp::Round => u(self, UnaryKind::Round),
            UnaryOp::Floor => u(self, UnaryKind::Floor),
            UnaryOp::Ceil => u(self, UnaryKind::Ceil),
            UnaryOp::Sign => u(self, UnaryKind::Sign),
            UnaryOp::Sqr => self.bin(BinaryKind::Mul, x, x),
            UnaryOp::Relu2 => {
                let r = self.g.unary(UnaryKind::Relu, x);
                self.bin(BinaryKind::Mul, r, r)
            }
            UnaryOp::Silu => {
                let s = self.g.unary(UnaryKind::Sigmoid, x);
                self.bin(BinaryKind::Mul, x, s)
            }
            UnaryOp::GeluTanh => {
                let x2 = self.bin(BinaryKind::Mul, x, x)?;
                let x3 = self.bin(BinaryKind::Mul, x2, x)?;
                let c = self.scalar(0.044715, x)?;
                let t = self.bin(BinaryKind::Mul, x3, c)?;
                let inner = self.bin(BinaryKind::Add, x, t)?;
                let k = self.scalar((2.0f32 / std::f32::consts::PI).sqrt(), x)?;
                let inner = self.bin(BinaryKind::Mul, inner, k)?;
                let th = self.g.unary(UnaryKind::Tanh, inner);
                let one = self.scalar(1.0, x)?;
                let p = self.bin(BinaryKind::Add, th, one)?;
                let hx = self.scalar(0.5, x)?;
                let hx = self.bin(BinaryKind::Mul, x, hx)?;
                self.bin(BinaryKind::Mul, hx, p)
            }
            UnaryOp::Clamp(lo, hi) => {
                let d = self.g.tensor(x);
                let (dt, s) = (d.dtype, d.shape.clone());
                Ok(self.g.node(dt, &s, |out| Op::Clamp { x, min: lo, max: hi, out }))
            }
            UnaryOp::Powf(p) => {
                let e = self.scalar(p, x)?;
                self.bin(BinaryKind::Pow, x, e)
            }
            UnaryOp::Affine(a, b) => {
                let mut y = x;
                if a != 1.0 {
                    let s = self.scalar(a, x)?;
                    y = self.bin(BinaryKind::Mul, y, s)?;
                }
                if b != 0.0 {
                    let s = self.scalar(b, x)?;
                    y = self.bin(BinaryKind::Add, y, s)?;
                }
                if y == x {
                    y = view::fresh(&mut self.g, x);
                }
                Ok(y)
            }
            UnaryOp::LeakyRelu(a) if a <= 1.0 => {
                let s = self.scalar(a, x)?;
                let ax = self.bin(BinaryKind::Mul, x, s)?;
                self.bin(BinaryKind::Max, x, ax)
            }
            _ => Err(SynaptixError::Unsupported("унарная операция на NPU")),
        }
    }

    fn iota(&mut self, n: usize, dt: DType) -> Result<TensorId> {
        let bytes: Vec<u8> = if dt == DType::F16 {
            (0..n).flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes()).collect()
        } else {
            (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect()
        };
        self.g.weight(dt, &[n as u32], &bytes).map_err(err)
    }

    fn cast(&mut self, x: TensorId, dt: DType) -> TensorId {
        if self.g.tensor(x).dtype == dt {
            return x;
        }
        let s = self.g.tensor(x).shape.clone();
        self.g.node(dt, &s, |out| Op::Cast { x, out })
    }

    fn clamp01(&mut self, x: TensorId) -> TensorId {
        let d = self.g.tensor(x);
        let (dt, s) = (d.dtype, d.shape.clone());
        self.g.node(dt, &s, |out| Op::Clamp { x, min: 0.0, max: 1.0, out })
    }

    fn reshape_to(&mut self, x: TensorId, lo: &Layout) -> TensorId {
        let shape = u32s(lo.dims());
        if self.g.tensor(x).shape == shape {
            return x;
        }
        let dt = self.g.tensor(x).dtype;
        self.g.node(dt, &shape, |out| Op::Reshape { x, out })
    }

    fn lower(&mut self, n: &Node) -> Result<TensorId> {
        let odt = npu_dtype(n.out_lo.dtype()).ok_or(SynaptixError::Unsupported("тип выхода на NPU"))?;
        let oshape = u32s(n.out_lo.dims());
        let t = match &n.op {
            NodeOp::Copy => {
                let x = self.input(&n.ins[0], true)?;
                view::fresh(&mut self.g, x)
            }
            NodeOp::Cast => {
                let x = self.input(&n.ins[0], true)?;
                self.g.node(odt, &oshape, |out| Op::Cast { x, out })
            }
            NodeOp::Concat(axis) => {
                let xs: Vec<TensorId> = n.ins.iter().map(|i| self.input(i, true)).collect::<Result<Vec<_>>>()?;
                let axis = *axis as u32;
                self.g.node(odt, &oshape, |out| Op::Concat { xs, axis, out })
            }
            NodeOp::Unary(op) => {
                let x = self.input(&n.ins[0], true)?;
                self.unary(*op, x)?
            }
            NodeOp::Binary(op) => {
                let a = self.input(&n.ins[0], false)?;
                let b = self.input(&n.ins[1], false)?;
                let kind = match op {
                    BinaryOp::Add => BinaryKind::Add,
                    BinaryOp::Sub => BinaryKind::Sub,
                    BinaryOp::Mul => BinaryKind::Mul,
                    BinaryOp::Div => BinaryKind::Div,
                    BinaryOp::Max => BinaryKind::Max,
                    BinaryOp::Min => BinaryKind::Min,
                };
                let y = self.bin(kind, a, b)?;
                if self.g.tensor(y).shape != oshape {
                    let dt = self.g.tensor(y).dtype;
                    let multiples: Vec<u32> =
                        oshape.iter().zip(&self.g.tensor(y).shape).map(|(&o, &c)| if c == 1 { o } else { 1 }).collect();
                    self.g.node(dt, &oshape, |out| Op::Tile { x: y, multiples, out })
                } else {
                    y
                }
            }
            NodeOp::MatMul => {
                let a = self.input(&n.ins[0], true)?;
                let b = self.input(&n.ins[1], true)?;
                self.g.matmul(a, b, false, false).map_err(err)?
            }
            NodeOp::Reduce(op, dims) => {
                let x = self.input(&n.ins[0], true)?;
                let mut kshape = self.g.tensor(x).shape.clone();
                for &d in dims {
                    kshape[d] = 1;
                }
                let axes = u32s(dims);
                let dt = self.g.tensor(x).dtype;
                let r = match op {
                    ReduceOp::ArgMax => {
                        let axis = axes[0];
                        self.g.node(odt, &kshape, |out| Op::ArgMax { x, axis, keep_dims: true, out })
                    }
                    _ => {
                        let kind = match op {
                            ReduceOp::Sum => ReduceKind::Sum,
                            ReduceOp::Mean => ReduceKind::Mean,
                            _ => ReduceKind::Max,
                        };
                        self.g.node(dt, &kshape, |out| Op::Reduce { kind, x, axes, keep_dims: true, out })
                    }
                };
                self.reshape_to(r, &n.out_lo)
            }
            NodeOp::Linear { bias, residual } => {
                let x = self.input(&n.ins[0], true)?;
                let w = self.input(&n.ins[1], true)?;
                let mut k = 2;
                let b = if *bias {
                    k += 1;
                    Some(self.input(&n.ins[2], true)?)
                } else {
                    None
                };
                let mut y = self.g.linear(x, w, b).map_err(err)?;
                if *residual {
                    let r = self.input(&n.ins[k], false)?;
                    y = self.bin(BinaryKind::Add, y, r)?;
                }
                y
            }
            NodeOp::LinearQ { n: rows, k: cols, scales } => {
                let x = self.input(&n.ins[0], true)?;
                let w = self.qweight(&n.ins[1], *rows, *cols, scales)?;
                self.g.linear(x, w, None).map_err(err)?
            }
            NodeOp::LayerNorm { eps, bias } => {
                let x = self.input(&n.ins[0], true)?;
                let gamma = self.input(&n.ins[1], true)?;
                let beta = if *bias { Some(self.input(&n.ins[2], true)?) } else { None };
                let axis = oshape.len() as u32 - 1;
                let eps = *eps;
                self.g.node(odt, &oshape, |out| Op::LayerNorm { x, gamma, beta, eps, axis, out })
            }
            NodeOp::RmsNorm { eps, qwen } => {
                let x = self.input(&n.ins[0], true)?;
                let mut gamma = self.input(&n.ins[1], true)?;
                if *qwen {
                    let one = self.scalar(1.0, gamma)?;
                    gamma = self.bin(BinaryKind::Add, gamma, one)?;
                }
                let axis = oshape.len() as u32 - 1;
                let eps = *eps;
                self.g.node(odt, &oshape, |out| Op::RmsNorm { x, gamma, eps, axis, out })
            }
            NodeOp::Attention { scale, causal } => {
                let q = self.input(&n.ins[0], true)?;
                let k = self.input(&n.ins[1], true)?;
                let v = self.input(&n.ins[2], true)?;
                let sc = self.scalar(*scale, q)?;
                let q = self.bin(BinaryKind::Mul, q, sc)?;
                let qs = self.g.tensor(q).shape.clone();
                let ks = self.g.tensor(k).shape.clone();
                let (nh, tq, tk) = (qs[1] as usize, qs[2] as usize, ks[2] as usize);
                let target: usize = std::env::var("SYNAPTIX_NPU_ATTN_ELEMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4_500_000);
                let chunk = std::env::var("SYNAPTIX_NPU_ATTN_HEADS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or_else(|| (target / (tq * tk).max(1)).max(1))
                    .clamp(1, nh);
                let mask = if *causal {
                    let dt = self.g.tensor(q).dtype;
                    let shift = tk.saturating_sub(tq);
                    let neg = if dt == DType::F16 { -60000.0f32 } else { -1e30 };
                    let mut bytes = Vec::with_capacity(tq * tk * 4);
                    for i in 0..tq {
                        for j in 0..tk {
                            let m = if j > i + shift { neg } else { 0.0 };
                            if dt == DType::F16 {
                                bytes.extend_from_slice(&half::f16::from_f32(m).to_le_bytes());
                            } else {
                                bytes.extend_from_slice(&m.to_le_bytes());
                            }
                        }
                    }
                    Some(self.g.weight(dt, &[tq as u32, tk as u32], &bytes).map_err(err)?)
                } else {
                    None
                };
                let mut parts = Vec::new();
                let mut h = 0;
                while h < nh {
                    let e = (h + chunk).min(nh);
                    let sl = |b: &mut Self, x: TensorId| {
                        if e - h == nh {
                            return x;
                        }
                        let sh = b.g.tensor(x).shape.clone();
                        let dt = b.g.tensor(x).dtype;
                        let mut os = sh.clone();
                        os[1] = (e - h) as u32;
                        let begin = vec![0, h as i32, 0, 0];
                        let end = vec![sh[0] as i32, e as i32, sh[2] as i32, sh[3] as i32];
                        b.g.node(dt, &os, |out| Op::Slice { x, begin, end, stride: vec![1; 4], out })
                    };
                    let (qh, kh, vh) = (sl(self, q), sl(self, k), sl(self, v));
                    let mut s = self.g.matmul(qh, kh, false, true).map_err(err)?;
                    if let Some(m) = mask {
                        s = self.bin(BinaryKind::Add, s, m)?;
                    }
                    let p = self.g.softmax(s);
                    parts.push(self.g.matmul(p, vh, false, false).map_err(err)?);
                    h = e;
                }
                if parts.len() == 1 {
                    parts[0]
                } else {
                    let dt = self.g.tensor(parts[0]).dtype;
                    let mut os = self.g.tensor(parts[0]).shape.clone();
                    os[1] = nh as u32;
                    self.g.node(dt, &os, |out| Op::Concat { xs: parts, axis: 1, out })
                }
            }
            NodeOp::EmbedGather => {
                let table = self.input(&n.ins[0], true)?;
                let ids = self.input(&n.ins[1], true)?;
                let ids = self.cast(ids, DType::I32);
                let ids = self.g.reshape(ids, &[self.g.tensor(ids).elems() as u32]).map_err(err)?;
                let ts = self.g.tensor(table).shape.clone();
                let rows = self.g.tensor(ids).shape[0];
                self.g.node(odt, &[rows, ts[ts.len() - 1]], |out| Op::Gather { x: table, indices: ids, axis: 0, out })
            }
            NodeOp::KvAppend if n.ins[1].lo.dims()[2] > 1 => {
                // T строк с позиции pos: one-hot oh[t, j] = [j = pos + t];
                // buf' = buf·(1 − Σₜ oh) + ohᵀ·k (копия точная: одна единица в строке).
                let buf = self.input(&n.ins[0], true)?;
                let k = self.input(&n.ins[1], true)?;
                let pos = self.input(&n.ins[2], true)?;
                let dt = self.g.tensor(buf).dtype;
                let cap = self.g.tensor(buf).shape[2] as usize;
                let t = self.g.tensor(k).shape[2] as usize;
                let iota = self.iota(cap, DType::F32)?;
                let iota = self.g.reshape(iota, &[1, cap as u32]).map_err(err)?;
                let it = self.iota(t, DType::F32)?;
                let it = self.g.reshape(it, &[t as u32, 1]).map_err(err)?;
                let p = self.cast(pos, DType::F32);
                let row = self.bin(BinaryKind::Add, it, p)?;
                let d = self.bin(BinaryKind::Sub, iota, row)?;
                let a = self.g.unary(UnaryKind::Abs, d);
                let one = self.scalar(1.0, a)?;
                let oh = self.bin(BinaryKind::Sub, one, a)?;
                let oh = self.clamp01(oh);
                let oh = self.cast(oh, dt);
                let oht = self.g.node(dt, &[cap as u32, t as u32], |out| Op::Transpose { x: oh, perm: vec![1, 0], out });
                let oht = self.g.reshape(oht, &[1, 1, cap as u32, t as u32]).map_err(err)?;
                let put = self.g.matmul(oht, k, false, false).map_err(err)?;
                let hit = self.g.node(dt, &[1, cap as u32], |out| Op::Reduce { kind: ReduceKind::Sum, x: oh, axes: vec![0], keep_dims: true, out });
                let one = self.scalar(1.0, hit)?;
                let keep = self.bin(BinaryKind::Sub, one, hit)?;
                let keep = self.g.reshape(keep, &[1, 1, cap as u32, 1]).map_err(err)?;
                let kept = self.bin(BinaryKind::Mul, buf, keep)?;
                self.bin(BinaryKind::Add, kept, put)?
            }
            NodeOp::KvAppend => {
                let buf = self.input(&n.ins[0], true)?;
                let k = self.input(&n.ins[1], true)?;
                let pos = self.input(&n.ins[2], true)?;
                let dt = self.g.tensor(buf).dtype;
                let cap = self.g.tensor(buf).shape[2] as usize;
                let iota = self.iota(cap, DType::F32)?;
                let p = self.cast(pos, DType::F32);
                let d = self.bin(BinaryKind::Sub, iota, p)?;
                let a = self.g.unary(UnaryKind::Abs, d);
                let one = self.scalar(1.0, a)?;
                let oh = self.bin(BinaryKind::Sub, one, a)?;
                let oh = self.clamp01(oh);
                let oh = self.cast(oh, dt);
                let oh = self.g.reshape(oh, &[1, 1, cap as u32, 1]).map_err(err)?;
                let diff = self.bin(BinaryKind::Sub, k, buf)?;
                let upd = self.bin(BinaryKind::Mul, diff, oh)?;
                self.bin(BinaryKind::Add, buf, upd)?
            }
            NodeOp::AttentionDev { scale, window, causal } => {
                let q = self.input(&n.ins[0], true)?;
                let k = self.input(&n.ins[1], true)?;
                let v = self.input(&n.ins[2], true)?;
                let t = self.input(&n.ins[3], true)?;
                let dt = self.g.tensor(q).dtype;
                let qs = self.g.tensor(q).shape.clone();
                let ks = self.g.tensor(k).shape.clone();
                let cap = ks[2] as usize;
                let (nh, nkv, hd, tq) = (qs[1], ks[1], qs[3], qs[2]);
                let sc = self.scalar(*scale, q)?;
                let q = self.bin(BinaryKind::Mul, q, sc)?;
                let q = if nh != nkv { self.g.reshape(q, &[qs[0], nkv, nh / nkv * tq, hd]).map_err(err)? } else { q };
                let s = self.g.matmul(q, k, false, true).map_err(err)?;
                let iota = self.iota(cap, DType::F32)?;
                let tf = self.cast(t, DType::F32);
                let m = if tq > 1 && *causal {
                    // Строка r видит j ≤ t_cache − T + r: m = clamp01(t_cache − T + 1 + r − j).
                    let ir = self.iota(tq as usize, DType::F32)?;
                    let ir = self.g.reshape(ir, &[tq, 1]).map_err(err)?;
                    let sh = self.scalar(1.0 - tq as f32, tf)?;
                    let base = self.bin(BinaryKind::Add, tf, sh)?;
                    let lim = self.bin(BinaryKind::Add, ir, base)?;
                    let iota2 = self.g.reshape(iota, &[1, cap as u32]).map_err(err)?;
                    self.bin(BinaryKind::Sub, lim, iota2)?
                } else {
                    self.bin(BinaryKind::Sub, tf, iota)?
                };
                let mut m = self.clamp01(m);
                if let Some(w) = window {
                    let lo = self.scalar((*w + 1) as f32, tf)?;
                    let start = self.bin(BinaryKind::Sub, tf, lo)?;
                    let d = self.bin(BinaryKind::Sub, iota, start)?;
                    let inw = self.clamp01(d);
                    m = self.bin(BinaryKind::Mul, m, inw)?;
                }
                let one = self.scalar(1.0, m)?;
                let m = self.bin(BinaryKind::Sub, m, one)?;
                let big = self.scalar(60000.0, m)?;
                let bias = self.bin(BinaryKind::Mul, m, big)?;
                let bias = self.cast(bias, dt);
                let s = if tq > 1 && *causal {
                    let bias = self.g.reshape(bias, &[1, 1, 1, tq, cap as u32]).map_err(err)?;
                    let ss = self.g.tensor(s).shape.clone();
                    let s5 = self.g.reshape(s, &[ss[0], ss[1], ss[2] / tq, tq, ss[3]]).map_err(err)?;
                    let s5 = self.bin(BinaryKind::Add, s5, bias)?;
                    self.g.reshape(s5, &ss).map_err(err)?
                } else {
                    let bias = self.g.reshape(bias, &[1, 1, 1, cap as u32]).map_err(err)?;
                    self.bin(BinaryKind::Add, s, bias)?
                };
                let p = self.g.softmax(s);
                let o = self.g.matmul(p, v, false, false).map_err(err)?;
                if nh != nkv { self.g.reshape(o, &qs).map_err(err)? } else { o }
            }
            NodeOp::RopeSplit => {
                // out = x·[cos, cos] + (x·R)·[sin, sin], R — поворот половин
                // (rotate_half = [−x₂, x₁]) постоянной матрицей D×D: поворот идёт
                // на матричном блоке, без срезов и Concat по последней оси.
                let x = self.input(&n.ins[0], true)?;
                let cos = self.input(&n.ins[1], true)?;
                let sin = self.input(&n.ins[2], true)?;
                let xs = self.g.tensor(x).shape.clone();
                let dt = self.g.tensor(x).dtype;
                let (s, d) = (xs[xs.len() - 2], xs[xs.len() - 1]);
                let half = d / 2;
                let full = |b: &mut Self, tab: TensorId| -> Result<TensorId> {
                    let t = b.cast(tab, dt);
                    let c = b.g.node(dt, &[s, d], |out| Op::Concat { xs: vec![t, t], axis: 1, out });
                    let mut bs = vec![1u32; xs.len()];
                    bs[xs.len() - 2] = s;
                    bs[xs.len() - 1] = d;
                    b.g.reshape(c, &bs).map_err(err)
                };
                let c2 = full(self, cos)?;
                let s2 = full(self, sin)?;
                let mut r = vec![0f32; (d * d) as usize];
                for i in 0..half as usize {
                    r[(i + half as usize) * d as usize + i] = -1.0;
                    r[i * d as usize + i + half as usize] = 1.0;
                }
                let bytes: Vec<u8> = if dt == DType::F16 {
                    r.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
                } else {
                    r.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let rw = self.g.weight(dt, &[d, d], &bytes).map_err(err)?;
                let rot = self.g.matmul(x, rw, false, false).map_err(err)?;
                let a = self.bin(BinaryKind::Mul, x, c2)?;
                let b2 = self.bin(BinaryKind::Mul, rot, s2)?;
                self.bin(BinaryKind::Add, a, b2)?
            }
            NodeOp::RopeDev { rotary_dim } => {
                let x = self.input(&n.ins[0], true)?;
                let cos = self.input(&n.ins[1], true)?;
                let sin = self.input(&n.ins[2], true)?;
                let pos = self.input(&n.ins[3], true)?;
                let xs = self.g.tensor(x).shape.clone();
                let dt = self.g.tensor(x).dtype;
                let rd = *rotary_dim as i32;
                let hd = xs[3] as i32;
                let half = rd / 2;
                let tq = xs[2];
                let pos = if tq > 1 {
                    // Строки pos..pos+T: индексы pos + iota(T) (в F32 точны до 2²⁴).
                    let p = self.cast(pos, DType::F32);
                    let it = self.iota(tq as usize, DType::F32)?;
                    let idx = self.bin(BinaryKind::Add, it, p)?;
                    self.cast(idx, DType::I32)
                } else {
                    let p = self.cast(pos, DType::I32);
                    self.g.reshape(p, &[1]).map_err(err)?
                };
                let gather = |b: &mut Self, tab: TensorId| -> Result<TensorId> {
                    let w = b.g.tensor(tab).shape[1];
                    let r = b.g.node(dt, &[tq, w], |out| Op::Gather { x: tab, indices: pos, axis: 0, out });
                    b.g.reshape(r, &[1, 1, tq, w]).map_err(err)
                };
                let c = gather(self, cos)?;
                let sn = gather(self, sin)?;
                let slice = |b: &mut Self, from: i32, to: i32| -> TensorId {
                    let mut os = xs.clone();
                    os[3] = (to - from) as u32;
                    let end = vec![xs[0] as i32, xs[1] as i32, xs[2] as i32, to];
                    b.g.node(dt, &os, |out| Op::Slice { x, begin: vec![0, 0, 0, from], end, stride: vec![1; 4], out })
                };
                let intel = crate::info().is_some_and(|i| i.model.vendor == synaptix_npu::Vendor::Intel);
                let xr = if rd == hd { x } else { slice(self, 0, rd) };
                let rot = if intel && rd == hd {
                    // Поворот половин — MatMul на постоянную матрицу: срезы и Concat
                    // по последней оси компилятор Intel NPU на части форм не собирает
                    // (K [1, 2, 32, 128] — ZE_RESULT_ERROR_INVALID_ARGUMENT).
                    let d = rd as usize;
                    let mut r = vec![0f32; d * d];
                    for i in 0..d / 2 {
                        r[(i + d / 2) * d + i] = -1.0;
                        r[i * d + i + d / 2] = 1.0;
                    }
                    let bytes: Vec<u8> = if dt == DType::F16 {
                        r.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
                    } else {
                        r.iter().flat_map(|v| v.to_le_bytes()).collect()
                    };
                    let rw = self.g.weight(dt, &[d as u32, d as u32], &bytes).map_err(err)?;
                    self.g.matmul(x, rw, false, false).map_err(err)?
                } else {
                    let x1 = slice(self, 0, half);
                    let x2 = slice(self, half, rd);
                    let nx2 = self.g.unary(UnaryKind::Neg, x2);
                    let mut hs = xs.clone();
                    hs[3] = rd as u32;
                    self.g.node(dt, &hs, |out| Op::Concat { xs: vec![nx2, x1], axis: 3, out })
                };
                let a = self.bin(BinaryKind::Mul, xr, c)?;
                let b2 = self.bin(BinaryKind::Mul, rot, sn)?;
                let y = self.bin(BinaryKind::Add, a, b2)?;
                if rd == hd {
                    y
                } else {
                    let rest = slice(self, rd, hd);
                    self.g.node(dt, &xs, |out| Op::Concat { xs: vec![y, rest], axis: 3, out })
                }
            }
            NodeOp::Conv2d { stride, padding, bias } => {
                let x = self.input(&n.ins[0], true)?;
                let w = self.input(&n.ins[1], true)?;
                let b = if *bias { Some(self.input(&n.ins[2], true)?) } else { None };
                let (s, p) = (*stride, *padding);
                self.g.node(odt, &oshape, |out| Op::Conv2d {
                    x,
                    w,
                    bias: b,
                    stride: [s.0 as u32, s.1 as u32],
                    pad: [[p.0 as u32, p.0 as u32], [p.1 as u32, p.1 as u32]],
                    dilation: [1, 1],
                    groups: 1,
                    out,
                })
            }
        };
        if self.g.tensor(t).dtype != odt {
            return Err(SynaptixError::Other(format!("NPU: тип выхода {:?} вместо {odt:?}", self.g.tensor(t).dtype)));
        }
        Ok(self.reshape_to(t, &n.out_lo))
    }
}

fn budget() -> (usize, usize, usize) {
    let mb = std::env::var("SYNAPTIX_NPU_SEGMENT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or((crate::npu().info().model.session_bytes / 16 >> 20) as usize);
    let nodes = std::env::var("SYNAPTIX_NPU_SEGMENT_NODES").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let melems: usize = std::env::var("SYNAPTIX_NPU_SEGMENT_MELEMS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    (mb << 20, nodes, melems * 1_000_000)
}

fn new_builder() -> Builder {
    Builder {
        g: Graph::new(),
        static_bytes: 0,
        act_elems: 0,
        by_serial: HashMap::new(),
        ready: HashMap::new(),
        bind_in: Vec::new(),
        used: Vec::new(),
    }
}

pub fn flush() -> Result<()> {
    let nodes = std::mem::take(&mut *REC.lock());
    if nodes.is_empty() {
        return Ok(());
    }
    let mut internal: HashMap<u64, usize> = HashMap::new();
    let mut last_use: HashMap<u64, usize> = HashMap::new();
    for (k, n) in nodes.iter().enumerate() {
        for i in &n.ins {
            *internal.entry(i.v.id).or_default() += 1;
            last_use.insert(i.v.id, k);
        }
    }
    let mut outv: Vec<Option<ValueRef>> = Vec::with_capacity(nodes.len());
    let mut ext: Vec<bool> = Vec::with_capacity(nodes.len());
    for n in &nodes {
        let v = n.out.upgrade().filter(|v| matches!(*v.state.lock(), State::Pending(s) if s == n.serial));
        let e = v.as_ref().is_some_and(|v| Arc::strong_count(v) - 1 - internal.get(&v.id).copied().unwrap_or(0) > 0);
        ext.push(e);
        outv.push(v);
    }

    let (max_bytes, max_nodes, max_elems) = budget();
    let mut b = new_builder();
    let mut start = 0;
    for k in 0..nodes.len() {
        let before = b.g.tensors.len();
        let t = b.lower(&nodes[k]).map_err(|e| {
            if std::env::var_os("SYNAPTIX_NPU_LOG").is_some() {
                eprintln!("npu: узел {} ({}), сегмент с {start}: {e}", nodes[k].serial, k);
                for i in &nodes[k].ins {
                    if let State::Pending(s) = &*i.v.state.lock() {
                        let at = nodes.iter().position(|n| n.serial == *s);
                        eprintln!("  вход ожидает узел {s}: позиция {at:?}, out жив {:?}", at.map(|a| outv[a].is_some()));
                    }
                }
            }
            e
        })?;
        b.act_elems += b.g.tensors[before..]
            .iter()
            .filter(|d| d.kind == synaptix_npu::TensorKind::Internal)
            .map(|d| d.elems())
            .sum::<usize>();
        let t = if b.g.tensor(t).kind != synaptix_npu::TensorKind::Internal { view::fresh(&mut b.g, t) } else { t };
        b.by_serial.insert(nodes[k].serial, t);
        let last = k + 1 == nodes.len();
        if last || b.static_bytes >= max_bytes || b.g.ops.len() >= max_nodes || b.act_elems >= max_elems {
            let mut outs = Vec::new();
            for i in start..=k {
                if let Some(v) = &outv[i] {
                    if ext[i] || last_use.get(&v.id).is_some_and(|&u| u > k) {
                        outs.push((b.by_serial[&nodes[i].serial], v.clone()));
                    }
                }
            }
            run_segment(std::mem::replace(&mut b, new_builder()), outs, k + 1 - start)?;
            start = k + 1;
        }
    }
    drop(nodes);
    Ok(())
}

fn run_segment(mut b: Builder, live: Vec<(TensorId, ValueRef)>, n_nodes: usize) -> Result<()> {
    if live.is_empty() {
        return Ok(());
    }
    let mut out_ids: Vec<(TensorId, ValueRef)> = Vec::new();
    for (t, v) in live {
        let t = if out_ids.iter().any(|(x, _)| *x == t) { view::fresh(&mut b.g, t) } else { t };
        b.g.output(t).map_err(err)?;
        out_ids.push((t, v));
    }

    let npu = crate::npu();
    let hash = b.g.structure_hash();
    let log = std::env::var_os("SYNAPTIX_NPU_LOG").is_some();
    let t_seg = std::time::Instant::now();
    let cached = GRAPHS.lock().get(hash);
    let was_cached = cached.is_some();
    let cg = match cached {
        Some(c) => {
            STATS.lock().cache_hits += 1;
            c
        }
        None => {
            let t0 = std::time::Instant::now();
            let c: Arc<dyn CompiledGraph> = Arc::from(npu.compile("synaptix", &b.g).map_err(err)?);
            let mut s = STATS.lock();
            s.compiles += 1;
            s.compile_ms += t0.elapsed().as_secs_f64() * 1e3;
            drop(s);
            GRAPHS.lock().insert(hash, c.clone(), b.static_bytes);
            c
        }
    };

    let mut out_bufs = Vec::with_capacity(out_ids.len());
    for (t, v) in &out_ids {
        let need = b.g.tensor(*t).bytes();
        if need != v.bytes {
            return Err(SynaptixError::Other(format!("NPU: выход {need} байт вместо {}", v.bytes)));
        }
        out_bufs.push(Arc::new(npu.alloc(need.max(64)).map_err(err)?));
    }
    let mut binds: Vec<Binding> = b.bind_in.iter().map(|(t, buf)| Binding { tensor: *t, buffer: buf, offset: 0 }).collect();
    for ((t, _), buf) in out_ids.iter().zip(&out_bufs) {
        binds.push(Binding { tensor: *t, buffer: buf, offset: 0 });
    }
    let t_prep = t_seg.elapsed();
    let t0 = std::time::Instant::now();
    cg.run(&binds).map_err(err)?;
    if log {
        eprintln!(
            "npu: сегмент {:016x}: {} узлов, веса {:.0} МБ, {} {:.0} мс, исполнение {:.1} мс",
            hash,
            n_nodes,
            b.static_bytes as f64 / 1048576.0,
            if was_cached { "в памяти" } else if cg.from_cache() { "открыт из кэша" } else { "собран" },
            t_prep.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    {
        let mut s = STATS.lock();
        s.flushes += 1;
        s.nodes += n_nodes as u64;
        s.exec_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    for ((_, v), buf) in out_ids.iter().zip(out_bufs) {
        *v.state.lock() = State::Ready(Ready { buf, origin: Origin::Graph, uses: 0, fingerprint: None });
    }
    for v in &b.used {
        if let State::Ready(r) = &mut *v.state.lock() {
            r.uses += 1;
        }
    }
    Ok(())
}

pub fn ensure_ready(v: &ValueRef) -> Result<()> {
    if matches!(*v.state.lock(), State::Pending(_)) {
        flush()?;
    }
    match &*v.state.lock() {
        State::Ready(_) => Ok(()),
        State::Pending(_) => Err(SynaptixError::Other("NPU: значение не вычислено после сброса графа".into())),
        State::Unset => Err(SynaptixError::Other("NPU: чтение незаписанного тензора".into())),
    }
}

pub fn read(v: &ValueRef) -> Result<Vec<u8>> {
    ensure_ready(v)?;
    let st = v.state.lock();
    let State::Ready(r) = &*st else { unreachable!() };
    r.buf.cpu_begin();
    let out = r.buf.as_slice()[..v.bytes].to_vec();
    r.buf.cpu_end();
    Ok(out)
}

pub fn origin_of(v: &ValueRef) -> Option<Origin> {
    match &*v.state.lock() {
        State::Ready(r) => Some(r.origin),
        _ => None,
    }
}

pub fn set_ready(v: &ValueRef, bytes: &[u8], origin: Origin) -> Result<()> {
    let npu = crate::npu();
    let mut buf = npu.alloc(bytes.len().max(64)).map_err(err)?;
    buf.cpu_begin();
    buf.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
    buf.cpu_end();
    *v.state.lock() = State::Ready(Ready { buf: Arc::new(buf), origin, uses: 0, fingerprint: None });
    Ok(())
}

pub fn new_ready(bytes: &[u8], origin: Origin) -> Result<ValueRef> {
    let v = Value::new(bytes.len(), State::Unset);
    set_ready(&v, bytes, origin)?;
    Ok(v)
}
