use crate::buffer::{NpuBuffer, Release};
use crate::graph::{DType, Graph, Op, Quant, TensorId, TensorKind};
use crate::models::{self, Vendor};
use crate::{Binding, CompiledGraph, IoTensor, Npu, NpuError, NpuInfo, Result};
use parking_lot::Mutex;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::Duration;

pub const SOCKET: &str = "/run/synnpu/qnn.sock";
const MAGIC: u32 = 0x5550_4e53;
const VERSION: u32 = 1;
const REPLY: u32 = 0x8000_0000;

const OP_HELLO: u32 = 1;
const OP_MEM_REGISTER: u32 = 2;
const OP_MEM_FREE: u32 = 3;
const OP_GRAPH_OPEN: u32 = 4;
const OP_GRAPH_BUILD: u32 = 5;
const OP_GRAPH_FREE: u32 = 6;
const OP_EXECUTE: u32 = 7;

const ST_NOT_FOUND: u32 = 1;

const KIND_INPUT: u8 = 0;
const KIND_OUTPUT: u8 = 1;
const KIND_STATIC: u8 = 2;
const KIND_NATIVE: u8 = 3;
const PARAM_SCALAR: u8 = 0;
const BUILD_FP16: u32 = 1;

const QNN_F32: u32 = 0x0232;
const QNN_F16: u32 = 0x0216;
const QNN_I32: u32 = 0x0032;
const QNN_U32: u32 = 0x0132;
const QNN_BOOL: u32 = 0x0508;
const QNN_UFX8: u32 = 0x0408;
const QNN_UFX16: u32 = 0x0416;
const QNN_SFX8: u32 = 0x0308;
const QNN_SFX4: u32 = 0x0304;
const QNN_U8: u32 = 0x0108;
const QNN_U16: u32 = 0x0116;
const QNN_I8: u32 = 0x0008;

fn qnn_dtype(d: DType, q: &Quant) -> u32 {
    let quant = !matches!(q, Quant::None);
    match (d, quant) {
        (DType::F32, _) => QNN_F32,
        (DType::F16, _) => QNN_F16,
        (DType::I32, _) => QNN_I32,
        (DType::U8, true) => QNN_UFX8,
        (DType::U16, true) => QNN_UFX16,
        (DType::I8, true) => QNN_SFX8,
        (DType::I4, _) => QNN_SFX4,
        (DType::U8, false) => QNN_U8,
        (DType::U16, false) => QNN_U16,
        (DType::I8, false) => QNN_I8,
    }
}

fn from_qnn_dtype(d: u32) -> Result<DType> {
    Ok(match d {
        QNN_F32 => DType::F32,
        QNN_F16 => DType::F16,
        QNN_I32 => DType::I32,
        QNN_UFX8 | QNN_U8 => DType::U8,
        QNN_UFX16 | QNN_U16 => DType::U16,
        QNN_SFX8 | QNN_I8 => DType::I8,
        QNN_SFX4 => DType::I4,
        _ => return Err(NpuError::Protocol(format!("тип QNN {d:#x}"))),
    })
}

#[derive(Default)]
struct Wr(Vec<u8>);

impl Wr {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct Rd<'a> {
    b: &'a [u8],
    off: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.off + n > self.b.len() {
            return Err(NpuError::Protocol("короткий ответ".into()));
        }
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

struct Conn {
    sock: UnixStream,
    seq: u32,
}

impl Conn {
    fn call(&mut self, op: u32, body: &[u8], fd: Option<RawFd>) -> Result<std::result::Result<Vec<u8>, (u32, String)>> {
        self.seq = self.seq.wrapping_add(1);
        let mut hdr = Wr::default();
        hdr.u32(MAGIC);
        hdr.u32(op);
        hdr.u32(self.seq);
        hdr.u32(0);
        hdr.u64(body.len() as u64);
        send_with_fd(&self.sock, &hdr.0, fd)?;
        self.sock.write_all(body)?;
        let mut rh = [0u8; 24];
        self.sock.read_exact(&mut rh)?;
        let mut r = Rd { b: &rh, off: 0 };
        let (magic, rop, rseq, _, len) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u64()?);
        if magic != MAGIC || rop != op | REPLY || rseq != self.seq {
            return Err(NpuError::Protocol(format!("ответ {rop:#x}/{rseq} на {op}/{}", self.seq)));
        }
        let mut b = vec![0u8; len as usize];
        self.sock.read_exact(&mut b)?;
        let mut r = Rd { b: &b, off: 0 };
        let status = r.u32()?;
        let msg = r.str()?;
        if status != 0 {
            return Ok(Err((status, msg)));
        }
        Ok(Ok(b[r.off..].to_vec()))
    }
}

fn send_with_fd(sock: &UnixStream, data: &[u8], fd: Option<RawFd>) -> Result<()> {
    let mut iov = libc::iovec { iov_base: data.as_ptr() as *mut _, iov_len: data.len() };
    let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    let mut cbuf = vec![0u8; space];
    let mut m: libc::msghdr = unsafe { std::mem::zeroed() };
    m.msg_iov = &mut iov;
    m.msg_iovlen = 1;
    if let Some(fd) = fd {
        m.msg_control = cbuf.as_mut_ptr() as *mut _;
        m.msg_controllen = space as _;
        unsafe {
            let c = libc::CMSG_FIRSTHDR(&m);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
            std::ptr::write_unaligned(libc::CMSG_DATA(c) as *mut RawFd, fd);
        }
    }
    let n = unsafe { libc::sendmsg(sock.as_raw_fd(), &m, libc::MSG_NOSIGNAL) };
    if n < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    if (n as usize) < data.len() {
        let mut s = sock;
        s.write_all(&data[n as usize..])?;
    }
    Ok(())
}

type Shared = Arc<Mutex<Conn>>;

fn call(conn: &Shared, op: u32, body: &[u8], fd: Option<RawFd>) -> Result<Vec<u8>> {
    conn.lock().call(op, body, fd)?.map_err(|(_, msg)| NpuError::Driver { driver: "qualcomm", msg })
}

pub struct QnnNpu {
    conn: Shared,
    info: NpuInfo,
}

impl QnnNpu {
    pub fn connect() -> Result<Self> {
        let path = std::env::var("SYNNPU_QNN_SOCKET").unwrap_or_else(|_| SOCKET.into());
        let sock = UnixStream::connect(&path).map_err(|e| NpuError::Unavailable(format!("{path}: {e}")))?;
        let conn = Arc::new(Mutex::new(Conn { sock, seq: 0 }));
        let b = call(&conn, OP_HELLO, &[], None)?;
        let mut r = Rd { b: &b, off: 0 };
        let (ver, arch, soc, vtcm) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        let runtime = r.str()?;
        if ver != VERSION {
            return Err(NpuError::Protocol(format!("версия демона {ver}, ждём {VERSION}")));
        }
        let model = models::find(Vendor::Qualcomm, arch)
            .ok_or_else(|| NpuError::Unavailable(format!("Hexagon HTP v{arch}: нет профиля модели NPU")))?;
        Ok(Self { conn, info: NpuInfo { model, soc, vtcm_mb: vtcm, runtime } })
    }

    fn register(&self, mut b: NpuBuffer) -> Result<NpuBuffer> {
        let mut w = Wr::default();
        w.u64(b.len() as u64);
        let r = call(&self.conn, OP_MEM_REGISTER, &w.0, Some(b.fd()))?;
        let handle = Rd { b: &r, off: 0 }.u32()?;
        let conn = Arc::downgrade(&self.conn);
        let release: Release = Arc::new(move |h| {
            if let Some(c) = conn.upgrade() {
                let _ = c.lock().call(OP_MEM_FREE, &h.to_le_bytes(), None);
            }
        });
        b.attach(handle, release);
        Ok(b)
    }

    fn check(&self, g: &Graph) -> Result<()> {
        let m = self.info.model;
        for t in &g.tensors {
            let ok = match t.kind {
                TensorKind::Weight { .. } => m.supports_weight(t.dtype),
                _ => m.supports_activation(t.dtype),
            };
            if !ok {
                return Err(NpuError::Unsupported(format!("тип {:?} тензора {}", t.dtype, t.id.0), m.id));
            }
        }
        Ok(())
    }

    fn lower(&self, key: &str, g: &Graph, weights: Option<&NpuBuffer>) -> Vec<u8> {
        let mut w = Wr::default();
        w.str(key);
        let fp16 = g.tensors.iter().any(|t| t.dtype == DType::F16);
        w.u32(if fp16 { BUILD_FP16 } else { 0 });
        w.u32(g.tensors.len() as u32);
        for t in &g.tensors {
            w.u32(t.id.0);
            w.u8(match t.kind {
                TensorKind::Input => KIND_INPUT,
                TensorKind::Output => KIND_OUTPUT,
                TensorKind::Weight { .. } => KIND_STATIC,
                TensorKind::Internal => KIND_NATIVE,
            });
            w.u32(qnn_dtype(t.dtype, &t.quant));
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
            if let TensorKind::Weight { offset } = t.kind {
                w.u32(weights.map(|b| b.handle).unwrap_or(0));
                w.u64(offset as u64);
                w.u64(t.bytes() as u64);
            }
        }
        w.u32(g.ops.len() as u32);
        for (i, op) in g.ops.iter().enumerate() {
            let name = format!("n{i}");
            let node = |w: &mut Wr, ty: &str, ins: &[TensorId], out: TensorId, params: &[(&str, u32, u64)]| {
                w.str(&name);
                w.str("qti.aisw");
                w.str(ty);
                w.u32(ins.len() as u32);
                for t in ins {
                    w.u32(t.0);
                }
                w.u32(1);
                w.u32(out.0);
                w.u32(params.len() as u32);
                for &(n, dt, v) in params {
                    w.str(n);
                    w.u8(PARAM_SCALAR);
                    w.u32(dt);
                    w.u64(v);
                }
            };
            match *op {
                Op::Linear { x, w: wt, bias, out } => {
                    let mut ins = vec![x, wt];
                    ins.extend(bias);
                    let keep = (g.tensor(x).shape.len() > 2) as u64;
                    node(&mut w, "FullyConnected", &ins, out, &[("keep_dims", QNN_BOOL, keep)]);
                }
                Op::MatMul { a, b, transpose_a, transpose_b, out } => node(
                    &mut w,
                    "MatMul",
                    &[a, b],
                    out,
                    &[("transpose_in0", QNN_BOOL, transpose_a as u64), ("transpose_in1", QNN_BOOL, transpose_b as u64)],
                ),
                Op::Add { a, b, out } => node(&mut w, "ElementWiseAdd", &[a, b], out, &[]),
                Op::Mul { a, b, out } => node(&mut w, "ElementWiseMultiply", &[a, b], out, &[]),
                Op::Softmax { x, out } => {
                    let axis = g.tensor(x).shape.len() as u64 - 1;
                    node(&mut w, "Softmax", &[x], out, &[("axis", QNN_U32, axis), ("beta", QNN_F32, 1f32.to_bits() as u64)])
                }
                Op::Gelu { x, out } => node(&mut w, "Gelu", &[x], out, &[]),
                Op::Sigmoid { x, out } => node(&mut w, "Sigmoid", &[x], out, &[]),
            }
        }
        w.0
    }
}

fn read_io(r: &mut Rd) -> Result<Vec<IoTensor>> {
    let n = r.u32()?;
    let mut v = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let id = TensorId(r.u32()?);
        let dtype = from_qnn_dtype(r.u32()?)?;
        let rank = r.u32()?;
        let shape = (0..rank).map(|_| r.u32()).collect::<Result<Vec<_>>>()?;
        v.push(IoTensor { id, dtype, shape });
    }
    Ok(v)
}

impl Npu for QnnNpu {
    fn info(&self) -> &NpuInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<NpuBuffer> {
        self.register(NpuBuffer::new(bytes)?)
    }

    fn compile(&self, key: &str, g: &Graph) -> Result<Box<dyn CompiledGraph>> {
        self.check(g)?;
        let key = format!("{key}#{:016x}", g.structure_hash());
        let mut w = Wr::default();
        w.str(&key);
        let opened = self.conn.lock().call(OP_GRAPH_OPEN, &w.0, None)?;
        let (b, from_cache, build_ms) = match opened {
            Ok(b) => (b, true, 0),
            Err((ST_NOT_FOUND, _)) => {
                let weights = if g.weights.is_empty() {
                    None
                } else {
                    let mut wb = self.alloc(g.weights.len())?;
                    wb.cpu_begin();
                    wb.as_mut_slice()[..g.weights.len()].copy_from_slice(&g.weights);
                    wb.cpu_end();
                    Some(wb)
                };
                let body = self.lower(&key, g, weights.as_ref());
                let b = call(&self.conn, OP_GRAPH_BUILD, &body, None)?;
                let ms = {
                    let mut r = Rd { b: &b, off: 4 };
                    read_io(&mut r)?;
                    read_io(&mut r)?;
                    r.u32()?
                };
                (b, false, ms)
            }
            Err((_, msg)) => return Err(NpuError::Driver { driver: "qualcomm", msg }),
        };
        let mut r = Rd { b: &b, off: 0 };
        let id = r.u32()?;
        let inputs = read_io(&mut r)?;
        let outputs = read_io(&mut r)?;
        Ok(Box::new(QnnGraph {
            conn: self.conn.clone(),
            id,
            inputs,
            outputs,
            from_cache,
            build: Duration::from_millis(build_ms as u64),
        }))
    }
}

pub struct QnnGraph {
    conn: Shared,
    id: u32,
    inputs: Vec<IoTensor>,
    outputs: Vec<IoTensor>,
    from_cache: bool,
    build: Duration,
}

impl CompiledGraph for QnnGraph {
    fn inputs(&self) -> &[IoTensor] {
        &self.inputs
    }

    fn outputs(&self) -> &[IoTensor] {
        &self.outputs
    }

    fn from_cache(&self) -> bool {
        self.from_cache
    }

    fn build_time(&self) -> Duration {
        self.build
    }

    fn run(&self, bindings: &[Binding<'_>]) -> Result<Duration> {
        let mut w = Wr::default();
        w.u32(self.id);
        w.u32(bindings.len() as u32);
        for b in bindings {
            if b.buffer.handle == 0 {
                return Err(NpuError::Graph("буфер не зарегистрирован в NPU (Npu::alloc)".into()));
            }
            w.u32(b.tensor.0);
            w.u32(b.buffer.handle);
            w.u64(b.offset as u64);
        }
        let r = call(&self.conn, OP_EXECUTE, &w.0, None)?;
        Ok(Duration::from_micros(Rd { b: &r, off: 0 }.u64()?))
    }
}

impl Drop for QnnGraph {
    fn drop(&mut self) {
        let _ = self.conn.lock().call(OP_GRAPH_FREE, &self.id.to_le_bytes(), None);
    }
}
