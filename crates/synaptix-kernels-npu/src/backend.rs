use crate::record::{self, Input, NodeOp};
use crate::value::{self, Origin, State, Value, ValueRef};
use synaptix_core::backend::{param_upload_active, Backend, BinaryOp, ReduceOp, UnaryOp};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CpuBuf, Storage};

pub struct NpuBackend {
    pub ordinal: usize,
}

fn cpu() -> &'static dyn Backend {
    synaptix_kernels_cpu::cpu_backend()
}

fn full(lo: &Layout, v: &ValueRef) -> bool {
    lo.is_contiguous() && lo.offset() == 0 && lo.dtype().bytes_for_numel(lo.numel()) == v.bytes
}

fn ok_dtype(lo: &Layout) -> bool {
    record::npu_dtype(lo.dtype()).is_some()
}

fn float(lo: &Layout) -> bool {
    matches!(lo.dtype(), DType::F32 | DType::F16)
}

fn inp(s: &Storage, lo: &Layout) -> Result<Input> {
    Ok(Input { v: value::value_of(s)?, lo: lo.clone() })
}

fn cpu_storage(s: &Storage) -> Result<Storage> {
    let v = value::value_of(s)?;
    Ok(Storage::Cpu(CpuBuf::from_vec(record::read(&v)?)))
}

fn cpu_dst(s: &Storage) -> Result<Storage> {
    let v = value::value_of(s)?;
    if matches!(*v.state.lock(), State::Unset) {
        return Ok(Storage::Cpu(CpuBuf::alloc_zeros(v.bytes)));
    }
    cpu_storage(s)
}

fn store(dst: &Storage, cpu: Storage, origin: Origin) -> Result<()> {
    let v = value::value_of(dst)?;
    let Storage::Cpu(b) = cpu else { unreachable!() };
    record::set_ready(&v, &b.as_bytes()[..v.bytes], origin)
}

fn host_origin(srcs: &[&Storage]) -> Option<Origin> {
    let mut o = Origin::Host;
    for s in srcs {
        let v = value::value_of(s).ok()?;
        match record::origin_of(&v)? {
            Origin::Graph => return None,
            Origin::Param => o = Origin::Param,
            Origin::Host => {}
        }
    }
    Some(o)
}

macro_rules! on_cpu {
    ($dst:expr, $origin:expr, |$c:ident, $d:ident| $call:expr) => {{
        record::note_cpu_fallback();
        let mut $d = cpu_dst($dst)?;
        let $c = cpu();
        $call?;
        store($dst, $d, $origin)
    }};
}

impl NpuBackend {
    fn rec(&self, op: NodeOp, ins: Vec<Input>, dst: (&mut Storage, &Layout)) -> Result<()> {
        let v = value::value_of(dst.0)?;
        record::record(op, ins, &v, dst.1.clone())
    }
}

impl Backend for NpuBackend {
    fn device_kind(&self) -> Device {
        Device::Npu(self.ordinal)
    }

    fn alloc_zeros(&self, n_bytes: usize, _device: Device) -> Result<Storage> {
        let v = record::new_ready(&vec![0u8; n_bytes], Origin::Host)?;
        Ok(value::storage(v, self.ordinal))
    }

    fn alloc_uninit(&self, n_bytes: usize, _device: Device) -> Result<Storage> {
        Ok(value::storage(Value::new(n_bytes, State::Unset), self.ordinal))
    }

    fn upload(&self, bytes: &[u8], _device: Device) -> Result<Storage> {
        let origin = if param_upload_active() { Origin::Param } else { Origin::Host };
        let v = record::new_ready(bytes, origin)?;
        Ok(value::storage(v, self.ordinal))
    }

    fn download(&self, src: &Storage) -> Result<Vec<u8>> {
        record::read(&value::value_of(src)?)
    }

    fn copy(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        let origin = host_origin(&[src.0]);
        if !full(dst.1, &dv) || !ok_dtype(src.1) || origin.is_some() {
            let cs = cpu_storage(src.0)?;
            let o = origin.unwrap_or(Origin::Graph);
            return on_cpu!(dst.0, o, |c, d| c.copy((&cs, src.1), (&mut d, dst.1), &Stream::CpuNoop));
        }
        self.rec(NodeOp::Copy, vec![inp(src.0, src.1)?], dst)
    }

    fn cast(&self, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        let origin = host_origin(&[src.0]);
        if !full(dst.1, &dv) || !ok_dtype(src.1) || !ok_dtype(dst.1) || origin.is_some() {
            let cs = cpu_storage(src.0)?;
            let o = origin.unwrap_or(Origin::Graph);
            return on_cpu!(dst.0, o, |c, d| c.cast((&cs, src.1), (&mut d, dst.1), &Stream::CpuNoop));
        }
        self.rec(NodeOp::Cast, vec![inp(src.0, src.1)?], dst)
    }

    fn unary(&self, op: UnaryOp, src: (&Storage, &Layout), dst: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        let supported = !matches!(op, UnaryOp::Erf | UnaryOp::StepGtZero)
            && !matches!(op, UnaryOp::LeakyRelu(a) if a > 1.0);
        if !full(dst.1, &dv) || !float(src.1) || !supported {
            let cs = cpu_storage(src.0)?;
            return on_cpu!(dst.0, Origin::Graph, |c, d| c.unary(op, (&cs, src.1), (&mut d, dst.1), &Stream::CpuNoop));
        }
        self.rec(NodeOp::Unary(op), vec![inp(src.0, src.1)?], dst)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        if !full(dst.1, &dv) || !float(a.1) || !float(b.1) || a.1.dtype() != b.1.dtype() {
            let (ca, cb) = (cpu_storage(a.0)?, cpu_storage(b.0)?);
            return on_cpu!(dst.0, Origin::Graph, |c, d| c.binary(
                op,
                (&ca, a.1),
                (&cb, b.1),
                (&mut d, dst.1),
                &Stream::CpuNoop
            ));
        }
        self.rec(NodeOp::Binary(op), vec![inp(a.0, a.1)?, inp(b.0, b.1)?], dst)
    }

    fn matmul(
        &self,
        lhs: (&Storage, &Layout),
        rhs: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        if !full(dst.1, &dv) || !float(lhs.1) || !float(rhs.1) {
            let (ca, cb) = (cpu_storage(lhs.0)?, cpu_storage(rhs.0)?);
            return on_cpu!(dst.0, Origin::Graph, |c, d| c.matmul((&ca, lhs.1), (&cb, rhs.1), (&mut d, dst.1), &Stream::CpuNoop));
        }
        self.rec(NodeOp::MatMul, vec![inp(lhs.0, lhs.1)?, inp(rhs.0, rhs.1)?], dst)
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        dims: &[usize],
        _stream: &Stream,
    ) -> Result<()> {
        let dv = value::value_of(dst.0)?;
        if !full(dst.1, &dv) || !float(src.1) || (op == ReduceOp::ArgMax && dims.len() != 1) || dims.is_empty() {
            let cs = cpu_storage(src.0)?;
            return on_cpu!(dst.0, Origin::Graph, |c, d| c.reduce(op, (&cs, src.1), (&mut d, dst.1), dims, &Stream::CpuNoop));
        }
        self.rec(NodeOp::Reduce(op, dims.to_vec()), vec![inp(src.0, src.1)?], dst)
    }

    fn linear(&self, x: (&Storage, &Layout), w: (&Storage, &Layout), out: (&mut Storage, &Layout), _stream: &Stream) -> Result<()> {
        if !float(x.1) || x.1.dtype() != w.1.dtype() || w.1.rank() != 2 {
            return Err(SynaptixError::Unsupported("linear на NPU: тип"));
        }
        self.rec(NodeOp::Linear { bias: false, residual: false }, vec![inp(x.0, x.1)?, inp(w.0, w.1)?], out)
    }

    fn linear_epilogue(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        residual: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        if !float(x.1) || x.1.dtype() != w.1.dtype() || w.1.rank() != 2 {
            return Err(SynaptixError::Unsupported("linear на NPU: тип"));
        }
        if let Some(b) = bias {
            if b.1.dtype() != x.1.dtype() || b.1.dims() != [w.1.dims()[0]] {
                return Err(SynaptixError::Unsupported("linear на NPU: bias"));
            }
        }
        let mut ins = vec![inp(x.0, x.1)?, inp(w.0, w.1)?];
        if let Some(b) = bias {
            ins.push(inp(b.0, b.1)?);
        }
        if let Some(r) = residual {
            if r.1.dtype() != x.1.dtype() {
                return Err(SynaptixError::Unsupported("linear на NPU: residual"));
            }
            ins.push(inp(r.0, r.1)?);
        }
        self.rec(NodeOp::Linear { bias: bias.is_some(), residual: residual.is_some() }, ins, out)
    }

    fn layer_norm(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        eps: f32,
        _stream: &Stream,
    ) -> Result<()> {
        if !float(x.1) || w.1.dtype() != x.1.dtype() || bias.is_some_and(|b| b.1.dtype() != x.1.dtype()) {
            return Err(SynaptixError::Unsupported("layer_norm на NPU: тип"));
        }
        let mut ins = vec![inp(x.0, x.1)?, inp(w.0, w.1)?];
        if let Some(b) = bias {
            ins.push(inp(b.0, b.1)?);
        }
        self.rec(NodeOp::LayerNorm { eps, bias: bias.is_some() }, ins, out)
    }

    fn rms_norm(
        &self,
        x: (&Storage, &Layout),
        w: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        eps: f32,
        qwen_gain: bool,
        _stream: &Stream,
    ) -> Result<()> {
        if !float(x.1) || w.1.dtype() != x.1.dtype() {
            return Err(SynaptixError::Unsupported("rms_norm на NPU: тип"));
        }
        self.rec(NodeOp::RmsNorm { eps, qwen: qwen_gain }, vec![inp(x.0, x.1)?, inp(w.0, w.1)?], out)
    }

    fn flash_attention(
        &self,
        q: (&Storage, &Layout),
        k: (&Storage, &Layout),
        v: (&Storage, &Layout),
        out: (&mut Storage, &Layout),
        scale: f32,
        causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        let same = q.1.rank() == 4 && k.1.rank() == 4 && v.1.rank() == 4 && q.1.dims()[1] == k.1.dims()[1];
        if !float(q.1) || k.1.dtype() != q.1.dtype() || v.1.dtype() != q.1.dtype() || !same {
            return Err(SynaptixError::Unsupported("attention на NPU"));
        }
        self.rec(NodeOp::Attention { scale, causal }, vec![inp(q.0, q.1)?, inp(k.0, k.1)?, inp(v.0, v.1)?], out)
    }

    fn conv2d(
        &self,
        input: (&Storage, &Layout),
        weight: (&Storage, &Layout),
        bias: Option<(&Storage, &Layout)>,
        out: (&mut Storage, &Layout),
        stride: (usize, usize),
        padding: (usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        let groups_ok = input.1.rank() == 4 && weight.1.rank() == 4 && input.1.dims()[1] == weight.1.dims()[1];
        if !float(input.1) || weight.1.dtype() != input.1.dtype() || !groups_ok {
            return Err(SynaptixError::Unsupported("conv2d на NPU"));
        }
        let mut ins = vec![inp(input.0, input.1)?, inp(weight.0, weight.1)?];
        if let Some(b) = bias {
            ins.push(inp(b.0, b.1)?);
        }
        self.rec(NodeOp::Conv2d { stride, padding, bias: bias.is_some() }, ins, out)
    }
}
