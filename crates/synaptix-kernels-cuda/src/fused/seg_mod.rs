//! Слитые ядра DiT с сегментной модуляцией (MiniMax-H3) и qk-нормой:
//! `gate_residual_seg` (out = x + y·g[ridx], один запуск на весь тензор) и
//! `qk_norm_rope` (RMS-норма головы с весом в F32 + RoPE split-partial,
//! вход и выход T — без F32-проходов). Бит-в-бит с прежними цепочками.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct SegModKernels {
    _module: Arc<CudaModule>,
    gate_bf16: CudaFunction,
    gate_f16: CudaFunction,
    qk_bf16: CudaFunction,
    qk_f16: CudaFunction,
    smq_bf16: CudaFunction,
    smq_f16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<SegModKernels>)>>> = OnceLock::new();

impl SegModKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let src = include_str!("../cu/fused/norm/seg_mod.cu");
        let module = compile_module_with_opts(ctx, src, "seg_mod.cu", &[], None)?;
        let new = Arc::new(Self {
            gate_bf16: load_fn(&module, "gate_residual_seg_bf16")?,
            gate_f16: load_fn(&module, "gate_residual_seg_f16")?,
            qk_bf16: load_fn(&module, "qk_norm_rope_bf16")?,
            qk_f16: load_fn(&module, "qk_norm_rope_f16")?,
            smq_bf16: load_fn(&module, "silu_mul_quant_mxfp8_bf16")?,
            smq_f16: load_fn(&module, "silu_mul_quant_mxfp8_f16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn view<'a, T: cudarc::driver::DeviceRepr>(s: &'a CudaSlice<u8>, off: usize, n: usize, what: &str) -> Result<cudarc::driver::CudaView<'a, T>> {
    let esz = std::mem::size_of::<T>();
    unsafe { s.slice(off..off + n * esz).transmute::<T>(n) }.ok_or_else(|| SynaptixError::Cuda(format!("seg_mod: transmute {what}")))
}

/// `out[m, d] = x + round(y · g[ridx[row]])`. Все буферы — байтовые, смещения в байтах.
#[allow(clippy::too_many_arguments)]
pub fn gate_residual_seg(
    k: &SegModKernels,
    stream: &Arc<CudaStream>,
    (x, x_off): (&CudaSlice<u8>, usize),
    (y, y_off): (&CudaSlice<u8>, usize),
    (g, g_off, g_rows): (&CudaSlice<u8>, usize, usize),
    (ridx, r_off): (&CudaSlice<u8>, usize),
    out: &mut CudaSlice<u8>,
    m: usize,
    d: usize,
    dtype: DType,
) -> Result<()> {
    if d % 8 != 0 {
        return Err(SynaptixError::Unsupported("gate_residual_seg: D%8 != 0"));
    }
    let numel = m * d;
    if numel == 0 {
        return Ok(());
    }
    let threads = numel.div_ceil(8);
    let cfg = LaunchConfig { grid_dim: (threads.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let r = view::<i32>(ridx, r_off, m, "ridx")?;
    let (numel_l, d_i) = (numel as i64, d as i32);
    macro_rules! go {
        ($t:ty, $f:expr) => {{
            let xv = view::<$t>(x, x_off, numel, "x")?;
            let yv = view::<$t>(y, y_off, numel, "y")?;
            let gv = view::<$t>(g, g_off, g_rows * d, "g")?;
            let mut os = out.slice_mut(0..numel * 2);
            let mut ov = unsafe { os.transmute_mut::<$t>(numel) }.ok_or_else(|| SynaptixError::Cuda("seg_mod: transmute out".into()))?;
            let mut b = stream.launch_builder($f);
            b.arg(&xv).arg(&yv).arg(&gv).arg(&r).arg(&mut ov).arg(&numel_l).arg(&d_i);
            unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch gate_residual_seg: {e:?}")))?;
        }};
    }
    match dtype {
        DType::BF16 => go!(half::bf16, &k.gate_bf16),
        DType::F16 => go!(half::f16, &k.gate_f16),
        _ => return Err(SynaptixError::Unsupported("gate_residual_seg: dtype")),
    }
    Ok(())
}

/// Голова `[rows = n·heads, d]` T → RMS-норма (вес F32 `[d]`) + RoPE (таблицы
/// F32 `[S, rot_dim/2]`, позиция `pos_off + row / heads`) → T.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    k: &SegModKernels,
    stream: &Arc<CudaStream>,
    (x, x_off): (&CudaSlice<u8>, usize),
    (w, w_off): (&CudaSlice<u8>, usize),
    (cos, cos_off): (&CudaSlice<u8>, usize),
    (sin, sin_off): (&CudaSlice<u8>, usize),
    table_rows: usize,
    out: &mut CudaSlice<u8>,
    rows: usize,
    heads: usize,
    d: usize,
    rot_dim: usize,
    pos_off: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    if d % 32 != 0 || d > 256 || rot_dim > d || rot_dim % 2 != 0 {
        return Err(SynaptixError::Unsupported("qk_norm_rope: D%32, D≤256, rot_dim"));
    }
    if rows == 0 {
        return Ok(());
    }
    let n = rows * d;
    let half = rot_dim / 2;
    let wv = view::<f32>(w, w_off, d, "w")?;
    let cv = view::<f32>(cos, cos_off, table_rows * half, "cos")?;
    let sv = view::<f32>(sin, sin_off, table_rows * half, "sin")?;
    let cfg = LaunchConfig { grid_dim: (rows.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let (rows_l, h, dd, rd, po) = (rows as i64, heads as i32, d as i32, rot_dim as i32, pos_off as i32);
    macro_rules! go {
        ($t:ty, $f:expr) => {{
            let xv = view::<$t>(x, x_off, n, "x")?;
            let mut os = out.slice_mut(0..n * 2);
            let mut ov = unsafe { os.transmute_mut::<$t>(n) }.ok_or_else(|| SynaptixError::Cuda("seg_mod: transmute out".into()))?;
            let mut b = stream.launch_builder($f);
            b.arg(&xv).arg(&wv).arg(&cv).arg(&sv).arg(&mut ov).arg(&rows_l).arg(&h).arg(&dd).arg(&rd).arg(&po).arg(&eps);
            unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch qk_norm_rope: {e:?}")))?;
        }};
    }
    match dtype {
        DType::BF16 => go!(half::bf16, &k.qk_bf16),
        DType::F16 => go!(half::f16, &k.qk_f16),
        _ => return Err(SynaptixError::Unsupported("qk_norm_rope: dtype")),
    }
    Ok(())
}

/// `h [m, 2F]` = `[gate | up]` (T = F16|BF16) → MXFP8 `silu(gate)·up`:
/// e4m3 `[m, F]` + natural E8M0 `[m, F/32]`.
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_quant_mxfp8(
    k: &SegModKernels,
    stream: &Arc<CudaStream>,
    (h, h_off): (&CudaSlice<u8>, usize),
    packed: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<u8>,
    m: usize,
    f: usize,
    dtype: DType,
) -> Result<()> {
    if f % 32 != 0 {
        return Err(SynaptixError::Unsupported("silu_mul_quant_mxfp8: F%32 != 0"));
    }
    let blocks = m * f / 32;
    if blocks == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (blocks.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let (mi, fi) = (m as i32, f as i32);
    macro_rules! go {
        ($t:ty, $fun:expr) => {{
            let hv = view::<$t>(h, h_off, m * 2 * f, "h")?;
            let mut b = stream.launch_builder($fun);
            b.arg(&hv).arg(&mut *packed).arg(&mut *scales).arg(&mi).arg(&fi);
            unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch silu_mul_quant_mxfp8: {e:?}")))?;
        }};
    }
    match dtype {
        DType::BF16 => go!(half::bf16, &k.smq_bf16),
        DType::F16 => go!(half::f16, &k.smq_f16),
        _ => return Err(SynaptixError::Unsupported("silu_mul_quant_mxfp8: dtype")),
    }
    Ok(())
}
