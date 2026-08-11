//! Partial Rotary Position Embedding (RoPE) — F16 / BF16 / F32.
//!
//! Один CUDA launch per (Q или K) layer вместо наивной композиции
//! (narrow + cat + broadcast_mul + neg). Device-resident `start_pos` →
//! совместимо с CUDA graph replay.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct RopeKernels {
    _module: Arc<CudaModule>,
    apply_partial_f16: CudaFunction,
    apply_partial_bf16: CudaFunction,
    apply_partial_f32: CudaFunction,
    split_f16: CudaFunction,
    split_bf16: CudaFunction,
    split_f32: CudaFunction,
    split_partial_f16: CudaFunction,
    split_partial_bf16: CudaFunction,
    split_partial_f32: CudaFunction,
    interleaved_f16: CudaFunction,
    interleaved_bf16: CudaFunction,
    interleaved_f32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<RopeKernels>)>>> = OnceLock::new();

impl RopeKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("../cu/elementwise/rope.cu");
        let module = compile_module(ctx, src, "rope.cu")?;
        let new = Arc::new(Self {
            apply_partial_f16: load_fn(&module, "rope_apply_partial_f16")?,
            apply_partial_bf16: load_fn(&module, "rope_apply_partial_bf16")?,
            apply_partial_f32: load_fn(&module, "rope_apply_partial_f32")?,
            split_f16: load_fn(&module, "rope_split_f16")?,
            split_bf16: load_fn(&module, "rope_split_bf16")?,
            split_f32: load_fn(&module, "rope_split_f32")?,
            split_partial_f16: load_fn(&module, "rope_split_partial_f16")?,
            split_partial_bf16: load_fn(&module, "rope_split_partial_bf16")?,
            split_partial_f32: load_fn(&module, "rope_split_partial_f32")?,
            interleaved_f16: load_fn(&module, "rope_interleaved_f16")?,
            interleaved_bf16: load_fn(&module, "rope_interleaved_bf16")?,
            interleaved_f32: load_fn(&module, "rope_interleaved_f32")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_partial<T: DeviceRepr>(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    out: &mut CudaSlice<T>,
    cos_table: &CudaSlice<T>,
    sin_table: &CudaSlice<T>,
    start_pos_dev: &CudaSlice<u32>,
    b: u32,
    h: u32,
    t: u32,
    head_dim: u32,
    rotary_dim: u32,
    dtype: DType,
) -> Result<()> {
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "rope_apply_partial: rotary_dim={rotary_dim} must be ≤ head_dim={head_dim} and even"
        )));
    }
    if head_dim > 1024 {
        return Err(SynaptixError::Cuda(format!(
            "rope_apply_partial: head_dim={head_dim} > 1024 (max block_dim)"
        )));
    }
    let func = match dtype {
        DType::F16 => &kernels.apply_partial_f16,
        DType::BF16 => &kernels.apply_partial_bf16,
        DType::F32 => &kernels.apply_partial_f32,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "rope_apply_partial: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * h * t, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(&mut *out)
        .arg(cos_table)
        .arg(sin_table)
        .arg(start_pos_dev)
        .arg(&b)
        .arg(&h)
        .arg(&t)
        .arg(&head_dim)
        .arg(&rotary_dim);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch rope_apply: {e:?}")))?;
    }
    Ok(())
}

pub fn apply_partial_f16(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    cos_table: &CudaSlice<f16>,
    sin_table: &CudaSlice<f16>,
    start_pos_dev: &CudaSlice<u32>,
    b: u32,
    h: u32,
    t: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    apply_partial::<f16>(
        kernels,
        stream,
        x,
        out,
        cos_table,
        sin_table,
        start_pos_dev,
        b,
        h,
        t,
        head_dim,
        rotary_dim,
        DType::F16,
    )
}

pub fn apply_partial_bf16(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    cos_table: &CudaSlice<bf16>,
    sin_table: &CudaSlice<bf16>,
    start_pos_dev: &CudaSlice<u32>,
    b: u32,
    h: u32,
    t: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    apply_partial::<bf16>(
        kernels,
        stream,
        x,
        out,
        cos_table,
        sin_table,
        start_pos_dev,
        b,
        h,
        t,
        head_dim,
        rotary_dim,
        DType::BF16,
    )
}

pub fn apply_partial_f32(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    cos_table: &CudaSlice<f32>,
    sin_table: &CudaSlice<f32>,
    start_pos_dev: &CudaSlice<u32>,
    b: u32,
    h: u32,
    t: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    apply_partial::<f32>(
        kernels,
        stream,
        x,
        out,
        cos_table,
        sin_table,
        start_pos_dev,
        b,
        h,
        t,
        head_dim,
        rotary_dim,
        DType::F32,
    )
}

/// Split RoPE из untyped `u8`-storage (для `Backend::rope_split`). `x`/`out` —
/// тип `dtype` [.., S, D] (rows = numel/D); `cos`/`sin` — F32 [S, half]. Один
/// launch на (rows) строк, ротация в F32. Принимает byte-offset'ы.
#[allow(clippy::too_many_arguments)]
pub fn run_rope_split_u8(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    cos: &CudaSlice<u8>,
    cos_off: usize,
    sin: &CudaSlice<u8>,
    sin_off: usize,
    rows: u32,
    s_len: u32,
    d: u32,
    dtype: DType,
) -> Result<()> {
    if rows == 0 || d == 0 {
        return Ok(());
    }
    if d % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "rope_split: head_dim={d} must be even"
        )));
    }
    if d > 1024 {
        return Err(SynaptixError::Cuda(format!(
            "rope_split: head_dim={d} > 1024"
        )));
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (rows as usize) * (d as usize);
    let cn = (s_len as usize) * (d as usize / 2);
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (d, 1, 1),
        shared_mem_bytes: 0,
    };
    let cos_v = unsafe {
        cos.slice(cos_off..cos_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_split: transmute cos".into()))?
    };
    let sin_v = unsafe {
        sin.slice(sin_off..sin_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_split: transmute sin".into()))?
    };

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_split: transmute x".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + xn * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_split: transmute out".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&x_v)
                .arg(&mut o_v)
                .arg(&cos_v)
                .arg(&sin_v)
                .arg(&s_len)
                .arg(&d);
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch rope_split: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F16 => go!(f16, &kernels.split_f16),
        DType::BF16 => go!(bf16, &kernels.split_bf16),
        DType::F32 => go!(f32, &kernels.split_f32),
        _ => return Err(SynaptixError::Unsupported("rope_split: dtype")),
    }
    Ok(())
}

/// Partial split RoPE из untyped `u8`: вращает первые `rot_dim` из `d`, остальные
/// пропускает. `x`/`out` — `dtype` [.., S, D] (rows = numel/D, позиция = row % S);
/// `cos`/`sin` — F32 [S, rot_dim/2]. Один launch на (rows) строк, ротация в F32.
#[allow(clippy::too_many_arguments)]
pub fn run_rope_split_partial_u8(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    cos: &CudaSlice<u8>,
    cos_off: usize,
    sin: &CudaSlice<u8>,
    sin_off: usize,
    rows: u32,
    s_len: u32,
    d: u32,
    rot_dim: u32,
    dtype: DType,
) -> Result<()> {
    if rows == 0 || d == 0 {
        return Ok(());
    }
    if rot_dim % 2 != 0 || rot_dim > d {
        return Err(SynaptixError::Cuda(format!(
            "rope_split_partial: rot_dim={rot_dim} vs head_dim={d}"
        )));
    }
    if d > 1024 {
        return Err(SynaptixError::Cuda(format!(
            "rope_split_partial: head_dim={d} > 1024"
        )));
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (rows as usize) * (d as usize);
    let cn = (s_len as usize) * (rot_dim as usize / 2);
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (d, 1, 1),
        shared_mem_bytes: 0,
    };
    let cos_v = unsafe {
        cos.slice(cos_off..cos_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_split_partial: transmute cos".into()))?
    };
    let sin_v = unsafe {
        sin.slice(sin_off..sin_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_split_partial: transmute sin".into()))?
    };

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_split_partial: transmute x".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + xn * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(xn).ok_or_else(|| {
                    SynaptixError::Cuda("rope_split_partial: transmute out".into())
                })?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&x_v)
                .arg(&mut o_v)
                .arg(&cos_v)
                .arg(&sin_v)
                .arg(&s_len)
                .arg(&d)
                .arg(&rot_dim);
            unsafe {
                b.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch rope_split_partial: {e:?}"))
                })?;
            }
        }};
    }

    match dtype {
        DType::F16 => go!(f16, &kernels.split_partial_f16),
        DType::BF16 => go!(bf16, &kernels.split_partial_bf16),
        DType::F32 => go!(f32, &kernels.split_partial_f32),
        _ => return Err(SynaptixError::Unsupported("rope_split_partial: dtype")),
    }
    Ok(())
}

/// Interleaved (adjacent-pair / FLUX) RoPE из untyped `u8`. `x`/`out` — `dtype`
/// [B,S,H,D] (rows = B*S*H); `cos`/`sin` — F32 ПОЛНАЯ таблица [S,D] (repeat_inter-
/// leave(2)). Один launch, ротация в F32. Заменяет ~10 decomposed-ops apply_rope.
#[allow(clippy::too_many_arguments)]
pub fn run_rope_interleaved_u8(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    cos: &CudaSlice<u8>,
    cos_off: usize,
    sin: &CudaSlice<u8>,
    sin_off: usize,
    rows: u32,
    h: u32,
    s_len: u32,
    d: u32,
    dtype: DType,
) -> Result<()> {
    if rows == 0 || d == 0 {
        return Ok(());
    }
    if d % 2 != 0 || d > 1024 {
        return Err(SynaptixError::Cuda(format!("rope_interleaved: bad head_dim={d}")));
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (rows as usize) * (d as usize);
    let cn = (s_len as usize) * (d as usize); // ПОЛНАЯ таблица [S,D]
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (d, 1, 1),
        shared_mem_bytes: 0,
    };
    let cos_v = unsafe {
        cos.slice(cos_off..cos_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_interleaved: transmute cos".into()))?
    };
    let sin_v = unsafe {
        sin.slice(sin_off..sin_off + cn * 4)
            .transmute::<f32>(cn)
            .ok_or_else(|| SynaptixError::Cuda("rope_interleaved: transmute sin".into()))?
    };
    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_interleaved: transmute x".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + xn * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_interleaved: transmute out".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&x_v).arg(&mut o_v).arg(&cos_v).arg(&sin_v).arg(&h).arg(&s_len).arg(&d);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch rope_interleaved: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => go!(f16, &kernels.interleaved_f16),
        DType::BF16 => go!(bf16, &kernels.interleaved_bf16),
        DType::F32 => go!(f32, &kernels.interleaved_f32),
        _ => return Err(SynaptixError::Unsupported("rope_interleaved: dtype")),
    }
    Ok(())
}

/// Partial RoPE из untyped `u8`-storage с device-резидентной `start_pos` (для
/// `Backend::rope_apply_dev` / CUDA-graph). В отличие от [`run_rope_split_u8`]:
/// `cos`/`sin` — в `dtype` `x` (НЕ F32) и в *дублированном* layout
/// `[max_seq, rotary_dim]` (ядро индексирует `cos[pos*rotary_dim + d]` для
/// `d∈[0,rotary_dim)`, поэтому половинки дублируются), `cos_n` = число
/// элементов таблицы (`max_seq*rotary_dim`). `start_pos_dev` — `&CudaView<u32>`
/// (1 элемент); launch config от значения не зависит → один граф для всех позиций.
#[allow(clippy::too_many_arguments)]
pub fn apply_partial_u8_dev(
    kernels: &RopeKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    cos: &CudaSlice<u8>,
    cos_off: usize,
    sin: &CudaSlice<u8>,
    sin_off: usize,
    cos_n: usize,
    start_pos_dev: &CudaView<u32>,
    b: u32,
    h: u32,
    t: u32,
    head_dim: u32,
    rotary_dim: u32,
    dtype: DType,
) -> Result<()> {
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "rope_apply u8_dev: rotary_dim={rotary_dim} must be ≤ head_dim={head_dim} and even"
        )));
    }
    if head_dim > 1024 {
        return Err(SynaptixError::Cuda(format!(
            "rope_apply u8_dev: head_dim={head_dim} > 1024 (max block_dim)"
        )));
    }
    if b * h * t == 0 || head_dim == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F16 => &kernels.apply_partial_f16,
        DType::BF16 => &kernels.apply_partial_bf16,
        DType::F32 => &kernels.apply_partial_f32,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "rope_apply u8_dev: unsupported dtype {other:?}"
            )))
        }
    };
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (b as usize) * (h as usize) * (t as usize) * (head_dim as usize);
    let cfg = LaunchConfig {
        grid_dim: (b * h * t, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! go {
        ($t:ty) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_apply u8_dev: transmute x".into()))?
            };
            let cos_v = unsafe {
                cos.slice(cos_off..cos_off + cos_n * esz)
                    .transmute::<$t>(cos_n)
                    .ok_or_else(|| SynaptixError::Cuda("rope_apply u8_dev: transmute cos".into()))?
            };
            let sin_v = unsafe {
                sin.slice(sin_off..sin_off + cos_n * esz)
                    .transmute::<$t>(cos_n)
                    .ok_or_else(|| SynaptixError::Cuda("rope_apply u8_dev: transmute sin".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + xn * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rope_apply u8_dev: transmute out".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&x_v)
                .arg(&mut o_v)
                .arg(&cos_v)
                .arg(&sin_v)
                .arg(start_pos_dev)
                .arg(&b)
                .arg(&h)
                .arg(&t)
                .arg(&head_dim)
                .arg(&rotary_dim);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch rope_apply u8_dev: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => go!(f16),
        DType::BF16 => go!(bf16),
        DType::F32 => go!(f32),
        _ => unreachable!(),
    }
    Ok(())
}
