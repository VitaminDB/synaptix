//! Causal depthwise conv1d (Mamba): `out[b,c,i] = sum_ki w[c,ki] * x_pad[b,c,i*stride+ki]`
//! с левым padding K-1. `x` [B,C,L], `weight` [C,1,K], `bias` [C], `out` [B,C,out_len],
//! `out_len = ceil(L/stride)`. F32/F16/BF16, f32-аккумулятор. Naive baseline (один
//! thread = один output). Семантика совпадает с `synaptix_ops::conv::causal_conv1d`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DeviceRepr, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 128;

pub struct CausalConv1dKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
    update_f32: CudaFunction,
    update_f16: CudaFunction,
    chunk_compute_f32: CudaFunction,
    chunk_compute_f16: CudaFunction,
    chunk_compute_bf16: CudaFunction,
    chunk_update_f32: CudaFunction,
    chunk_update_f16: CudaFunction,
    chunk_update_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<CausalConv1dKernels>)>>> = OnceLock::new();

impl CausalConv1dKernels {
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
        let src_base = include_str!("../cu/conv/causal_conv1d.cu");
        let src_chunk = include_str!("../cu/conv/causal_conv1d_chunk.cu");
        let src = format!("{src_base}\n{src_chunk}");
        let module = compile_module(ctx, &src, "causal_conv1d.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "causal_conv1d_f32")?,
            f16: load_fn(&module, "causal_conv1d_f16")?,
            bf16: load_fn(&module, "causal_conv1d_bf16")?,
            update_f32: load_fn(&module, "causal_conv1d_update_f32")?,
            update_f16: load_fn(&module, "causal_conv1d_update_f16")?,
            chunk_compute_f32: load_fn(&module, "causal_conv1d_chunk_compute_f32")?,
            chunk_compute_f16: load_fn(&module, "causal_conv1d_chunk_compute_f16")?,
            chunk_compute_bf16: load_fn(&module, "causal_conv1d_chunk_compute_bf16")?,
            chunk_update_f32: load_fn(&module, "causal_conv1d_chunk_update_state_f32")?,
            chunk_update_f16: load_fn(&module, "causal_conv1d_chunk_update_state_f16")?,
            chunk_update_bf16: load_fn(&module, "causal_conv1d_chunk_update_state_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Сырой handle update-F16-ядра (для оркестратора linear-decode).
    pub(crate) fn update_f16_fn(&self) -> &CudaFunction {
        &self.update_f16
    }

    /// Stateful single-step (decode T=1) causal depthwise conv1d. `x` `[conv_dim]`
    /// — новый сэмпл; `state` `[(K-1), conv_dim]` (FIFO oldest-first) обновляется
    /// in-place; `w` `[conv_dim, K]`; `out` `[conv_dim]`. `silu` → выход
    /// `act = a / (1 + e^-a)`. Один thread = один канал. Capture-safe (нет host
    /// round-trip, alloc'ов и переменной launch-config).
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_update_f32(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        conv_dim: u32,
        k: u32,
        silu: bool,
    ) -> Result<()> {
        self.update_launch(
            &self.update_f32,
            stream,
            x,
            state,
            w,
            out,
            conv_dim,
            k,
            silu,
        )
    }

    /// F16-вариант [`Self::causal_conv1d_update_f32`].
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_update_f16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f16>,
        state: &mut CudaSlice<f16>,
        w: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        conv_dim: u32,
        k: u32,
        silu: bool,
    ) -> Result<()> {
        self.update_launch(
            &self.update_f16,
            stream,
            x,
            state,
            w,
            out,
            conv_dim,
            k,
            silu,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn update_launch<T: DeviceRepr>(
        &self,
        func: &CudaFunction,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<T>,
        state: &mut CudaSlice<T>,
        w: &CudaSlice<T>,
        out: &mut CudaSlice<T>,
        conv_dim: u32,
        k: u32,
        silu: bool,
    ) -> Result<()> {
        if conv_dim == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (conv_dim.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (cd_i, k_i, silu_i) = (conv_dim as i32, k as i32, if silu { 1i32 } else { 0i32 });
        let mut bld = stream.launch_builder(func);
        bld.arg(x)
            .arg(&mut *state)
            .arg(w)
            .arg(&mut *out)
            .arg(&cd_i)
            .arg(&k_i)
            .arg(&silu_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch causal_conv1d_update: {e:?}")))?;
        }
        Ok(())
    }
}

/// Stateful chunked causal depthwise conv1d (prefill T≥1) для GatedDeltaNet.
/// Layout = host-эталон `synaptix_ops::conv::causal_conv1d_stateful` (time-major):
/// `x` `[T, conv_dim]`, `state` `[(K-1), conv_dim]` (FIFO oldest-first) in-place,
/// `w` `[conv_dim, K]`, `out` `[T, conv_dim]`. `silu` → apply `a/(1+e^-a)`
/// поверх out. Два launch'а: compute (один thread = (c, t)) + update_state
/// (один thread = c). f32-аккумулятор. Совместимо c semантикой update T=1
/// (`Self::causal_conv1d_update_f32`), но без host round-trip и со scaling по T.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunk_f32(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaView<f32>,
    state: &mut CudaViewMut<f32>,
    w: &CudaView<f32>,
    out: &mut CudaViewMut<f32>,
    t_in: u32,
    conv_dim: u32,
    k: u32,
    silu: bool,
) -> Result<()> {
    chunk_launch_compute_f32(
        &kernels.chunk_compute_f32,
        stream,
        x,
        state,
        w,
        out,
        t_in,
        conv_dim,
        k,
        silu,
    )?;
    chunk_launch_update_f32(
        &kernels.chunk_update_f32,
        stream,
        x,
        state,
        t_in,
        conv_dim,
        k,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunk_f16(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaView<f16>,
    state: &mut CudaViewMut<f16>,
    w: &CudaView<f16>,
    out: &mut CudaViewMut<f16>,
    t_in: u32,
    conv_dim: u32,
    k: u32,
    silu: bool,
) -> Result<()> {
    chunk_launch_compute_f16(
        &kernels.chunk_compute_f16,
        stream,
        x,
        state,
        w,
        out,
        t_in,
        conv_dim,
        k,
        silu,
    )?;
    chunk_launch_update_f16(
        &kernels.chunk_update_f16,
        stream,
        x,
        state,
        t_in,
        conv_dim,
        k,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunk_bf16(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaView<bf16>,
    state: &mut CudaViewMut<bf16>,
    w: &CudaView<bf16>,
    out: &mut CudaViewMut<bf16>,
    t_in: u32,
    conv_dim: u32,
    k: u32,
    silu: bool,
) -> Result<()> {
    chunk_launch_compute_bf16(
        &kernels.chunk_compute_bf16,
        stream,
        x,
        state,
        w,
        out,
        t_in,
        conv_dim,
        k,
        silu,
    )?;
    chunk_launch_update_bf16(
        &kernels.chunk_update_bf16,
        stream,
        x,
        state,
        t_in,
        conv_dim,
        k,
    )
}

// Внутренние launch'ы — typed per dtype (PushKernelArg<&CudaView<T>> работает,
// но дженерик-bound на DevicePtr не пропустит arg() trait). Дублирование
// очевидно и тривиально-проверяемо.

macro_rules! impl_chunk_launch {
    ($name_compute:ident, $name_update:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name_compute(
            func: &CudaFunction,
            stream: &Arc<CudaStream>,
            x: &CudaView<$T>,
            state: &mut CudaViewMut<$T>,
            w: &CudaView<$T>,
            out: &mut CudaViewMut<$T>,
            t_in: u32,
            conv_dim: u32,
            k: u32,
            silu: bool,
        ) -> Result<()> {
            if t_in == 0 || conv_dim == 0 {
                return Ok(());
            }
            let (t_i, c_i, k_i, silu_i) = (
                t_in as i32,
                conv_dim as i32,
                k as i32,
                if silu { 1i32 } else { 0i32 },
            );
            let cfg = LaunchConfig {
                grid_dim: (conv_dim.div_ceil(BLOCK), t_in, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            // state читается read-only на этом launch'е → reborrow в CudaView.
            let state_ro: CudaView<$T> = state.as_view();
            let mut bld = stream.launch_builder(func);
            bld.arg(x)
                .arg(&state_ro)
                .arg(w)
                .arg(&mut *out)
                .arg(&t_i)
                .arg(&c_i)
                .arg(&k_i)
                .arg(&silu_i);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch causal_conv1d_chunk_compute: {e:?}"))
                })?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn $name_update(
            func: &CudaFunction,
            stream: &Arc<CudaStream>,
            x: &CudaView<$T>,
            state: &mut CudaViewMut<$T>,
            t_in: u32,
            conv_dim: u32,
            k: u32,
        ) -> Result<()> {
            if conv_dim == 0 {
                return Ok(());
            }
            let (t_i, c_i, k_i) = (t_in as i32, conv_dim as i32, k as i32);
            let cfg = LaunchConfig {
                grid_dim: (conv_dim.div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(x).arg(&mut *state).arg(&t_i).arg(&c_i).arg(&k_i);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch causal_conv1d_chunk_update_state: {e:?}"))
                })?;
            }
            Ok(())
        }
    };
}

impl_chunk_launch!(chunk_launch_compute_f32, chunk_launch_update_f32, f32);
impl_chunk_launch!(chunk_launch_compute_f16, chunk_launch_update_f16, f16);
impl_chunk_launch!(chunk_launch_compute_bf16, chunk_launch_update_bf16, bf16);

/// Длина выхода causal conv1d: `ceil(L / stride)`.
pub fn out_len(l: usize, stride: usize) -> usize {
    let s = stride.max(1);
    l.div_ceil(s)
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d<T: DeviceRepr>(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    out: &mut CudaSlice<T>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
    dtype: DType,
) -> Result<()> {
    let stride = stride.max(1);
    let o = out_len(l as usize, stride as usize) as u32;
    if o == 0 || b == 0 || c == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "causal_conv1d: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * c, o.div_ceil(BLOCK), 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (b_i, c_i, l_i, k_i, s_i, o_i) = (
        b as i32,
        c as i32,
        l as i32,
        k as i32,
        stride as i32,
        o as i32,
    );
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let bias_ptr = bias.unwrap_or(x);
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(weight)
        .arg(bias_ptr)
        .arg(&has_bias_i)
        .arg(&mut *out)
        .arg(&b_i)
        .arg(&c_i)
        .arg(&l_i)
        .arg(&k_i)
        .arg(&s_i)
        .arg(&o_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch causal_conv1d: {e:?}")))?;
    }
    Ok(())
}

pub fn causal_conv1d_f32(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    out: &mut CudaSlice<f32>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
) -> Result<()> {
    causal_conv1d::<f32>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        DType::F32,
    )
}

pub fn causal_conv1d_f16(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: Option<&CudaSlice<f16>>,
    out: &mut CudaSlice<f16>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
) -> Result<()> {
    causal_conv1d::<f16>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        DType::F16,
    )
}

pub fn causal_conv1d_bf16(
    kernels: &CausalConv1dKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    out: &mut CudaSlice<bf16>,
    b: u32,
    c: u32,
    l: u32,
    k: u32,
    stride: u32,
) -> Result<()> {
    causal_conv1d::<bf16>(
        kernels,
        stream,
        x,
        weight,
        bias,
        out,
        b,
        c,
        l,
        k,
        stride,
        DType::BF16,
    )
}
