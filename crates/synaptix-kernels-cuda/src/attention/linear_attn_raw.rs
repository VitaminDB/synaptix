//! Raw prep-ядра для GatedDeltaNet decode T=1 (linear attention).
//!
//! Портировано из `ai-quant/src/kernels/linear_attn_raw.rs` (валидировано
//! bit-exact в проде Qwen3.6). Заменяет 8-12 поэлементных capture-небезопасных
//! ops на 4 fused launch'а. Все ядра — F32 internal, F16 in/out для большинства
//! тензоров (g / β — F32 out для матчинга с `gated_delta_rule` API).
//!
//! Состав:
//! - [`LinearAttnRawKernels::softplus_neg_exp_g`] — pre-exp log-decay per head.
//! - [`LinearAttnRawKernels::sigmoid_f16_to_f32`] — β = sigmoid(b).
//! - [`LinearAttnRawKernels::repeat_interleave_cast_f16_to_f32`] — Q/K repeat
//!   (n_rep) + cast или V cast (n_rep = 1).
//! - [`LinearAttnRawKernels::rms_norm_gated_f32_in_f16_out`] — fused
//!   `RMSNorm(x_f32) * silu(gate_f16) * weight_f16 → out_f16`.
//! - [`LinearAttnRawKernels::linear_attn_prep_fused`] — 5 → 1 launch.
//!
//! Исходник CUDA: `src/cu/fused/attention/linear_attn_raw.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK_DIM: u32 = 256;

pub struct LinearAttnRawKernels {
    _module: Arc<CudaModule>,
    softplus_neg_exp_g: CudaFunction,
    sigmoid_f16_to_f32: CudaFunction,
    repeat_interleave_cast: CudaFunction,
    rms_norm_gated: CudaFunction,
    prep_fused: CudaFunction,
    prep_scatter_f16: CudaFunction,
    prep_scatter_bf16: CudaFunction,
    prep_scatter_f32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<LinearAttnRawKernels>)>>> = OnceLock::new();

impl LinearAttnRawKernels {
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
        let src_base = include_str!("../cu/fused/attention/linear_attn_raw.cu");
        let src_chunk = include_str!("../cu/fused/ssm/linear_attn_prep_scatter.cu");
        let src = format!("{src_base}\n{src_chunk}");
        let module = compile_module(ctx, &src, "linear_attn_raw.cu")?;
        let new = Arc::new(Self {
            softplus_neg_exp_g: load_fn(&module, "softplus_neg_exp_g")?,
            sigmoid_f16_to_f32: load_fn(&module, "sigmoid_f16_to_f32")?,
            repeat_interleave_cast: load_fn(&module, "repeat_interleave_cast_f16_to_f32")?,
            rms_norm_gated: load_fn(&module, "rms_norm_gated_f32_in_f16_out")?,
            prep_fused: load_fn(&module, "linear_attn_prep_fused_f16")?,
            prep_scatter_f16: load_fn(&module, "linear_attn_prep_scatter_f16")?,
            prep_scatter_bf16: load_fn(&module, "linear_attn_prep_scatter_bf16")?,
            prep_scatter_f32: load_fn(&module, "linear_attn_prep_scatter_f32")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Сырой handle prep-fused-ядра (для оркестратора linear-decode).
    pub(crate) fn prep_fused_fn(&self) -> &CudaFunction {
        &self.prep_fused
    }

    /// `g_out[i] = softplus(a[i] + dt_bias[i]) * (-exp(A_log[i]))` для i в [0, num_v).
    pub fn softplus_neg_exp_g(
        &self,
        stream: &Arc<CudaStream>,
        a_f16: &CudaSlice<f16>,
        dt_bias_f32: &CudaSlice<f32>,
        a_log_f32: &CudaSlice<f32>,
        g_out_f32: &mut CudaSlice<f32>,
        num_v: u32,
    ) -> Result<()> {
        let grid = num_v.div_ceil(BLOCK_DIM);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.softplus_neg_exp_g);
        b.arg(a_f16)
            .arg(dt_bias_f32)
            .arg(a_log_f32)
            .arg(g_out_f32)
            .arg(&num_v);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch softplus_neg_exp_g: {e:?}")))?;
        }
        Ok(())
    }

    /// `out[i] = sigmoid(in[i])` для i в [0, n). F16 → F32.
    pub fn sigmoid_f16_to_f32(
        &self,
        stream: &Arc<CudaStream>,
        in_f16: &CudaSlice<f16>,
        out_f32: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let grid = n.div_ceil(BLOCK_DIM);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.sigmoid_f16_to_f32);
        b.arg(in_f16).arg(out_f32).arg(&n);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch sigmoid_f16_to_f32: {e:?}")))?;
        }
        Ok(())
    }

    /// `out[h_out, d] = (float)in[in_offset + (h_out / n_rep) * dim + d]`.
    ///
    /// `in_offset` — смещение в элементах от начала `in_f16` (для slicing'а
    /// post_conv [Q|K|V] без отдельных CudaView'ов). При `n_rep = 1` — straight
    /// cast F16→F32; при `n_rep > 1` — Q/K repeat_interleave по axis=2.
    pub fn repeat_interleave_cast_f16_to_f32(
        &self,
        stream: &Arc<CudaStream>,
        in_f16: &CudaSlice<f16>,
        in_offset: u32,
        out_f32: &mut CudaSlice<f32>,
        h_in: u32,
        n_rep: u32,
        dim: u32,
    ) -> Result<()> {
        let h_out = h_in * n_rep;
        let grid_d = dim.div_ceil(BLOCK_DIM);
        let cfg = LaunchConfig {
            grid_dim: (h_out, grid_d, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.repeat_interleave_cast);
        b.arg(in_f16)
            .arg(&in_offset)
            .arg(out_f32)
            .arg(&n_rep)
            .arg(&dim);
        unsafe {
            b.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch repeat_interleave_cast_f16_to_f32: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Fused RMSNorm + silu(gate) gating + weight gain. F32 input → F16 output.
    ///
    /// `x`: `(n_rows, dim)` F32. `gate`: `(n_rows, dim)` F16. `weight`: `(dim,)` F16.
    /// `out`: `(n_rows, dim)` F16. Алгоритм:
    /// `out = weight * x / sqrt(mean(x²) + eps) * silu(gate)`.
    ///
    /// Block: round_up_pow2(dim) ≤ 1024. Один block per row.
    pub fn rms_norm_gated_f32_in_f16_out(
        &self,
        stream: &Arc<CudaStream>,
        x_f32: &CudaSlice<f32>,
        gate_f16: &CudaSlice<f16>,
        weight_f16: &CudaSlice<f16>,
        out_f16: &mut CudaSlice<f16>,
        eps: f64,
        n_rows: u32,
        dim: u32,
    ) -> Result<()> {
        let block = {
            let mut b = 1u32;
            while b < dim {
                b <<= 1;
            }
            b.min(1024)
        };
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let eps_f32 = eps as f32;
        let mut b = stream.launch_builder(&self.rms_norm_gated);
        b.arg(x_f32)
            .arg(gate_f16)
            .arg(weight_f16)
            .arg(out_f16)
            .arg(&eps_f32)
            .arg(&dim);
        unsafe {
            b.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch rms_norm_gated_f32_in_f16_out: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Fused kernel: 5 → 1 launch для подготовки SSM-входов одного linear_attn
    /// слоя. Заменяет sigmoid(b)→beta, softplus(a,dt_bias,a_log)→g, и три
    /// repeat_interleave_cast (Q/K с n_rep, V с n_rep=1). Bit-exact с отдельными.
    ///
    /// Layout: grid `(num_v, 1, 4)`, block `(max(hk, hv), 1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_prep_fused(
        &self,
        stream: &Arc<CudaStream>,
        b_f16: &CudaSlice<f16>,
        a_f16: &CudaSlice<f16>,
        dt_bias_f32: &CudaSlice<f32>,
        a_log_f32: &CudaSlice<f32>,
        beta_f32: &mut CudaSlice<f32>,
        g_out_f32: &mut CudaSlice<f32>,
        post_conv_f16: &CudaSlice<f16>,
        q_out_f32: &mut CudaSlice<f32>,
        k_out_f32: &mut CudaSlice<f32>,
        v_out_f32: &mut CudaSlice<f32>,
        num_v: u32,
        num_k: u32,
        n_rep: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        debug_assert_eq!(num_k * n_rep, num_v, "num_k*n_rep must equal num_v");
        let key_dim = num_k * hk;
        let block = hk.max(hv);
        let cfg = LaunchConfig {
            grid_dim: (num_v, 1, 4),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.prep_fused);
        b.arg(b_f16)
            .arg(a_f16)
            .arg(dt_bias_f32)
            .arg(a_log_f32)
            .arg(beta_f32)
            .arg(g_out_f32)
            .arg(post_conv_f16)
            .arg(q_out_f32)
            .arg(k_out_f32)
            .arg(v_out_f32)
            .arg(&num_v)
            .arg(&n_rep)
            .arg(&hk)
            .arg(&hv)
            .arg(&key_dim);
        unsafe {
            b.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch linear_attn_prep_fused_f16: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Chunk-prefill версия `linear_attn_prep_fused` (T≥1). Bit-exact с host-loop
    /// в `LinearAttn::forward` (model.rs:879-907 + gated_delta_decay_beta).
    /// Layout выходов идентичен тому, что ожидает `chunk_gated_delta_rule`
    /// (BH = num_v, T, HK = hk).
    ///
    /// `conv_out` параметризован: F16 (после chunk-conv1d compute), BF16 или F32.
    /// `a`/`b` — F16, `dt_bias`/`a_log` — F32, выходы (β/g/q/k/v) — F32.
    /// Grid: `(num_v, T, 4)`, block: `(max(hk, hv, num_v), 1, 1)`.
    ///
    /// Inputs (read-only) — `&CudaView<T>` (единый тип для CudaSlice через
    /// `.as_view()` и transmute-результата в backend). Outputs — `&mut CudaSlice<f32>`
    /// (alloc локально в caller'е).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_prep_scatter_f16(
        &self,
        stream: &Arc<CudaStream>,
        b_f16: &CudaView<f16>,
        a_f16: &CudaView<f16>,
        dt_bias_f32: &CudaView<f32>,
        a_log_f32: &CudaView<f32>,
        beta_f32: &mut CudaSlice<f32>,
        g_out_f32: &mut CudaSlice<f32>,
        conv_out_f16: &CudaView<f16>,
        q_out_f32: &mut CudaSlice<f32>,
        k_out_f32: &mut CudaSlice<f32>,
        v_out_f32: &mut CudaSlice<f32>,
        t_in: u32,
        t_out: u32,
        num_v: u32,
        num_k: u32,
        n_rep: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        prep_scatter_launch_f16(
            &self.prep_scatter_f16,
            stream,
            b_f16,
            a_f16,
            dt_bias_f32,
            a_log_f32,
            beta_f32,
            g_out_f32,
            conv_out_f16,
            q_out_f32,
            k_out_f32,
            v_out_f32,
            t_in,
            t_out,
            num_v,
            num_k,
            n_rep,
            hk,
            hv,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_prep_scatter_bf16(
        &self,
        stream: &Arc<CudaStream>,
        b_f16: &CudaView<f16>,
        a_f16: &CudaView<f16>,
        dt_bias_f32: &CudaView<f32>,
        a_log_f32: &CudaView<f32>,
        beta_f32: &mut CudaSlice<f32>,
        g_out_f32: &mut CudaSlice<f32>,
        conv_out_bf16: &CudaView<bf16>,
        q_out_f32: &mut CudaSlice<f32>,
        k_out_f32: &mut CudaSlice<f32>,
        v_out_f32: &mut CudaSlice<f32>,
        t_in: u32,
        t_out: u32,
        num_v: u32,
        num_k: u32,
        n_rep: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        prep_scatter_launch_bf16(
            &self.prep_scatter_bf16,
            stream,
            b_f16,
            a_f16,
            dt_bias_f32,
            a_log_f32,
            beta_f32,
            g_out_f32,
            conv_out_bf16,
            q_out_f32,
            k_out_f32,
            v_out_f32,
            t_in,
            t_out,
            num_v,
            num_k,
            n_rep,
            hk,
            hv,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_prep_scatter_f32(
        &self,
        stream: &Arc<CudaStream>,
        b_f16: &CudaView<f16>,
        a_f16: &CudaView<f16>,
        dt_bias_f32: &CudaView<f32>,
        a_log_f32: &CudaView<f32>,
        beta_f32: &mut CudaSlice<f32>,
        g_out_f32: &mut CudaSlice<f32>,
        conv_out_f32: &CudaView<f32>,
        q_out_f32: &mut CudaSlice<f32>,
        k_out_f32: &mut CudaSlice<f32>,
        v_out_f32: &mut CudaSlice<f32>,
        t_in: u32,
        t_out: u32,
        num_v: u32,
        num_k: u32,
        n_rep: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        prep_scatter_launch_f32(
            &self.prep_scatter_f32,
            stream,
            b_f16,
            a_f16,
            dt_bias_f32,
            a_log_f32,
            beta_f32,
            g_out_f32,
            conv_out_f32,
            q_out_f32,
            k_out_f32,
            v_out_f32,
            t_in,
            t_out,
            num_v,
            num_k,
            n_rep,
            hk,
            hv,
        )
    }
}

// Три параллельные launch функции (раздельные T у conv_out → разные PushKernelArg
// impl). Тело одно, dtype-specifix только сигнатура conv_out_*.
macro_rules! impl_prep_scatter_launch {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            func: &CudaFunction,
            stream: &Arc<CudaStream>,
            b_f16: &CudaView<f16>,
            a_f16: &CudaView<f16>,
            dt_bias_f32: &CudaView<f32>,
            a_log_f32: &CudaView<f32>,
            beta_f32: &mut CudaSlice<f32>,
            g_out_f32: &mut CudaSlice<f32>,
            conv_out: &CudaView<$T>,
            q_out_f32: &mut CudaSlice<f32>,
            k_out_f32: &mut CudaSlice<f32>,
            v_out_f32: &mut CudaSlice<f32>,
            t_in: u32,
            t_out: u32,
            num_v: u32,
            num_k: u32,
            n_rep: u32,
            hk: u32,
            hv: u32,
        ) -> Result<()> {
            debug_assert_eq!(num_k * n_rep, num_v, "num_k*n_rep must equal num_v");
            debug_assert!(t_out >= t_in, "t_out must be >= t_in");
            if t_in == 0 || num_v == 0 {
                return Ok(());
            }
            let key_dim = num_k * hk;
            let block = hk.max(hv).max(num_v);
            let cfg = LaunchConfig {
                grid_dim: (num_v, t_in, 4),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(b_f16)
                .arg(a_f16)
                .arg(dt_bias_f32)
                .arg(a_log_f32)
                .arg(beta_f32)
                .arg(g_out_f32)
                .arg(conv_out)
                .arg(q_out_f32)
                .arg(k_out_f32)
                .arg(v_out_f32)
                .arg(&t_in)
                .arg(&t_out)
                .arg(&num_v)
                .arg(&n_rep)
                .arg(&hk)
                .arg(&hv)
                .arg(&key_dim);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch linear_attn_prep_scatter: {e:?}"))
                })?;
            }
            Ok(())
        }
    };
}

impl_prep_scatter_launch!(prep_scatter_launch_f16, f16);
impl_prep_scatter_launch!(prep_scatter_launch_bf16, bf16);
impl_prep_scatter_launch!(prep_scatter_launch_f32, f32);
