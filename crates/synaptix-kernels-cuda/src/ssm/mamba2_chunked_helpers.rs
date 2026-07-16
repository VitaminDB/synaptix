//! Mamba2 chunked-SSD helpers (permute / cumsum / cast / transpose /
//! decay-mask / exp-mul / state-decay / add-inplace / yoff-accumulate / post).
//!
//! Все light-weight kernels для multi-kernel pipeline chunked-SSD. См.
//! `src/cu/ssm/mamba2_chunked_helpers.cu` и план
//! `plan/mamba2_chunked_stage2_handover.md`.
//!
//! Принимают raw `CudaSlice<T>` без size-checks (caller-orchestrator знает
//! workspace layout). Каждый kernel — простой launch, формула в Rust-доке.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct Mamba2ChunkedHelpersKernels {
    _module: Arc<CudaModule>,
    // 1. alpha_cum
    alpha_cum_f32: CudaFunction,
    alpha_cum_f16: CudaFunction,
    alpha_cum_bf16: CudaFunction,
    // 2. permute (B, L, H, X) -> (BH, T, Q, X) bf16
    permute_blhx_f32: CudaFunction,
    permute_blhx_f16: CudaFunction,
    permute_blhx_bf16: CudaFunction,
    // 3. dt_x
    dt_x_f32: CudaFunction,
    dt_x_f16: CudaFunction,
    dt_x_bf16: CudaFunction,
    // 4. transpose bf16
    transpose_bf16: CudaFunction,
    // 5. decay mask
    decay_mask: CudaFunction,
    // 6. broadcast exp mul (по R-оси и Q-оси)
    col_bcast_exp_mul: CudaFunction,
    row_bcast_exp_mul: CudaFunction,
    // 7. state linear decay
    state_decay: CudaFunction,
    // 8. add inplace f32
    add_inplace: CudaFunction,
    // 9. add yoff chunk
    add_yoff: CudaFunction,
    // 10. state cast f32->bf16
    state_cast: CudaFunction,
    // 11. post
    post_f32: CudaFunction,
    post_f16: CudaFunction,
    post_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Mamba2ChunkedHelpersKernels>)>>> = OnceLock::new();

impl Mamba2ChunkedHelpersKernels {
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
        let src = include_str!("../cu/ssm/mamba2_chunked_helpers.cu");
        let module = compile_module(ctx, src, "mamba2_chunked_helpers.cu")?;
        let new = Arc::new(Self {
            alpha_cum_f32: load_fn(&module, "mamba2_alpha_cum_f32_in")?,
            alpha_cum_f16: load_fn(&module, "mamba2_alpha_cum_f16_in")?,
            alpha_cum_bf16: load_fn(&module, "mamba2_alpha_cum_bf16_in")?,
            permute_blhx_f32: load_fn(&module, "mamba2_permute_blhx_f32_to_bf16")?,
            permute_blhx_f16: load_fn(&module, "mamba2_permute_blhx_f16_to_bf16")?,
            permute_blhx_bf16: load_fn(&module, "mamba2_permute_blhx_bf16_to_bf16")?,
            dt_x_f32: load_fn(&module, "mamba2_compute_dt_x_f32_to_bf16")?,
            dt_x_f16: load_fn(&module, "mamba2_compute_dt_x_f16_to_bf16")?,
            dt_x_bf16: load_fn(&module, "mamba2_compute_dt_x_bf16_to_bf16")?,
            transpose_bf16: load_fn(&module, "mamba2_transpose_bf16")?,
            decay_mask: load_fn(&module, "mamba2_apply_decay_mask_to_bf16")?,
            col_bcast_exp_mul: load_fn(&module, "mamba2_col_broadcast_exp_mul_bf16")?,
            row_bcast_exp_mul: load_fn(&module, "mamba2_row_broadcast_exp_mul_bf16")?,
            state_decay: load_fn(&module, "mamba2_state_linear_decay_f32")?,
            add_inplace: load_fn(&module, "mamba2_add_inplace_f32")?,
            add_yoff: load_fn(&module, "mamba2_add_yoff_chunk_f32")?,
            state_cast: load_fn(&module, "mamba2_state_cast_f32_to_bf16")?,
            post_f32: load_fn(&module, "mamba2_post_f32")?,
            post_f16: load_fn(&module, "mamba2_post_f16")?,
            post_bf16: load_fn(&module, "mamba2_post_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// 1. `alpha_cum[bh, t, j] = Σ_{k≤j} (A[h] * dt[b, t*Q+k, h])`.
    ///
    /// Shapes: `dt [B, L, H]`, `a [H]`, `alpha_cum [BH, T, Q]` f32.
    /// `L = T*Q`. Один block = (bh, t), threads = Q (`Q ≤ 1024`).
    #[allow(clippy::too_many_arguments)]
    pub fn alpha_cum<T>(
        &self,
        stream: &Arc<CudaStream>,
        dt: &CudaSlice<T>,
        a: &CudaSlice<T>,
        alpha_cum: &mut CudaSlice<f32>,
        b: u32,
        h: u32,
        t: u32,
        q: u32,
        dtype: DType,
    ) -> Result<()> {
        let func = match dtype {
            DType::F32 => &self.alpha_cum_f32,
            DType::F16 => &self.alpha_cum_f16,
            DType::BF16 => &self.alpha_cum_bf16,
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "alpha_cum: unsupported dtype {other:?}"
                )))
            }
        };
        let cfg = LaunchConfig {
            grid_dim: (b * h, t, 1),
            block_dim: (q, 1, 1),
            shared_mem_bytes: q * std::mem::size_of::<f32>() as u32,
        };
        let (b_i, h_i, t_i, q_i) = (b as i32, h as i32, t as i32, q as i32);
        let mut bld = stream.launch_builder(func);
        bld.arg(dt)
            .arg(a)
            .arg(&mut *alpha_cum)
            .arg(&b_i)
            .arg(&h_i)
            .arg(&t_i)
            .arg(&q_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch alpha_cum: {e:?}")))?;
        }
        Ok(())
    }

    /// 2. Permute `src [B, L, H, X] → dst [BH, T, Q, X]` с cast в bf16. `L = T*Q`.
    #[allow(clippy::too_many_arguments)]
    pub fn permute_blhx_to_bhtqx<T>(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<bf16>,
        b: u32,
        l: u32,
        h: u32,
        x: u32,
        q: u32,
        dtype: DType,
    ) -> Result<()> {
        let func = match dtype {
            DType::F32 => &self.permute_blhx_f32,
            DType::F16 => &self.permute_blhx_f16,
            DType::BF16 => &self.permute_blhx_bf16,
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "permute_blhx: unsupported dtype {other:?}"
                )))
            }
        };
        let tx = 32u32.min(x);
        let grid_x = x.div_ceil(tx);
        let (grid_y, grid_z) = split_blh(b * l * h);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, grid_z),
            block_dim: (tx, 1, 1),
            shared_mem_bytes: 0,
        };
        let (b_i, l_i, h_i, x_i, q_i) = (b as i32, l as i32, h as i32, x as i32, q as i32);
        let mut bld = stream.launch_builder(func);
        bld.arg(src)
            .arg(&mut *dst)
            .arg(&b_i)
            .arg(&l_i)
            .arg(&h_i)
            .arg(&x_i)
            .arg(&q_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch permute_blhx: {e:?}")))?;
        }
        Ok(())
    }

    /// 3. `dt_x[bh, t, q, p] = dt[b, l, h] * x[b, l, h, p]`. `L = T*Q`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_dt_x<T>(
        &self,
        stream: &Arc<CudaStream>,
        dt: &CudaSlice<T>,
        x: &CudaSlice<T>,
        dt_x: &mut CudaSlice<bf16>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        q: u32,
        dtype: DType,
    ) -> Result<()> {
        let func = match dtype {
            DType::F32 => &self.dt_x_f32,
            DType::F16 => &self.dt_x_f16,
            DType::BF16 => &self.dt_x_bf16,
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "compute_dt_x: unsupported dtype {other:?}"
                )))
            }
        };
        let tp = 32u32.min(p);
        let grid_p = p.div_ceil(tp);
        let (grid_y, grid_z) = split_blh(b * l * h);
        let cfg = LaunchConfig {
            grid_dim: (grid_p, grid_y, grid_z),
            block_dim: (tp, 1, 1),
            shared_mem_bytes: 0,
        };
        let (b_i, l_i, h_i, p_i, q_i) = (b as i32, l as i32, h as i32, p as i32, q as i32);
        let mut bld = stream.launch_builder(func);
        bld.arg(dt)
            .arg(x)
            .arg(&mut *dt_x)
            .arg(&b_i)
            .arg(&l_i)
            .arg(&h_i)
            .arg(&p_i)
            .arg(&q_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch dt_x: {e:?}")))?;
        }
        Ok(())
    }

    /// 4. Per-batch transpose: `src [BAT, R, C] → dst [BAT, C, R]` bf16.
    pub fn transpose_bf16(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<bf16>,
        dst: &mut CudaSlice<bf16>,
        bat: u32,
        r: u32,
        c: u32,
    ) -> Result<()> {
        let tx = 16u32.min(c);
        let ty = 16u32.min(r);
        let grid_x = c.div_ceil(tx);
        let grid_y = r.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bat),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bat_i, r_i, c_i) = (bat as i32, r as i32, c as i32);
        let mut bld = stream.launch_builder(&self.transpose_bf16);
        bld.arg(src).arg(&mut *dst).arg(&bat_i).arg(&r_i).arg(&c_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch transpose_bf16: {e:?}")))?;
        }
        Ok(())
    }

    /// 5. `A_decayed[bht, i, j] = A_intra[bht, i, j] * exp(α_cum[bht,i]-α_cum[bht,j]) * [j≤i]`.
    /// Cast в bf16.
    pub fn apply_decay_mask(
        &self,
        stream: &Arc<CudaStream>,
        a_intra: &CudaSlice<f32>,
        alpha_cum: &CudaSlice<f32>,
        a_decayed: &mut CudaSlice<bf16>,
        bht: u32,
        q: u32,
    ) -> Result<()> {
        let tx = 16u32.min(q);
        let ty = 16u32.min(q);
        let grid_x = q.div_ceil(tx);
        let grid_y = q.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bht),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bht_i, q_i) = (bht as i32, q as i32);
        let mut bld = stream.launch_builder(&self.decay_mask);
        bld.arg(a_intra)
            .arg(alpha_cum)
            .arg(&mut *a_decayed)
            .arg(&bht_i)
            .arg(&q_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch decay_mask: {e:?}")))?;
        }
        Ok(())
    }

    /// 6a. `dst[bat, r, c] = src[bat, r, c] * exp(vec[bat, r])` (`from_end=false`)
    /// или `* exp(vec[bat, R-1] - vec[bat, r])` (`from_end=true`).
    /// Used: `C_QN * exp(α_cum_chunk)` (from_end=false, R=Q, C=N).
    #[allow(clippy::too_many_arguments)]
    pub fn col_broadcast_exp_mul(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<bf16>,
        vec: &CudaSlice<f32>,
        dst: &mut CudaSlice<bf16>,
        bat: u32,
        r: u32,
        c: u32,
        from_end: bool,
    ) -> Result<()> {
        let tx = 16u32.min(c);
        let ty = 16u32.min(r);
        let grid_x = c.div_ceil(tx);
        let grid_y = r.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bat),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bat_i, r_i, c_i, fe_i) = (bat as i32, r as i32, c as i32, from_end as i32);
        let mut bld = stream.launch_builder(&self.col_bcast_exp_mul);
        bld.arg(src)
            .arg(vec)
            .arg(&mut *dst)
            .arg(&bat_i)
            .arg(&r_i)
            .arg(&c_i)
            .arg(&fe_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch col_bcast_exp_mul: {e:?}")))?;
        }
        Ok(())
    }

    /// 6b. Row-broadcast по C-оси: `dst[bat, r, c] = src[bat, r, c] * exp(vec[bat, c])`
    /// или `* exp(vec[bat, Q_vec-1] - vec[bat, c])` (`from_end=true`).
    /// Used: `dt_x_PQ * exp(α_end - α_cum)` (from_end=true, R=P, C=Q, Q_vec=Q).
    #[allow(clippy::too_many_arguments)]
    pub fn row_broadcast_exp_mul(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<bf16>,
        vec: &CudaSlice<f32>,
        dst: &mut CudaSlice<bf16>,
        bat: u32,
        r: u32,
        c: u32,
        q_vec: u32,
        from_end: bool,
    ) -> Result<()> {
        let tx = 16u32.min(c);
        let ty = 16u32.min(r);
        let grid_x = c.div_ceil(tx);
        let grid_y = r.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bat),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bat_i, r_i, c_i, qv_i, fe_i) = (
            bat as i32,
            r as i32,
            c as i32,
            q_vec as i32,
            from_end as i32,
        );
        let mut bld = stream.launch_builder(&self.row_bcast_exp_mul);
        bld.arg(src)
            .arg(vec)
            .arg(&mut *dst)
            .arg(&bat_i)
            .arg(&r_i)
            .arg(&c_i)
            .arg(&qv_i)
            .arg(&fe_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch row_bcast_exp_mul: {e:?}")))?;
        }
        Ok(())
    }

    /// 7. `state[bh, p, n] *= exp(alpha_cum[bh, chunk, Q-1])`.
    #[allow(clippy::too_many_arguments)]
    pub fn state_linear_decay(
        &self,
        stream: &Arc<CudaStream>,
        state: &mut CudaSlice<f32>,
        alpha_cum: &CudaSlice<f32>,
        bh: u32,
        p: u32,
        n: u32,
        t: u32,
        q: u32,
        chunk: u32,
    ) -> Result<()> {
        let tx = 16u32.min(n);
        let ty = 16u32.min(p);
        let grid_x = n.div_ceil(tx);
        let grid_y = p.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bh),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bh_i, p_i, n_i, t_i, q_i, chunk_i) = (
            bh as i32,
            p as i32,
            n as i32,
            t as i32,
            q as i32,
            chunk as i32,
        );
        let mut bld = stream.launch_builder(&self.state_decay);
        bld.arg(&mut *state)
            .arg(alpha_cum)
            .arg(&bh_i)
            .arg(&p_i)
            .arg(&n_i)
            .arg(&t_i)
            .arg(&q_i)
            .arg(&chunk_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch state_decay: {e:?}")))?;
        }
        Ok(())
    }

    /// 8. `dst[0..n] += src[0..n]` (f32, in-place).
    pub fn add_inplace_f32(
        &self,
        stream: &Arc<CudaStream>,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        n: u64,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let threads = 256u32;
        let grid_x = ((n + threads as u64 - 1) / threads as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_sz: usize = n as usize;
        let mut bld = stream.launch_builder(&self.add_inplace);
        bld.arg(&mut *dst).arg(src).arg(&n_sz);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch add_inplace: {e:?}")))?;
        }
        Ok(())
    }

    /// 9. `Y_intra[bh, chunk, q, p] += Y_off_chunk[bh, q, p]`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_yoff_chunk(
        &self,
        stream: &Arc<CudaStream>,
        y_intra: &mut CudaSlice<f32>,
        y_off_chunk: &CudaSlice<f32>,
        bh: u32,
        t: u32,
        q: u32,
        p: u32,
        chunk: u32,
    ) -> Result<()> {
        let tx = 16u32.min(p);
        let ty = 16u32.min(q);
        let grid_x = p.div_ceil(tx);
        let grid_y = q.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bh),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bh_i, t_i, q_i, p_i, chunk_i) =
            (bh as i32, t as i32, q as i32, p as i32, chunk as i32);
        let mut bld = stream.launch_builder(&self.add_yoff);
        bld.arg(&mut *y_intra)
            .arg(y_off_chunk)
            .arg(&bh_i)
            .arg(&t_i)
            .arg(&q_i)
            .arg(&p_i)
            .arg(&chunk_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch add_yoff: {e:?}")))?;
        }
        Ok(())
    }

    /// 10. `state_bf16[i] = bf16(state_f32[i])` — простой cast, без транспозы.
    pub fn state_cast_f32_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<bf16>,
        n: u64,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let threads = 256u32;
        let grid_x = ((n + threads as u64 - 1) / threads as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_sz: usize = n as usize;
        let mut bld = stream.launch_builder(&self.state_cast);
        bld.arg(src).arg(&mut *dst).arg(&n_sz);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch state_cast: {e:?}")))?;
        }
        Ok(())
    }

    /// 11. `y_out[b, l, h, p] = Y_intra[bh, t, q, p] + (has_d ? D[h] * x[b, l, h, p] : 0)`.
    /// Unpermute + skip-D. Output dtype = `dtype`.
    #[allow(clippy::too_many_arguments)]
    pub fn post<T>(
        &self,
        stream: &Arc<CudaStream>,
        y_intra: &CudaSlice<f32>,
        x: &CudaSlice<T>,
        d_skip: Option<&CudaSlice<T>>,
        y_out: &mut CudaSlice<T>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        q: u32,
        dtype: DType,
    ) -> Result<()> {
        let func = match dtype {
            DType::F32 => &self.post_f32,
            DType::F16 => &self.post_f16,
            DType::BF16 => &self.post_bf16,
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "post: unsupported dtype {other:?}"
                )))
            }
        };
        let tp = 32u32.min(p);
        let grid_p = p.div_ceil(tp);
        let (grid_y, grid_z) = split_blh(b * l * h);
        let cfg = LaunchConfig {
            grid_dim: (grid_p, grid_y, grid_z),
            block_dim: (tp, 1, 1),
            shared_mem_bytes: 0,
        };
        let has_d_i = d_skip.is_some() as i32;
        let (b_i, l_i, h_i, p_i, q_i) = (b as i32, l as i32, h as i32, p as i32, q as i32);
        let d_ptr = d_skip.unwrap_or(x);
        let mut bld = stream.launch_builder(func);
        bld.arg(y_intra)
            .arg(x)
            .arg(d_ptr)
            .arg(&has_d_i)
            .arg(&mut *y_out)
            .arg(&b_i)
            .arg(&l_i)
            .arg(&h_i)
            .arg(&p_i)
            .arg(&q_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch post: {e:?}")))?;
        }
        Ok(())
    }
}

/// Split одномерного `blh = B*L*H` на пару `(grid.y, grid.z)` с обоими
/// измерениями ≤ 65535. Использует grid.y до `MAX_GRID_DIM_YZ = 65535`,
/// остаток уходит в grid.z. Kernel вычисляет `blh = z * gridDim.y + y`.
const MAX_GRID_DIM_YZ: u32 = 65535;
fn split_blh(blh: u32) -> (u32, u32) {
    if blh <= MAX_GRID_DIM_YZ {
        (blh, 1)
    } else {
        // grid.y фиксируем на MAX, grid.z = ceil(blh / MAX).
        let grid_z = blh.div_ceil(MAX_GRID_DIM_YZ);
        (MAX_GRID_DIM_YZ, grid_z)
    }
}

// ── View-варианты для chunk-slicing в orchestrator ─────────────────────────

impl Mamba2ChunkedHelpersKernels {
    /// View-вариант [`Self::col_broadcast_exp_mul`] — для chunk-aware вызова
    /// из orchestrator (src/dst — chunk-views, vec — chunk-view).
    #[allow(clippy::too_many_arguments)]
    pub fn col_broadcast_exp_mul_view(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaView<'_, bf16>,
        vec: &CudaView<'_, f32>,
        dst: &mut CudaViewMut<'_, bf16>,
        bat: u32,
        r: u32,
        c: u32,
        from_end: bool,
    ) -> Result<()> {
        let tx = 16u32.min(c);
        let ty = 16u32.min(r);
        let grid_x = c.div_ceil(tx);
        let grid_y = r.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bat),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bat_i, r_i, c_i, fe_i) = (bat as i32, r as i32, c as i32, from_end as i32);
        let mut bld = stream.launch_builder(&self.col_bcast_exp_mul);
        bld.arg(src)
            .arg(vec)
            .arg(&mut *dst)
            .arg(&bat_i)
            .arg(&r_i)
            .arg(&c_i)
            .arg(&fe_i);
        unsafe {
            bld.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch col_bcast_exp_mul_view: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// View-вариант [`Self::row_broadcast_exp_mul`].
    #[allow(clippy::too_many_arguments)]
    pub fn row_broadcast_exp_mul_view(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaView<'_, bf16>,
        vec: &CudaView<'_, f32>,
        dst: &mut CudaViewMut<'_, bf16>,
        bat: u32,
        r: u32,
        c: u32,
        q_vec: u32,
        from_end: bool,
    ) -> Result<()> {
        let tx = 16u32.min(c);
        let ty = 16u32.min(r);
        let grid_x = c.div_ceil(tx);
        let grid_y = r.div_ceil(ty);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, bat),
            block_dim: (tx, ty, 1),
            shared_mem_bytes: 0,
        };
        let (bat_i, r_i, c_i, qv_i, fe_i) = (
            bat as i32,
            r as i32,
            c as i32,
            q_vec as i32,
            from_end as i32,
        );
        let mut bld = stream.launch_builder(&self.row_bcast_exp_mul);
        bld.arg(src)
            .arg(vec)
            .arg(&mut *dst)
            .arg(&bat_i)
            .arg(&r_i)
            .arg(&c_i)
            .arg(&qv_i)
            .arg(&fe_i);
        unsafe {
            bld.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch row_bcast_exp_mul_view: {e:?}"))
            })?;
        }
        Ok(())
    }
}

// Compile-time check: f32, f16, bf16 — единственные допустимые dtype.
#[doc(hidden)]
pub fn _ensure_dtype_link() -> (DType, DType, DType) {
    (DType::F32, DType::F16, DType::BF16)
}

#[doc(hidden)]
pub fn _ensure_half_link() -> (f32, f16, bf16) {
    (0.0, f16::ZERO, bf16::ZERO)
}
