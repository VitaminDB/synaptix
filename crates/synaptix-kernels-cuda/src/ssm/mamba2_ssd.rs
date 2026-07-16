//! Mamba2 State-Space Duality (SSD) — рекуррентная форма.
//!
//! Скалярный decay A на голову (vs матрица [D,N] в Mamba1), структура голов
//! (H × P). Корректный рекуррент (functional, bit-exact против CPU-эталона).
//! Один block = (b, h, p), block_dim = N (степень двойки, ≤ 1024). State
//! держится в регистре, sequential по L.
//!
//! **Chunked-SSD (segment-sum, blocked matmuls по chunks) — будущая perf-
//! оптимизация для long-sequence Mamba2.** На длинных L recurrence по
//! одному элементу за раз даёт O(L) латентность; chunked-режим даёт
//! O(L/chunk_size) синхронизаций ценой O(chunk_size²·N) flops. Сейчас не
//! реализовано — рекуррентная форма используется как baseline и для
//! коротких L.
//!
//! Исходник CUDA: `src/cu/fused/ssm/mamba2_ssd.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct Mamba2SsdKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Mamba2SsdKernels>)>>> = OnceLock::new();

impl Mamba2SsdKernels {
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
        let src = include_str!("../cu/fused/ssm/mamba2_ssd.cu");
        let module = compile_module(ctx, src, "mamba2_ssd.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "mamba2_ssd_f32")?,
            f16: load_fn(&module, "mamba2_ssd_f16")?,
            bf16: load_fn(&module, "mamba2_ssd_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Mamba2 SSD forward.
    /// - `x`      `(B, L, H, P)` row-major
    /// - `dt`     `(B, L, H)` — timestep (>0)
    /// - `a`      `(H,)` — скалярный decay на голову (обычно < 0)
    /// - `b_in`   `(B, L, H, N)` — selective input projection
    /// - `c_in`   `(B, L, H, N)` — selective output projection
    /// - `d_skip` `(H,)` optional (None ⟹ no skip)
    /// - `y`      `(B, L, H, P)` row-major
    ///
    /// `n_state` (N) — степень двойки, ≤ 1024 (block tree-reduce).
    #[allow(clippy::too_many_arguments)]
    pub fn ssd<T: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<T>,
        dt: &CudaSlice<T>,
        a: &CudaSlice<T>,
        b_in: &CudaSlice<T>,
        c_in: &CudaSlice<T>,
        d_skip: Option<&CudaSlice<T>>,
        y: &mut CudaSlice<T>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        n_state: u32,
        dtype: DType,
    ) -> Result<()> {
        if n_state == 0 || (n_state & (n_state - 1)) != 0 || n_state > 1024 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd: n_state={n_state} должно быть степенью двойки ≤ 1024"
            )));
        }
        let func = match dtype {
            DType::F32 => &self.f32,
            DType::F16 => &self.f16,
            DType::BF16 => &self.bf16,
            other => {
                return Err(SynaptixError::Cuda(format!(
                    "mamba2_ssd: unsupported dtype {other:?}"
                )))
            }
        };
        let cfg = LaunchConfig {
            grid_dim: (b * h * p, 1, 1),
            block_dim: (n_state, 1, 1),
            shared_mem_bytes: n_state * std::mem::size_of::<f32>() as u32,
        };
        let has_d_i: i32 = d_skip.is_some() as i32;
        let (b_i, l_i, h_i, p_i, n_i) = (b as i32, l as i32, h as i32, p as i32, n_state as i32);
        let d_skip_ptr = d_skip.unwrap_or(a);
        let mut bld = stream.launch_builder(func);
        bld.arg(x)
            .arg(dt)
            .arg(a)
            .arg(b_in)
            .arg(c_in)
            .arg(d_skip_ptr)
            .arg(&has_d_i)
            .arg(&mut *y)
            .arg(&b_i)
            .arg(&l_i)
            .arg(&h_i)
            .arg(&p_i)
            .arg(&n_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch mamba2_ssd: {e:?}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ssd_f32(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b_in: &CudaSlice<f32>,
        c_in: &CudaSlice<f32>,
        d_skip: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        n_state: u32,
    ) -> Result<()> {
        self.ssd::<f32>(
            stream,
            x,
            dt,
            a,
            b_in,
            c_in,
            d_skip,
            y,
            b,
            l,
            h,
            p,
            n_state,
            DType::F32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ssd_f16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f16>,
        dt: &CudaSlice<f16>,
        a: &CudaSlice<f16>,
        b_in: &CudaSlice<f16>,
        c_in: &CudaSlice<f16>,
        d_skip: Option<&CudaSlice<f16>>,
        y: &mut CudaSlice<f16>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        n_state: u32,
    ) -> Result<()> {
        self.ssd::<f16>(
            stream,
            x,
            dt,
            a,
            b_in,
            c_in,
            d_skip,
            y,
            b,
            l,
            h,
            p,
            n_state,
            DType::F16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ssd_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<bf16>,
        dt: &CudaSlice<bf16>,
        a: &CudaSlice<bf16>,
        b_in: &CudaSlice<bf16>,
        c_in: &CudaSlice<bf16>,
        d_skip: Option<&CudaSlice<bf16>>,
        y: &mut CudaSlice<bf16>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        n_state: u32,
    ) -> Result<()> {
        self.ssd::<bf16>(
            stream,
            x,
            dt,
            a,
            b_in,
            c_in,
            d_skip,
            y,
            b,
            l,
            h,
            p,
            n_state,
            DType::BF16,
        )
    }
}
