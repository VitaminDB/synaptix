//! Mamba2 chunked-SSD helper: batched BF16 matmul с F32 accumulator.
//!
//! Computes `C[b, m, n] = sum_k A[b, m, k] * B[b, n, k]` где
//!  - A `[batch, M, K]` BF16 row-major
//!  - B `[batch, N, K]` BF16 row-major (т.е. B физически = (N, K))
//!  - C `[batch, M, N]` F32  row-major
//!
//! Ограничения: `M % 16 == 0`, `N % 8 == 0`, `K % 16 == 0`. Это фундаментально
//! для `mma.sync m16n8k16`. Для Mamba2 chunked SSD shape-ов (Q=16/32/64,
//! P=64, N_state=128) выполнено по построению.
//!
//! Адаптивный выбор тайла `WARPS_M × WARPS_N`:
//!  - (1, 4) — короткий M (M ≤ 16): tile 16×32, 4 warps.
//!  - (2, 4) — M = 32: tile 32×32, 8 warps.
//!  - (4, 2) — большой M / маленький N: tile 64×16, 8 warps.
//!  - (4, 4) — большие M и N: tile 64×32, 16 warps.
//!  - (2, 2) — fallback: tile 32×16, 4 warps.
//!
//! Исходник CUDA: `src/cu/ssm/mamba2_bmm.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg,
};
use half::bf16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct Mamba2BmmKernels {
    _module: Arc<CudaModule>,
    bmm_1x4: CudaFunction,
    bmm_2x2: CudaFunction,
    bmm_2x4: CudaFunction,
    bmm_4x2: CudaFunction,
    bmm_4x4: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Mamba2BmmKernels>)>>> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
struct BmmConfig {
    warps_m: u32,
    warps_n: u32,
}

impl BmmConfig {
    fn tile_m(self) -> u32 {
        self.warps_m * 16
    }
    fn tile_n(self) -> u32 {
        self.warps_n * 8
    }
    fn threads(self) -> u32 {
        self.warps_m * self.warps_n * 32
    }
    fn smem_bytes(self) -> u32 {
        // (warps_m*16 + warps_n*8) × 16 × sizeof(bf16=2)
        (self.warps_m * 16 + self.warps_n * 8) * 16 * 2
    }
}

fn pick_config(m: u32, n: u32) -> BmmConfig {
    if m <= 16 {
        BmmConfig {
            warps_m: 1,
            warps_n: 4,
        }
    } else if m == 32 {
        BmmConfig {
            warps_m: 2,
            warps_n: 4,
        }
    } else if n <= 16 {
        BmmConfig {
            warps_m: 4,
            warps_n: 2,
        }
    } else if m >= 64 && n >= 32 {
        BmmConfig {
            warps_m: 4,
            warps_n: 4,
        }
    } else {
        BmmConfig {
            warps_m: 2,
            warps_n: 2,
        }
    }
}

impl Mamba2BmmKernels {
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
        let src = include_str!("../cu/ssm/mamba2_bmm.cu");
        let module = compile_module(ctx, src, "mamba2_bmm.cu")?;
        let new = Arc::new(Self {
            bmm_1x4: load_fn(&module, "mamba2_bmm_bf16_f32acc_1x4")?,
            bmm_2x2: load_fn(&module, "mamba2_bmm_bf16_f32acc_2x2")?,
            bmm_2x4: load_fn(&module, "mamba2_bmm_bf16_f32acc_2x4")?,
            bmm_4x2: load_fn(&module, "mamba2_bmm_bf16_f32acc_4x2")?,
            bmm_4x4: load_fn(&module, "mamba2_bmm_bf16_f32acc_4x4")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn pick_fn(&self, cfg: BmmConfig) -> &CudaFunction {
        match (cfg.warps_m, cfg.warps_n) {
            (1, 4) => &self.bmm_1x4,
            (2, 2) => &self.bmm_2x2,
            (2, 4) => &self.bmm_2x4,
            (4, 2) => &self.bmm_4x2,
            (4, 4) => &self.bmm_4x4,
            _ => unreachable!("pick_config returns only configured variants"),
        }
    }

    /// `c[b, m, n] = sum_k a[b, m, k] * b_tensor[b, n, k]`.
    ///
    /// Размеры: `a` ≥ `batch*M*K`, `b_tensor` ≥ `batch*N*K`, `c` ≥ `batch*M*N`.
    /// Требования: `M % 16 == 0`, `N % 8 == 0`, `K % 16 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn bmm(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<bf16>,
        b_tensor: &CudaSlice<bf16>,
        c: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 || batch == 0 {
            return Ok(());
        }
        if m % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: M={m} должно быть кратно 16"
            )));
        }
        if n % 8 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: N={n} должно быть кратно 8"
            )));
        }
        if k % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: K={k} должно быть кратно 16"
            )));
        }
        let needed_a = (batch as usize) * (m as usize) * (k as usize);
        let needed_b = (batch as usize) * (n as usize) * (k as usize);
        let needed_c = (batch as usize) * (m as usize) * (n as usize);
        if a.len() < needed_a {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: a slice {} < required {needed_a}",
                a.len()
            )));
        }
        if b_tensor.len() < needed_b {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: b slice {} < required {needed_b}",
                b_tensor.len()
            )));
        }
        if c.len() < needed_c {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm: c slice {} < required {needed_c}",
                c.len()
            )));
        }

        let cfg = pick_config(m, n);
        let func = self.pick_fn(cfg);

        let grid_x = m.div_ceil(cfg.tile_m());
        let grid_y = n.div_ceil(cfg.tile_n());
        let launch = LaunchConfig {
            grid_dim: (grid_x, grid_y, batch),
            block_dim: (cfg.threads(), 1, 1),
            shared_mem_bytes: cfg.smem_bytes(),
        };

        let mut bld = stream.launch_builder(func);
        bld.arg(a)
            .arg(b_tensor)
            .arg(&mut *c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&batch);
        unsafe {
            bld.launch(launch)
                .map_err(|e| SynaptixError::Cuda(format!("launch mamba2_bmm: {e:?}")))?;
        }
        Ok(())
    }

    /// View-based вариант [`Self::bmm`]: принимает CudaView/CudaViewMut для
    /// работы с подсегментами workspace (per-chunk slicing в orchestrator).
    /// Без size-checks — caller гарантирует достаточный размер view'ов.
    #[allow(clippy::too_many_arguments)]
    pub fn bmm_view(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaView<'_, bf16>,
        b_tensor: &CudaView<'_, bf16>,
        c: &mut CudaViewMut<'_, f32>,
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 || batch == 0 {
            return Ok(());
        }
        if m % 16 != 0 || n % 8 != 0 || k % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_bmm_view: M={m}/N={n}/K={k} должны быть кратны 16/8/16"
            )));
        }

        let cfg = pick_config(m, n);
        let func = self.pick_fn(cfg);

        let grid_x = m.div_ceil(cfg.tile_m());
        let grid_y = n.div_ceil(cfg.tile_n());
        let launch = LaunchConfig {
            grid_dim: (grid_x, grid_y, batch),
            block_dim: (cfg.threads(), 1, 1),
            shared_mem_bytes: cfg.smem_bytes(),
        };

        let mut bld = stream.launch_builder(func);
        bld.arg(a)
            .arg(b_tensor)
            .arg(&mut *c)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&batch);
        unsafe {
            bld.launch(launch)
                .map_err(|e| SynaptixError::Cuda(format!("launch mamba2_bmm_view: {e:?}")))?;
        }
        Ok(())
    }
}
