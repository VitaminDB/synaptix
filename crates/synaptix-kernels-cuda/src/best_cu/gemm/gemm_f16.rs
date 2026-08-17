use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};
use crate::wsalloc::WsAlloc;

const BM: u32 = 128;
const BN: u32 = 128;
const THREADS: u32 = 256;
const SWIZZLE_STRIDE: u32 = 2048;

fn warn_partial_tile(m: u32, n: u32) {
    if m % BM == 0 && n % BN == 0 {
        return;
    }
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u64>>> = OnceLock::new();
    let set = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let key = ((m as u64) << 32) | n as u64;
    if set.lock().insert(key) {
        eprintln!(
            "[best_cu gemm_f16] WARNING: M={m} N={n} не кратно 128 → партиал-тайл \
             (ядро корректно, но эффективная TF ниже: работа считается на полный \
             128-тайл; малый M лучше через GEMV/cuBLAS)"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmConfig {
    pub stages: u32,
    pub swizzle: bool,
    suffix: &'static str,
}

impl GemmConfig {
    pub const S3_SWZ: Self = Self {
        stages: 3,
        swizzle: true,
        suffix: "s3_swz",
    };
    pub const S2_SWZ: Self = Self {
        stages: 2,
        swizzle: true,
        suffix: "s2_swz",
    };
    pub const S4_SWZ: Self = Self {
        stages: 4,
        swizzle: true,
        suffix: "s4_swz",
    };
    pub const S3_NOSWZ: Self = Self {
        stages: 3,
        swizzle: false,
        suffix: "s3_noswz",
    };

    const ALL: [Self; 4] = [Self::S3_SWZ, Self::S2_SWZ, Self::S4_SWZ, Self::S3_NOSWZ];

    fn idx(&self) -> Option<usize> {
        Self::ALL.iter().position(|c| c.suffix == self.suffix)
    }
}

pub fn pick_config(m: u32) -> GemmConfig {
    if m <= 768 {
        GemmConfig::S4_SWZ
    } else {
        GemmConfig::S3_SWZ
    }
}

pub struct GemmF16Kernels {
    _module: Arc<CudaModule>,
    fns: [Option<CudaFunction>; 4],
    part: Option<CudaFunction>,
    bf16_fns: [Option<CudaFunction>; 4],
    bf16_part: Option<CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemmF16Kernels>)>>> = OnceLock::new();

// Грузит NN-набор (4 cfg + part) под префиксом. f16 и bf16 — один модуль
// gemm_f16.cu (общий hgemm_wmma_stages_impl<T>), разные prefix.
fn load_nn_set(
    module: &Arc<CudaModule>,
    prefix: &str,
) -> Result<([Option<CudaFunction>; 4], Option<CudaFunction>)> {
    let mut fns: [Option<CudaFunction>; 4] = std::array::from_fn(|_| None);
    for (ci, cfg) in GemmConfig::ALL.iter().enumerate() {
        fns[ci] = Some(load_fn(module, &format!("{prefix}_{}", cfg.suffix))?);
    }
    let part = Some(load_fn(module, &format!("{prefix}_part"))?);
    Ok((fns, part))
}

impl GemmF16Kernels {
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
        let src = include_str!("gemm_f16.cu");
        let module = compile_module_with_opts(ctx, src, "gemm_f16.cu", &[], Some("sm_120a"))?;
        let (fns, part) = load_nn_set(&module, "gemm_wmma_f16")?;
        let (bf16_fns, bf16_part) = load_nn_set(&module, "gemm_wmma_bf16")?;
        let new = Arc::new(Self {
            fns,
            part,
            bf16_fns,
            bf16_part,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn fnref(&self, cfg: GemmConfig) -> Option<&CudaFunction> {
        self.fns[cfg.idx()?].as_ref()
    }
    fn fnref_part(&self) -> Option<&CudaFunction> {
        self.part.as_ref()
    }
    fn fnref_bf16(&self, cfg: GemmConfig) -> Option<&CudaFunction> {
        self.bf16_fns[cfg.idx()?].as_ref()
    }
    fn fnref_bf16_part(&self) -> Option<&CudaFunction> {
        self.bf16_part.as_ref()
    }
}

fn launch_cfg(m: u32, n: u32, swizzle: bool) -> LaunchConfig {
    let (gx, gy, gz) = if swizzle {
        let n_swz = n.div_ceil(SWIZZLE_STRIDE).max(1);
        (n.div_ceil(BN).div_ceil(n_swz), m.div_ceil(BM), n_swz)
    } else {
        (n.div_ceil(BN), m.div_ceil(BM), 1)
    };
    LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_f16_cfg(
    kernels: &GemmF16Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f16>,
    b: &CudaSlice<f16>,
    c: &mut CudaSlice<f16>,
    m: u32,
    n: u32,
    k: u32,
    cfg: GemmConfig,
) -> Result<()> {
    if k % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "gemm_f16: K={k} must be multiple of 16"
        )));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    warn_partial_tile(m, n);
    let partial = m % BM != 0 || n % BN != 0;
    let (kfn, swizzle) = if partial {
        let f = kernels
            .part
            .as_ref()
            .ok_or_else(|| SynaptixError::Cuda("gemm_f16: нет part-ядра".into()))?;
        (f, false)
    } else {
        let f = kernels
            .fnref(cfg)
            .ok_or_else(|| SynaptixError::Cuda(format!("gemm_f16: нет ядра для {cfg:?}")))?;
        (f, cfg.swizzle)
    };
    let launch = launch_cfg(m, n, swizzle);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let mut bld = stream.launch_builder(kfn);
    bld.arg(a).arg(b).arg(&mut *c).arg(&mi).arg(&ni).arg(&ki);
    unsafe {
        bld.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch gemm_f16: {e:?}")))?;
    }
    Ok(())
}

pub fn gemm_f16(
    kernels: &GemmF16Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f16>,
    b: &CudaSlice<f16>,
    c: &mut CudaSlice<f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    gemm_f16_cfg(kernels, stream, a, b, c, m, n, k, pick_config(m))
}

// NN GEMM C[b,M,N] = A[b,M,K] @ B[(b|1),K,N] на байтовых буферах (f16/bf16,
// float-acc) — для Backend::matmul. Ограничения: K%16==0, N%8==0 (векторный
// cp.async B вдоль N); M/N любые (partial-тайл). Иначе Unsupported → cutlass.
// batched: per-batch launch со смещениями (batch модест для bmm, не storm);
// b_broadcast → B-offset 0 (один B на все батчи).
#[allow(clippy::too_many_arguments)]
pub fn gemm_nn_u8(
    kernels: &GemmF16Kernels,
    dtype: DType,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u8>,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    b_broadcast: bool,
) -> Result<()> {
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err(SynaptixError::Unsupported("gemm_nn_u8: dtype не F16/BF16"));
    }
    if m == 0 || n == 0 || batch == 0 {
        return Ok(());
    }
    // K-tail / малый K: NN-ядро WMMA шагает K по 16, drain требует
    // NUM_K_TILES >= K_STAGE-1 (S4 → K>=48). При K%16!=0 ИЛИ K<48 паддим K нулями:
    // A [b,M,K]→[b,M,K_pad] (pad K-колонок), B [(b|1),K,N]→[(b|1),K_pad,N] (pad
    // K-строк). Нули не вносят вклад. Путь редкий (matmul K большой; нужен conv im2col).
    const MIN_K: u32 = 16 * 3;
    let bb = if b_broadcast { 1 } else { batch };
    let (a_hold, b_hold, k) = if k % 16 == 0 && k >= MIN_K {
        (None, None, k)
    } else {
        let k_pad = (k.div_ceil(16) * 16).max(MIN_K);
        let mut a_pad = stream
            .ws_alloc_zeros::<u8>((batch as usize) * (m as usize) * (k_pad as usize) * 2)
            .map_err(|e| SynaptixError::Cuda(format!("gemm_nn K-pad alloc a: {e:?}")))?;
        let mut b_pad = stream
            .ws_alloc_zeros::<u8>((bb as usize) * (k_pad as usize) * (n as usize) * 2)
            .map_err(|e| SynaptixError::Cuda(format!("gemm_nn K-pad alloc b: {e:?}")))?;
        crate::best_cu::gemm::gemm_bf16::pad_copy_k_bytes(stream, a, &mut a_pad, batch * m, k, k_pad)?;
        crate::best_cu::gemm::gemm_bf16::pad_copy_k_bytes(stream, b, &mut b_pad, bb, k * n, k_pad * n)?;
        (Some(a_pad), Some(b_pad), k_pad)
    };
    let a = a_hold.as_ref().unwrap_or(a);
    let b = b_hold.as_ref().unwrap_or(b);
    // N-pad: B-load векторизуется вдоль N (16 байт) → нужен N%8==0. При N%8!=0
    // паддим B [(b|1),K,N]→[(b|1),K,N_eff] (нули-колонки), считаем в C_pad
    // [b,M,N_eff], затем unpad первые N колонок → C. Путь редкий (N=Cout редко %8≠0).
    let need_n = n % 8 != 0;
    let n_eff = if need_n { n.div_ceil(8) * 8 } else { n };
    let b_npad = if need_n {
        let mut b_n = stream
            .ws_alloc_zeros::<u8>((bb as usize) * (k as usize) * (n_eff as usize) * 2)
            .map_err(|e| SynaptixError::Cuda(format!("gemm_nn N-pad alloc b: {e:?}")))?;
        crate::best_cu::gemm::gemm_bf16::pad_copy_k_bytes(stream, b, &mut b_n, bb * k, n, n_eff)?;
        Some(b_n)
    } else {
        None
    };
    let b = b_npad.as_ref().unwrap_or(b);
    let mut c_pad = if need_n {
        Some(
            stream
                .ws_alloc_zeros::<u8>((batch as usize) * (m as usize) * (n_eff as usize) * 2)
                .map_err(|e| SynaptixError::Cuda(format!("gemm_nn N-pad alloc c: {e:?}")))?,
        )
    } else {
        None
    };

    let partial = m % BM != 0 || n_eff % BN != 0;
    let cfg = pick_config(m);
    let (kfn, swizzle) = if partial {
        let f = match dtype {
            DType::F16 => kernels.fnref_part(),
            _ => kernels.fnref_bf16_part(),
        }
        .ok_or_else(|| SynaptixError::Cuda("gemm_nn_u8: нет part-ядра".into()))?;
        (f, false)
    } else {
        let f = match dtype {
            DType::F16 => kernels.fnref(cfg),
            _ => kernels.fnref_bf16(cfg),
        }
        .ok_or_else(|| SynaptixError::Cuda("gemm_nn_u8: нет ядра".into()))?;
        (f, cfg.swizzle)
    };
    let (am, bk, cm) = ((m * k) as usize, (k * n_eff) as usize, (m * n_eff) as usize);
    let (mi, ni, ki) = (m as i32, n_eff as i32, k as i32);
    {
        let c_target: &mut CudaSlice<u8> = c_pad.as_mut().unwrap_or(c);
        for bi in 0..batch as usize {
            let a_off = bi * am * 2;
            let b_off = if b_broadcast { 0 } else { bi * bk * 2 };
            let c_off = bi * cm * 2;
            let a_v = unsafe { a.slice(a_off..a_off + am * 2).transmute::<f16>(am) }
                .ok_or_else(|| SynaptixError::Cuda("gemm_nn_u8: transmute a".into()))?;
            let b_v = unsafe { b.slice(b_off..b_off + bk * 2).transmute::<f16>(bk) }
                .ok_or_else(|| SynaptixError::Cuda("gemm_nn_u8: transmute b".into()))?;
            let mut c_s = c_target.slice_mut(c_off..c_off + cm * 2);
            let mut c_v = unsafe { c_s.transmute_mut::<f16>(cm) }
                .ok_or_else(|| SynaptixError::Cuda("gemm_nn_u8: transmute c".into()))?;
            let launch = launch_cfg(m, n_eff, swizzle);
            let mut bld = stream.launch_builder(kfn);
            bld.arg(&a_v)
                .arg(&b_v)
                .arg(&mut c_v)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki);
            unsafe {
                bld.launch(launch)
                    .map_err(|e| SynaptixError::Cuda(format!("launch gemm_nn_u8: {e:?}")))?;
            }
        }
    }
    if let Some(cp) = c_pad.as_ref() {
        // unpad C_pad[b,M,N_eff] → C[b,M,N]: первые N колонок каждой строки.
        crate::best_cu::gemm::gemm_bf16::copy_2d_bytes(
            stream,
            cp,
            c,
            batch * m,
            (n as usize) * 2,
            (n_eff as usize) * 2,
            (n as usize) * 2,
            0,
        )?;
    }
    Ok(())
}
