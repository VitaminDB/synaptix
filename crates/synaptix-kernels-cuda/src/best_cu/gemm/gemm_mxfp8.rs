use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};
use crate::wsalloc::WsAlloc;

// КОРРЕКТНОЕ MXFP8 GEMM (sm_120a), порт gau-nernst/learn-cuda 09a_block_scaled_mm_sm120 v1:
// cp.async-конвейер (БЕЗ TMA), натуральные E8M0-scale, cos=0.999999 на outlier. См. gemm_mxfp8.cu.
// Только 128×128×128-кратные формы; иначе вызывающий делает fallback.
const MXFP8_BLOCK: u32 = 128;
fn mxfp8_smem() -> u32 {
    (MXFP8_BLOCK + MXFP8_BLOCK) * (MXFP8_BLOCK + MXFP8_BLOCK / 32) * 3 // (BM+BN)*(BK+BK/32)*NSTAGES
}

// ROT-ядро (рецепт CUTLASS-порта nvfp4): тайл 128×128, 2 TMA-стадии + барьеры.
const ROT_BM: u32 = 128;
const ROT_BN: u32 = 128;
const ROT_STAGES: u32 = 2;
fn mxfp8_rot_smem() -> u32 {
    // данные (BM+BN)*128 + SF-окна по 16Б/строку (TMA inner-dim минимум).
    ROT_STAGES * ((ROT_BM + ROT_BN) * (128 + 16)) + 2 * ROT_STAGES * 8
}

// drot-конфиги: 128×128 (база) + 64-тайлы для средних M (attn-256: 128-тайл
// даёт 64 CTA — полмашины) + широкий 64×256 (bf16-урок b256t).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotCfg {
    pub bm: u32,
    pub bn: u32,
    pub stages: u32,
    fname: &'static str,
    pub suffix: &'static str,
}

impl RotCfg {
    pub const D128: Self = Self { bm: 128, bn: 128, stages: 2, fname: "gn_mxfp8_drot_128x128_s2", suffix: "d128" };
    pub const D64X128S2: Self = Self { bm: 64, bn: 128, stages: 2, fname: "gn_mxfp8_drot_64x128_s2", suffix: "d64x128s2" };
    pub const D64X128S3: Self = Self { bm: 64, bn: 128, stages: 3, fname: "gn_mxfp8_drot_64x128_s3", suffix: "d64x128s3" };
    pub const D64X256S2: Self = Self { bm: 64, bn: 256, stages: 2, fname: "gn_mxfp8_drot_64x256_s2", suffix: "d64x256s2" };
    pub const ALL: [Self; 4] = [Self::D128, Self::D64X128S2, Self::D64X128S3, Self::D64X256S2];
    fn smem(&self) -> u32 {
        self.stages * ((self.bm + self.bn) * (128 + 16)) + 2 * self.stages * 8
    }
    pub fn fits(&self, m: u32, n: u32, k: u32) -> bool {
        // m ЛЮБОЕ >=1: TMA зануляет OOB-чтения хвостового M-ряда, эпилог
        // гардит сторы (рецепт bf16 BT-ядер).
        m >= 1 && n % self.bn == 0 && k % 512 == 0 && k / 128 >= self.stages
    }
}

pub struct GemmMxFp8Kernels {
    _module: Arc<CudaModule>,
    f: CudaFunction,
    f_rot: CudaFunction,
    f_drot: CudaFunction,
    f_drot_cfgs: [CudaFunction; 4],
    f_drot_sk: CudaFunction,
    f_sk_reduce: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemmMxFp8Kernels>)>>> = OnceLock::new();

impl GemmMxFp8Kernels {
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
        let src = include_str!("gemm_mxfp8.cu");
        let module = compile_module_with_opts(ctx, src, "gemm_mxfp8.cu", &[], Some("sm_120a"))?;
        let f = load_fn(&module, "gn_mxfp8_128x128")?;
        f.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            mxfp8_smem() as i32,
        )
        .map_err(|e| SynaptixError::Cuda(format!("set_attr mxfp8: {e:?}")))?;
        let f_rot = load_fn(&module, "gn_mxfp8_rot_128x128_s2")?;
        f_rot
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mxfp8_rot_smem() as i32,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attr mxfp8_rot: {e:?}")))?;
        let f_drot = load_fn(&module, "gn_mxfp8_drot_128x128_s2")?;
        f_drot
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mxfp8_rot_smem() as i32,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attr mxfp8_drot: {e:?}")))?;
        let mut f_drot_cfgs: Vec<CudaFunction> = Vec::with_capacity(4);
        for cfg in RotCfg::ALL {
            let fc = load_fn(&module, cfg.fname)?;
            fc.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                cfg.smem() as i32,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attr {}: {e:?}", cfg.fname)))?;
            f_drot_cfgs.push(fc);
        }
        let f_drot_cfgs: [CudaFunction; 4] = f_drot_cfgs
            .try_into()
            .map_err(|_| SynaptixError::Cuda("mxfp8: drot cfgs".into()))?;
        let f_drot_sk = load_fn(&module, "gn_mxfp8_drot_128x128_s2_sk")?;
        f_drot_sk
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                RotCfg::D128.smem() as i32,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attr mxfp8_drot_sk: {e:?}")))?;
        let f_sk_reduce = load_fn(&module, "mxfp8_sk_reduce")?;
        let new = Arc::new(Self {
            f,
            f_rot,
            f_drot,
            f_drot_cfgs,
            f_drot_sk,
            f_sk_reduce,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// ROT GEMM на предквантованных операндах (natural-лейауты как gemm_mxfp8).
/// M%128==N%128==0, K%512==0, K/128>=2. Bit-exact к gn_mxfp8_128x128.
/// drot: выделенный producer-warpgroup (384 потока) вместо fused-tid0.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8_rot(
    kernels: &GemmMxFp8Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    sb: &CudaSlice<u8>,
    y: &mut cudarc::driver::CudaViewMut<'_, f16>,
    m: u32,
    n: u32,
    k: u32,
    drot: bool,
) -> Result<()> {
    use super::gemm_nvfp4::cached_desc_2d;
    use cudarc::driver::sys::CUtensorMapSwizzle;
    use cudarc::driver::DevicePtr;
    // k%512: SF-окна TMA по 16Б = 16 k32-блоков (K/32 кратно 16).
    // Конфиг drot: pick_rot_cfg (data-driven по форме).
    let cfg = if drot {
        pick_rot_cfg(m, n, k)
    } else {
        RotCfg::D128
    };
    if !cfg.fits(m, n, k) {
        return Err(SynaptixError::Cuda(format!(
            "gemm_mxfp8_rot: {m}x{n}x{k} не подходит"
        )));
    }
    let (a_ptr, _ra) = a.device_ptr(stream);
    let (b_ptr, _rb) = b.device_ptr(stream);
    let (sa_ptr, _rsa) = sa.device_ptr(stream);
    let (sb_ptr, _rsb) = sb.device_ptr(stream);
    let swz = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    const NOSWZ: CUtensorMapSwizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    let a_desc = cached_desc_2d(stream, a_ptr, m, k, cfg.bm, 128, swz)?;
    let b_desc = cached_desc_2d(stream, b_ptr, n, k, cfg.bn, 128, swz)?;
    let sfa_desc = cached_desc_2d(stream, sa_ptr, m, k / 32, cfg.bm, 16, NOSWZ)?;
    let sfb_desc = cached_desc_2d(stream, sb_ptr, n, k / 32, cfg.bn, 16, NOSWZ)?;
    let (kfn, threads) = if drot {
        let idx = RotCfg::ALL.iter().position(|c| *c == cfg).unwrap_or(0);
        (&kernels.f_drot_cfgs[idx], 384)
    } else {
        (&kernels.f_rot, 256)
    };
    // L2-растр: окно волны raster_gr N-тайлов × M-полоса. Вес > L2 (ff_up 64MB) →
    // узкое окно 8 (180→249 TF); вес ≤ L2 (attn 8MB) → широкое 32 (264→273).
    let tiles_n = n / cfg.bn;
    let mut raster_gr: u32 = if (n as u64) * (k as u64) <= 24 * 1024 * 1024 {
        32
    } else {
        8
    };
    while raster_gr > 1 && tiles_n % raster_gr != 0 {
        raster_gr /= 2;
    }
    let raster_gr = raster_gr.max(1);
    // TMA-store эпилог: дескриптор выхода [m, n*2Б], бокс = варп-регион
    // [WARP_M=bm/2, WARP_N*2Б=bn/4*2]; свизл слота по ширине строки.
    let out_ptr = {
        let (p, _ro) = y.device_ptr(stream);
        p
    };
    let wn_bytes = (cfg.bn / 4) * 2;
    let out_swz = match wn_bytes {
        128 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        64 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        _ => NOSWZ,
    };
    let out_desc = cached_desc_2d(stream, out_ptr, m, n * 2, cfg.bm / 2, wn_bytes, out_swz)?;
    let mut bld = stream.launch_builder(kfn);
    bld.arg(&*a_desc)
        .arg(&*b_desc)
        .arg(&*sfa_desc)
        .arg(&*sfb_desc)
        .arg(&mut *y)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&raster_gr)
        .arg(&*out_desc);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (tiles_n * m.div_ceil(cfg.bm), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: cfg.smem(),
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8_rot: {e:?}")))?;
    }
    Ok(())
}

// Data-driven выбор drot-конфига (events-свип 2026-06-06, zoo
// bench/results/retest_mxfp8_cfg_*.txt): малые M — 64-тайлы (грид ×2:
// attn-32 +25%, ffd-64 +33%, ffd-128 +22%), глубокий K → s3 (конвейер),
// M=192 — ffd d64x256 +7.8% / ffu s3 +13.2%; M>=256 везде D128.
fn pick_rot_cfg(m: u32, n: u32, k: u32) -> RotCfg {
    let deep_k = k >= 16384;
    let cand = if m <= 32 {
        if deep_k { RotCfg::D64X128S2 } else { RotCfg::D64X128S3 }
    } else if m <= 64 {
        if deep_k { RotCfg::D64X128S3 } else { RotCfg::D64X128S2 }
    } else if m <= 128 {
        if deep_k {
            RotCfg::D64X128S3
        } else if n <= 4096 {
            RotCfg::D64X128S2
        } else {
            RotCfg::D128
        }
    } else if m <= 192 {
        if deep_k {
            RotCfg::D64X256S2
        } else if n >= 16384 {
            RotCfg::D64X128S3
        } else {
            RotCfg::D128
        }
    } else {
        RotCfg::D128
    };
    if cand.fits(m, n, k) {
        cand
    } else {
        RotCfg::D128
    }
}

/// split-K вариант drot 128×128: грид (tiles, splits), f32-ws + reduce f16.
/// kt_chunk кратен 4 (SF-окно TMA по 4 стадии). Детерминизм: фикс-порядок в reduce.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8_rot_splitk(
    kernels: &GemmMxFp8Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    sb: &CudaSlice<u8>,
    y: &mut cudarc::driver::CudaViewMut<'_, f16>,
    m: u32,
    n: u32,
    k: u32,
    splits: u32,
) -> Result<()> {
    use super::gemm_nvfp4::cached_desc_2d;
    use cudarc::driver::sys::CUtensorMapSwizzle;
    use cudarc::driver::DevicePtr;
    let cfg = RotCfg::D128;
    if !cfg.fits(m, n, k) || splits < 2 {
        return Err(SynaptixError::Cuda("mxfp8 splitk: форма/splits".into()));
    }
    let total_kt = k / 128;
    let kt_chunk = total_kt.div_ceil(splits).next_multiple_of(4);
    let used = total_kt.div_ceil(kt_chunk);
    let mn = (m as usize) * (n as usize);
    let mut ws: CudaSlice<f32> = unsafe { stream.alloc(mn * used as usize) }
        .map_err(|e| SynaptixError::Cuda(format!("mxfp8 splitk ws: {e:?}")))?;
    let (a_ptr, _ra) = a.device_ptr(stream);
    let (b_ptr, _rb) = b.device_ptr(stream);
    let (sa_ptr, _rsa) = sa.device_ptr(stream);
    let (sb_ptr, _rsb) = sb.device_ptr(stream);
    let swz = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    const NOSWZ: CUtensorMapSwizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    let a_desc = cached_desc_2d(stream, a_ptr, m, k, cfg.bm, 128, swz)?;
    let b_desc = cached_desc_2d(stream, b_ptr, n, k, cfg.bn, 128, swz)?;
    let sfa_desc = cached_desc_2d(stream, sa_ptr, m, k / 32, cfg.bm, 16, NOSWZ)?;
    let sfb_desc = cached_desc_2d(stream, sb_ptr, n, k / 32, cfg.bn, 16, NOSWZ)?;
    let tiles_n = n / cfg.bn;
    let mut raster_gr: u32 = if (n as u64) * (k as u64) <= 24 * 1024 * 1024 { 32 } else { 8 };
    while raster_gr > 1 && tiles_n % raster_gr != 0 {
        raster_gr /= 2;
    }
    let raster_gr = raster_gr.max(1);
    let mut bld = stream.launch_builder(&kernels.f_drot_sk);
    bld.arg(&*a_desc)
        .arg(&*b_desc)
        .arg(&*sfa_desc)
        .arg(&*sfb_desc)
        .arg(&mut ws)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&raster_gr)
        .arg(&kt_chunk);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (tiles_n * m.div_ceil(cfg.bm), used, 1),
            block_dim: (384, 1, 1),
            shared_mem_bytes: cfg.smem(),
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8_rot_sk: {e:?}")))?;
    }
    let (mn_ll, splits_i) = (mn as i64, used as i32);
    let mut bld = stream.launch_builder(&kernels.f_sk_reduce);
    bld.arg(&ws).arg(&mut *y).arg(&mn_ll).arg(&splits_i);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: ((mn as u32).div_ceil(256).min(4096), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8_sk_reduce: {e:?}")))?;
    }
    Ok(())
}

/// GEMM на ПРЕДКВАНТОВАННЫХ операндах: a/b = e4m3 [M,K]/[N,K] (natural row-major),
/// sa/sb = натуральные E8M0 [rows,K/32], y = f16 [M,N]. M%128==N%128==K%128==0.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8(
    kernels: &GemmMxFp8Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    sb: &CudaSlice<u8>,
    y: &mut cudarc::driver::CudaViewMut<'_, f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m % MXFP8_BLOCK != 0 || n % MXFP8_BLOCK != 0 || k % MXFP8_BLOCK != 0 {
        return Err(SynaptixError::Cuda(format!(
            "gemm_mxfp8: {m}x{n}x{k} не кратно {MXFP8_BLOCK}"
        )));
    }
    let grid = m.div_ceil(MXFP8_BLOCK) * n.div_ceil(MXFP8_BLOCK);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let mut bld = stream.launch_builder(&kernels.f);
    bld.arg(a).arg(b).arg(sa).arg(sb).arg(&mut *y).arg(&mi).arg(&ni).arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: mxfp8_smem(),
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8: {e:?}")))?;
    }
    Ok(())
}

/// Высокоуровневый MXFP8 linear: Y[M,N] f16 = X[M,K] f16 @ W[N,K]ᵀ. Натуральный device-quant
/// обоих операндов + корректное ядро (cp.async). M%128==N%128==K%128==0.
pub fn gemm_mxfp8_linear(
    qk: &crate::elementwise::quant::Mxfp8QuantKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    w: &CudaSlice<f16>,
    y: &mut cudarc::driver::CudaViewMut<'_, f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use crate::elementwise::quant::mxfp8_quant_natural;
    let ctx = stream.context();
    let gk = GemmMxFp8Kernels::for_context(&ctx)?;
    let mut xq = stream
        .ws_alloc_zeros::<u8>((m * k) as usize)
        .map_err(|e| SynaptixError::Cuda(format!("mxfp8 alloc xq: {e:?}")))?;
    let mut wq = stream
        .ws_alloc_zeros::<u8>((n * k) as usize)
        .map_err(|e| SynaptixError::Cuda(format!("mxfp8 alloc wq: {e:?}")))?;
    let mut sa = stream
        .ws_alloc_zeros::<u8>((m * k / 32) as usize)
        .map_err(|e| SynaptixError::Cuda(format!("mxfp8 alloc sa: {e:?}")))?;
    let mut sb = stream
        .ws_alloc_zeros::<u8>((n * k / 32) as usize)
        .map_err(|e| SynaptixError::Cuda(format!("mxfp8 alloc sb: {e:?}")))?;
    mxfp8_quant_natural(qk, stream, &x.as_view(), &mut xq, &mut sa, m, k)?;
    mxfp8_quant_natural(qk, stream, &w.as_view(), &mut wq, &mut sb, n, k)?;
    gemm_mxfp8(&gk, stream, &xq, &wq, &sa, &sb, y, m, n, k)
}
