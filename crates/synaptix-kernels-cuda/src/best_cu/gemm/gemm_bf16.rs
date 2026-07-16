use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::bf16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 16;
const WARP_TILE_K: u32 = 2;
const THREADS: u32 = 256;
const SWIZZLE_STRIDE: u32 = 2048;

static BF16_CFG_OVERRIDE: Mutex<Option<String>> = Mutex::new(None);

pub fn set_bf16_cfg_override(cfg: Option<&str>) {
    *BF16_CFG_OVERRIDE.lock() = cfg.map(|s| s.to_string());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16Config {
    pub stages: u32,
    pub bm: u32,
    pub bn: u32,
    suffix: &'static str,
}

impl Bf16Config {
    pub const S2: Self = Self {
        stages: 2,
        bm: 128,
        bn: 128,
        suffix: "s2",
    };
    pub const S3: Self = Self {
        stages: 3,
        bm: 128,
        bn: 128,
        suffix: "s3",
    };
    pub const S4: Self = Self {
        stages: 4,
        bm: 128,
        bn: 128,
        suffix: "s4",
    };
    pub const S5: Self = Self {
        stages: 5,
        bm: 128,
        bn: 128,
        suffix: "s5",
    };
    pub const S6: Self = Self {
        stages: 6,
        bm: 128,
        bn: 128,
        suffix: "s6",
    };
    pub const B256S3: Self = Self {
        stages: 3,
        bm: 256,
        bn: 128,
        suffix: "b256s3",
    };
    pub const B256S4: Self = Self {
        stages: 4,
        bm: 256,
        bn: 128,
        suffix: "b256s4",
    };
    pub const S64S3: Self = Self {
        stages: 3,
        bm: 64,
        bn: 64,
        suffix: "s64s3",
    };
    pub const S64S4: Self = Self {
        stages: 4,
        bm: 64,
        bn: 64,
        suffix: "s64s4",
    };
    pub const S64S6: Self = Self {
        stages: 6,
        bm: 64,
        bn: 64,
        suffix: "s64s6",
    };
    pub const S64S8: Self = Self {
        stages: 8,
        bm: 64,
        bn: 64,
        suffix: "s64s8",
    };
    pub const B256TS3: Self = Self {
        stages: 3,
        bm: 128,
        bn: 256,
        suffix: "b256ts3",
    };
    pub const B256TS4: Self = Self {
        stages: 4,
        bm: 128,
        bn: 256,
        suffix: "b256ts4",
    };
    const ALL: [Self; 13] = [
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5,
        Self::S6,
        Self::B256S3,
        Self::B256S4,
        Self::B256TS3,
        Self::B256TS4,
        Self::S64S3,
        Self::S64S4,
        Self::S64S6,
        Self::S64S8,
    ];

    fn from_override() -> Option<Self> {
        let g = BF16_CFG_OVERRIDE.lock();
        let v = g.as_deref()?;
        Self::ALL.iter().copied().find(|c| c.suffix == v)
    }

    fn idx(&self) -> Option<usize> {
        Self::ALL.iter().position(|c| c.suffix == self.suffix)
    }
    fn smem_bytes(&self) -> u32 {
        self.stages * (self.bm + self.bn) * BK * WARP_TILE_K * 2
    }
}

pub struct BestGemmBf16Kernels {
    _module: Arc<CudaModule>,
    fns: [Option<CudaFunction>; 13],
    parts: [Option<CudaFunction>; 13],
    f16_fns: [Option<CudaFunction>; 13],
    f16_parts: [Option<CudaFunction>; 13],
    splitk: Option<CudaFunction>,
    splitk_reduce: Option<CudaFunction>,
    f16_splitk: Option<CudaFunction>,
    f16_splitk_reduce: Option<CudaFunction>,
    splitk_b256: Option<CudaFunction>,
    f16_splitk_b256: Option<CudaFunction>,
    splitk_cfg: Bf16Config,
    splitk_align: u32,
    k128: Option<CudaFunction>,
    f16_k128: Option<CudaFunction>,
    k64: Option<CudaFunction>,
    f16_k64: Option<CudaFunction>,
    k128m32l2: Option<CudaFunction>,
    f16_k128m32l2: Option<CudaFunction>,
    tma_s3: Option<CudaFunction>,
    tma_s5: Option<CudaFunction>,
    tma_s5d: Option<CudaFunction>,
    f16_tma_s3: Option<CudaFunction>,
    f16_tma_s5: Option<CudaFunction>,
    f16_tma_s5d: Option<CudaFunction>,
    tma_b64x128_s3d: Option<CudaFunction>,
    f16_tma_b64x128_s3d: Option<CudaFunction>,
    // кэш TMA-дескрипторов: ключ (ptr, rows, cols_b) — дескриптор зависит только
    // от адреса/лейаута (не от данных); веса стабильны между вызовами → хиты.
    tma_descs: Mutex<std::collections::HashMap<(u64, u32, u32), CudaSlice<u8>>>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<BestGemmBf16Kernels>)>>> = OnceLock::new();

// Грузит TN-набор (swz_* + part_*) под заданным префиксом entry-точек.
// bf16 и f16 — один модуль gemm_bf16.cu (общий gemm_bf16_impl<T>), разные prefix.
// part-инстанциации есть только для s2/s3 (s3 = историческое имя `_part` без суффикса).
fn load_tn_set(
    module: &Arc<CudaModule>,
    prefix: &str,
) -> Result<([Option<CudaFunction>; 13], [Option<CudaFunction>; 13])> {
    let mut fns: [Option<CudaFunction>; 13] = std::array::from_fn(|_| None);
    let mut parts: [Option<CudaFunction>; 13] = std::array::from_fn(|_| None);
    for (ci, cfg) in Bf16Config::ALL.iter().enumerate() {
        let name = format!("{prefix}_swz_{}", cfg.suffix);
        let f = load_fn(module, &name)?;
        f.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            cfg.smem_bytes() as i32,
        )
        .map_err(|e| SynaptixError::Cuda(format!("set smem {name}: {e:?}")))?;
        let _ = f.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            100,
        );
        fns[ci] = Some(f);
        let part_name = match cfg.suffix {
            "s3" => format!("{prefix}_part"),
            s => format!("{prefix}_part_{s}"),
        };
        if let Ok(pf) = load_fn(module, &part_name) {
            pf.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                cfg.smem_bytes() as i32,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set smem {part_name}: {e:?}")))?;
            parts[ci] = Some(pf);
        }
    }
    Ok((fns, parts))
}

impl BestGemmBf16Kernels {
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
        let module = {
            let src = include_str!("gemm_bf16.cu");
            compile_module_with_opts(ctx, src, "gemm_bf16.cu", &[], Some("sm_120a"))?
        };
        let (fns, parts) = load_tn_set(&module, "gemm_bf16")?;
        let (f16_fns, f16_parts) = load_tn_set(&module, "gemm_f16tn")?;
        let load_opt = |name: &str| -> Option<CudaFunction> {
            let f = load_fn(&module, name).ok()?;
            let smem = if name.ends_with("k64") {
                Bf16Config::S64S4.smem_bytes()
            } else {
                Bf16Config::S64S8.smem_bytes()
            };
            let _ = f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            );
            Some(f)
        };
        let load_b256sk = |name: &str| -> Option<CudaFunction> {
            let f = load_fn(&module, name).ok()?;
            let _ = f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                Bf16Config::B256S3.smem_bytes() as i32,
            );
            Some(f)
        };
        // TMA-входы (порт рецепта mxfp8-rot) — в том же модуле gemm_bf16.cu.
        let tma_smem = |stages: u32| stages * (64 + 64) * 128 + 2 * stages * 8;
        let load_tma = |name: &str, stages: u32| -> Option<CudaFunction> {
            let f = load_fn(&module, name).ok()?;
            let _ = f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                tma_smem(stages) as i32,
            );
            Some(f)
        };
        // generic TMA-загрузчик: smem от (bm, bn, stages) — крупные тайлы.
        let load_tma_g = |name: &str, bm: u32, bn: u32, stages: u32| -> Option<CudaFunction> {
            let f = load_fn(&module, name).ok()?;
            let _ = f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                (stages * (bm + bn) * 128 + 2 * stages * 8) as i32,
            );
            Some(f)
        };
        let new = Arc::new(Self {
            fns,
            parts,
            f16_fns,
            f16_parts,
            tma_s3: load_tma("gn_bf16_tma_64x64_s3", 3),
            tma_s5: load_tma("gn_bf16_tma_64x64_s5", 5),
            tma_s5d: load_tma("gn_bf16_tma_64x64_s5d", 5),
            f16_tma_s3: load_tma("gn_f16_tma_64x64_s3", 3),
            f16_tma_s5: load_tma("gn_f16_tma_64x64_s5", 5),
            f16_tma_s5d: load_tma("gn_f16_tma_64x64_s5d", 5),
            tma_b64x128_s3d: load_tma_g("gn_bf16_tma_64x128_s3d", 64, 128, 3),
            f16_tma_b64x128_s3d: load_tma_g("gn_f16_tma_64x128_s3d", 64, 128, 3),
            tma_descs: Mutex::new(std::collections::HashMap::new()),
            splitk: load_opt("gemm_bf16_splitk_k128"),
            splitk_reduce: load_opt("gemm_bf16_splitk_reduce"),
            f16_splitk: load_opt("gemm_f16tn_splitk_k128"),
            f16_splitk_reduce: load_opt("gemm_f16tn_splitk_reduce"),
            splitk_b256: load_b256sk("gemm_bf16_splitk_b256s3"),
            f16_splitk_b256: load_b256sk("gemm_f16tn_splitk_b256s3"),
            splitk_cfg: Bf16Config::S64S8,
            splitk_align: 128,
            k128: load_opt("gemm_bf16_k128"),
            f16_k128: load_opt("gemm_f16tn_k128"),
            k64: load_opt("gemm_bf16_k64"),
            f16_k64: load_opt("gemm_f16tn_k64"),
            k128m32l2: load_opt("gemm_bf16_k128m32l2"),
            f16_k128m32l2: load_opt("gemm_f16tn_k128m32l2"),
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn fnref(&self, cfg: Bf16Config) -> Option<&CudaFunction> {
        Some(self.fns[cfg.idx()?].as_ref()?)
    }
    fn fnref_part_cfg(&self, cfg: Bf16Config) -> Option<&CudaFunction> {
        Some(self.parts[cfg.idx()?].as_ref()?)
    }
    fn fnref_part(&self) -> Option<&CudaFunction> {
        self.fnref_part_cfg(Bf16Config::S3)
    }
    fn fnref_f16(&self, cfg: Bf16Config) -> Option<&CudaFunction> {
        Some(self.f16_fns[cfg.idx()?].as_ref()?)
    }
    fn fnref_f16_part_cfg(&self, cfg: Bf16Config) -> Option<&CudaFunction> {
        Some(self.f16_parts[cfg.idx()?].as_ref()?)
    }
}

// Растр (N-чанкинг через grid.z) теперь и у part-гибрида — грид одинаков
// для обоих путей; PARTIAL-ядро гасит overshoot bx*BN>=N через b_valid.
fn launch_for(m: u32, n: u32, _partial: bool, smem: u32, bm: u32, bn: u32) -> LaunchConfig {
    let n_swz = n.div_ceil(SWIZZLE_STRIDE).max(1);
    LaunchConfig {
        grid_dim: (n.div_ceil(bn).div_ceil(n_swz), m.div_ceil(bm), n_swz),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: smem,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn best_gemm_bf16_cfg(
    kernels: &BestGemmBf16Kernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    m: u32,
    n: u32,
    k: u32,
    cfg: Bf16Config,
) -> Result<()> {
    if k % (BK * WARP_TILE_K) != 0 {
        return Err(SynaptixError::Cuda(format!(
            "best_gemm_bf16: K={k} must be multiple of {}",
            BK * WARP_TILE_K
        )));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    let partial = m % cfg.bm != 0 || n % cfg.bn != 0;
    let (kfn, run_cfg) = if partial {
        (
            kernels
                .fnref_part()
                .ok_or_else(|| SynaptixError::Cuda("best_gemm_bf16: нет part-ядра".into()))?,
            Bf16Config::S3,
        )
    } else {
        (
            kernels.fnref(cfg).ok_or_else(|| {
                SynaptixError::Cuda(format!("best_gemm_bf16: нет ядра для {cfg:?}"))
            })?,
            cfg,
        )
    };
    let launch = launch_for(m, n, partial, run_cfg.smem_bytes(), run_cfg.bm, run_cfg.bn);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let (no_bias, no_res) = (0i32, 0i32);
    let mut bld = stream.launch_builder(kfn);
    bld.arg(x).arg(w).arg(&mut *y).arg(&mi).arg(&ni).arg(&ki)
        .arg(w).arg(&no_bias).arg(w).arg(&no_res);
    unsafe {
        bld.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch best_gemm_bf16: {e:?}")))?;
    }
    Ok(())
}

pub fn best_gemm_bf16(
    kernels: &BestGemmBf16Kernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    best_gemm_bf16_cfg(kernels, stream, x, w, y, m, n, k, Bf16Config::S3)
}

pub fn fits(m: u32, n: u32, k: u32) -> bool {
    m % BM == 0 && n % BN == 0 && k % (BK * WARP_TILE_K) == 0
}

// TN linear out[m,n] = sum_k x[m,k]*w[n,k] на best_cu (W = [N,K], без транспонир.).
// BF16-путь — для callsite #2 BF16; F16-путь — для F16 (Whisper и т.п.).
#[allow(clippy::too_many_arguments)]
pub fn best_gemm_bf16_linear_u8(
    kernels: &BestGemmBf16Kernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
    m: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
) -> Result<()> {
    let env = Bf16Config::from_override();
    let part_cfg = env
        .filter(|c| kernels.fnref_part_cfg(*c).is_some())
        .unwrap_or_else(|| pick_part_cfg(m, n, k));
    let part = kernels
        .fnref_part_cfg(part_cfg)
        .ok_or_else(|| SynaptixError::Cuda("bf16 linear_u8: нет part-ядра".into()))?;
    let splitk = kernels
        .splitk
        .as_ref()
        .zip(kernels.splitk_reduce.as_ref())
        .map(|(a, b)| (a, b, kernels.splitk_cfg, kernels.splitk_align))
        .filter(|_| env.is_none());
    let swz_cfg = env.unwrap_or_else(|| pick_swz_cfg(m, n, k));
    let cfg = kernels
        .fnref(swz_cfg)
        .ok_or_else(|| SynaptixError::Cuda("bf16 linear_u8: нет ядра".into()))?;
    let k128 = pick_ksub(m)
        .and_then(|sub| pick_kvariant(kernels, sub, false, m))
        .filter(|_| env.is_none());
    if env.is_none() {
        let sb = splitk_b256_splits(m, n, k);
        if sb > 1 {
            if let (Some(skfn), Some(redfn)) =
                (kernels.splitk_b256.as_ref(), kernels.splitk_reduce.as_ref())
            {
                return launch_splitk_b256(
                    skfn, redfn, "gemm_bf16", stream, w, x, out, n, k, m, sb, bias, residual,
                );
            }
        }
        if tma_base_ok(n, k, bias, residual) && tma_zone(m) {
            // (fn, stages, threads, bm, bn)
            let tma = if m <= 64 {
                kernels.tma_s5.as_ref().map(|f| (f, 5u32, 128u32, 64u32, 64u32))
            } else if big_tile_zone(m, n) {
                // 64×128 s3d (зеркало cuBLAS: 256 потоков, smem 73.7KB):
                // attn-128/256 63.6/75.3→73.3/79.4, ff_down-128/256
                // ~71/83.0→84.8/87.1; ff_up (N=16384) хуже — s3/s5d.
                kernels.tma_b64x128_s3d.as_ref().map(|f| (f, 3u32, 256u32, 64u32, 128u32))
            } else if (129..=256).contains(&m) {
                // s5d (дед-продьюсер): 192 +3%, 256 attn/ff_up/ff_down
                // 76.1/98.0/83.0 vs 69/75/79 (свип 2026-06-05 вечер)
                kernels.tma_s5d.as_ref().map(|f| (f, 5u32, 256u32, 64u32, 64u32))
            } else {
                kernels.tma_s3.as_ref().map(|f| (f, 3u32, 128u32, 64u32, 64u32))
            };
            if let Some((tfn, stages, threads, bm, bn)) = tma {
                return launch_tma_64(kernels, tfn, stages, threads, bm, bn, "gemm_bf16", stream, w, x, out, n, k, m);
            }
        }
    }
    launch_tn_u8(part, part_cfg, cfg, swz_cfg, splitk, k128, "gemm_bf16", stream, w, x, out, n, k, m, bias, residual)
}

#[allow(clippy::too_many_arguments)]
pub fn best_gemm_f16tn_linear_u8(
    kernels: &BestGemmBf16Kernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
    m: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
) -> Result<()> {
    let env = Bf16Config::from_override();
    let part_cfg = env
        .filter(|c| kernels.fnref_f16_part_cfg(*c).is_some())
        .unwrap_or_else(|| pick_part_cfg(m, n, k));
    let part = kernels
        .fnref_f16_part_cfg(part_cfg)
        .ok_or_else(|| SynaptixError::Cuda("f16tn linear_u8: нет part-ядра".into()))?;
    let splitk = kernels
        .f16_splitk
        .as_ref()
        .zip(kernels.f16_splitk_reduce.as_ref())
        .map(|(a, b)| (a, b, kernels.splitk_cfg, kernels.splitk_align))
        .filter(|_| env.is_none());
    let swz_cfg = env.unwrap_or_else(|| pick_swz_cfg(m, n, k));
    let cfg = kernels
        .fnref_f16(swz_cfg)
        .ok_or_else(|| SynaptixError::Cuda("f16tn linear_u8: нет ядра".into()))?;
    let k128 = pick_ksub(m)
        .and_then(|sub| pick_kvariant(kernels, sub, true, m))
        .filter(|_| env.is_none());
    if env.is_none() {
        let sb = splitk_b256_splits(m, n, k);
        if sb > 1 {
            if let (Some(skfn), Some(redfn)) =
                (kernels.f16_splitk_b256.as_ref(), kernels.f16_splitk_reduce.as_ref())
            {
                return launch_splitk_b256(
                    skfn, redfn, "gemm_f16tn", stream, w, x, out, n, k, m, sb, bias, residual,
                );
            }
        }
        if tma_base_ok(n, k, bias, residual) && tma_zone(m) {
            // (fn, stages, threads, bm, bn)
            let tma = if m <= 64 {
                kernels.f16_tma_s5.as_ref().map(|f| (f, 5u32, 128u32, 64u32, 64u32))
            } else if big_tile_zone(m, n) {
                kernels.f16_tma_b64x128_s3d.as_ref().map(|f| (f, 3u32, 256u32, 64u32, 128u32))
            } else if (129..=256).contains(&m) {
                kernels.f16_tma_s5d.as_ref().map(|f| (f, 5u32, 256u32, 64u32, 64u32))
            } else {
                kernels.f16_tma_s3.as_ref().map(|f| (f, 3u32, 128u32, 64u32, 64u32))
            };
            if let Some((tfn, stages, threads, bm, bn)) = tma {
                return launch_tma_64(kernels, tfn, stages, threads, bm, bn, "gemm_f16tn", stream, w, x, out, n, k, m);
            }
        }
    }
    launch_tn_u8(part, part_cfg, cfg, swz_cfg, splitk, k128, "gemm_f16tn", stream, w, x, out, n, k, m, bias, residual)
}

// Data-driven выбор тайла (свип 2026-06-04, bench_ltx_gemm): большой тайл 256×128
// (b256s4, путь cutlass 256x128_32x3) выигрывает large-M на всех LTX-формах
// (+13..56%, энергоэффективнее cuBLAS: меньше L2-трафика/FLOP → выше DVFS-клок),
// но проигрывает при коротком гриде (attn M≤2048: 43-86 vs 80-93 у 128-тайла).
fn use_b256(m: u32, n: u32, k: u32) -> bool {
    m >= 4000 || (n >= 16384 && m >= 2048)
}

// M≤192: 128-тайл даёт мало CTA (attn: 32) → SM пустые (warps 17%, DRAM 20%);
// cuBLAS здесь берёт skinny 32x32_128x2 (128 CTA). s64 = наш ответ (64×64).
fn use_s64(m: u32) -> bool {
    m <= 192
}

// SUB супер-конвейера: m<=64 → k128 (макс. глубина, латентность);
// 128..192 → k64 (32KB → 2 CTA/SM, грид уже заполняет машину).
fn pick_ksub(m: u32) -> Option<u32> {
    if !use_s64(m) {
        None
    } else if m <= 64 {
        Some(4)
    } else {
        Some(2)
    }
}

#[allow(clippy::type_complexity)]
fn pick_kvariant<'a>(
    kernels: &'a BestGemmBf16Kernels,
    sub: u32,
    f16: bool,
    m: u32,
) -> Option<(&'a CudaFunction, u32, u32, u32, u32, u32)> {
    // m<=32: тайл 32×32 (m32) — единственный вариант без холостых M-строк,
    // 128 CTA на attn (свип 2026-06-05: 18.4 vs 17.8 у k128); l2-вариант
    // (L2::256B prefetch на W-стриме) поднимает DRAM 73→77.6% = уровень
    // cuBLAS-skinny (ncu 2026-06-05), attn-32 19.4→20.0.
    if m <= 32 {
        let f = if f16 { kernels.f16_k128m32l2.as_ref() } else { kernels.k128m32l2.as_ref() };
        f.map(|f| (f, 4, 128, 32, 32, 2 * 4 * (32 + 32) * BK * 2 * 2))
    } else {
        let f = match (sub, f16) {
            (4, false) => kernels.k128.as_ref(),
            (_, false) => kernels.k64.as_ref(),
            (4, true) => kernels.f16_k128.as_ref(),
            _ => kernels.f16_k64.as_ref(),
        };
        f.map(|f| (f, sub, 256, 64, 64, 2 * sub * (64 + 64) * BK * 2 * 2))
    }
}

fn pick_part_cfg(m: u32, n: u32, k: u32) -> Bf16Config {
    if use_s64(m) {
        Bf16Config::S64S6
    } else if use_b256(m, n, k) {
        // b256t (128×256, рецепт cuBLAS ff_up-4992): large-M +8-12% по всей
        // зоне (свип 2026-06-05: 4992+ все формы 102-109 vs 94-98 у b256s4;
        // ff_up-2048 108.6 vs 98.7). attn/ff_down-2048 вне зоны (88 < 93-95).
        Bf16Config::B256TS3
    } else {
        Bf16Config::S3
    }
}

fn pick_swz_cfg(m: u32, n: u32, k: u32) -> Bf16Config {
    if use_s64(m) {
        Bf16Config::S64S6
    } else if use_b256(m, n, k) {
        Bf16Config::B256TS3
    } else if m % BM == 0 && n % BN == 0 {
        if k >= 16384 {
            Bf16Config::S6
        } else if k >= 160 {
            Bf16Config::S5
        } else {
            Bf16Config::S4
        }
    } else {
        Bf16Config::S3
    }
}

// Зона крупного TMA-тайла 64×128 (свип 2026-06-05): подтверждённые точки
// M=128/256 на attn+ff_down; M=192 (tiles_m=3, дисбаланс волны) и ff_up
// (N=16384: грид и так широкий, s3/s5d быстрее) — мимо. BN=128 требует n%128.
fn big_tile_zone(m: u32, n: u32) -> bool {
    ((97..=128).contains(&m) || (193..=256).contains(&m)) && n < 16384 && n % 128 == 0
}

// Зона TMA (свип 2026-06-05): 33..=512 кроме 256 — на 256 кратность 128
// кормит старые swz-пути (attn 0.97/ff_down s6 0.95 > TMA 0.94/0.90);
// m<=32 — m32-тайл cp.async быстрее (рамп mbarrier-конвейера на коротком ядре).
fn tma_zone(m: u32) -> bool {
    (33..=512).contains(&m)
}

fn tma_base_ok(
    n: u32,
    k: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
) -> bool {
    k % 64 == 0 && n % 64 == 0 && bias.is_none() && residual.is_none()
}

// TMA-ядро 64×64 (порт mxfp8-rot): дескрипторы кэшируются, растр рецепта rot.
#[allow(clippy::too_many_arguments)]
fn launch_tma_64(
    kernels: &BestGemmBf16Kernels,
    kfn: &CudaFunction,
    stages: u32,
    threads: u32,
    bm: u32,
    bn: u32,
    tag: &str,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
    m: u32,
) -> Result<()> {
    use crate::tma::make_tma_desc_2d_u8_swz_l2;
    use cudarc::driver::sys::{CUtensorMapL2promotion, CUtensorMapSwizzle};
    use cudarc::driver::DevicePtr;
    let (x_ptr, _rx) = x.device_ptr(stream);
    let (w_ptr, _rw) = w.device_ptr(stream);
    let swz = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    let desc_for = |ptr: u64, rows: u32, box_r: u32| -> Result<CudaSlice<u8>> {
        let key = (ptr, rows, k * 2 + box_r);
        let mut g = kernels.tma_descs.lock();
        if let Some(d) = g.get(&key) {
            return Ok(d.clone());
        }
        // L2-promotion 256B: стриминговое чтение W — DRAM-эффективность.
        let l2 = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
        let d = make_tma_desc_2d_u8_swz_l2(stream, ptr, rows, k * 2, box_r, 128, swz, l2)?;
        if g.len() >= 512 {
            g.clear();
        }
        g.insert(key, d.clone());
        Ok(d)
    };
    let a_desc = desc_for(x_ptr, m, bm)?;
    let b_desc = desc_for(w_ptr, n, bn)?;
    let mn = (m as usize) * (n as usize);
    let mut out_s = out.slice_mut(..mn * 2);
    let mut y_v = unsafe { out_s.transmute_mut::<bf16>(mn) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} tma: transmute out")))?;
    let tiles_n = n / bn;
    let tiles_m = m.div_ceil(bm);
    let mut gr: u32 = if (n as u64) * (k as u64) * 2 <= 24 * 1024 * 1024 { 32 } else { 8 };
    while gr > 1 && tiles_n % gr != 0 {
        gr /= 2;
    }
    let gr = gr.max(1);
    let smem = stages * (bm + bn) * 128 + 2 * stages * 8;
    let mut bld = stream.launch_builder(kfn);
    bld.arg(&a_desc)
        .arg(&b_desc)
        .arg(&mut y_v)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&gr);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (tiles_n * tiles_m, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch {tag} tma: {e:?}")))?;
    }
    Ok(())
}

// b256-сплит (256×128, s3 — рецепт cuBLAS ff_down-256 «128x256_32x3 split-5»:
// их грид (8,4,5)=160 CTA, smem 73.73KB, SM 91%): глубокий K при коротком
// гриде — m 193..256, k>=8192. ff_up (n=16384) не трогаем: грид и так широкий.
fn splitk_b256_splits(m: u32, n: u32, k: u32) -> u32 {
    if !(193..=256).contains(&m) || n % 128 != 0 || n >= 16384 || k < 8192 || k % 32 != 0 {
        return 1;
    }
    let tiles = (n / 128) * m.div_ceil(256);
    // ЦЕЛОЕ число волн (свип 2026-06-05: 5→90.0, 4→78.8, 6→76.8, 8→73.7 —
    // 32 тайла × 5 = 160 CTA = 1.95 волны; div_ceil давал 6 → хвост-волна).
    let splits = (164u32 / tiles.max(1)).clamp(2, 8);
    if splits <= 1 || tiles >= 164 {
        return 1;
    }
    splits
}

#[allow(clippy::too_many_arguments)]
fn launch_splitk_b256(
    skfn: &CudaFunction,
    redfn: &CudaFunction,
    tag: &str,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
    m: u32,
    splits: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
) -> Result<()> {
    let (nk, mk) = ((n * k) as usize, (m * k) as usize);
    let mn = (m as usize) * (n as usize);
    let w_v = unsafe { w.slice(..nk * 2).transmute::<bf16>(nk) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} splitk_b256: transmute w")))?;
    let x_v = unsafe { x.slice(..mk * 2).transmute::<bf16>(mk) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} splitk_b256: transmute x")))?;
    let mut out_s = out.slice_mut(..mn * 2);
    let mut y_v = unsafe { out_s.transmute_mut::<bf16>(mn) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} splitk_b256: transmute out")))?;
    let bias_v = match bias {
        Some(bb) => unsafe { bb.slice(..n as usize * 2).transmute::<bf16>(n as usize) }
            .ok_or_else(|| SynaptixError::Cuda(format!("{tag} splitk_b256: transmute bias")))?,
        None => unsafe { w.slice(..n as usize * 2).transmute::<bf16>(n as usize) }.unwrap(),
    };
    let resid_v = match residual {
        Some(rr) => unsafe { rr.slice(..mn * 2).transmute::<bf16>(mn) }
            .ok_or_else(|| SynaptixError::Cuda(format!("{tag} splitk_b256: transmute residual")))?,
        None => unsafe { w.slice(..mn.min(nk) * 2).transmute::<bf16>(mn.min(nk)) }.unwrap(),
    };
    let (has_bias_i, has_res_i): (i32, i32) =
        (if bias.is_some() { 1 } else { 0 }, if residual.is_some() { 1 } else { 0 });
    let chunk = k.div_ceil(splits).next_multiple_of(32);
    let used = k.div_ceil(chunk);
    // ws без memset: каждый f32-партиал пишется ядром ровно один раз.
    let mut ws: CudaSlice<f32> = unsafe { stream.alloc(mn * used as usize) }
        .map_err(|e| SynaptixError::Cuda(format!("{tag} splitk_b256 ws: {e:?}")))?;
    let launch = LaunchConfig {
        grid_dim: (n / 128, m.div_ceil(256), used),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: Bf16Config::B256S3.smem_bytes(),
    };
    let (mi, ni, ki, chunk_i) = (m as i32, n as i32, k as i32, chunk as i32);
    let mut bld = stream.launch_builder(skfn);
    bld.arg(&x_v).arg(&w_v).arg(&mut ws).arg(&mi).arg(&ni).arg(&ki).arg(&chunk_i);
    unsafe {
        bld.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch {tag} splitk_b256: {e:?}")))?;
    }
    let (mn_ll, splits_i) = (mn as i64, used as i32);
    let red_launch = LaunchConfig {
        grid_dim: ((mn as u32).div_ceil(THREADS).min(4096), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(redfn);
    bld.arg(&ws)
        .arg(&mut y_v)
        .arg(&mn_ll)
        .arg(&ni)
        .arg(&splits_i)
        .arg(&bias_v)
        .arg(&has_bias_i)
        .arg(&resid_v)
        .arg(&has_res_i);
    unsafe {
        bld.launch(red_launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch {tag} splitk_b256-reduce: {e:?}")))?;
    }
    Ok(())
}

// splits>1 при малых M: грид s64-классов не заполняет 82 SM (attn M=32: 64 CTA,
// warps_active 16.6%, DRAM 50% vs 55% у cuBLAS skinny-ядра). Цель ≈2 CTA/SM.
fn pick_splitk(m: u32, n: u32, k: u32, stages: u32, align: u32) -> (u32, u32) {
    if k % align != 0 {
        return (1, k);
    }
    if !use_s64(m) || k >= 8192 {
        // K>=16384: 64 CTA уже глубоко стримят K, сплит добавляет волну
        // partial-CTA и редьюс — замер 2026-06-05: ff_down 33.3 vs 28.9 без сплита.
        return (1, k);
    }
    let blocks = n.div_ceil(Bf16Config::S64S3.bn) * m.div_ceil(Bf16Config::S64S3.bm);
    if blocks >= 128 {
        return (1, k);
    }
    let min_chunk = (stages * BK * WARP_TILE_K).max(align);
    let mut splits = 164u32.div_ceil(blocks).clamp(2, 8);
    loop {
        let chunk = k.div_ceil(splits).next_multiple_of(align);
        let used = k.div_ceil(chunk);
        let last = k - (used - 1) * chunk;
        if (last >= min_chunk && chunk >= min_chunk) || splits <= 2 {
            if used < 2 || last < min_chunk || chunk < min_chunk {
                return (1, k);
            }
            return (used, chunk);
        }
        splits -= 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_tn_u8(
    part_fn: &CudaFunction,
    part_cfg: Bf16Config,
    cfg_fn: &CudaFunction,
    swz_cfg: Bf16Config,
    splitk_fns: Option<(&CudaFunction, &CudaFunction, Bf16Config, u32)>,
    k128_fn: Option<(&CudaFunction, u32, u32, u32, u32, u32)>,
    tag: &str,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    n: u32,
    k: u32,
    m: u32,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<u8>>,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    // K-tail / малый K: ядро шагает K тайлами по 32; multistage-пролог префетчит
    // K_STAGE тайлов, поэтому нужно NUM_K_TILES >= stages (иначе пролог читает за
    // границей K → мусор; так ломался x_embedder K=64 при S4: 2 тайла < 4 стадий,
    // cos 0.91). Паддим K нулями до max(round-up-32, stages*32) во временные буферы
    // когда K%32 != 0 ЛИБО NUM_K_TILES < stages. Нули не вносят вклад в MMA.
    let partial = m % swz_cfg.bm != 0 || n % swz_cfg.bn != 0;
    let run_cfg = if partial { part_cfg } else { swz_cfg };
    let k_step = BK * WARP_TILE_K;
    let num_k_tiles = (k + k_step - 1) / k_step;
    let (x_pad_hold, w_pad_hold, k) = if k % k_step == 0 && num_k_tiles >= run_cfg.stages {
        (None, None, k)
    } else {
        let k_pad = ((k + k_step - 1) / k_step).max(run_cfg.stages) * k_step;
        let mut xp = stream
            .alloc_zeros::<u8>((m as usize) * (k_pad as usize) * 2)
            .map_err(|e| SynaptixError::Cuda(format!("{tag} K-pad alloc x: {e:?}")))?;
        let mut wp = stream
            .alloc_zeros::<u8>((n as usize) * (k_pad as usize) * 2)
            .map_err(|e| SynaptixError::Cuda(format!("{tag} K-pad alloc w: {e:?}")))?;
        pad_copy_k_bytes(stream, x, &mut xp, m, k, k_pad)?;
        pad_copy_k_bytes(stream, w, &mut wp, n, k, k_pad)?;
        (Some(xp), Some(wp), k_pad)
    };
    let x = x_pad_hold.as_ref().unwrap_or(x);
    let w = w_pad_hold.as_ref().unwrap_or(w);
    let kfn = if partial { part_fn } else { cfg_fn };
    // Аргументы — 16-битный typed-view над байтами: bf16 и f16 делят gemm_bf16_impl<T>,
    // указатель один и тот же, ядро интерпретирует биты per-dtype (по своему entry).
    let (nk, mk, mn) = ((n * k) as usize, (m * k) as usize, (m * n) as usize);
    let w_v = unsafe { w.slice(..nk * 2).transmute::<bf16>(nk) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} linear_u8: transmute w")))?;
    let x_v = unsafe { x.slice(..mk * 2).transmute::<bf16>(mk) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} linear_u8: transmute x")))?;
    let mut out_s = out.slice_mut(..mn * 2);
    let mut y_v = unsafe { out_s.transmute_mut::<bf16>(mn) }
        .ok_or_else(|| SynaptixError::Cuda(format!("{tag} linear_u8: transmute out")))?;
    let bias_v = match bias {
        Some(bb) => unsafe { bb.slice(..n as usize * 2).transmute::<bf16>(n as usize) }
            .ok_or_else(|| SynaptixError::Cuda(format!("{tag} linear_u8: transmute bias")))?,
        None => unsafe { w.slice(..n as usize * 2).transmute::<bf16>(n as usize) }.unwrap(),
    };
    let resid_v = match residual {
        Some(rr) => unsafe { rr.slice(..mn * 2).transmute::<bf16>(mn) }
            .ok_or_else(|| SynaptixError::Cuda(format!("{tag} linear_u8: transmute residual")))?,
        None => unsafe { w.slice(..mn.min(nk) * 2).transmute::<bf16>(mn.min(nk)) }.unwrap(),
    };
    let has_bias_i: i32 = if bias.is_some() { 1 } else { 0 };
    let has_res_i: i32 = if residual.is_some() { 1 } else { 0 };
    if let Some((kfn128, sub, threads, bm, bn, smem)) = k128_fn {
        if use_s64(m) && k % (sub * 32) == 0 {
            let mut launch = launch_for(m, n, true, smem, bm, bn);
            launch.block_dim = (threads, 1, 1);
            let (mi, ni, ki) = (m as i32, n as i32, k as i32);
            let mut bld = stream.launch_builder(kfn128);
            bld.arg(&x_v)
                .arg(&w_v)
                .arg(&mut y_v)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&bias_v)
                .arg(&has_bias_i)
                .arg(&resid_v)
                .arg(&has_res_i);
            unsafe {
                bld.launch(launch)
                    .map_err(|e| SynaptixError::Cuda(format!("launch {tag} k128: {e:?}")))?;
            }
            return Ok(());
        }
    }
    if let Some((skfn, redfn, sk_cfg, sk_align)) = splitk_fns {
        let (splits, chunk) = pick_splitk(m, n, k, sk_cfg.stages, sk_align);
        if splits > 1 {
            let s64 = sk_cfg;
            let mn = (m as usize) * (n as usize);
            // ws без memset: каждый f32-партиал в [0, splits*MN) пишется ядром
            // ровно один раз (пары (col,col+1) на границе N принадлежат соседней
            // строке-владельцу), reduce читает только этот диапазон.
            let mut ws: CudaSlice<f32> = unsafe { stream.alloc(mn * splits as usize) }
                .map_err(|e| SynaptixError::Cuda(format!("{tag} splitk ws: {e:?}")))?;
            let launch = LaunchConfig {
                grid_dim: (n.div_ceil(s64.bn), m.div_ceil(s64.bm), splits),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: s64.smem_bytes(),
            };
            let (mi, ni, ki, chunk_i) = (m as i32, n as i32, k as i32, chunk as i32);
            let mut bld = stream.launch_builder(skfn);
            bld.arg(&x_v)
                .arg(&w_v)
                .arg(&mut ws)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&chunk_i);
            unsafe {
                bld.launch(launch)
                    .map_err(|e| SynaptixError::Cuda(format!("launch {tag} splitk: {e:?}")))?;
            }
            let (mn_ll, splits_i) = (mn as i64, splits as i32);
            let red_launch = LaunchConfig {
                grid_dim: ((mn as u32).div_ceil(THREADS).min(4096), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = stream.launch_builder(redfn);
            bld.arg(&ws)
                .arg(&mut y_v)
                .arg(&mn_ll)
                .arg(&ni)
                .arg(&splits_i)
                .arg(&bias_v)
                .arg(&has_bias_i)
                .arg(&resid_v)
                .arg(&has_res_i);
            unsafe {
                bld.launch(red_launch)
                    .map_err(|e| SynaptixError::Cuda(format!("launch {tag} splitk-reduce: {e:?}")))?;
            }
            return Ok(());
        }
    }
    let smem = run_cfg.smem_bytes();
    let launch = launch_for(m, n, partial, smem, run_cfg.bm, run_cfg.bn);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let mut bld = stream.launch_builder(kfn);
    bld.arg(&x_v)
        .arg(&w_v)
        .arg(&mut y_v)
        .arg(&mi)
        .arg(&ni)
        .arg(&ki)
        .arg(&bias_v)
        .arg(&has_bias_i)
        .arg(&resid_v)
        .arg(&has_res_i);
    unsafe {
        bld.launch(launch)
            .map_err(|e| SynaptixError::Cuda(format!("launch {tag} linear_u8: {e:?}")))?;
    }
    Ok(())
}

/// Копирует bf16-матрицу [rows, k] → зануленный буфер [rows, k_pad] (k_pad >= k)
/// одной pitched-копией cuMemcpy2D (без построчного launch-storm). Колонки
/// [k, k_pad) остаются нулями (буфер пришёл из alloc_zeros).
// Общая pitched-копия одним cuMemcpy2D: `rows` строк по `width_bytes`, из src
// (шаг src_pitch_bytes) в dst (шаг dst_pitch_bytes). Используется для pad (src
// узкий → dst широкий) и unpad (src широкий → берём первые width_bytes).
pub(crate) fn copy_2d_bytes(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    rows: u32,
    width_bytes: usize,
    src_pitch_bytes: usize,
    dst_pitch_bytes: usize,
) -> Result<()> {
    use cudarc::driver::{sys, DevicePtr, DevicePtrMut};
    let src_p: sys::CUdeviceptr = {
        let (p, _g) = src.device_ptr(stream);
        p
    };
    let dst_p: sys::CUdeviceptr = {
        let (p, _g) = dst.device_ptr_mut(stream);
        p
    };
    let copy = sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src_p,
        srcArray: std::ptr::null_mut(),
        srcPitch: src_pitch_bytes,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst_p,
        dstArray: std::ptr::null_mut(),
        dstPitch: dst_pitch_bytes,
        WidthInBytes: width_bytes,
        Height: rows as usize,
    };
    unsafe {
        sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream())
            .result()
            .map_err(|e| SynaptixError::Cuda(format!("copy_2d cuMemcpy2D: {e:?}")))?;
    }
    Ok(())
}

// Pad: [rows, k] (2-байтовые элементы) → [rows, k_pad] (правый паддинг нулями).
pub(crate) fn pad_copy_k_bytes(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    rows: u32,
    k: u32,
    k_pad: u32,
) -> Result<()> {
    let row_bytes = (k as usize) * 2;
    copy_2d_bytes(stream, src, dst, rows, row_bytes, row_bytes, (k_pad as usize) * 2)
}
