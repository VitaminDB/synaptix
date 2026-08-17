use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

static NVFP4_QUANT_SLOW: AtomicBool = AtomicBool::new(false);
static MXFP8_QUANT_SLOW: AtomicBool = AtomicBool::new(false);
static MXFP8_QUANT_FAST4: AtomicBool = AtomicBool::new(true);

pub fn set_nvfp4_quant_slow(slow: bool) {
    NVFP4_QUANT_SLOW.store(slow, Ordering::Relaxed);
}

pub fn set_mxfp8_quant_slow(slow: bool) {
    MXFP8_QUANT_SLOW.store(slow, Ordering::Relaxed);
}

pub fn set_mxfp8_quant_fast4(fast4: bool) {
    MXFP8_QUANT_FAST4.store(fast4, Ordering::Relaxed);
}

// ──────────────────────────────────────────────────────────────────────
// NVFP4 (4-bit E2M1 + FP8 E4M3 block scale, block=16, tile-major layout)
// ──────────────────────────────────────────────────────────────────────

pub struct Nvfp4QuantKernels {
    _module: Arc<CudaModule>,
    quantize_f16_to_nvfp4: CudaFunction,
    quantize_f16_to_nvfp4_fast: CudaFunction,
    silu_mul_quantize_nvfp4_fast: CudaFunction,
    nvfp4_dequant_f16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4QuantKernels>)>>> = OnceLock::new();
static CACHE_BF16: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4QuantKernels>)>>> = OnceLock::new();

impl Nvfp4QuantKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, CACHE.get_or_init(|| Mutex::new(Vec::new())), &[], "nvfp4_quant.cu")
    }

    pub fn for_context_bf16(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(
            ctx,
            CACHE_BF16.get_or_init(|| Mutex::new(Vec::new())),
            &["-DSYN_IN_BF16"],
            "nvfp4_quant_bf16.cu",
        )
    }

    fn build(
        ctx: &Arc<CudaContext>,
        cache: &Mutex<Vec<(usize, Arc<Nvfp4QuantKernels>)>>,
        opts: &[&str],
        name: &'static str,
    ) -> Result<Arc<Self>> {
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("../cu/elementwise/nvfp4_quant.cu");
        let module = compile_module_with_opts(ctx, src, name, opts, Some("sm_80"))?;
        let new = Arc::new(Self {
            quantize_f16_to_nvfp4: load_fn(&module, "quantize_f16_to_nvfp4")?,
            quantize_f16_to_nvfp4_fast: load_fn(&module, "quantize_f16_to_nvfp4_fast")?,
            silu_mul_quantize_nvfp4_fast: load_fn(&module, "silu_mul_quantize_nvfp4_fast")?,
            nvfp4_dequant_f16: load_fn(&module, "nvfp4_dequant_f16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// Размер scale buffer в байтах (cuBLASLt 13 spec §3.1.4.4.2 tile-major).
/// `outer_dim` = M для A (или N для B), `inner_dim` = K.
/// Caller обязан выделить буфер этого размера И обнулить — out-of-bounds
/// scale bytes должны быть 0.
pub fn nvfp4_scale_buffer_size(outer_dim: usize, inner_dim: usize) -> usize {
    let s_rows = inner_dim.div_ceil(64) * 4;
    let s_cols = outer_dim.div_ceil(128) * 128;
    s_rows * s_cols
}

const QUANT_CHUNK: u32 = 32_768;

pub fn quantize_f16_to_nvfp4(
    kernels: &Nvfp4QuantKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    packed: &mut CudaSlice<u8>,
    scales_e4m3: &mut CudaSlice<u8>,
    outer_dim: u32,
    inner_dim: u32,
) -> Result<()> {
    let xv = x.as_view();
    quantize_f16_to_nvfp4_view(
        kernels,
        stream,
        &xv,
        packed,
        scales_e4m3,
        outer_dim,
        inner_dim,
    )
}

/// Как [`quantize_f16_to_nvfp4`], но вход — `CudaView<f16>` (zero-copy из
/// F16-storage без отдельного буфера).
pub fn quantize_f16_to_nvfp4_view(
    kernels: &Nvfp4QuantKernels,
    stream: &Arc<CudaStream>,
    x: &CudaView<f16>,
    packed: &mut CudaSlice<u8>,
    scales_e4m3: &mut CudaSlice<u8>,
    outer_dim: u32,
    inner_dim: u32,
) -> Result<()> {
    if inner_dim % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "quantize_f16_to_nvfp4: inner_dim={inner_dim} must be multiple of 16"
        )));
    }
    let num_blocks_per_row = inner_dim / 16;
    let sf_inner_dim = inner_dim.div_ceil(64) * 4;

    // Быстрый путь: 1 поток = 1 группа (uint4-лоады требуют 16B-выравнивания x;
    // выравнивание строк гарантирует inner_dim%16==0). Арифметика бит-в-бит
    // совпадает со старым ядром. set_nvfp4_quant_slow(true) — аварийный откат.
    let x_ptr_aligned = {
        use cudarc::driver::DevicePtr;
        let (p, _g) = x.device_ptr(stream);
        p % 16 == 0
    };
    if x_ptr_aligned && !NVFP4_QUANT_SLOW.load(Ordering::Relaxed) {
        // outer_cov: ядро зануляет scale-хвост 128-тайла раскладки — буфер
        // скейлов полностью определён без CE-memset вызывающего.
        let outer_cov = outer_dim.div_ceil(128) * 128;
        let total = (outer_cov as u64) * (num_blocks_per_row as u64);
        let blocks = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&kernels.quantize_f16_to_nvfp4_fast);
        b.arg(x)
            .arg(&mut *packed)
            .arg(&mut *scales_e4m3)
            .arg(&outer_dim)
            .arg(&inner_dim)
            .arg(&sf_inner_dim)
            .arg(&outer_cov);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch quantize_nvfp4_fast: {e:?}")))?;
        }
        return Ok(());
    }

    // Контракт обёртки: после вызова буфер скейлов ПОЛНОСТЬЮ определён (хвост
    // 128-тайла = 0). Fast-путь зануляет хвост сам (outer_cov); slow-ядро пишет
    // только реальные строки → зануляем буфер целиком (редкий fallback-путь).
    stream
        .memset_zeros(&mut *scales_e4m3)
        .map_err(|e| SynaptixError::Cuda(format!("quantize_nvfp4: zero scales: {e:?}")))?;
    let mut outer_offset: u32 = 0;
    while outer_offset < outer_dim {
        let remaining = outer_dim - outer_offset;
        let rows_this = remaining.min(QUANT_CHUNK);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks_per_row, rows_this, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 16 * 4 + 16,
        };
        let mut b = stream.launch_builder(&kernels.quantize_f16_to_nvfp4);
        b.arg(x)
            .arg(&mut *packed)
            .arg(&mut *scales_e4m3)
            .arg(&outer_dim)
            .arg(&inner_dim)
            .arg(&sf_inner_dim)
            .arg(&outer_offset);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch quantize_nvfp4: {e:?}")))?;
        }
        outer_offset += rows_this;
    }
    Ok(())
}

pub fn silu_mul_quantize_nvfp4_u8(
    kernels: &Nvfp4QuantKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off_bytes: usize,
    packed: &mut CudaSlice<u8>,
    scales_e4m3: &mut CudaSlice<u8>,
    outer_dim: u32,
    inner_dim: u32,
    inv_pre: f32,
) -> Result<()> {
    if inner_dim % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "silu_mul_quantize_nvfp4: inner_dim={inner_dim} must be multiple of 16"
        )));
    }
    let n = (outer_dim as usize) * (inner_dim as usize) * 2;
    let x_v = unsafe {
        x.slice(x_off_bytes..x_off_bytes + n * 2)
            .transmute::<f16>(n)
            .ok_or_else(|| SynaptixError::Cuda("silu_mul_quantize_nvfp4: transmute x".into()))?
    };
    let num_blocks_per_row = inner_dim / 16;
    let sf_inner_dim = inner_dim.div_ceil(64) * 4;
    let outer_cov = outer_dim.div_ceil(128) * 128;
    let total = (outer_cov as u64) * (num_blocks_per_row as u64);
    let blocks = total.div_ceil(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kernels.silu_mul_quantize_nvfp4_fast);
    b.arg(&x_v)
        .arg(&mut *packed)
        .arg(&mut *scales_e4m3)
        .arg(&outer_dim)
        .arg(&inner_dim)
        .arg(&sf_inner_dim)
        .arg(&outer_cov)
        .arg(&inv_pre);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch silu_mul_quantize_nvfp4: {e:?}")))?;
    }
    Ok(())
}

pub fn nvfp4_dequant_f16(
    kernels: &Nvfp4QuantKernels,
    stream: &Arc<CudaStream>,
    packed: &CudaSlice<u8>,
    scales_e4m3: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    outer_dim: u32,
    inner_dim: u32,
) -> Result<()> {
    if inner_dim % 16 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_dequant_f16: inner_dim={inner_dim} must be multiple of 16"
        )));
    }
    let num_blocks_per_row = inner_dim / 16;
    let sf_inner_dim = inner_dim.div_ceil(64) * 4;

    let mut outer_offset: u32 = 0;
    while outer_offset < outer_dim {
        let remaining = outer_dim - outer_offset;
        let rows_this = remaining.min(QUANT_CHUNK);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks_per_row, rows_this, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&kernels.nvfp4_dequant_f16);
        b.arg(packed)
            .arg(scales_e4m3)
            .arg(&mut *out)
            .arg(&outer_dim)
            .arg(&inner_dim)
            .arg(&sf_inner_dim)
            .arg(&outer_offset);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_dequant: {e:?}")))?;
        }
        outer_offset += rows_this;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// MXFP8 (Blackwell-нативный block-scale FP8): per-32-block E8M0 + E4M3.
// Заменил legacy per-tensor FP8 E4M3 (был под cuBLASLt, убран).
// ──────────────────────────────────────────────────────────────────────

pub struct Mxfp8QuantKernels {
    _module: Arc<CudaModule>,
    quant_natural: CudaFunction,
    quant_natural_fast: CudaFunction,
    quant_natural_fast4: CudaFunction,
    dequant: CudaFunction,
}

static MXFP8_CACHE: OnceLock<Mutex<Vec<(usize, Arc<Mxfp8QuantKernels>)>>> = OnceLock::new();

impl Mxfp8QuantKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = MXFP8_CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("../cu/elementwise/mxfp8_quant.cu");
        let module = compile_module_with_opts(ctx, src, "mxfp8_quant.cu", &[], Some("sm_80"))?;
        let new = Arc::new(Self {
            quant_natural: load_fn(&module, "mxfp8_quant_natural")?,
            quant_natural_fast: load_fn(&module, "mxfp8_quant_natural_fast")?,
            quant_natural_fast4: load_fn(&module, "mxfp8_quant_natural_fast4")?,
            dequant: load_fn(&module, "mxfp8_dequant_f16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// MXFP8 natural квант [rows,K] f16 → fp8 e4m3 + natural E8M0 scales [rows,K/32]
/// (для gemv decode + хранения веса).
pub fn mxfp8_quant_natural(
    kernels: &Mxfp8QuantKernels,
    stream: &Arc<CudaStream>,
    in_f16: &CudaView<f16>,
    out_fp8: &mut CudaSlice<u8>,
    out_scales: &mut CudaSlice<u8>,
    rows: u32,
    k: u32,
) -> Result<()> {
    let block = 256u32;
    let (ri, ki) = (rows as i32, k as i32);
    // Векторизованное ядро при 16B-выровненном входе (бит-в-бит та же арифметика);
    // set_mxfp8_quant_slow(true) — аварийный откат. fast4: поток = 8 эл +
    // shfl-amax (×4 потоков — латентность на малых M; bit-same scale).
    let aligned = {
        use cudarc::driver::DevicePtr;
        let (p, _g) = in_f16.device_ptr(stream);
        p % 16 == 0
    };
    let slow = MXFP8_QUANT_SLOW.load(Ordering::Relaxed);
    let use4 = aligned && !slow && MXFP8_QUANT_FAST4.load(Ordering::Relaxed);
    let (kfn, grid) = if use4 {
        (&kernels.quant_natural_fast4, (rows * (k / 32) * 4).div_ceil(block))
    } else if aligned && !slow {
        (&kernels.quant_natural_fast, (rows * (k / 32)).div_ceil(block))
    } else {
        (&kernels.quant_natural, (rows * (k / 32)).div_ceil(block))
    };
    let mut bld = stream.launch_builder(kfn);
    bld.arg(in_f16).arg(&mut *out_fp8).arg(&mut *out_scales).arg(&ri).arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8_quant_natural: {e:?}")))?;
    }
    Ok(())
}

/// MXFP8 → f16 dequant [rows,K]: value = e4m3(packed) * 2^(E8M0[row,k/32]-127).
/// Деквант MXFP8 → f16. `packed`/`scales` — вьюхи: позволяют дековантовать
/// полосу строк веса (натуральная раскладка [n,k] / [n,k/32] — строки лежат
/// подряд, полоса = байтовое подокно), не материализуя W целиком.
pub fn mxfp8_dequant_f16(
    kernels: &Mxfp8QuantKernels,
    stream: &Arc<CudaStream>,
    packed: &CudaView<'_, u8>,
    scales: &CudaView<'_, u8>,
    out: &mut CudaViewMut<f16>,
    rows: u32,
    k: u32,
) -> Result<()> {
    let block = 256u32;
    let grid = (rows * k).div_ceil(block);
    let (ri, ki) = (rows as i32, k as i32);
    let mut bld = stream.launch_builder(&kernels.dequant);
    bld.arg(packed).arg(scales).arg(&mut *out).arg(&ri).arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch mxfp8_dequant_f16: {e:?}")))?;
    }
    Ok(())
}
