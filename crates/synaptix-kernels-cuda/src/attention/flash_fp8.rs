//! FP8-внимание BSHD, head_dim 128, без маски (видео/картиночные DiT):
//! QKᵀ на e4m3-mma (квант Q/K по токену, у K вычитается среднее), P·V —
//! bf16 (`full = false`) или e4m3 (`full = true`, V по каналу). См.
//! `cu/fused/attention/flash_fp8.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_req, load_fn};

pub struct FlashFp8Kernels {
    _module: Arc<CudaModule>,
    col_sum: CudaFunction,
    col_amax: CudaFunction,
    quant_rows: CudaFunction,
    quant_vt: CudaFunction,
    main_qk: CudaFunction,
    main_full: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashFp8Kernels>)>>> = OnceLock::new();

const SMEM_QK: u32 = 8192 + 16384 + 32768;
const SMEM_FULL: u32 = 8192 + 16384 + 16384;

impl FlashFp8Kernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let src = include_str!("../cu/fused/attention/flash_fp8.cu");
        let module = compile_module_req(ctx, src, "flash_fp8.cu", &[], crate::caps::Feature::Fp8Mma)?;
        let main_qk = load_fn(&module, "flash_fp8_qk_hd128_bshd")?;
        let main_full = load_fn(&module, "flash_fp8_full_hd128_bshd")?;
        for (f, s) in [(&main_qk, SMEM_QK), (&main_full, SMEM_FULL)] {
            f.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, s as i32)
                .map_err(|e| SynaptixError::Cuda(format!("flash_fp8 smem: {e:?}")))?;
        }
        let new = Arc::new(Self {
            col_sum: load_fn(&module, "fa8_col_sum")?,
            col_amax: load_fn(&module, "fa8_col_amax")?,
            quant_rows: load_fn(&module, "fa8_quant_rows")?,
            quant_vt: load_fn(&module, "fa8_quant_vt")?,
            main_qk,
            main_full,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch_err(what: &str) -> impl Fn(cudarc::driver::DriverError) -> SynaptixError + '_ {
    move |e| SynaptixError::Cuda(format!("launch {what}: {e:?}"))
}

/// `q [B,Sq,H,128]`, `k/v [B,Skv,H,128]` bf16 (байтовые срезы со смещением),
/// `out [B,Sq,H,128]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn flash_fp8_bshd(
    kern: &FlashFp8Kernels,
    stream: &Arc<CudaStream>,
    (q, q_off): (&CudaSlice<u8>, usize),
    (k, k_off): (&CudaSlice<u8>, usize),
    (v, v_off): (&CudaSlice<u8>, usize),
    out: &mut CudaSlice<u8>,
    b: usize,
    h: usize,
    sq: usize,
    skv: usize,
    scale: f32,
    full: bool,
) -> Result<()> {
    if b == 0 || h == 0 || sq == 0 || skv == 0 {
        return Ok(());
    }
    let d = 128usize;
    let alloc = |n: usize| -> Result<CudaSlice<u8>> {
        unsafe { stream.alloc::<u8>(n.max(1)) }.map_err(|e| SynaptixError::Cuda(format!("flash_fp8 alloc: {e:?}")))
    };
    fn view(s: &CudaSlice<u8>, off: usize, n: usize) -> Result<cudarc::driver::CudaView<'_, half::bf16>> {
        unsafe { s.slice(off..off + n * 2).transmute::<half::bf16>(n) }.ok_or_else(|| SynaptixError::Cuda("flash_fp8: transmute".into()))
    }
    let (qv, kv, vv) = (view(q, q_off, b * sq * h * d)?, view(k, k_off, b * skv * h * d)?, view(v, v_off, b * skv * h * d)?);
    let (bi, hi, sqi, skvi) = (b as i32, h as i32, sq as i32, skv as i32);
    let spad = skv.div_ceil(64) * 64;
    let spadi = spad as i32;

    // Среднее K по токенам (сумма; деление — в кванте).
    let mut ksum: CudaSlice<f32> = stream.alloc_zeros(b * h * d).map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    let split = (skv / 256).clamp(1, 32) as u32;
    let red_cfg = LaunchConfig { grid_dim: ((b * h) as u32, split, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    {
        let mut lb = stream.launch_builder(&kern.col_sum);
        lb.arg(&kv).arg(&mut ksum).arg(&skvi).arg(&hi);
        unsafe { lb.launch(red_cfg) }.map_err(launch_err("fa8_col_sum"))?;
    }
    // Квант Q (шкала · scale · log2e) и K (минус среднее).
    let mut qq = alloc(b * sq * h * d)?;
    let mut kq = alloc(b * skv * h * d)?;
    let mut qs: CudaSlice<f32> = unsafe { stream.alloc(b * h * sq) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    let mut ks: CudaSlice<f32> = unsafe { stream.alloc(b * h * skv) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    let mul_q = scale * std::f32::consts::LOG2_E;
    let one = 1.0f32;
    let null: u64 = 0;
    for (src, n_s, dst, sc, mul, sub) in [(&qv, sq, &mut qq, &mut qs, mul_q, false), (&kv, skv, &mut kq, &mut ks, one, true)] {
        let rows = b * n_s * h;
        let cfg = LaunchConfig { grid_dim: (rows.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let n_si = n_s as i32;
        let mut lb = stream.launch_builder(&kern.quant_rows);
        lb.arg(src);
        if sub {
            lb.arg(&ksum);
        } else {
            lb.arg(&null);
        }
        lb.arg(dst).arg(sc).arg(&bi).arg(&n_si).arg(&hi).arg(&mul);
        unsafe { lb.launch(cfg) }.map_err(launch_err("fa8_quant_rows"))?;
    }

    let grid = LaunchConfig { grid_dim: (sq.div_ceil(64) as u32, (b * h) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let mut out_s = out.slice_mut(0..b * sq * h * d * 2);
    let mut ov = unsafe { out_s.transmute_mut::<half::bf16>(b * sq * h * d) }
        .ok_or_else(|| SynaptixError::Cuda("flash_fp8: transmute out".into()))?;
    if full {
        let mut vmax: CudaSlice<f32> = stream.alloc_zeros(b * h * d).map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
        {
            let mut lb = stream.launch_builder(&kern.col_amax);
            lb.arg(&vv).arg(&mut vmax).arg(&skvi).arg(&hi);
            unsafe { lb.launch(red_cfg) }.map_err(launch_err("fa8_col_amax"))?;
        }
        let mut vt = alloc(b * h * d * spad)?;
        let mut vsc: CudaSlice<f32> = unsafe { stream.alloc(b * h * d) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
        {
            let cfg = LaunchConfig { grid_dim: ((spad / 64) as u32, (b * h) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let mut lb = stream.launch_builder(&kern.quant_vt);
            lb.arg(&vv).arg(&vmax).arg(&mut vt).arg(&mut vsc).arg(&skvi).arg(&hi).arg(&spadi);
            unsafe { lb.launch(cfg) }.map_err(launch_err("fa8_quant_vt"))?;
        }
        let mut lb = stream.launch_builder(&kern.main_full);
        lb.arg(&qq).arg(&qs).arg(&kq).arg(&ks).arg(&vt).arg(&vsc).arg(&mut ov).arg(&bi).arg(&hi).arg(&sqi).arg(&skvi).arg(&spadi);
        unsafe { lb.launch(LaunchConfig { shared_mem_bytes: SMEM_FULL, ..grid }) }.map_err(launch_err("flash_fp8_full"))?;
    } else {
        let mut lb = stream.launch_builder(&kern.main_qk);
        lb.arg(&qq).arg(&qs).arg(&kq).arg(&ks).arg(&vv).arg(&null).arg(&mut ov).arg(&bi).arg(&hi).arg(&sqi).arg(&skvi).arg(&spadi);
        unsafe { lb.launch(LaunchConfig { shared_mem_bytes: SMEM_QK, ..grid }) }.map_err(launch_err("flash_fp8_qk"))?;
    }
    Ok(())
}
