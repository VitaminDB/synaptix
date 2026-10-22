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
    col_total: CudaFunction,
    quant_vt: CudaFunction,
    main_qk: CudaFunction,
    main_full: CudaFunction,
    main_a16: CudaFunction,
}

/// Режим FP8-внимания.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fa8Mode {
    /// QKᵀ e4m3, P·V bf16.
    Qk,
    /// Обе свёртки e4m3, f32-накопление.
    Full,
    /// Обе свёртки e4m3 с f16-накоплением (вдвое быстрее mma на sm_120);
    /// кванты Q/K/V и P сужены, чтобы f16 не переполнялся.
    Acc16,
}

/// `full` → [`Fa8Mode::Acc16`], если не выключено `SYN_FA8_ACC16=0`.
pub fn full_mode() -> Fa8Mode {
    static M: OnceLock<Fa8Mode> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("SYN_FA8_ACC16").as_deref() {
        Ok("0") => Fa8Mode::Full,
        _ => Fa8Mode::Acc16,
    })
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashFp8Kernels>)>>> = OnceLock::new();
static CACHE_F16: OnceLock<Mutex<Vec<(usize, Arc<FlashFp8Kernels>)>>> = OnceLock::new();

const SMEM_QK: u32 = 8192 + 16384 + 32768;
const SMEM_FULL: u32 = 16384 + 16384 + 512;
const SMEM_A16: u32 = 16384 + 16384 + 128;

impl FlashFp8Kernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        Self::build(ctx, &CACHE, false)
    }

    /// Ядра под тип входа/выхода: BF16 или F16 (`-DFA8_F16`).
    pub fn for_dtype(ctx: &Arc<CudaContext>, dt: synaptix_core::dtype::DType) -> Result<Arc<Self>> {
        if dt == synaptix_core::dtype::DType::F16 {
            Self::build(ctx, &CACHE_F16, true)
        } else {
            Self::build(ctx, &CACHE, false)
        }
    }

    fn build(ctx: &Arc<CudaContext>, cache: &OnceLock<Mutex<Vec<(usize, Arc<FlashFp8Kernels>)>>>, f16: bool) -> Result<Arc<Self>> {
        let cache = cache.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let src = include_str!("../cu/fused/attention/flash_fp8.cu");
        let (name, opts): (&str, &[&str]) = if f16 { ("flash_fp8_f16.cu", &["-DFA8_F16"]) } else { ("flash_fp8.cu", &[]) };
        let module = compile_module_req(ctx, src, name, opts, crate::caps::Feature::Fp8Mma)?;
        let main_qk = load_fn(&module, "flash_fp8_qk_hd128_bshd")?;
        let main_full = load_fn(&module, "flash_fp8_full_hd128_bshd")?;
        let main_a16 = load_fn(&module, "flash_fp8_a16_hd128_bshd")?;
        for (f, s) in [(&main_qk, SMEM_QK), (&main_full, SMEM_FULL), (&main_a16, SMEM_A16)] {
            f.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, s as i32)
                .map_err(|e| SynaptixError::Cuda(format!("flash_fp8 smem: {e:?}")))?;
        }
        let new = Arc::new(Self {
            col_sum: load_fn(&module, "fa8_col_sum")?,
            col_amax: load_fn(&module, "fa8_col_amax")?,
            quant_rows: load_fn(&module, "fa8_quant_rows")?,
            col_total: load_fn(&module, "fa8_col_total")?,
            quant_vt: load_fn(&module, "fa8_quant_vt")?,
            main_qk,
            main_full,
            main_a16,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch_err(what: &str) -> impl Fn(cudarc::driver::DriverError) -> SynaptixError + '_ {
    move |e| SynaptixError::Cuda(format!("launch {what}: {e:?}"))
}

/// `q [B,Sq,H,128]`, `k/v [B,Skv,H,128]` bf16 или f16 — по ядрам `kern`
/// (байтовые срезы со смещением), `out [B,Sq,H,128]` того же типа.
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
    let mode = if full { full_mode() } else { Fa8Mode::Qk };
    flash_fp8_bshd_mode(kern, stream, (q, q_off), (k, k_off), (v, v_off), out, b, h, sq, skv, scale, mode)
}

/// То же с явным режимом.
#[allow(clippy::too_many_arguments)]
pub fn flash_fp8_bshd_mode(
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
    mode: Fa8Mode,
) -> Result<()> {
    let full = mode != Fa8Mode::Qk;
    let a16 = mode == Fa8Mode::Acc16;
    let qmax: f32 = if a16 { 16.0 } else { 448.0 };
    let (vq, ps): (f32, f32) = if a16 { (3.9, 1.0) } else { (448.0, 448.0) };
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

    // Среднее K по токенам: частичные суммы сплитов (каждый пишет свою
    // часть целиком — обнулять нечего), затем одна сумма на канал.
    let split = (skv / 256).clamp(1, 32) as u32;
    let mut ksum: CudaSlice<f32> = unsafe { stream.alloc(split as usize * b * h * d) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    let red_cfg = LaunchConfig { grid_dim: ((b * h) as u32, split, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    {
        let mut lb = stream.launch_builder(&kern.col_sum);
        lb.arg(&kv).arg(&mut ksum).arg(&skvi).arg(&hi);
        unsafe { lb.launch(red_cfg) }.map_err(launch_err("fa8_col_sum"))?;
    }
    let mut kmean: CudaSlice<f32> = unsafe { stream.alloc(b * h * d) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    {
        let (bhi, ns) = ((b * h) as i32, split as i32);
        let cfg = LaunchConfig { grid_dim: ((b * h * d).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut lb = stream.launch_builder(&kern.col_total);
        lb.arg(&ksum).arg(&mut kmean).arg(&bhi).arg(&ns);
        unsafe { lb.launch(cfg) }.map_err(launch_err("fa8_col_total"))?;
    }
    // Квант Q (шкала · scale · log2e) и K (минус среднее).
    let mut qq = alloc(b * sq * h * d)?;
    let mut kq = alloc(b * skv * h * d)?;
    let mut qs: CudaSlice<f32> = unsafe { stream.alloc(b * h * sq) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    // Шкалы K шагом spad: ядро PV8 грузит их плитками по 64 (cp.async 16 Б).
    let ks_stride = if full { spad } else { skv };
    let mut ks: CudaSlice<f32> = unsafe { stream.alloc(b * h * ks_stride) }.map_err(|e| SynaptixError::Cuda(format!("{e:?}")))?;
    let mul_q = scale * std::f32::consts::LOG2_E;
    let one = 1.0f32;
    let null: u64 = 0;
    // ACC16: экспоненты шкал K байтами (шаг spad), ядро читает их вместо ks.
    let mut ke = alloc(if a16 { b * h * spad } else { 1 })?;
    for (src, n_s, dst, sc, mul, sub, stride) in [(&qv, sq, &mut qq, &mut qs, mul_q, false, sq), (&kv, skv, &mut kq, &mut ks, one, true, ks_stride)] {
        let rows = b * n_s * h;
        let cfg = LaunchConfig { grid_dim: (rows.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let n_si = n_s as i32;
        let mut lb = stream.launch_builder(&kern.quant_rows);
        lb.arg(src);
        if sub {
            lb.arg(&kmean);
        } else {
            lb.arg(&null);
        }
        let stride = stride as i32;
        lb.arg(dst).arg(sc).arg(&bi).arg(&n_si).arg(&hi).arg(&mul).arg(&qmax).arg(&stride);
        if sub && a16 {
            lb.arg(&mut ke);
        } else {
            lb.arg(&null);
        }
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
            lb.arg(&vv).arg(&vmax).arg(&mut vt).arg(&mut vsc).arg(&skvi).arg(&hi).arg(&spadi).arg(&vq).arg(&ps);
            unsafe { lb.launch(cfg) }.map_err(launch_err("fa8_quant_vt"))?;
        }
        let mut lb = stream.launch_builder(if a16 { &kern.main_a16 } else { &kern.main_full });
        lb.arg(&qq).arg(&qs).arg(&kq);
        if a16 {
            lb.arg(&ke);
        } else {
            lb.arg(&ks);
        }
        lb.arg(&vt).arg(&vsc).arg(&mut ov).arg(&bi).arg(&hi).arg(&sqi).arg(&skvi).arg(&spadi);
        unsafe { lb.launch(LaunchConfig { shared_mem_bytes: if a16 { SMEM_A16 } else { SMEM_FULL }, ..grid }) }.map_err(launch_err("flash_fp8_full"))?;
    } else {
        let mut lb = stream.launch_builder(&kern.main_qk);
        lb.arg(&qq).arg(&qs).arg(&kq).arg(&ks).arg(&vv).arg(&null).arg(&mut ov).arg(&bi).arg(&hi).arg(&sqi).arg(&skvi).arg(&spadi);
        unsafe { lb.launch(LaunchConfig { shared_mem_bytes: SMEM_QK, ..grid }) }.map_err(launch_err("flash_fp8_qk"))?;
    }
    Ok(())
}
