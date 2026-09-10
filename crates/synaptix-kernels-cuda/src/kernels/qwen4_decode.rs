//! Слитые ядра мелких этапов Qwen4Exp: групповой RMS, gated-residual
//! (mix/inject), масштаб+активация, взвешенная сумма строк MoE.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

pub struct Qwen4DecodeKernels {
    _module: Arc<CudaModule>,
    group_rms: CudaFunction,
    hc_mix: CudaFunction,
    hc_inject: CudaFunction,
    scale_act: CudaFunction,
    weighted_rows_sum: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Qwen4DecodeKernels>)>>> = OnceLock::new();

impl Qwen4DecodeKernels {
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
        let src = include_str!("../cu/fused/qwen4_decode.cu");
        let module = compile_module(ctx, src, "qwen4_decode.cu")?;
        let new = Arc::new(Self {
            group_rms: load_fn(&module, "group_rms_f16")?,
            hc_mix: load_fn(&module, "hc_mix_f16")?,
            hc_inject: load_fn(&module, "hc_inject_f16")?,
            scale_act: load_fn(&module, "scale_act_f16")?,
            weighted_rows_sum: load_fn(&module, "weighted_rows_sum_f16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// Буфер с байтовым смещением начала данных.
pub type In<'a> = (&'a CudaSlice<u8>, usize);
pub type Out<'a> = (&'a mut CudaSlice<u8>, usize);

fn grid1(n: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as u64).max(1) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_err(what: &str, e: cudarc::driver::DriverError) -> SynaptixError {
    SynaptixError::Cuda(format!("launch {what}: {e:?}"))
}

#[allow(clippy::too_many_arguments)]
pub fn group_rms_f16(
    k: &Qwen4DecodeKernels,
    stream: &Arc<CudaStream>,
    x: In,
    w: In,
    out: Out,
    rows: u32,
    groups: u32,
    group: u32,
    eps: f32,
) -> Result<()> {
    let n = rows as usize * groups as usize * group as usize * 2;
    let xv = x.0.slice(x.1..x.1 + n);
    let wv = w.0.slice(w.1..w.1 + groups as usize * group as usize * 2);
    let mut ov = out.0.slice_mut(out.1..out.1 + n);
    let cfg = LaunchConfig {
        grid_dim: (rows * groups, 1, 1),
        block_dim: (BLOCK.min(group.next_power_of_two()).max(32), 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&k.group_rms);
    bld.arg(&xv).arg(&wv).arg(&mut ov).arg(&rows).arg(&groups).arg(&group).arg(&eps);
    unsafe { bld.launch(cfg).map_err(|e| launch_err("group_rms_f16", e))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn hc_mix_f16(
    k: &Qwen4DecodeKernels,
    stream: &Arc<CudaStream>,
    up: In,
    normed: In,
    out: Out,
    rows: u32,
    hc: u32,
    h: u32,
) -> Result<()> {
    let n_in = rows as usize * hc as usize * h as usize * 2;
    let n_out = rows as usize * h as usize * 2;
    let uv = up.0.slice(up.1..up.1 + n_in);
    let nv = normed.0.slice(normed.1..normed.1 + n_in);
    let mut ov = out.0.slice_mut(out.1..out.1 + n_out);
    let mut bld = stream.launch_builder(&k.hc_mix);
    bld.arg(&uv).arg(&nv).arg(&mut ov).arg(&rows).arg(&hc).arg(&h);
    unsafe { bld.launch(grid1(rows as u64 * h as u64)).map_err(|e| launch_err("hc_mix_f16", e))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn hc_inject_f16(
    k: &Qwen4DecodeKernels,
    stream: &Arc<CudaStream>,
    hyper: In,
    block: In,
    w: In,
    out: Out,
    rows: u32,
    hc: u32,
    h: u32,
) -> Result<()> {
    let n_all = rows as usize * hc as usize * h as usize * 2;
    let hv = hyper.0.slice(hyper.1..hyper.1 + n_all);
    let bv = block.0.slice(block.1..block.1 + rows as usize * h as usize * 2);
    let wv = w.0.slice(w.1..w.1 + rows as usize * hc as usize * 2);
    let mut ov = out.0.slice_mut(out.1..out.1 + n_all);
    let mut bld = stream.launch_builder(&k.hc_inject);
    bld.arg(&hv).arg(&bv).arg(&wv).arg(&mut ov).arg(&rows).arg(&hc).arg(&h);
    unsafe {
        bld.launch(grid1(rows as u64 * hc as u64 * h as u64))
            .map_err(|e| launch_err("hc_inject_f16", e))?
    };
    Ok(())
}

pub fn scale_act_f16(
    k: &Qwen4DecodeKernels,
    stream: &Arc<CudaStream>,
    x: In,
    out: Out,
    n: u32,
    scale: f32,
    act: u32,
) -> Result<()> {
    let xv = x.0.slice(x.1..x.1 + n as usize * 2);
    let mut ov = out.0.slice_mut(out.1..out.1 + n as usize * 2);
    let mut bld = stream.launch_builder(&k.scale_act);
    bld.arg(&xv).arg(&mut ov).arg(&n).arg(&scale).arg(&act);
    unsafe { bld.launch(grid1(n as u64)).map_err(|e| launch_err("scale_act_f16", e))? };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn weighted_rows_sum_f16(
    k: &Qwen4DecodeKernels,
    stream: &Arc<CudaStream>,
    parts: In,
    w: In,
    out: Out,
    t: u32,
    kk: u32,
    h: u32,
) -> Result<()> {
    let pv = parts.0.slice(parts.1..parts.1 + t as usize * kk as usize * h as usize * 2);
    let wv = w.0.slice(w.1..w.1 + t as usize * kk as usize * 4);
    let mut ov = out.0.slice_mut(out.1..out.1 + t as usize * h as usize * 2);
    let mut bld = stream.launch_builder(&k.weighted_rows_sum);
    bld.arg(&pv).arg(&wv).arg(&mut ov).arg(&t).arg(&kk).arg(&h);
    unsafe {
        bld.launch(grid1(t as u64 * h as u64)).map_err(|e| launch_err("weighted_rows_sum_f16", e))?
    };
    Ok(())
}
