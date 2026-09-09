//! Flash-decode одного запроса с GQA (см. `cu/fused/attention/flash_decode_gqa.cu`):
//! split по ключам с фиксированной сеткой (под CUDA-граф) и слияние partials.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub const FDG_SPLIT_MAX: u32 = 32;

pub struct FlashDecodeGqaKernels {
    _module: Arc<CudaModule>,
    split: HashMap<(u32, u32), CudaFunction>,
    merge: HashMap<u32, CudaFunction>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashDecodeGqaKernels>)>>> = OnceLock::new();

impl FlashDecodeGqaKernels {
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
        let src = include_str!("../cu/fused/attention/flash_decode_gqa.cu");
        let module = compile_module(ctx, src, "flash_decode_gqa.cu")?;
        let mut split = HashMap::new();
        for (hd, nrep) in [(256u32, 1u32), (256, 2), (256, 4), (256, 8), (512, 2), (512, 4), (512, 8)] {
            split.insert((hd, nrep), load_fn(&module, &format!("fdg_split_hd{hd}_r{nrep}"))?);
        }
        let mut merge = HashMap::new();
        for hd in [256u32, 512] {
            merge.insert(hd, load_fn(&module, &format!("fdg_merge_hd{hd}"))?);
        }
        let new = Arc::new(Self { split, merge, _module: module });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    pub fn supports(&self, hd: u32, nrep: u32) -> bool {
        self.split.contains_key(&(hd, nrep))
    }

    /// Число partials на голову при `splits` сегментах: внутри блока ключи
    /// делятся ещё на `4 / WPH` частей (см. ядро).
    pub fn partials_per_head(nrep: u32, splits: u32) -> u32 {
        let hpw = nrep.div_ceil(4);
        let wph = nrep.div_ceil(hpw);
        splits * (4 / wph)
    }
}

/// Запуск split + merge. Все указатели — device-адреса; partials — f32
/// скретчи `[nh, S]`, `[nh, S]`, `[nh, S, hd]` с `S = partials_per_head`.
/// Выход: bf16 `out` [nh·hd] (0 — не нужен) и/или MXFP8-пара `mx_packed`
/// [nh·hd] + `mx_scales` [nh·hd/32] — вход o_proj без отдельного кванта.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gqa(
    k: &FlashDecodeGqaKernels,
    stream: &Arc<CudaStream>,
    q: u64,
    kc: u64,
    vc: u64,
    tkv_ptr: u64,
    scale: f32,
    window: u32,
    cap: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    splits: u32,
    part_m: u64,
    part_l: u64,
    part_acc: u64,
    out: u64,
    mx_packed: u64,
    mx_scales: u64,
) -> Result<()> {
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!("flash_decode_gqa: nh={nh} кратно nkv={nkv}")));
    }
    let nrep = nh / nkv;
    let f = k
        .split
        .get(&(hd, nrep))
        .ok_or_else(|| SynaptixError::Unsupported("flash_decode_gqa: голова/группа не поддержаны"))?;
    let mf = k.merge.get(&hd).ok_or(SynaptixError::Unsupported("flash_decode_gqa: merge hd"))?;
    if splits == 0 || splits > FDG_SPLIT_MAX {
        return Err(SynaptixError::Cuda(format!("flash_decode_gqa: splits={splits}")));
    }
    let s = FlashDecodeGqaKernels::partials_per_head(nrep, splits) as i32;
    let (wi, capi, spl) = (window as i32, cap as i32, splits as i32);
    {
        let mut b = stream.launch_builder(f);
        b.arg(&q)
            .arg(&kc)
            .arg(&vc)
            .arg(&tkv_ptr)
            .arg(&scale)
            .arg(&wi)
            .arg(&capi)
            .arg(&part_m)
            .arg(&part_l)
            .arg(&part_acc)
            .arg(&spl);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nkv, splits, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| SynaptixError::Cuda(format!("launch fdg_split: {e:?}")))?;
    }
    let mut b = stream.launch_builder(mf);
    b.arg(&part_m).arg(&part_l).arg(&part_acc).arg(&out).arg(&mx_packed).arg(&mx_scales).arg(&s);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (nh, 1, 1),
            block_dim: (hd, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| SynaptixError::Cuda(format!("launch fdg_merge: {e:?}")))?;
    Ok(())
}
