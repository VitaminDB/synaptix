//! Attention по таблице блоков KV (разреженный путь QSA).
//!
//! Запрос смотрит на свой набор блоков по `ratio` подряд идущих позиций плюс
//! хвост. Ядро читает KV прямо по таблице, поэтому собранного буфера не нужно
//! вовсе: раньше KV проходил через память трижды (чтение исходного, запись
//! собранного, чтение собранного ядром внимания).
//!
//!   q     (B, NH, D)     по одному запросу на строку
//!   k/v   (NKV, CAP, D)  общий KV-буфер слоя
//!   table (B, NB)        индексы блоков, `tail_from`/`tail_len` — хвост
//!   out   (B, NH, D)

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, compile_module_with_opts, load_fn};

const BLOCK: u32 = 128;
const WARPS: u32 = 4;
/// Столько q-голов на одну kv-голову ядро держит в регистрах.
const MAX_REP: u32 = 16;
/// Столько элементов строки держит один лейн (`D / 32`).
const MAX_VEC: u32 = 8;

pub struct FlashBlocksKernels {
    _module: Arc<CudaModule>,
    _fast: Option<Arc<CudaModule>>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
    f32_h128_r8: CudaFunction,
    f16_h128_r8: CudaFunction,
    bf16_h128_r8: CudaFunction,
    f32_h256_r6: CudaFunction,
    f16_h256_r6: CudaFunction,
    bf16_h256_r6: CudaFunction,
    q8_f32: CudaFunction,
    q8_f16: CudaFunction,
    q8_bf16: CudaFunction,
    q8_f32_h128_r8: CudaFunction,
    q8_f16_h128_r8: CudaFunction,
    q8_bf16_h128_r8: CudaFunction,
    q8_f32_h256_r6: CudaFunction,
    q8_f16_h256_r6: CudaFunction,
    q8_bf16_h256_r6: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashBlocksKernels>)>>> = OnceLock::new();

impl FlashBlocksKernels {
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
        let src = include_str!("../cu/fused/attention/flash_blocks.cu");
        let module = compile_module(ctx, src, "flash_blocks.cu")?;
        // Тот же исходник, но под sm_120a: там деквант E4M3 — одна
        // инструкция, а на sm_80 он разворачивается программно и ядро по
        // квантованному KV выходит вдвое медленнее плотного. На картах, где
        // модуль не поднимается, остаются функции из основного.
        let fast = compile_module_with_opts(ctx, src, "flash_blocks.cu", &[], Some("sm_120a")).ok();
        let q8 = fast.as_ref().unwrap_or(&module);
        let new = Arc::new(Self {
            f32: load_fn(&module, "flash_blocks_f32")?,
            f16: load_fn(&module, "flash_blocks_f16")?,
            bf16: load_fn(&module, "flash_blocks_bf16")?,
            f32_h128_r8: load_fn(&module, "flash_blocks_f32_h128_r8")?,
            f16_h128_r8: load_fn(&module, "flash_blocks_f16_h128_r8")?,
            bf16_h128_r8: load_fn(&module, "flash_blocks_bf16_h128_r8")?,
            f32_h256_r6: load_fn(&module, "flash_blocks_f32_h256_r6")?,
            f16_h256_r6: load_fn(&module, "flash_blocks_f16_h256_r6")?,
            bf16_h256_r6: load_fn(&module, "flash_blocks_bf16_h256_r6")?,
            q8_f32: load_fn(q8, "flash_blocks_mxfp8_f32")?,
            q8_f16: load_fn(q8, "flash_blocks_mxfp8_f16")?,
            q8_bf16: load_fn(q8, "flash_blocks_mxfp8_bf16")?,
            q8_f32_h128_r8: load_fn(q8, "flash_blocks_mxfp8_f32_h128_r8")?,
            q8_f16_h128_r8: load_fn(q8, "flash_blocks_mxfp8_f16_h128_r8")?,
            q8_bf16_h128_r8: load_fn(q8, "flash_blocks_mxfp8_bf16_h128_r8")?,
            q8_f32_h256_r6: load_fn(q8, "flash_blocks_mxfp8_f32_h256_r6")?,
            q8_f16_h256_r6: load_fn(q8, "flash_blocks_mxfp8_f16_h256_r6")?,
            q8_bf16_h256_r6: load_fn(q8, "flash_blocks_mxfp8_bf16_h256_r6")?,
            _fast: fast,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flash_blocks_u8(
    kernels: &FlashBlocksKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    table: &CudaSlice<u8>,
    tail_from: &CudaSlice<u8>,
    tail_len: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    b: u32,
    nh: u32,
    nkv: u32,
    cap: u32,
    d: u32,
    nb: u32,
    ratio: u32,
    scale: f32,
    row_offset: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || d == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks: NH={nh} не кратно NKV={nkv}"
        )));
    }
    let n_rep = nh / nkv;
    if n_rep > MAX_REP {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks: {n_rep} голов на kv-голову, потолок {MAX_REP}"
        )));
    }
    if ratio == 0 || cap % ratio != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks: ёмкость {cap} не кратна блоку {ratio}"
        )));
    }
    if d % 32 != 0 || d / 32 > MAX_VEC {
        return Err(SynaptixError::Unsupported("flash_blocks: голова не кратна 32"));
    }
    // Ходовые формы идут специализациями: границы циклов известны
    // компилятору, и q с аккумулятором остаются в регистрах. Голов на проход
    // берётся столько, сколько их состояние выдержит.
    let rep_tile = if d == 128 && n_rep % 8 == 0 {
        8
    } else if d == 256 && n_rep % 6 == 0 {
        6
    } else {
        4
    };
    let func = match (dtype, d, rep_tile) {
        (DType::F32, 128, 8) => &kernels.f32_h128_r8,
        (DType::F16, 128, 8) => &kernels.f16_h128_r8,
        (DType::BF16, 128, 8) => &kernels.bf16_h128_r8,
        (DType::F32, 256, 6) => &kernels.f32_h256_r6,
        (DType::F16, 256, 6) => &kernels.f16_h256_r6,
        (DType::BF16, 256, 6) => &kernels.bf16_h256_r6,
        (DType::F32, _, _) => &kernels.f32,
        (DType::F16, _, _) => &kernels.f16,
        (DType::BF16, _, _) => &kernels.bf16,
        (other, _, _) => {
            return Err(SynaptixError::Cuda(format!(
                "flash_blocks: dtype {other:?} не поддержан"
            )))
        }
    };

    let groups = n_rep.div_ceil(rep_tile);
    let smem = (WARPS * rep_tile.min(n_rep) * (d + 2)) * 4;
    let cfg = LaunchConfig {
        grid_dim: (b * nkv, groups, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let (b_i, nh_i, nkv_i, cap_i, d_i, nb_i, ratio_i, off_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        cap as i32,
        d as i32,
        nb as i32,
        ratio as i32,
        row_offset as i32,
    );
    let mut bld = stream.launch_builder(func);
    bld.arg(q)
        .arg(k)
        .arg(v)
        .arg(table)
        .arg(tail_from)
        .arg(tail_len)
        .arg(out)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&nkv_i)
        .arg(&cap_i)
        .arg(&d_i)
        .arg(&nb_i)
        .arg(&ratio_i)
        .arg(&scale)
        .arg(&off_i);
    unsafe {
        bld.launch(cfg).map_err(|e| {
            SynaptixError::Cuda(format!(
                "launch flash_blocks: {e:?} (b={b} nh={nh} nkv={nkv} cap={cap} d={d} nb={nb} ratio={ratio} smem={smem} rep_tile={rep_tile})"
            ))
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn flash_blocks_mxfp8_u8(
    kernels: &FlashBlocksKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    k_scale: &CudaSlice<u8>,
    v_scale: &CudaSlice<u8>,
    table: &CudaSlice<u8>,
    tail_from: &CudaSlice<u8>,
    tail_len: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    b: u32,
    nh: u32,
    nkv: u32,
    cap: u32,
    d: u32,
    nb: u32,
    ratio: u32,
    scale: f32,
    row_offset: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || d == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks mxfp8: NH={nh} не кратно NKV={nkv}"
        )));
    }
    let n_rep = nh / nkv;
    if n_rep > MAX_REP {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks mxfp8: {n_rep} голов на kv-голову, потолок {MAX_REP}"
        )));
    }
    if ratio == 0 || cap % ratio != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_blocks mxfp8: ёмкость {cap} не кратна блоку {ratio}"
        )));
    }
    if d % 32 != 0 || d / 32 > MAX_VEC {
        return Err(SynaptixError::Unsupported("flash_blocks mxfp8: голова не кратна 32"));
    }
    let rep_tile = if d == 128 && n_rep % 8 == 0 {
        8
    } else if d == 256 && n_rep % 6 == 0 {
        6
    } else {
        4
    };
    let func = match (dtype, d, rep_tile) {
        (DType::F32, 128, 8) => &kernels.q8_f32_h128_r8,
        (DType::F16, 128, 8) => &kernels.q8_f16_h128_r8,
        (DType::BF16, 128, 8) => &kernels.q8_bf16_h128_r8,
        (DType::F32, 256, 6) => &kernels.q8_f32_h256_r6,
        (DType::F16, 256, 6) => &kernels.q8_f16_h256_r6,
        (DType::BF16, 256, 6) => &kernels.q8_bf16_h256_r6,
        (DType::F32, _, _) => &kernels.q8_f32,
        (DType::F16, _, _) => &kernels.q8_f16,
        (DType::BF16, _, _) => &kernels.q8_bf16,
        (other, _, _) => {
            return Err(SynaptixError::Cuda(format!(
                "flash_blocks mxfp8: dtype {other:?} не поддержан"
            )))
        }
    };

    let groups = n_rep.div_ceil(rep_tile);
    let smem = (WARPS * rep_tile.min(n_rep) * (d + 2)) * 4;
    let cfg = LaunchConfig {
        grid_dim: (b * nkv, groups, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let (b_i, nh_i, nkv_i, cap_i, d_i, nb_i, ratio_i, off_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        cap as i32,
        d as i32,
        nb as i32,
        ratio as i32,
        row_offset as i32,
    );
    let mut bld = stream.launch_builder(func);
    bld.arg(q)
        .arg(k)
        .arg(v)
        .arg(k_scale)
        .arg(v_scale)
        .arg(table)
        .arg(tail_from)
        .arg(tail_len)
        .arg(out)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&nkv_i)
        .arg(&cap_i)
        .arg(&d_i)
        .arg(&nb_i)
        .arg(&ratio_i)
        .arg(&scale)
        .arg(&off_i);
    unsafe {
        bld.launch(cfg).map_err(|e| {
            SynaptixError::Cuda(format!(
                "launch flash_blocks mxfp8: {e:?} (b={b} nh={nh} nkv={nkv} cap={cap} d={d} nb={nb} ratio={ratio} smem={smem} rep_tile={rep_tile})"
            ))
        })?;
    }
    Ok(())
}
