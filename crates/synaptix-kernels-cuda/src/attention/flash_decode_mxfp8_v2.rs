//! MXFP8-KV flash-decode v2 (sm_120a): GQA-групповое split-K ядро для малых Tq
//! (decode Tq=1, MTP-verify Tq=2..8). Блок обслуживает GROUP query-голов одной
//! KV-головы: KV-сегмент читается из DRAM один раз на группу, деквант E4M3 —
//! аппаратным cvt (см. `flash_decode_mxfp8_v2.cu`).
//!
//! Модуль компилируется под sm_120a; на более старых архитектурах загрузка
//! падает — [`FlashDecodeMxfp8V2Kernels::try_for_context`] возвращает `None`,
//! и диспетчер остаётся на скалярном ядре из `flash_decode.cu`.
//!
//! Ограничения v2: `d ∈ {128, 256}`, q/out — F16/BF16, k/v-байты выровнены на
//! 16 (uint4-загрузки), `split_k ≤ 64`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};
use crate::wsalloc::WsAlloc;

const BLOCK: u32 = 128;
pub const V2_SPLIT_K_MAX: u32 = 64;
/// GROUP-инстансы ядра; выбирается наибольший делитель n_rep.
const GROUPS: [u32; 4] = [6, 4, 2, 1];

pub struct FlashDecodeMxfp8V2Kernels {
    _module: Arc<CudaModule>,
    // [dtype(f16=0,bf16=1)][group(1,2,4,6)][d(128=0,256=1)], split и _dev.
    split: [[[CudaFunction; 2]; 4]; 2],
    split_dev: [[[CudaFunction; 2]; 4]; 2],
    merge_f16: CudaFunction,
    merge_bf16: CudaFunction,
}

/// Кэш по контексту; `None` — компиляция/загрузка не удалась (не sm_120a),
/// чтобы не перекомпилировать на каждый вызов.
static CACHE: OnceLock<Mutex<Vec<(usize, Option<Arc<FlashDecodeMxfp8V2Kernels>>)>>> =
    OnceLock::new();

fn group_slot(group: u32) -> usize {
    match group {
        1 => 0,
        2 => 1,
        4 => 2,
        6 => 3,
        _ => unreachable!("v2 group"),
    }
}

impl FlashDecodeMxfp8V2Kernels {
    pub fn try_for_context(ctx: &Arc<CudaContext>) -> Option<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return v.clone();
                }
            }
        }
        let built = Self::build(ctx)
            .map_err(|e| {
                eprintln!("[flash_decode_mxfp8_v2] недоступен, fallback на скалярное ядро: {e}");
                e
            })
            .ok();
        cache.lock().push((key, built.clone()));
        built
    }

    fn build(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let src = include_str!("../cu/fused/attention/flash_decode_mxfp8_v2.cu");
        let module =
            compile_module_with_opts(ctx, src, "flash_decode_mxfp8_v2.cu", &[], Some("sm_120a"))?;
        let load3 = |t: &str, g: u32, d: u32, dev: bool| -> Result<CudaFunction> {
            let suffix = if dev { "_dev" } else { "" };
            load_fn(&module, &format!("fd2_{t}_g{g}_d{d}{suffix}"))
        };
        let mut split: Vec<CudaFunction> = Vec::with_capacity(16);
        let mut split_dev: Vec<CudaFunction> = Vec::with_capacity(16);
        for t in ["f16", "bf16"] {
            for g in [1u32, 2, 4, 6] {
                for d in [128u32, 256] {
                    split.push(load3(t, g, d, false)?);
                    split_dev.push(load3(t, g, d, true)?);
                }
            }
        }
        let mut sit = split.into_iter();
        let mut dit = split_dev.into_iter();
        let take = |it: &mut std::vec::IntoIter<CudaFunction>| -> [[[CudaFunction; 2]; 4]; 2] {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| it.next().unwrap()))
            })
        };
        let split = take(&mut sit);
        let split_dev = take(&mut dit);
        Ok(Arc::new(Self {
            merge_f16: load_fn(&module, "fd2_merge_f16")?,
            merge_bf16: load_fn(&module, "fd2_merge_bf16")?,
            split,
            split_dev,
            _module: module,
        }))
    }

    fn pick(&self, dtype: DType, group: u32, d: u32, dev: bool) -> Result<&CudaFunction> {
        let ti = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => return Err(SynaptixError::Unsupported("flash mxfp8 v2: q dtype")),
        };
        let di = match d {
            128 => 0,
            256 => 1,
            _ => return Err(SynaptixError::Unsupported("flash mxfp8 v2: d")),
        };
        let table = if dev { &self.split_dev } else { &self.split };
        Ok(&table[ti][group_slot(group)][di])
    }
}

/// Наибольший GROUP-инстанс, делящий n_rep (=nh/nkv).
pub fn v2_group(nh: u32, nkv: u32) -> u32 {
    let n_rep = nh / nkv.max(1);
    GROUPS.iter().copied().find(|g| n_rep % g == 0).unwrap_or(1)
}

/// Пригодность формы/типов для v2 (без учёта наличия sm_120a-модуля).
pub fn v2_shape_ok(d: u32, t_q: u32, q_dtype: DType, k_off: usize, v_off: usize) -> bool {
    (d == 128 || d == 256)
        && t_q <= 8
        && matches!(q_dtype, DType::F16 | DType::BF16)
        && k_off % 16 == 0
        && v_off % 16 == 0
}

/// split_k: целимся в ~256 блоков (латентность DRAM прячется параллелизмом),
/// но сегмент держим ≥ ~512 токенов, чтобы не платить за staging тайлов.
fn v2_split_k(gx_blocks: u32, t_len: u32) -> u32 {
    let occ = (256 / gx_blocks.max(1)).max(1);
    let cap_len = (t_len / 512).max(1);
    occ.min(cap_len).clamp(1, V2_SPLIT_K_MAX)
}

struct V2Launch<'a> {
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    group: u32,
    q_dtype: DType,
    q: &'a CudaSlice<u8>,
    q_off: usize,
    k: &'a CudaSlice<u8>,
    k_off: usize,
    v: &'a CudaSlice<u8>,
    v_off: usize,
    k_scale: &'a CudaSlice<u8>,
    ks_off: usize,
    v_scale: &'a CudaSlice<u8>,
    vs_off: usize,
    out: &'a mut CudaSlice<u8>,
    out_off: usize,
}

enum TkvArg<'a> {
    Host(u32),
    Dev(&'a CudaView<'a, u32>),
}

#[allow(clippy::too_many_arguments)]
fn launch(
    kernels: &FlashDecodeMxfp8V2Kernels,
    stream: &Arc<CudaStream>,
    p: V2Launch<'_>,
    tkv: TkvArg<'_>,
) -> Result<()> {
    let V2Launch {
        b, nh, nkv, t_q, d, scale, causal, split_k, t_stride, group, q_dtype,
        q, q_off, k, k_off, v, v_off, k_scale, ks_off, v_scale, vs_off, out, out_off,
    } = p;
    if !(1..=V2_SPLIT_K_MAX).contains(&split_k) {
        return Err(SynaptixError::Cuda(format!("flash mxfp8 v2: split_k {split_k}")));
    }
    let n_rep = nh / nkv;
    if n_rep % group != 0 {
        return Err(SynaptixError::Cuda(format!("flash mxfp8 v2: group {group} vs n_rep {n_rep}")));
    }
    let nsub = n_rep / group;
    let esz_q = (q_dtype.size_in_bits() / 8) as usize;
    let nb = (d / 32) as usize;
    let t_stride_eff = t_stride as usize;
    let q_n = (b * nh * t_q * d) as usize;
    let kv_n = (b * nkv) as usize * t_stride_eff * d as usize;
    let scale_n = (b * nkv) as usize * t_stride_eff * nb;
    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);

    let mut partial_acc = stream
        .ws_alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .ws_alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .ws_alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;

    let (b_i, nh_i, nkv_i, tq_i, sk_i) =
        (b as i32, nh as i32, nkv as i32, t_q as i32, split_k as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    // dyn smem: q half2 + p float + 2 массива скейлов float (см. .cu).
    let smem = group * d * 2 + group * 128 * 4 + 2 * (d / 32) * 128 * 4;
    let gx = b * nkv * nsub * t_q;
    let split_cfg = LaunchConfig {
        grid_dim: (gx, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let split_fn = kernels.pick(q_dtype, group, d, matches!(tkv, TkvArg::Dev(_)))?;
    let k_v = k.slice(k_off..k_off + kv_n);
    let v_v = v.slice(v_off..v_off + kv_n);
    let ks_v = k_scale.slice(ks_off..ks_off + scale_n);
    let vs_v = v_scale.slice(vs_off..vs_off + scale_n);

    macro_rules! run {
        ($t:ty, $merge:expr) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz_q)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash mxfp8 v2: transmute q".into()))?
            };
            {
                let mut bld = stream.launch_builder(split_fn);
                bld.arg(&q_v)
                    .arg(&k_v)
                    .arg(&v_v)
                    .arg(&ks_v)
                    .arg(&vs_v)
                    .arg(&mut partial_acc)
                    .arg(&mut partial_m)
                    .arg(&mut partial_l)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&tq_i);
                let tkv_host;
                match &tkv {
                    TkvArg::Host(t) => {
                        tkv_host = *t as i32;
                        bld.arg(&tkv_host);
                    }
                    TkvArg::Dev(view) => {
                        bld.arg(*view);
                    }
                }
                bld.arg(&scale).arg(&causal_i).arg(&sk_i).arg(&ts_i);
                unsafe {
                    bld.launch(split_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 v2 split: {e:?}"))
                    })?;
                }
            }
            {
                let d_i = d as i32;
                let mut o_s = out.slice_mut(out_off..out_off + q_n * esz_q);
                let mut o_v = unsafe {
                    o_s.transmute_mut::<$t>(q_n)
                        .ok_or_else(|| SynaptixError::Cuda("flash mxfp8 v2: transmute out".into()))?
                };
                let mut bld = stream.launch_builder($merge);
                bld.arg(&partial_acc)
                    .arg(&partial_m)
                    .arg(&partial_l)
                    .arg(&mut o_v)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&tq_i)
                    .arg(&d_i)
                    .arg(&sk_i);
                unsafe {
                    bld.launch(merge_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 v2 merge: {e:?}"))
                    })?;
                }
            }
        }};
    }

    match q_dtype {
        DType::F16 => run!(f16, &kernels.merge_f16),
        DType::BF16 => run!(bf16, &kernels.merge_bf16),
        _ => return Err(SynaptixError::Unsupported("flash mxfp8 v2: q dtype")),
    }
    Ok(())
}

/// v2-аналог `flash_decode_mxfp8_u8`: Tkv — host-значение.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_mxfp8_v2_u8(
    kernels: &FlashDecodeMxfp8V2Kernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    k_scale: &CudaSlice<u8>,
    ks_off: usize,
    v_scale: &CudaSlice<u8>,
    vs_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    t_stride: u32,
    q_dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 {
        return Ok(());
    }
    let group = v2_group(nh, nkv);
    let nsub = (nh / nkv.max(1)) / group;
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv };
    let split_k = v2_split_k(b * nkv * nsub * t_q, t_kv);
    launch(
        kernels,
        stream,
        V2Launch {
            b, nh, nkv, t_q, d, scale, causal, split_k,
            t_stride: t_stride_eff, group, q_dtype,
            q, q_off, k, k_off, v, v_off, k_scale, ks_off, v_scale, vs_off, out, out_off,
        },
        TkvArg::Host(t_kv),
    )
}

/// v2-аналог `flash_decode_mxfp8_u8_dev`: Tkv — device-скаляр (`*tkv_dev`),
/// split_k выводится из физического `t_stride` — конфигурация запуска не
/// зависит от текущей длины (валидно под CUDA-graph).
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_mxfp8_v2_u8_dev(
    kernels: &FlashDecodeMxfp8V2Kernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    k_scale: &CudaSlice<u8>,
    ks_off: usize,
    v_scale: &CudaSlice<u8>,
    vs_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    tkv_dev: &CudaView<u32>,
    d: u32,
    scale: f32,
    causal: bool,
    t_stride: u32,
    q_dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 {
        return Ok(());
    }
    if t_stride == 0 {
        return Err(SynaptixError::Cuda(
            "flash mxfp8 v2 dev: требуется preallocated буфер (t_stride > 0)".into(),
        ));
    }
    let group = v2_group(nh, nkv);
    let nsub = (nh / nkv.max(1)) / group;
    let split_k = v2_split_k(b * nkv * nsub * t_q, t_stride);
    launch(
        kernels,
        stream,
        V2Launch {
            b, nh, nkv, t_q, d, scale, causal, split_k, t_stride, group, q_dtype,
            q, q_off, k, k_off, v, v_off, k_scale, ks_off, v_scale, vs_off, out, out_off,
        },
        TkvArg::Dev(tkv_dev),
    )
}
