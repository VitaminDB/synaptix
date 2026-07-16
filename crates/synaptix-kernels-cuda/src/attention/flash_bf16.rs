//! FlashAttention-2 forward BF16 (GQA + online softmax).
//!
//! BF16 flash-attn-2 (скалярный FA-2; диагностический путь SYN_ATTN=fa2)
//! input/output без cast в F16. Алгоритм identical с F16 variant
//! (F32 accumulator + online softmax + GQA inside kernel).
//!
//! Два kernel'я:
//! - `flash_attn2_fwd_bf16`        — single-row (decode T_chunk=1)
//! - `flash_attn2_fwd_bf16_tiled`  ×{m32, m64} — BLOCK_M tiling (prefill)
//!
//! NVRTC compile: объединяем base `flash_attn_v2.cu` + `flash_attn_v2_bf16.cu`
//! (NVRTC не имеет linker'а, поэтому концатенируем source строки).

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::bf16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::compile_module_with_opts;

const NULL_DEV_PTR: u64 = 0;

pub struct FlashAttnBf16Kernels {
    _module_default: Arc<CudaModule>,
    _module_m64: Arc<CudaModule>,
    flash_attn2_fwd_bf16: CudaFunction,
    flash_attn2_fwd_bf16_tiled_m32: CudaFunction,
    flash_attn2_fwd_bf16_tiled_m64: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashAttnBf16Kernels>)>>> = OnceLock::new();

impl FlashAttnBf16Kernels {
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
        let new = Arc::new(Self::compile(ctx)?);
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    fn compile(ctx: &Arc<CudaContext>) -> Result<Self> {
        let base = include_str!("../cu/fused/attention/flash_attn_v2.cu");
        let bf16_src = include_str!("../cu/fused/attention/flash_attn_v2_bf16.cu");
        let combined = format!("{base}\n\n{bf16_src}");

        let load = |m: &Arc<CudaModule>, name: &str| -> Result<CudaFunction> {
            m.load_function(name)
                .map_err(|e| SynaptixError::Cuda(format!("load_function {name}: {e:?}")))
        };

        let module_default =
            compile_module_with_opts(ctx, &combined, "flash_attn_v2_bf16.cu", &[], None)?;
        let flash_attn2_fwd_bf16 = load(&module_default, "flash_attn2_fwd_bf16")?;
        let flash_attn2_fwd_bf16_tiled_m32 = load(&module_default, "flash_attn2_fwd_bf16_tiled")?;

        let module_m64 = compile_module_with_opts(
            ctx,
            &combined,
            "flash_attn_v2_bf16.cu(m64)",
            &["-DBLOCK_M=64"],
            None,
        )?;
        let flash_attn2_fwd_bf16_tiled_m64 = load(&module_m64, "flash_attn2_fwd_bf16_tiled")?;
        flash_attn2_fwd_bf16_tiled_m64
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                64 * 1024,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attribute bf16 m64: {e:?}")))?;

        Ok(Self {
            _module_default: module_default,
            _module_m64: module_m64,
            flash_attn2_fwd_bf16,
            flash_attn2_fwd_bf16_tiled_m32,
            flash_attn2_fwd_bf16_tiled_m64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        t_stride: u32,
    ) -> Result<()> {
        const BLOCK_D: u32 = 128;
        const BLOCK_KV: u32 = 64;
        if hd % BLOCK_D != 0 || hd > 512 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16: hd must be multiple of 128 and ≤512, got {hd}"
            )));
        }
        let shared_bytes = hd * std::mem::size_of::<bf16>() as u32
            + BLOCK_KV * std::mem::size_of::<f32>() as u32
            + 3 * std::mem::size_of::<f32>() as u32;
        let cfg = LaunchConfig {
            grid_dim: (b * nh, t_chunk, 1),
            block_dim: (BLOCK_D, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut bld = stream.launch_builder(&self.flash_attn2_fwd_bf16);
        bld.arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut *out)
            .arg(&scale)
            .arg(&b)
            .arg(&nh)
            .arg(&nkv)
            .arg(&t_chunk)
            .arg(&t_cache)
            .arg(&hd)
            .arg(&n_rep)
            .arg(&q_pos_base)
            .arg(&causal)
            .arg(&t_stride)
            .arg(&NULL_DEV_PTR);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch flash_attn2_fwd_bf16: {e:?}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd_bf16_tiled(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        block_m: u32,
        t_stride: u32,
    ) -> Result<()> {
        const BLOCK_D: u32 = 128;
        const BLOCK_KV: u32 = 64;
        if hd != 128 && hd != 256 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16_tiled: hd must be 128 or 256, got {hd}"
            )));
        }
        if block_m != 32 && block_m != 64 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16_tiled: block_m must be 32 or 64, got {block_m}"
            )));
        }
        let shared_bytes = block_m * hd * std::mem::size_of::<bf16>() as u32
            + block_m * BLOCK_KV * std::mem::size_of::<f32>() as u32
            + 3 * block_m * std::mem::size_of::<f32>() as u32;
        let grid_y = t_chunk.div_ceil(block_m);
        let cfg = LaunchConfig {
            grid_dim: (b * nh, grid_y, 1),
            block_dim: (BLOCK_D, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let func = if block_m == 64 {
            &self.flash_attn2_fwd_bf16_tiled_m64
        } else {
            &self.flash_attn2_fwd_bf16_tiled_m32
        };
        let mut bld = stream.launch_builder(func);
        bld.arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut *out)
            .arg(&scale)
            .arg(&b)
            .arg(&nh)
            .arg(&nkv)
            .arg(&t_chunk)
            .arg(&t_cache)
            .arg(&hd)
            .arg(&n_rep)
            .arg(&q_pos_base)
            .arg(&causal)
            .arg(&t_stride)
            .arg(&NULL_DEV_PTR);
        unsafe {
            bld.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!(
                    "launch flash_attn2_fwd_bf16_tiled(m{block_m}): {e:?}"
                ))
            })?;
        }
        Ok(())
    }

    /// Tiled prefill из untyped `u8`-storage (для `Backend::flash_attention`).
    /// Принимает byte-offset'ы (q/out — contiguous views; k/v — могут быть strided
    /// preallocated KV-буфер через `t_stride`). Транслитерирует u8→bf16 view +
    /// запускает `flash_attn2_fwd_bf16_tiled` (block_m=32|64).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd_bf16_tiled_u8(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<u8>,
        q_off: usize,
        k: &CudaSlice<u8>,
        k_off: usize,
        v: &CudaSlice<u8>,
        v_off: usize,
        out: &mut CudaSlice<u8>,
        out_off: usize,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        block_m: u32,
        t_stride: u32,
    ) -> Result<()> {
        const BLOCK_D: u32 = 128;
        const BLOCK_KV: u32 = 64;
        if hd != 128 && hd != 256 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16_tiled_u8: hd must be 128 or 256, got {hd}"
            )));
        }
        if block_m != 32 && block_m != 64 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16_tiled_u8: block_m must be 32 or 64, got {block_m}"
            )));
        }
        let esz = std::mem::size_of::<bf16>();
        let t_stride_eff = if t_stride > 0 { t_stride } else { t_cache } as usize;
        let q_n = (b as usize) * (nh as usize) * (t_chunk as usize) * (hd as usize);
        let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (hd as usize);
        let q_v = unsafe {
            q.slice(q_off..q_off + q_n * esz)
                .transmute::<bf16>(q_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 tiled: transmute q".into()))?
        };
        let k_v = unsafe {
            k.slice(k_off..k_off + kv_n * esz)
                .transmute::<bf16>(kv_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 tiled: transmute k".into()))?
        };
        let v_v = unsafe {
            v.slice(v_off..v_off + kv_n * esz)
                .transmute::<bf16>(kv_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 tiled: transmute v".into()))?
        };
        let shared_bytes = block_m * hd * std::mem::size_of::<bf16>() as u32
            + block_m * BLOCK_KV * std::mem::size_of::<f32>() as u32
            + 3 * block_m * std::mem::size_of::<f32>() as u32;
        let grid_y = t_chunk.div_ceil(block_m);
        let cfg = LaunchConfig {
            grid_dim: (b * nh, grid_y, 1),
            block_dim: (BLOCK_D, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let func = if block_m == 64 {
            &self.flash_attn2_fwd_bf16_tiled_m64
        } else {
            &self.flash_attn2_fwd_bf16_tiled_m32
        };
        let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
        let mut o_v = unsafe {
            o_s.transmute_mut::<bf16>(q_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 tiled: transmute out".into()))?
        };
        let mut bld = stream.launch_builder(func);
        bld.arg(&q_v)
            .arg(&k_v)
            .arg(&v_v)
            .arg(&mut o_v)
            .arg(&scale)
            .arg(&b)
            .arg(&nh)
            .arg(&nkv)
            .arg(&t_chunk)
            .arg(&t_cache)
            .arg(&hd)
            .arg(&n_rep)
            .arg(&q_pos_base)
            .arg(&causal)
            .arg(&t_stride)
            .arg(&NULL_DEV_PTR);
        unsafe {
            bld.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!(
                    "launch flash_attn2_fwd_bf16_tiled_u8(m{block_m}): {e:?}"
                ))
            })?;
        }
        Ok(())
    }

    /// Single-row FA-2 из untyped `u8`-storage (decode / forced FA-2 на Tq=1).
    /// q/out contiguous views (offset), k/v — strided KV-буфер через `t_stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd_bf16_u8(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<u8>,
        q_off: usize,
        k: &CudaSlice<u8>,
        k_off: usize,
        v: &CudaSlice<u8>,
        v_off: usize,
        out: &mut CudaSlice<u8>,
        out_off: usize,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        t_stride: u32,
    ) -> Result<()> {
        const BLOCK_D: u32 = 128;
        const BLOCK_KV: u32 = 64;
        if hd % BLOCK_D != 0 || hd > 512 {
            return Err(SynaptixError::Cuda(format!(
                "flash_attn2_fwd_bf16_u8: hd must be multiple of 128 and ≤512, got {hd}"
            )));
        }
        let esz = std::mem::size_of::<bf16>();
        let t_stride_eff = if t_stride > 0 { t_stride } else { t_cache } as usize;
        let q_n = (b as usize) * (nh as usize) * (t_chunk as usize) * (hd as usize);
        let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (hd as usize);
        let q_v = unsafe {
            q.slice(q_off..q_off + q_n * esz)
                .transmute::<bf16>(q_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 single-row: transmute q".into()))?
        };
        let k_v = unsafe {
            k.slice(k_off..k_off + kv_n * esz)
                .transmute::<bf16>(kv_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 single-row: transmute k".into()))?
        };
        let v_v = unsafe {
            v.slice(v_off..v_off + kv_n * esz)
                .transmute::<bf16>(kv_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 single-row: transmute v".into()))?
        };
        let shared_bytes = hd * esz as u32
            + BLOCK_KV * std::mem::size_of::<f32>() as u32
            + 3 * std::mem::size_of::<f32>() as u32;
        let cfg = LaunchConfig {
            grid_dim: (b * nh, t_chunk, 1),
            block_dim: (BLOCK_D, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
        let mut o_v = unsafe {
            o_s.transmute_mut::<bf16>(q_n)
                .ok_or_else(|| SynaptixError::Cuda("flash_bf16 single-row: transmute out".into()))?
        };
        let mut bld = stream.launch_builder(&self.flash_attn2_fwd_bf16);
        bld.arg(&q_v)
            .arg(&k_v)
            .arg(&v_v)
            .arg(&mut o_v)
            .arg(&scale)
            .arg(&b)
            .arg(&nh)
            .arg(&nkv)
            .arg(&t_chunk)
            .arg(&t_cache)
            .arg(&hd)
            .arg(&n_rep)
            .arg(&q_pos_base)
            .arg(&causal)
            .arg(&t_stride)
            .arg(&NULL_DEV_PTR);
        unsafe {
            bld.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch flash_attn2_fwd_bf16_u8: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Combined u8-диспетчер (forced FA-2): single-row при `t_chunk==1`, иначе
    /// tiled (block_m=64). Для `Backend::flash_attention` mode=Fa2.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd_u8(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<u8>,
        q_off: usize,
        k: &CudaSlice<u8>,
        k_off: usize,
        v: &CudaSlice<u8>,
        v_off: usize,
        out: &mut CudaSlice<u8>,
        out_off: usize,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        t_stride: u32,
    ) -> Result<()> {
        if t_chunk == 1 || (hd != 128 && hd != 256) {
            self.flash_attn2_fwd_bf16_u8(
                stream, q, q_off, k, k_off, v, v_off, out, out_off, scale, b, nh, nkv, t_chunk,
                t_cache, hd, n_rep, q_pos_base, causal, t_stride,
            )
        } else {
            self.flash_attn2_fwd_bf16_tiled_u8(
                stream, q, q_off, k, k_off, v, v_off, out, out_off, scale, b, nh, nkv, t_chunk,
                t_cache, hd, n_rep, q_pos_base, causal, 64, t_stride,
            )
        }
    }

    /// Dispatcher: single-row для decode (t_chunk=1) или non-supported hd,
    /// иначе tiled m32 (prefill).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn2_fwd(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        scale: f32,
        b: u32,
        nh: u32,
        nkv: u32,
        t_chunk: u32,
        t_cache: u32,
        hd: u32,
        n_rep: u32,
        q_pos_base: u32,
        causal: i32,
        t_stride: u32,
    ) -> Result<()> {
        if t_chunk == 1 || (hd != 128 && hd != 256) {
            self.flash_attn2_fwd_bf16(
                stream, q, k, v, out, scale, b, nh, nkv, t_chunk, t_cache, hd, n_rep, q_pos_base,
                causal, t_stride,
            )
        } else {
            self.flash_attn2_fwd_bf16_tiled(
                stream, q, k, v, out, scale, b, nh, nkv, t_chunk, t_cache, hd, n_rep, q_pos_base,
                causal, 32, t_stride,
            )
        }
    }
}
