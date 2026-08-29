//! Chunk-FLA helper-ядра для Gated DeltaNet prefill (chunked linear attention).
//!
//! Портировано из `ai-quant/src/kernels/fla.rs` (валидировано bit-exact в проде
//! Qwen3.6). Эти ядра покрывают chunk-aware element-wise части алгоритма
//! `torch_chunk_gated_delta_rule`; батчевые GEMM делает оркестратор
//! (см. `crate::scan::chunk_scan`).
//!
//! Исходник CUDA: `src/cu/fused/attention/chunk_fla.cu`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct ChunkFlaKernels {
    _module: Arc<CudaModule>,
    compute_chunk_attn: CudaFunction,
    state_update_decay: CudaFunction,
    mul_decay_mask: CudaFunction,
    sub_inplace: CudaFunction,
    add_inplace: CudaFunction,
    scale_by_exp_diff: CudaFunction,
    sub_chunk: CudaFunction,
    mul_decay_mask_chunk: CudaFunction,
    scale_k_decayed_chunk: CudaFunction,
    scale_k_decayed_all: CudaFunction,
    gdn_chunk_scan: CudaFunction,
    mul_inplace: CudaFunction,
    state_decay_from_gcumsum_chunk: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ChunkFlaKernels>)>>> = OnceLock::new();

impl ChunkFlaKernels {
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
        let src = include_str!("../cu/fused/attention/chunk_fla.cu");
        let module = compile_module(ctx, src, "chunk_fla.cu")?;
        let new = Arc::new(Self {
            compute_chunk_attn: load_fn(&module, "compute_chunk_attn_f32")?,
            state_update_decay: load_fn(&module, "state_update_decay_f32")?,
            mul_decay_mask: load_fn(&module, "mul_decay_mask_f32")?,
            sub_inplace: load_fn(&module, "sub_inplace_f32")?,
            add_inplace: load_fn(&module, "add_inplace_f32")?,
            scale_by_exp_diff: load_fn(&module, "scale_by_exp_diff_f32")?,
            sub_chunk: load_fn(&module, "sub_chunk_f32")?,
            mul_decay_mask_chunk: load_fn(&module, "mul_decay_mask_chunk_f32")?,
            scale_k_decayed_chunk: load_fn(&module, "scale_k_decayed_chunk_f32")?,
            scale_k_decayed_all: load_fn(&module, "scale_k_decayed_all_f32")?,
            gdn_chunk_scan: load_fn(&module, "gdn_chunk_scan_f32")?,
            mul_inplace: load_fn(&module, "mul_inplace_f32")?,
            state_decay_from_gcumsum_chunk: load_fn(&module, "state_decay_from_gcumsum_chunk_f32")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// Pre-compute decay_mask + intra-chunk attn (closed-form cumprod). Все
    /// тензоры F32, row-major. `k_beta`/`key` `(BH,NC,CS,HK)`, `g_cumsum`
    /// `(BH,NC,CS)`, `attn_out`/`decay_mask_out` `(BH,NC,CS,CS)`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_chunk_attn(
        &self,
        stream: &Arc<CudaStream>,
        k_beta: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        g_cumsum: &CudaSlice<f32>,
        attn_out: &mut CudaSlice<f32>,
        decay_mask_out: &mut CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        hk: u32,
    ) -> Result<()> {
        let shared_bytes = ((cs + cs * cs) as usize * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (bh, nc, 1),
            block_dim: (cs, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut b = stream.launch_builder(&self.compute_chunk_attn);
        b.arg(k_beta)
            .arg(key)
            .arg(g_cumsum)
            .arg(attn_out)
            .arg(decay_mask_out)
            .arg(&bh)
            .arg(&nc)
            .arg(&cs)
            .arg(&hk);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch compute_chunk_attn: {e:?}")))?;
        }
        Ok(())
    }

    /// `state[b,:,:] *= exp(g_last[b])`. State `(BH, HK, HV)`, g_last `(BH,)`.
    pub fn state_update_decay(
        &self,
        stream: &Arc<CudaStream>,
        state: &mut CudaSlice<f32>,
        g_last: &CudaSlice<f32>,
        bh: u32,
        hk: u32,
        hv: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_v = hv.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, hk, grid_v),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.state_update_decay);
        b.arg(state).arg(g_last).arg(&bh).arg(&hk).arg(&hv);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch state_update_decay: {e:?}")))?;
        }
        Ok(())
    }

    /// `attn_intra *= decay_mask_i`. Оба `(BH, CS, CS)`.
    pub fn mul_decay_mask(
        &self,
        stream: &Arc<CudaStream>,
        attn_intra: &mut CudaSlice<f32>,
        decay_mask_i: &CudaSlice<f32>,
        bh: u32,
        cs: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_j = cs.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, cs, grid_j),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.mul_decay_mask);
        b.arg(attn_intra).arg(decay_mask_i).arg(&bh).arg(&cs);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch mul_decay_mask: {e:?}")))?;
        }
        Ok(())
    }

    /// `a -= b` поэлементно, n элементов.
    pub fn sub_inplace(
        &self,
        stream: &Arc<CudaStream>,
        a: &mut CudaSlice<f32>,
        b_in: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        self.binary_inplace(stream, &self.sub_inplace, a, b_in, n, "sub_inplace")
    }

    /// `a += b` поэлементно, n элементов.
    pub fn add_inplace(
        &self,
        stream: &Arc<CudaStream>,
        a: &mut CudaSlice<f32>,
        b_in: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        self.binary_inplace(stream, &self.add_inplace, a, b_in, n, "add_inplace")
    }

    fn binary_inplace(
        &self,
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        a: &mut CudaSlice<f32>,
        b_in: &CudaSlice<f32>,
        n: u32,
        tag: &str,
    ) -> Result<()> {
        let block: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(func);
        b.arg(a).arg(b_in).arg(&n);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch {tag}: {e:?}")))?;
        }
        Ok(())
    }

    /// Common scaling: `mode=0` → `out = in * exp(vec_g[row])`; `mode=1` →
    /// `out = in * exp(scalar_g[row/cs_in] - vec_g[row])`. `scalar_g` для mode=0
    /// можно передать `None` (kernel игнорирует — передаётся `vec_g`-заглушка).
    #[allow(clippy::too_many_arguments)]
    pub fn scale_by_exp_diff(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        scalar_g: Option<&CudaSlice<f32>>,
        vec_g: &CudaSlice<f32>,
        total_rows: u32,
        d: u32,
        cs_in: u32,
        mode: u32,
    ) -> Result<()> {
        let block: u32 = 128;
        let grid_c = d.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (total_rows, grid_c, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let scalar_arg = scalar_g.unwrap_or(vec_g);
        let mut b = stream.launch_builder(&self.scale_by_exp_diff);
        b.arg(out)
            .arg(input)
            .arg(scalar_arg)
            .arg(vec_g)
            .arg(&total_rows)
            .arg(&d)
            .arg(&cs_in)
            .arg(&mode);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch scale_by_exp_diff: {e:?}")))?;
        }
        Ok(())
    }

    /// `value_proc[:, ci, :, :] -= v_prime`. value_proc `(BH,NC,CS,HV)`,
    /// v_prime `(BH,CS,HV)`.
    #[allow(clippy::too_many_arguments)]
    pub fn sub_chunk(
        &self,
        stream: &Arc<CudaStream>,
        value_proc: &mut CudaSlice<f32>,
        v_prime: &CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        hv: u32,
        chunk_idx: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_v = hv.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, cs, grid_v),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.sub_chunk);
        b.arg(value_proc)
            .arg(v_prime)
            .arg(&bh)
            .arg(&nc)
            .arg(&cs)
            .arg(&hv)
            .arg(&chunk_idx);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch sub_chunk: {e:?}")))?;
        }
        Ok(())
    }

    /// `attn_intra *= decay_mask[:, ci, :, :]`. attn_intra `(BH,CS,CS)`,
    /// decay_mask `(BH,NC,CS,CS)`.
    pub fn mul_decay_mask_chunk(
        &self,
        stream: &Arc<CudaStream>,
        attn_intra: &mut CudaSlice<f32>,
        decay_mask: &CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        chunk_idx: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_j = cs.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, cs, grid_j),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.mul_decay_mask_chunk);
        b.arg(attn_intra)
            .arg(decay_mask)
            .arg(&bh)
            .arg(&nc)
            .arg(&cs)
            .arg(&chunk_idx);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch mul_decay_mask_chunk: {e:?}")))?;
        }
        Ok(())
    }

    /// `k_decayed = k[:, ci, :, :] * exp(g_last - g_cumsum[:, ci, :])`.
    /// k `(BH,NC,CS,HK)`, g_cumsum `(BH,NC,CS)`, k_decayed_out `(BH,CS,HK)`.
    #[allow(clippy::too_many_arguments)]
    /// Формы, на которые рассчитан слитый цикл чанк-скана.
    pub fn chunk_scan_fits(cs: u32, hk: u32, hv: u32) -> bool {
        cs == 64 && hk == 128 && hv % 64 == 0
    }

    /// Весь главный цикл чанк-скана одним запуском: блок берёт голову и
    /// полосу выходных каналов, держит состояние в shared и проходит по всем
    /// чанкам сам.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_scan(
        &self,
        stream: &Arc<CudaStream>,
        q_scaled: &CudaSlice<f32>,
        k_cumdecay: &CudaSlice<f32>,
        k_decayed: &CudaSlice<f32>,
        value_proc: &CudaSlice<f32>,
        attn: &CudaSlice<f32>,
        g_cumsum: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        bh: u32,
        nc: u32,
        hv: u32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (bh, hv / 64, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.gdn_chunk_scan);
        b.arg(q_scaled)
            .arg(k_cumdecay)
            .arg(k_decayed)
            .arg(value_proc)
            .arg(attn)
            .arg(g_cumsum)
            .arg(state)
            .arg(out)
            .arg(&nc)
            .arg(&hv);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch gdn_chunk_scan: {e:?}")))?;
        }
        Ok(())
    }

    /// `k_decayed` сразу по всем чанкам: шаг от состояния не зависит, а
    /// поштучно это был запуск на чанк с сеткой в `BH` блоков.
    pub fn scale_k_decayed_all(
        &self,
        stream: &Arc<CudaStream>,
        k_decayed_out: &mut CudaSlice<f32>,
        k: &CudaSlice<f32>,
        g_cumsum: &CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        hk: u32,
    ) -> Result<()> {
        let block: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: (bh * nc * cs, hk.div_ceil(block), 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.scale_k_decayed_all);
        b.arg(k_decayed_out).arg(k).arg(g_cumsum).arg(&bh).arg(&nc).arg(&cs).arg(&hk);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch scale_k_decayed_all: {e:?}")))?;
        }
        Ok(())
    }

    /// Поэлементное `a *= b` по всему буферу.
    pub fn mul_inplace(
        &self,
        stream: &Arc<CudaStream>,
        a: &mut CudaSlice<f32>,
        b_in: &CudaSlice<f32>,
        n: u64,
    ) -> Result<()> {
        let block: u32 = 256;
        let grid = (n.div_ceil(block as u64)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = stream.launch_builder(&self.mul_inplace);
        bld.arg(a).arg(b_in).arg(&n);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch mul_inplace: {e:?}")))?;
        }
        Ok(())
    }

    pub fn scale_k_decayed_chunk(
        &self,
        stream: &Arc<CudaStream>,
        k_decayed_out: &mut CudaSlice<f32>,
        k: &CudaSlice<f32>,
        g_cumsum: &CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        hk: u32,
        chunk_idx: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_d = hk.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, cs, grid_d),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.scale_k_decayed_chunk);
        b.arg(k_decayed_out)
            .arg(k)
            .arg(g_cumsum)
            .arg(&bh)
            .arg(&nc)
            .arg(&cs)
            .arg(&hk)
            .arg(&chunk_idx);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch scale_k_decayed_chunk: {e:?}")))?;
        }
        Ok(())
    }

    /// `state *= exp(g_cumsum[:, ci, CS-1])` — device-only. State `(BH,HK,HV)`.
    #[allow(clippy::too_many_arguments)]
    pub fn state_decay_from_gcumsum_chunk(
        &self,
        stream: &Arc<CudaStream>,
        state: &mut CudaSlice<f32>,
        g_cumsum: &CudaSlice<f32>,
        bh: u32,
        nc: u32,
        cs: u32,
        hk: u32,
        hv: u32,
        chunk_idx: u32,
    ) -> Result<()> {
        let block: u32 = 64;
        let grid_v = hv.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (bh, hk, grid_v),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.state_decay_from_gcumsum_chunk);
        b.arg(state)
            .arg(g_cumsum)
            .arg(&bh)
            .arg(&nc)
            .arg(&cs)
            .arg(&hk)
            .arg(&hv)
            .arg(&chunk_idx);
        unsafe {
            b.launch(cfg).map_err(|e| {
                SynaptixError::Cuda(format!("launch state_decay_from_gcumsum_chunk: {e:?}"))
            })?;
        }
        Ok(())
    }
}
