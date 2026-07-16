//! Chunk-scan оркестратор для Gated DeltaNet prefill (chunked linear attention).
//!
//! Реализует chunked-форму gated delta rule поверх helper-ядер
//! [`crate::attention::chunk_fla`] + локальных f32-примитивов (`src/cu/scan/chunk_scan.cu`):
//! cumsum, L2-norm, построчное умножение, наивный strided-batched GEMM (cuBLAS
//! выпилен из synaptix). Все операции device-resident, без host-roundtrip.
//!
//! Cumsum считается **пер-чанково** (decay сбрасывается в начале чанка), что
//! доказуемо эквивалентно рекуррентному [`crate::ssm::gated_delta_rule`] —
//! см. parity-тест `tests/cuda_chunk_scan.rs` (проверяется и nc>1).
//!
//! Требование: `T % CS == 0` (без паддинга). Layout входов — `(BH, T, *)`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::attention::chunk_fla::ChunkFlaKernels;
use crate::kernels::compile::{compile_module, load_fn};

pub struct ChunkScanKernels {
    _module: Arc<CudaModule>,
    cumsum: CudaFunction,
    l2norm_scale: CudaFunction,
    mul_rowwise: CudaFunction,
    bmm: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ChunkScanKernels>)>>> = OnceLock::new();

impl ChunkScanKernels {
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
        let src = include_str!("../cu/scan/chunk_scan.cu");
        let module = compile_module(ctx, src, "chunk_scan.cu")?;
        let new = Arc::new(Self {
            cumsum: load_fn(&module, "cumsum_lastdim_f32")?,
            l2norm_scale: load_fn(&module, "l2norm_scale_lastdim_f32")?,
            mul_rowwise: load_fn(&module, "mul_rowwise_f32")?,
            bmm: load_fn(&module, "bmm_f32")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }

    /// `out[row, i] = Σ_{j≤i} in[row, i]`, rows строк длины `n` каждая.
    pub fn cumsum_lastdim(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: u32,
        n: u32,
    ) -> Result<()> {
        let block = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.cumsum);
        b.arg(input).arg(out).arg(&rows).arg(&n);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch cumsum_lastdim: {e:?}")))?;
        }
        Ok(())
    }

    /// `out = in / sqrt(Σ x² + eps) * scale` по последней оси (`dim`).
    pub fn l2norm_scale_lastdim(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: u32,
        dim: u32,
        scale: f32,
        eps: f32,
    ) -> Result<()> {
        let block = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.l2norm_scale);
        b.arg(input)
            .arg(out)
            .arg(&rows)
            .arg(&dim)
            .arg(&scale)
            .arg(&eps);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch l2norm_scale: {e:?}")))?;
        }
        Ok(())
    }

    /// `out[row, d] = in[row, d] * scal[row]`.
    pub fn mul_rowwise(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        scal: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: u32,
        dim: u32,
    ) -> Result<()> {
        let block = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, dim.div_ceil(block), 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.mul_rowwise);
        b.arg(input).arg(scal).arg(out).arg(&rows).arg(&dim);
        unsafe {
            b.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch mul_rowwise: {e:?}")))?;
        }
        Ok(())
    }

    /// `C = alpha · op(A) · op(B) + beta · C` (strided-batched). Offsets/strides
    /// в элементах. op(A)=(M,K), op(B)=(K,N), C=(M,N) row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn bmm(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        off_a: u32,
        b_mat: &CudaSlice<f32>,
        off_b: u32,
        c: &mut CudaSlice<f32>,
        off_c: u32,
        trans_a: bool,
        trans_b: bool,
        m: u32,
        n: u32,
        k: u32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch: u32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let block = 256u32;
        let total = (batch as u64) * (m as u64) * (n as u64);
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let ta: i32 = trans_a as i32;
        let tb: i32 = trans_b as i32;
        let mut bld = stream.launch_builder(&self.bmm);
        bld.arg(a)
            .arg(&off_a)
            .arg(b_mat)
            .arg(&off_b)
            .arg(c)
            .arg(&off_c)
            .arg(&ta)
            .arg(&tb)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .arg(&stride_a)
            .arg(&stride_b)
            .arg(&stride_c)
            .arg(&batch)
            .arg(&alpha)
            .arg(&beta);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| SynaptixError::Cuda(format!("launch bmm: {e:?}")))?;
        }
        Ok(())
    }
}

/// Chunked gated delta rule (prefill). Layout входов `(BH, T, *)`, state
/// `(BH, HK, HV)` in/out, out `(BH, T, HV)`. Требует `T % cs == 0`.
///
/// Эквивалентно рекуррентному `gated_delta_rule_step`, применённому T раз:
/// q/k нормализуются L2 по hk, q·=q_scale, decay g_t=exp(g) накапливается
/// пер-чанково.
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule(
    cfk: &ChunkFlaKernels,
    csk: &ChunkScanKernels,
    stream: &Arc<CudaStream>,
    q_in: &CudaSlice<f32>,      // (BH, T, HK)
    k_in: &CudaSlice<f32>,      // (BH, T, HK)
    v_in: &CudaSlice<f32>,      // (BH, T, HV)
    g_in: &CudaSlice<f32>,      // (BH, T)
    beta_in: &CudaSlice<f32>,   // (BH, T)
    state: &mut CudaSlice<f32>, // (BH, HK, HV) in/out
    out: &mut CudaSlice<f32>,   // (BH, T, HV)
    q_scale: f32,
    bh: u32,
    t: u32,
    hk: u32,
    hv: u32,
    cs: u32,
) -> Result<()> {
    if t % cs != 0 {
        return Err(SynaptixError::Cuda(format!(
            "chunk_gated_delta_rule: T={t} не кратно cs={cs}"
        )));
    }
    let nc = t / cs;
    let (bh_u, t_u, hk_u, hv_u, cs_u, nc_u) = (
        bh as usize,
        t as usize,
        hk as usize,
        hv as usize,
        cs as usize,
        nc as usize,
    );

    let alloc = |n: usize| -> Result<CudaSlice<f32>> {
        stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| SynaptixError::Cuda(format!("alloc chunk_scan ws: {e:?}")))
    };

    // ── Препроцессинг.
    let mut q_n = alloc(bh_u * t_u * hk_u)?;
    let mut k_n = alloc(bh_u * t_u * hk_u)?;
    let mut v_beta = alloc(bh_u * t_u * hv_u)?;
    let mut k_beta = alloc(bh_u * t_u * hk_u)?;
    let mut g_cumsum = alloc(bh_u * t_u)?;

    csk.l2norm_scale_lastdim(stream, q_in, &mut q_n, bh * t, hk, q_scale, 1e-6)?;
    csk.l2norm_scale_lastdim(stream, k_in, &mut k_n, bh * t, hk, 1.0, 1e-6)?;
    csk.mul_rowwise(stream, v_in, beta_in, &mut v_beta, bh * t, hv)?;
    csk.mul_rowwise(stream, &k_n, beta_in, &mut k_beta, bh * t, hk)?;
    // g_in (BH, T) = (BH*NC, CS) — cumsum по чанку.
    csk.cumsum_lastdim(stream, g_in, &mut g_cumsum, bh * nc, cs)?;

    // ── Workspace.
    let mut attn = alloc(bh_u * nc_u * cs_u * cs_u)?;
    let mut dm = alloc(bh_u * nc_u * cs_u * cs_u)?;
    let mut value_proc = alloc(bh_u * nc_u * cs_u * hv_u)?;
    let mut k_cumdecay = alloc(bh_u * nc_u * cs_u * hk_u)?;
    let mut k_cumdecay_input = alloc(bh_u * nc_u * cs_u * hk_u)?;
    let mut q_scaled = alloc(bh_u * nc_u * cs_u * hk_u)?;
    let mut v_prime = alloc(bh_u * cs_u * hv_u)?;
    let mut attn_intra = alloc(bh_u * cs_u * cs_u)?;
    let mut k_decayed = alloc(bh_u * cs_u * hk_u)?;

    // ── intra-chunk attn + decay_mask.
    cfk.compute_chunk_attn(
        stream, &k_beta, &k_n, &g_cumsum, &mut attn, &mut dm, bh, nc, cs, hk,
    )?;

    // ── k_cumdecay_input = k_beta * exp(g_cumsum); q_scaled = q_n * exp(g_cumsum).
    cfk.scale_by_exp_diff(
        stream,
        &mut k_cumdecay_input,
        &k_beta,
        None,
        &g_cumsum,
        bh * nc * cs,
        hk,
        cs,
        0,
    )?;
    cfk.scale_by_exp_diff(
        stream,
        &mut q_scaled,
        &q_n,
        None,
        &g_cumsum,
        bh * nc * cs,
        hk,
        cs,
        0,
    )?;

    // ── value_proc = attn @ v_beta; k_cumdecay = attn @ k_cumdecay_input. (batch BH*NC)
    let bnc = bh * nc;
    csk.bmm(
        stream,
        &attn,
        0,
        &v_beta,
        0,
        &mut value_proc,
        0,
        false,
        false,
        cs,
        hv,
        cs,
        (cs * cs) as i64,
        (cs * hv) as i64,
        (cs * hv) as i64,
        bnc,
        1.0,
        0.0,
    )?;
    csk.bmm(
        stream,
        &attn,
        0,
        &k_cumdecay_input,
        0,
        &mut k_cumdecay,
        0,
        false,
        false,
        cs,
        hk,
        cs,
        (cs * cs) as i64,
        (cs * hk) as i64,
        (cs * hk) as i64,
        bnc,
        1.0,
        0.0,
    )?;

    // ── Главный цикл по чанкам (state-зависимость).
    for ci in 0..nc {
        let off_hk = ci * cs * hk;
        let off_hv = ci * cs * hv;

        // 9.1 v_prime = k_cumdecay[:, ci] @ state. (CS,HK)@(HK,HV) per-BH.
        csk.bmm(
            stream,
            &k_cumdecay,
            off_hk,
            state,
            0,
            &mut v_prime,
            0,
            false,
            false,
            cs,
            hv,
            hk,
            (nc * cs * hk) as i64,
            (hk * hv) as i64,
            (cs * hv) as i64,
            bh,
            1.0,
            0.0,
        )?;

        // 9.2 value_proc[:, ci] -= v_prime.
        cfk.sub_chunk(stream, &mut value_proc, &v_prime, bh, nc, cs, hv, ci)?;

        // 9.3 out[:, ci] = q_scaled[:, ci] @ state.
        csk.bmm(
            stream,
            &q_scaled,
            off_hk,
            state,
            0,
            out,
            off_hv,
            false,
            false,
            cs,
            hv,
            hk,
            (nc * cs * hk) as i64,
            (hk * hv) as i64,
            (nc * cs * hv) as i64,
            bh,
            1.0,
            0.0,
        )?;

        // 9.4 attn_intra = q_n[:, ci] @ k_n[:, ci]^T.
        csk.bmm(
            stream,
            &q_n,
            off_hk,
            &k_n,
            off_hk,
            &mut attn_intra,
            0,
            false,
            true,
            cs,
            cs,
            hk,
            (nc * cs * hk) as i64,
            (nc * cs * hk) as i64,
            (cs * cs) as i64,
            bh,
            1.0,
            0.0,
        )?;

        // 9.5 attn_intra *= decay_mask[:, ci].
        cfk.mul_decay_mask_chunk(stream, &mut attn_intra, &dm, bh, nc, cs, ci)?;

        // 9.6 out[:, ci] += attn_intra @ v_new (v_new = value_proc[:, ci]).
        csk.bmm(
            stream,
            &attn_intra,
            0,
            &value_proc,
            off_hv,
            out,
            off_hv,
            false,
            false,
            cs,
            hv,
            cs,
            (cs * cs) as i64,
            (nc * cs * hv) as i64,
            (nc * cs * hv) as i64,
            bh,
            1.0,
            1.0,
        )?;

        // 9.7 state *= exp(g_cumsum[:, ci, CS-1]).
        cfk.state_decay_from_gcumsum_chunk(stream, state, &g_cumsum, bh, nc, cs, hk, hv, ci)?;

        // 9.8 k_decayed = k_n[:, ci] * exp(g_last - g_cumsum[:, ci]).
        cfk.scale_k_decayed_chunk(stream, &mut k_decayed, &k_n, &g_cumsum, bh, nc, cs, hk, ci)?;

        // 9.9 state += k_decayed^T @ v_new.
        csk.bmm(
            stream,
            &k_decayed,
            0,
            &value_proc,
            off_hv,
            state,
            0,
            true,
            false,
            hk,
            hv,
            cs,
            (cs * hk) as i64,
            (nc * cs * hv) as i64,
            (hk * hv) as i64,
            bh,
            1.0,
            1.0,
        )?;
    }

    Ok(())
}
