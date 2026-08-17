//! Mamba2 State-Space Duality (SSD) — chunked-form (multi-kernel pipeline).
//!
//! Stage 2 chunked-form Mamba2 SSD по плану
//! `plan/mamba2_chunked_stage2_handover.md`. Реализован через свой
//! `mma.sync.m16n8k16.f32.bf16.bf16.f32` WMMA bmm
//! ([`Mamba2BmmKernels`](super::mamba2_bmm)) + 11 light-weight helper kernels
//! ([`Mamba2ChunkedHelpersKernels`](super::mamba2_chunked_helpers)).
//!
//! Pipeline (упрощённо):
//!  1. `alpha_cum[t, bh, j] = Σ_{k≤j} (A[h] * dt[b, t·Q+k, h])`.
//!  2. Permute `c_in`/`b_in` `(B,L,H,N) → (T,BH,Q,N)` bf16;
//!     compute `dt_x = dt * x` в layout `(T,BH,Q,P)` bf16.
//!  3. Transpose `B_QN → B_NQ` `(N,Q)`, `dt_x_QP → dt_x_PQ` `(P,Q)`.
//!  4. `A_intra = C_QN @ B_QN^T` (bmm с `M=Q, K=N, N=Q`).
//!  5. `A_decayed = A_intra * exp(α_i - α_j) * [j≤i]` (bf16).
//!  6. `Y_intra = A_decayed @ dt_x_PQ` (bmm с `M=Q, K=Q, N=P`).
//!  7. Chunk-loop `t ∈ [0, T)`:
//!      - cast `state_PN_f32 → state_PN_bf16`,
//!      - `c_scaled = C_QN[t] * exp(α_cum[t])` (bf16),
//!      - `Y_off = c_scaled @ state_PN_bf16^T` (bmm `M=Q, K=N, N=P`),
//!      - `Y_intra[t] += Y_off`,
//!      - `dt_x_scaled = dt_x_PQ[t] * exp(α_end - α_cum[t,q])` (bf16),
//!      - `state_upd = dt_x_scaled @ B_NQ[t]^T` (bmm `M=P, K=Q, N=N_state`),
//!      - `state_PN *= exp(α_end)`,
//!      - `state_PN += state_upd`.
//!  8. `y_out = unpermute(Y_intra) + (has_d ? D * x : 0)`.
//!
//! Layout convention: chunked workspace всегда `(T, BH, *)` row-major —
//! chunk-axis самый внешний, чтобы per-chunk slice одной операцией
//! `buf[c*per_chunk..(c+1)*per_chunk]` давал contiguous view.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};
use half::bf16;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use super::mamba2_bmm::Mamba2BmmKernels;
use super::mamba2_chunked_helpers::Mamba2ChunkedHelpersKernels;
use crate::wsalloc::WsAlloc;

pub struct Mamba2SsdChunkedKernels {
    bmm: Arc<Mamba2BmmKernels>,
    helpers: Arc<Mamba2ChunkedHelpersKernels>,
}

impl Mamba2SsdChunkedKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let bmm = Mamba2BmmKernels::for_context(ctx)?;
        let helpers = Mamba2ChunkedHelpersKernels::for_context(ctx)?;
        Ok(Arc::new(Self { bmm, helpers }))
    }

    /// Mamba2 SSD chunked-form forward.
    ///
    /// Параметры аналогичны
    /// [`super::mamba2_ssd::Mamba2SsdKernels::ssd`] плюс `q` — размер chunk.
    /// Требования:
    ///  - `l % q == 0` (число chunks `T = l / q`);
    ///  - `q % 16 == 0` (требование bmm m16n8k16);
    ///  - `n % 16 == 0` (K = n в bmm A_intra), `n % 8 == 0` (n_out в state_update);
    ///  - `p % 16 == 0` (M в bmm state_update), `p % 8 == 0` (n_out в Y_intra/Y_off).
    ///
    /// Для типовых Mamba2 shape'ов (Q=16/32/64, P=64, N=128) выполнено.
    #[allow(clippy::too_many_arguments)]
    pub fn ssd<T: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<T>,
        dt: &CudaSlice<T>,
        a: &CudaSlice<T>,
        b_in: &CudaSlice<T>,
        c_in: &CudaSlice<T>,
        d_skip: Option<&CudaSlice<T>>,
        y: &mut CudaSlice<T>,
        b: u32,
        l: u32,
        h: u32,
        p: u32,
        n: u32,
        q: u32,
        dtype: DType,
    ) -> Result<()> {
        // ── Проверки ──
        if l % q != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd_chunked: L={l} не делится на Q={q}"
            )));
        }
        if q % 16 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd_chunked: Q={q} должно быть кратно 16 (bmm m16n8k16)"
            )));
        }
        if n % 16 != 0 || n % 8 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd_chunked: N={n} должно быть кратно 16"
            )));
        }
        if p % 16 != 0 || p % 8 != 0 {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd_chunked: P={p} должно быть кратно 16"
            )));
        }
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(SynaptixError::Cuda(format!(
                "mamba2_ssd_chunked: unsupported dtype {dtype:?}"
            )));
        }

        let t = l / q;
        let bh = b * h;
        let bht = bh * t;

        // ── Workspace alloc ──
        let alpha_cum_n = (bht * q) as usize;
        let qn_per_chunk_per_bh = (q * n) as usize;
        let pq_per_chunk_per_bh = (p * q) as usize;
        let qq_per_chunk_per_bh = (q * q) as usize;
        let qp_per_chunk_per_bh = (q * p) as usize;
        let nq_per_chunk_per_bh = (n * q) as usize;
        let bh_qn = (bh as usize) * qn_per_chunk_per_bh;
        let bh_pq = (bh as usize) * pq_per_chunk_per_bh;
        let bh_pn = (bh as usize) * (p as usize) * (n as usize);
        let bh_qp = (bh as usize) * qp_per_chunk_per_bh;
        let bh_qn_total = (t as usize) * bh_qn;
        let bh_pq_total = (t as usize) * bh_pq;
        let bh_nq_total = (t as usize) * (bh as usize) * nq_per_chunk_per_bh;
        let bh_qq_total = (t as usize) * (bh as usize) * qq_per_chunk_per_bh;
        let bh_qp_total = (t as usize) * bh_qp;

        let mut alpha_cum = stream.ws_alloc_zeros::<f32>(alpha_cum_n).map_err(cuda_err)?;

        let mut c_qn = stream.ws_alloc_zeros::<bf16>(bh_qn_total).map_err(cuda_err)?;
        let mut b_qn = stream.ws_alloc_zeros::<bf16>(bh_qn_total).map_err(cuda_err)?;
        let mut dt_x_qp = stream.ws_alloc_zeros::<bf16>(bh_qp_total).map_err(cuda_err)?;
        let mut b_nq = stream.ws_alloc_zeros::<bf16>(bh_nq_total).map_err(cuda_err)?;
        let mut dt_x_pq = stream.ws_alloc_zeros::<bf16>(bh_pq_total).map_err(cuda_err)?;

        let mut a_intra = stream.ws_alloc_zeros::<f32>(bh_qq_total).map_err(cuda_err)?;
        let mut a_decayed = stream.ws_alloc_zeros::<bf16>(bh_qq_total).map_err(cuda_err)?;
        let mut y_intra = stream.ws_alloc_zeros::<f32>(bh_qp_total).map_err(cuda_err)?;

        let mut state_pn = stream.ws_alloc_zeros::<f32>(bh_pn).map_err(cuda_err)?;
        let mut state_pn_bf16 = stream.ws_alloc_zeros::<bf16>(bh_pn).map_err(cuda_err)?;
        let mut c_scaled = stream.ws_alloc_zeros::<bf16>(bh_qn).map_err(cuda_err)?;
        let mut dt_x_scaled_pq = stream.ws_alloc_zeros::<bf16>(bh_pq).map_err(cuda_err)?;
        let mut y_off_chunk = stream.ws_alloc_zeros::<f32>(bh_qp).map_err(cuda_err)?;
        let mut state_upd_chunk = stream.ws_alloc_zeros::<f32>(bh_pn).map_err(cuda_err)?;

        // ── 1. alpha_cum ──
        self.helpers
            .alpha_cum(stream, dt, a, &mut alpha_cum, b, h, t, q, dtype)?;

        // ── 2. permute c_in, b_in; compute dt_x ──
        self.helpers
            .permute_blhx_to_bhtqx(stream, c_in, &mut c_qn, b, l, h, n, q, dtype)?;
        self.helpers
            .permute_blhx_to_bhtqx(stream, b_in, &mut b_qn, b, l, h, n, q, dtype)?;
        self.helpers
            .compute_dt_x(stream, dt, x, &mut dt_x_qp, b, l, h, p, q, dtype)?;

        // ── 3. transposes ──
        // B_QN (BHT, Q, N) → B_NQ (BHT, N, Q).
        self.helpers
            .transpose_bf16(stream, &b_qn, &mut b_nq, bht, q, n)?;
        // dt_x_QP (BHT, Q, P) → dt_x_PQ (BHT, P, Q).
        self.helpers
            .transpose_bf16(stream, &dt_x_qp, &mut dt_x_pq, bht, q, p)?;

        // ── 4. bmm A_intra: M=Q, K=N, N_out=Q ──
        // A=C_QN (BHT, Q, N) row, B=B_QN (BHT, Q, N) row (= B-operand (N_out=Q, K=N)).
        self.bmm
            .bmm(stream, &c_qn, &b_qn, &mut a_intra, q, q, n, bht)?;

        // ── 5. apply_decay_mask: A_decayed[i,j] = A_intra[i,j] * exp(α_i-α_j) * [j≤i] ──
        self.helpers
            .apply_decay_mask(stream, &a_intra, &alpha_cum, &mut a_decayed, bht, q)?;

        // ── 6. bmm Y_intra: M=Q, K=Q, N_out=P ──
        // A=A_decayed (BHT, Q, Q), B=dt_x_PQ (BHT, P, Q) (= B-operand (P, Q)).
        self.bmm
            .bmm(stream, &a_decayed, &dt_x_pq, &mut y_intra, q, p, q, bht)?;

        // ── 7. chunk loop ──
        for chunk in 0..t {
            // a. state_PN_f32 → state_PN_bf16 (no transpose; layout (BH, P, N)).
            self.helpers.state_cast_f32_to_bf16(
                stream,
                &state_pn,
                &mut state_pn_bf16,
                bh_pn as u64,
            )?;

            // b. c_scaled[bh, q, n] = C_QN[chunk, bh, q, n] * exp(α_cum[chunk, bh, q]).
            //    Layout (T, BH, Q, N): chunk slice is contiguous bh_qn elements
            //    starting at `chunk * bh_qn`.
            let off_qn = (chunk as usize) * bh_qn;
            let off_aq = (chunk as usize) * (bh as usize) * (q as usize);
            let c_qn_chunk = c_qn.slice(off_qn..off_qn + bh_qn);
            let alpha_chunk = alpha_cum.slice(off_aq..off_aq + (bh as usize) * (q as usize));
            {
                let mut c_scaled_v = c_scaled.slice_mut(..);
                self.helpers.col_broadcast_exp_mul_view(
                    stream,
                    &c_qn_chunk,
                    &alpha_chunk,
                    &mut c_scaled_v,
                    bh,
                    q,
                    n,
                    false,
                )?;
            }

            // c. bmm Y_off: c_scaled (BH, Q, N) @ state_PN_bf16 (BH, P, N) as B-operand.
            //    M=Q, K=N, N_out=P, batch=BH.
            {
                let c_scaled_v = c_scaled.slice(..);
                let state_bf16_v = state_pn_bf16.slice(..);
                let mut y_off_v = y_off_chunk.slice_mut(..);
                self.bmm.bmm_view(
                    stream,
                    &c_scaled_v,
                    &state_bf16_v,
                    &mut y_off_v,
                    q,
                    p,
                    n,
                    bh,
                )?;
            }

            // d. Y_intra[chunk] += Y_off_chunk.
            self.helpers
                .add_yoff_chunk(stream, &mut y_intra, &y_off_chunk, bh, t, q, p, chunk)?;

            // e. dt_x_scaled_PQ[bh, p, q] = dt_x_PQ[chunk, bh, p, q] * exp(α_end - α_cum[chunk, bh, q]).
            //    Layout dt_x_PQ (T, BH, P, Q): chunk slice = bh_pq starting at chunk*bh_pq.
            let off_pq = (chunk as usize) * bh_pq;
            let dt_x_pq_chunk = dt_x_pq.slice(off_pq..off_pq + bh_pq);
            let alpha_chunk2 = alpha_cum.slice(off_aq..off_aq + (bh as usize) * (q as usize));
            {
                let mut scaled_v = dt_x_scaled_pq.slice_mut(..);
                self.helpers.row_broadcast_exp_mul_view(
                    stream,
                    &dt_x_pq_chunk,
                    &alpha_chunk2,
                    &mut scaled_v,
                    bh,
                    p,
                    q,
                    q,
                    true,
                )?;
            }

            // f. bmm state_update: dt_x_scaled_PQ (BH, P, Q) @ B_NQ[chunk] (BH, N, Q) as B-operand.
            //    M=P, K=Q, N_out=N, batch=BH.
            let off_nq = (chunk as usize) * (bh as usize) * nq_per_chunk_per_bh;
            let b_nq_chunk = b_nq.slice(off_nq..off_nq + (bh as usize) * nq_per_chunk_per_bh);
            {
                let scaled_v = dt_x_scaled_pq.slice(..);
                let mut state_upd_v = state_upd_chunk.slice_mut(..);
                self.bmm.bmm_view(
                    stream,
                    &scaled_v,
                    &b_nq_chunk,
                    &mut state_upd_v,
                    p,
                    n,
                    q,
                    bh,
                )?;
            }

            // g. state_PN *= exp(α_end[chunk]).
            self.helpers.state_linear_decay(
                stream,
                &mut state_pn,
                &alpha_cum,
                bh,
                p,
                n,
                t,
                q,
                chunk,
            )?;

            // h. state_PN += state_upd_chunk.
            self.helpers
                .add_inplace_f32(stream, &mut state_pn, &state_upd_chunk, bh_pn as u64)?;
        }

        // ── 8. post: unpermute Y_intra + (has_d ? D*x : 0) → y_out ──
        self.helpers
            .post(stream, &y_intra, x, d_skip, y, b, l, h, p, q, dtype)?;

        // Силcuanced (workspace-allocations not held after this point) — explicit drop
        // to keep memory ownership clear.
        drop((
            alpha_cum,
            c_qn,
            b_qn,
            dt_x_qp,
            b_nq,
            dt_x_pq,
            a_intra,
            a_decayed,
            y_intra,
            state_pn,
            state_pn_bf16,
            c_scaled,
            dt_x_scaled_pq,
            y_off_chunk,
            state_upd_chunk,
        ));
        Ok(())
    }
}

fn cuda_err(e: cudarc::driver::DriverError) -> SynaptixError {
    SynaptixError::Cuda(format!("mamba2_ssd_chunked alloc/launch: {e:?}"))
}
