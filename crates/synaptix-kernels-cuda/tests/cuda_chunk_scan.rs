
//! Parity: chunked gated delta rule (chunk_scan) vs рекуррентный per-step
//! (gated_delta_rule). Проверяется многочанковый путь (nc>1) — то, что эталонный
//! ai-quant parity-тест не покрывал.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::attention::chunk_fla::ChunkFlaKernels;
use synaptix_kernels_cuda::scan::chunk_scan::{chunk_gated_delta_rule, ChunkScanKernels};
use synaptix_kernels_cuda::ssm::gated_delta_rule::GatedDeltaRuleKernels;

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale + offset
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn chunked_vs_per_step_multichunk() {
    let Some((ctx, stream)) = setup() else { return };
    let cfk = ChunkFlaKernels::for_context(&ctx).expect("chunk_fla");
    let csk = ChunkScanKernels::for_context(&ctx).expect("chunk_scan");
    let gdr = GatedDeltaRuleKernels::for_context(&ctx).expect("gated_delta_rule");

    let bh = 2usize;
    let hk = 16usize;
    let hv = 16usize;
    let cs = 8usize;
    let nc = 3usize;
    let t = nc * cs; // 24 — nc>1
    let q_scale = (hk as f32).powf(-0.5);

    // Полные входы (BH, T, *) и (BH, T).
    let q = det_f32(0x10, bh * t * hk, 1.0, 0.0);
    let k = det_f32(0x20, bh * t * hk, 1.0, 0.0);
    let v = det_f32(0x30, bh * t * hv, 1.0, 0.0);
    // g < 0 (log-decay), малый диапазон → exp(g) ∈ (0,1).
    let g = det_f32(0x40, bh * t, 0.15, -0.15);
    let beta = det_f32(0x50, bh * t, 0.3, 0.5);

    // ── Chunked.
    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
    let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
    let mut state_c: CudaSlice<f32> = stream.alloc_zeros(bh * hk * hv).unwrap();
    let mut out_c: CudaSlice<f32> = stream.alloc_zeros(bh * t * hv).unwrap();
    chunk_gated_delta_rule(
        &cfk,
        &csk,
        &stream,
        &dq,
        &dk,
        &dv,
        &dg,
        &dbeta,
        &mut state_c,
        &mut out_c,
        q_scale,
        bh as u32,
        t as u32,
        hk as u32,
        hv as u32,
        cs as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let out_chunk: Vec<f32> = stream.clone_dtoh(&out_c).unwrap();
    let state_chunk: Vec<f32> = stream.clone_dtoh(&state_c).unwrap();

    // ── Recurrent (per-step, b=bh, h=1).
    let mut state_r: CudaSlice<f32> = stream.alloc_zeros(bh * hk * hv).unwrap();
    let mut out_rec = vec![0.0_f32; bh * t * hv];
    for ti in 0..t {
        // Собираем per-step срезы (BH, *).
        let mut q_step = vec![0.0_f32; bh * hk];
        let mut k_step = vec![0.0_f32; bh * hk];
        let mut v_step = vec![0.0_f32; bh * hv];
        let mut g_step = vec![0.0_f32; bh];
        let mut beta_step = vec![0.0_f32; bh];
        for b in 0..bh {
            for d in 0..hk {
                q_step[b * hk + d] = q[(b * t + ti) * hk + d];
                k_step[b * hk + d] = k[(b * t + ti) * hk + d];
            }
            for d in 0..hv {
                v_step[b * hv + d] = v[(b * t + ti) * hv + d];
            }
            g_step[b] = g[b * t + ti];
            beta_step[b] = beta[b * t + ti];
        }
        let dqs: CudaSlice<f32> = stream.clone_htod(&q_step).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&k_step).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&v_step).unwrap();
        let dgs: CudaSlice<f32> = stream.clone_htod(&g_step).unwrap();
        let dbs: CudaSlice<f32> = stream.clone_htod(&beta_step).unwrap();
        let mut dos: CudaSlice<f32> = stream.alloc_zeros(bh * hv).unwrap();
        gdr.gated_delta_rule_step(
            &stream,
            &dqs,
            &dks,
            &dvs,
            &dgs,
            &dbs,
            &mut state_r,
            &mut dos,
            q_scale,
            bh as u32,
            1,
            hk as u32,
            hv as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let o: Vec<f32> = stream.clone_dtoh(&dos).unwrap();
        for b in 0..bh {
            for d in 0..hv {
                out_rec[(b * t + ti) * hv + d] = o[b * hv + d];
            }
        }
    }
    let state_rec: Vec<f32> = stream.clone_dtoh(&state_r).unwrap();

    let m_out = max_abs(&out_chunk, &out_rec);
    let m_state = max_abs(&state_chunk, &state_rec);
    eprintln!("[chunk_scan parity nc={nc}] out_max_abs={m_out:.6} state_max_abs={m_state:.6}");
    assert!(m_out < 1e-3, "out parity max_abs={m_out}");
    assert!(m_state < 1e-3, "state parity max_abs={m_state}");
}
