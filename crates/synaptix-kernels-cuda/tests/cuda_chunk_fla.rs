#![cfg(feature = "cuda")]

//! Unit-тесты для chunk-FLA helper-ядер против CPU-f32 эталонов.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::attention::chunk_fla::ChunkFlaKernels;

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
fn compute_chunk_attn_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = ChunkFlaKernels::for_context(&ctx).expect("compile");
    let (bh, nc, cs, hk) = (2usize, 2usize, 8usize, 4usize);
    let k_beta = det_f32(0x11, bh * nc * cs * hk, 0.5, 0.0);
    let key = det_f32(0x22, bh * nc * cs * hk, 0.5, 0.0);
    // g_cumsum монотонно убывает (decay), небольшой диапазон.
    let g = det_f32(0x33, bh * nc * cs, 0.2, -0.2);

    // CPU-эталон, повторяет ядро (включая последовательный cumprod).
    let mut attn_exp = vec![0.0_f32; bh * nc * cs * cs];
    let mut dm_exp = vec![0.0_f32; bh * nc * cs * cs];
    for b in 0..bh {
        for c in 0..nc {
            let base_kv = (b * nc + c) * cs * hk;
            let base_g = (b * nc + c) * cs;
            let base_a = (b * nc + c) * cs * cs;
            let mut attn = vec![0.0_f32; cs * cs];
            for i in 0..cs {
                for j in 0..cs {
                    let dm = if j <= i {
                        (g[base_g + i] - g[base_g + j]).exp()
                    } else {
                        0.0
                    };
                    dm_exp[base_a + i * cs + j] = dm;
                    if j < i {
                        let mut acc = 0.0_f32;
                        for d in 0..hk {
                            acc += k_beta[base_kv + i * hk + d] * key[base_kv + j * hk + d];
                        }
                        attn[i * cs + j] = -acc * dm;
                    }
                }
            }
            for row in 1..cs {
                for j in 0..row {
                    let mut acc = 0.0_f32;
                    for l in 0..row {
                        acc += attn[row * cs + l] * attn[l * cs + j];
                    }
                    attn[row * cs + j] += acc;
                }
            }
            for i in 0..cs {
                attn[i * cs + i] += 1.0;
            }
            for x in 0..cs * cs {
                attn_exp[base_a + x] = attn[x];
            }
        }
    }

    let d_kb: CudaSlice<f32> = stream.clone_htod(&k_beta).unwrap();
    let d_key: CudaSlice<f32> = stream.clone_htod(&key).unwrap();
    let d_g: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
    let mut d_attn: CudaSlice<f32> = stream.alloc_zeros(bh * nc * cs * cs).unwrap();
    let mut d_dm: CudaSlice<f32> = stream.alloc_zeros(bh * nc * cs * cs).unwrap();
    kern.compute_chunk_attn(
        &stream,
        &d_kb,
        &d_key,
        &d_g,
        &mut d_attn,
        &mut d_dm,
        bh as u32,
        nc as u32,
        cs as u32,
        hk as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let attn_got: Vec<f32> = stream.clone_dtoh(&d_attn).unwrap();
    let dm_got: Vec<f32> = stream.clone_dtoh(&d_dm).unwrap();
    let ma = max_abs(&attn_got, &attn_exp);
    let md = max_abs(&dm_got, &dm_exp);
    eprintln!("[compute_chunk_attn] attn={ma:.6} decay_mask={md:.6}");
    assert!(ma < 1e-4 && md < 1e-4, "attn={ma} dm={md}");
}

#[test]
fn scale_by_exp_diff_both_modes() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = ChunkFlaKernels::for_context(&ctx).expect("compile");
    let (bh, nc, cs, d) = (2usize, 2usize, 8usize, 4usize);
    let total_rows = bh * nc * cs;
    let input = det_f32(0x44, total_rows * d, 1.0, 0.0);
    let vec_g = det_f32(0x55, total_rows, 0.2, -0.1);
    // scalar_g индексируется как row / cs_in, cs_in = cs → (bh*nc) элементов.
    let scalar_g = det_f32(0x66, bh * nc, 0.2, -0.1);

    // mode 0: out = in * exp(vec_g[row]).
    let mut exp0 = vec![0.0_f32; total_rows * d];
    for r in 0..total_rows {
        let f = vec_g[r].exp();
        for c in 0..d {
            exp0[r * d + c] = input[r * d + c] * f;
        }
    }
    let d_in: CudaSlice<f32> = stream.clone_htod(&input).unwrap();
    let d_vg: CudaSlice<f32> = stream.clone_htod(&vec_g).unwrap();
    let mut d_out: CudaSlice<f32> = stream.alloc_zeros(total_rows * d).unwrap();
    kern.scale_by_exp_diff(
        &stream,
        &mut d_out,
        &d_in,
        None,
        &d_vg,
        total_rows as u32,
        d as u32,
        cs as u32,
        0,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got0: Vec<f32> = stream.clone_dtoh(&d_out).unwrap();
    let m0 = max_abs(&got0, &exp0);

    // mode 1: out = in * exp(scalar_g[row/cs] - vec_g[row]).
    let mut exp1 = vec![0.0_f32; total_rows * d];
    for r in 0..total_rows {
        let f = (scalar_g[r / cs] - vec_g[r]).exp();
        for c in 0..d {
            exp1[r * d + c] = input[r * d + c] * f;
        }
    }
    let d_sg: CudaSlice<f32> = stream.clone_htod(&scalar_g).unwrap();
    let mut d_out1: CudaSlice<f32> = stream.alloc_zeros(total_rows * d).unwrap();
    kern.scale_by_exp_diff(
        &stream,
        &mut d_out1,
        &d_in,
        Some(&d_sg),
        &d_vg,
        total_rows as u32,
        d as u32,
        cs as u32,
        1,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got1: Vec<f32> = stream.clone_dtoh(&d_out1).unwrap();
    let m1 = max_abs(&got1, &exp1);
    eprintln!("[scale_by_exp_diff] mode0={m0:.6} mode1={m1:.6}");
    assert!(m0 < 1e-4 && m1 < 1e-4, "mode0={m0} mode1={m1}");
}

#[test]
fn chunk_aware_elementwise() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = ChunkFlaKernels::for_context(&ctx).expect("compile");
    let (bh, nc, cs, hk, hv) = (2usize, 3usize, 8usize, 4usize, 4usize);
    let ci = 1usize;

    // sub_chunk: value_proc[:, ci] -= v_prime.
    let value_proc = det_f32(0x71, bh * nc * cs * hv, 1.0, 0.0);
    let v_prime = det_f32(0x72, bh * cs * hv, 0.5, 0.0);
    let mut vp_exp = value_proc.clone();
    for b in 0..bh {
        for t in 0..cs {
            for d in 0..hv {
                let off = ((b * nc + ci) * cs + t) * hv + d;
                let offp = (b * cs + t) * hv + d;
                vp_exp[off] -= v_prime[offp];
            }
        }
    }
    let mut d_vp: CudaSlice<f32> = stream.clone_htod(&value_proc).unwrap();
    let d_vpr: CudaSlice<f32> = stream.clone_htod(&v_prime).unwrap();
    kern.sub_chunk(
        &stream, &mut d_vp, &d_vpr, bh as u32, nc as u32, cs as u32, hv as u32, ci as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let vp_got: Vec<f32> = stream.clone_dtoh(&d_vp).unwrap();
    let m_sub = max_abs(&vp_got, &vp_exp);

    // scale_k_decayed_chunk: k_decayed = k[:, ci] * exp(g_last - g_cumsum[:, ci]).
    let k = det_f32(0x73, bh * nc * cs * hk, 1.0, 0.0);
    let g = det_f32(0x74, bh * nc * cs, 0.2, -0.1);
    let mut kd_exp = vec![0.0_f32; bh * cs * hk];
    for b in 0..bh {
        let base_g = (b * nc + ci) * cs;
        let g_last = g[base_g + cs - 1];
        for t in 0..cs {
            let f = (g_last - g[base_g + t]).exp();
            for d in 0..hk {
                let off_k = ((b * nc + ci) * cs + t) * hk + d;
                kd_exp[(b * cs + t) * hk + d] = k[off_k] * f;
            }
        }
    }
    let d_k: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let d_g: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
    let mut d_kd: CudaSlice<f32> = stream.alloc_zeros(bh * cs * hk).unwrap();
    kern.scale_k_decayed_chunk(
        &stream, &mut d_kd, &d_k, &d_g, bh as u32, nc as u32, cs as u32, hk as u32, ci as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let kd_got: Vec<f32> = stream.clone_dtoh(&d_kd).unwrap();
    let m_kd = max_abs(&kd_got, &kd_exp);

    // state_decay_from_gcumsum_chunk: state *= exp(g_cumsum[:, ci, CS-1]).
    let state = det_f32(0x75, bh * hk * hv, 1.0, 0.0);
    let mut st_exp = state.clone();
    for b in 0..bh {
        let decay = g[(b * nc + ci) * cs + cs - 1].exp();
        for x in 0..hk * hv {
            st_exp[b * hk * hv + x] *= decay;
        }
    }
    let mut d_st: CudaSlice<f32> = stream.clone_htod(&state).unwrap();
    kern.state_decay_from_gcumsum_chunk(
        &stream, &mut d_st, &d_g, bh as u32, nc as u32, cs as u32, hk as u32, hv as u32, ci as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let st_got: Vec<f32> = stream.clone_dtoh(&d_st).unwrap();
    let m_st = max_abs(&st_got, &st_exp);

    eprintln!("[chunk elementwise] sub={m_sub:.6} k_decayed={m_kd:.6} state_decay={m_st:.6}");
    assert!(m_sub == 0.0, "sub={m_sub}");
    assert!(
        m_kd < 1e-4 && m_st < 1e-4,
        "k_decayed={m_kd} state_decay={m_st}"
    );
}
