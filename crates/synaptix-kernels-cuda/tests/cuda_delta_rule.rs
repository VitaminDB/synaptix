
//! Bit-exact (F32-эталон) тесты для DeltaNet рекуррентного шага.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::ssm::delta_rule::DeltaRuleKernels;

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

#[allow(clippy::too_many_arguments)]
fn cpu_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    state: &mut [f32],
    b: usize,
    h: usize,
    hk: usize,
    hv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * h * hv];
    for bi in 0..b {
        for hi in 0..h {
            let base_qk = (bi * h + hi) * hk;
            let base_v = (bi * h + hi) * hv;
            let base_state = (bi * h + hi) * hk * hv;
            let beta_t = beta[bi * h + hi];
            for vi in 0..hv {
                let mut kv_mem = 0.0_f32;
                let mut st = vec![0.0_f32; hk];
                for kk in 0..hk {
                    st[kk] = state[base_state + kk * hv + vi];
                    kv_mem += st[kk] * k[base_qk + kk];
                }
                let delta = (v[base_v + vi] - kv_mem) * beta_t;
                let mut o = 0.0_f32;
                for kk in 0..hk {
                    let new_st = st[kk] + k[base_qk + kk] * delta;
                    state[base_state + kk * hv + vi] = new_st;
                    o += new_st * q[base_qk + kk];
                }
                out[base_v + vi] = o;
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn delta_rule_step_multistep() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = DeltaRuleKernels::for_context(&ctx).expect("compile");
    let (b, h, hk, hv) = (2usize, 4usize, 16usize, 16usize);
    let steps = 5usize;

    let mut cpu_state = vec![0.0_f32; b * h * hk * hv];
    let mut dev_state: CudaSlice<f32> = stream.alloc_zeros(b * h * hk * hv).unwrap();

    for t in 0..steps {
        let q = det_f32(0x100 + t as u64, b * h * hk, 0.5, 0.0);
        let k = det_f32(0x200 + t as u64, b * h * hk, 0.5, 0.0);
        let v = det_f32(0x300 + t as u64, b * h * hv, 0.5, 0.0);
        let beta = det_f32(0x500 + t as u64, b * h, 0.3, 0.5);

        let expected = cpu_step(&q, &k, &v, &beta, &mut cpu_state, b, h, hk, hv);

        let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
        let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
        let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
        let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * h * hv).unwrap();
        kern.delta_rule_step(
            &stream,
            &dq,
            &dk,
            &dv,
            &dbeta,
            &mut dev_state,
            &mut dout,
            b as u32,
            h as u32,
            hk as u32,
            hv as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
        let m = max_abs(&got, &expected);
        eprintln!("[delta_rule step {t}] max_abs={m:.6}");
        assert!(m < 1e-4, "step {t}: max_abs={m}");
    }
    let got_state: Vec<f32> = stream.clone_dtoh(&dev_state).unwrap();
    let ms = max_abs(&got_state, &cpu_state);
    eprintln!("[delta_rule] state_max_abs={ms:.6}");
    assert!(ms < 1e-4, "state max_abs={ms}");
}
