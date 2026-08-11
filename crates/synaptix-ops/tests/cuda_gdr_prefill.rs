
//! End-to-end: `Tensor::gated_delta_rule_prefill` (CUDA: copy-bridge → chunk
//! orchestrator → writeback) vs host `gated_delta_net_recurrent`. Покрывает
//! Backend-плумбинг и in-place обновление `ssm_state`. Чистую математику
//! chunk-vs-recurrent проверяет `synaptix-kernels-cuda/tests/cuda_chunk_scan.rs`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::linear::gated_delta_net_recurrent;

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 * scale - scale + offset
        })
        .collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn prefill_op_matches_recurrent() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let (bh, hk, hv, cs, nc) = (2usize, 16usize, 16usize, 8usize, 3usize);
    let t = nc * cs;
    let q_scale = (hk as f32).powf(-0.5);
    let q = det_f32(0x10, bh * t * hk, 1.0, 0.0);
    let k = det_f32(0x20, bh * t * hk, 1.0, 0.0);
    let v = det_f32(0x30, bh * t * hv, 1.0, 0.0);
    let g = det_f32(0x40, bh * t, 0.15, -0.15);
    let beta = det_f32(0x50, bh * t, 0.3, 0.5);

    let q_t = Tensor::from_vec(q.clone(), vec![bh, t, hk], dev).unwrap();
    let k_t = Tensor::from_vec(k.clone(), vec![bh, t, hk], dev).unwrap();
    let v_t = Tensor::from_vec(v.clone(), vec![bh, t, hv], dev).unwrap();
    let g_t = Tensor::from_vec(g.clone(), vec![bh, t], dev).unwrap();
    let beta_t = Tensor::from_vec(beta.clone(), vec![bh, t], dev).unwrap();
    let mut ss = Tensor::from_vec(vec![0.0f32; bh * hk * hv], vec![bh, hk, hv], dev).unwrap();

    let out = q_t
        .gated_delta_rule_prefill(&k_t, &v_t, &g_t, &beta_t, &mut ss, q_scale, cs)
        .expect("gated_delta_rule_prefill");
    let out_dev = host(&out);
    let state_dev = host(&ss);

    // Reference: рекуррент (bh трактуется как heads, t как seq) — тот же layout.
    let mut state_ref = vec![0.0f32; bh * hk * hv];
    let out_ref =
        gated_delta_net_recurrent(&mut state_ref, &q, &k, &v, &g, &beta, bh, t, hk, hv, q_scale);

    let m_out = max_abs(&out_dev, &out_ref);
    let m_st = max_abs(&state_dev, &state_ref);
    eprintln!("[gdr_prefill op] out_max_abs={m_out:.6} state_max_abs={m_st:.6}");
    assert!(m_out < 1e-3, "out parity max_abs={m_out}");
    assert!(m_st < 1e-3, "state parity max_abs={m_st}");
}
