//! Слитый decode-шаг GDN (`gdn_decode_step_f16`) против трёх ядер
//! (conv-update + prep + step/rms) на случайных данных: выход и состояния
//! должны совпадать с точностью f32-перестановки сумм.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale + offset
        })
        .collect()
}

fn run(fused: bool, num_k: usize, n_rep: usize, hk: usize, conv_k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dev = Device::Cuda(0);
    let hv = hk;
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let value_dim = num_v * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let f16 = |v: Vec<f32>, shape: Vec<usize>| Tensor::from_vec(v, shape, dev).unwrap().to_dtype(DType::F16).unwrap();
    let f32t = |v: Vec<f32>, shape: Vec<usize>| Tensor::from_vec(v, shape, dev).unwrap();
    let qkv = f16(det_f32(1, conv_dim, 1.0, 0.0), vec![1, conv_dim]);
    let conv_w = f16(det_f32(2, conv_dim * conv_k, 0.5, 0.0), vec![conv_dim, conv_k]);
    let a = f16(det_f32(3, num_v, 2.0, 0.0), vec![1, num_v]);
    let b = f16(det_f32(4, num_v, 2.0, 0.0), vec![1, num_v]);
    let dt_bias = f32t(det_f32(5, num_v, 0.5, 0.0), vec![num_v]);
    let a_log = f32t(det_f32(6, num_v, 0.5, -1.0), vec![num_v]);
    let z = f16(det_f32(7, value_dim, 1.0, 0.0), vec![1, value_dim]);
    let norm_w = f16(det_f32(8, hv, 0.2, 1.0), vec![hv]);
    let mut conv_state = f16(det_f32(9, (conv_k - 1) * conv_dim, 1.0, 0.0), vec![conv_k - 1, conv_dim]);
    let mut ssm_state = f32t(det_f32(10, num_v * hk * hv, 0.1, 0.0), vec![num_v, hk, hv]);
    std::env::set_var("SYN_GDN_FUSED", if fused { "1" } else { "0" });
    let out = qkv
        .linear_attn_decode_step(
            &conv_w, &a, &b, &dt_bias, &a_log, &z, &norm_w, &mut conv_state, &mut ssm_state,
            num_k, num_v, hk, hv, conv_k, 1.0 / (hk as f32).sqrt(), 1e-6, false,
        )
        .unwrap();
    (flat(&out), flat(&conv_state), flat(&ssm_state))
}

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.to_dtype(DType::F32).unwrap().reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn gdn_decode_step_fused_matches_three_kernels() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    for (num_k, n_rep) in [(16usize, 1usize), (16, 2), (16, 3), (4, 2)] {
        let (o0, c0, s0) = run(false, num_k, n_rep, 128, 4);
        let (o1, c1, s1) = run(true, num_k, n_rep, 128, 4);
        let (dout, dconv, dstate) = (max_diff(&o0, &o1), max_diff(&c0, &c1), max_diff(&s0, &s1));
        let omax = o0.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        eprintln!("num_k {num_k} n_rep {n_rep}: |out| ≤ {omax:.4}, Δout {dout:.3e}, Δconv {dconv:.3e}, Δstate {dstate:.3e}");
        assert_eq!(dconv, 0.0, "состояние свёртки: n_rep {n_rep}");
        assert!(dstate < 1e-5, "ssm-состояние: n_rep {n_rep}, Δ {dstate}");
        assert!(dout < 2e-3 * omax.max(1.0), "выход: n_rep {n_rep}, Δ {dout}");
    }
}
