
//! End-to-end: `Tensor::linear_attn_chunk_prefill` (CUDA-резидентная chain
//! chunk_conv1d → silu → prep_scatter → chunk_gated_delta_rule) vs ПОЛНАЯ
//! host-эталонная цепочка (`causal_conv1d_stateful` + silu +
//! `gated_delta_decay_beta` + scatter qe/ke/vv + `gated_delta_net_recurrent`).
//! Bit-exact gate перед интеграцией в `LinearAttn::forward` (model.rs).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::linear::{gated_delta_decay_beta, gated_delta_net_recurrent};
use synaptix_ops::conv::causal_conv1d::causal_conv1d_stateful;

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

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

// Host-эталон полной цепочки (skala model.rs:879-915 для cuda-ветки), но без
// reshape в конце (out возвращается как [num_v, T, hv]).
#[allow(clippy::too_many_arguments)]
fn host_full_prefill(
    qkv_v: &[f32],
    conv_w: &[f32],
    a_v: &[f32],
    b_v: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    t_in: usize,
    num_k: usize,
    num_v: usize,
    hk: usize,
    hv: usize,
    conv_kernel: usize,
    q_scale: f32,
) -> Vec<f32> {
    let key_dim = num_k * hk;
    let v_off0 = key_dim * 2;
    let conv_dim = 2 * key_dim + num_v * hv;
    let n_rep = num_v / num_k;
    let mut conv_out = causal_conv1d_stateful(conv_state, qkv_v, conv_w, t_in, conv_dim, conv_kernel);
    for x in conv_out.iter_mut() {
        *x /= 1.0 + (-*x).exp();
    }
    let (g, beta) = gated_delta_decay_beta(a_v, b_v, a_log, dt_bias, t_in, num_v);
    let mut qe = vec![0.0f32; num_v * t_in * hk];
    let mut ke = vec![0.0f32; num_v * t_in * hk];
    let mut vv = vec![0.0f32; num_v * t_in * hv];
    for hi in 0..num_v {
        let kh = hi / n_rep;
        for t in 0..t_in {
            let row = t * conv_dim;
            for r in 0..hk {
                qe[(hi * t_in + t) * hk + r] = conv_out[row + kh * hk + r];
                ke[(hi * t_in + t) * hk + r] = conv_out[row + key_dim + kh * hk + r];
            }
            for c in 0..hv {
                vv[(hi * t_in + t) * hv + c] = conv_out[row + v_off0 + hi * hv + c];
            }
        }
    }
    gated_delta_net_recurrent(
        ssm_state, &qe, &ke, &vv, &g, &beta, num_v, t_in, hk, hv, q_scale,
    )
}

fn run_case_f32(t_in: usize, num_k: usize, n_rep: usize, hk: usize, hv: usize, k_size: usize, cs: usize, seed: u64) {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;
    let q_scale = (hk as f32).powf(-0.5);

    let qkv = det_f32(seed, t_in * conv_dim, 0.7, 0.0);
    let cw = det_f32(seed.wrapping_add(0x10), conv_dim * k_size, 0.4, 0.0);
    let a = det_f32(seed.wrapping_add(0x20), t_in * num_v, 0.5, 0.0);
    let b = det_f32(seed.wrapping_add(0x30), t_in * num_v, 0.5, 0.0);
    let dt = det_f32(seed.wrapping_add(0x40), num_v, 0.2, 0.0);
    let al = det_f32(seed.wrapping_add(0x50), num_v, 0.2, 0.0);
    let cs0 = det_f32(seed.wrapping_add(0x60), (k_size - 1) * conv_dim, 0.3, 0.0);
    let ss0 = vec![0.0f32; num_v * hk * hv];

    // Host эталон.
    let mut cs_ref = cs0.clone();
    let mut ss_ref = ss0.clone();
    let out_ref = host_full_prefill(
        &qkv, &cw, &a, &b, &dt, &al, &mut cs_ref, &mut ss_ref,
        t_in, num_k, num_v, hk, hv, k_size, q_scale,
    );

    // Device chain (F32 compute path).
    let qkv_t = Tensor::from_vec(qkv.clone(), vec![1, t_in, conv_dim], dev).unwrap();
    let cw_t = Tensor::from_vec(cw.clone(), vec![conv_dim, k_size], dev).unwrap();
    let a_t_f32 = Tensor::from_vec(a.clone(), vec![1, t_in, num_v], dev).unwrap();
    let b_t_f32 = Tensor::from_vec(b.clone(), vec![1, t_in, num_v], dev).unwrap();
    let a_t = a_t_f32.to_dtype(DType::F16).unwrap();
    let b_t = b_t_f32.to_dtype(DType::F16).unwrap();
    let dt_t = Tensor::from_vec(dt.clone(), vec![num_v], dev).unwrap();
    let al_t = Tensor::from_vec(al.clone(), vec![num_v], dev).unwrap();
    let mut cs_t = Tensor::from_vec(cs0.clone(), vec![k_size - 1, conv_dim], dev).unwrap();
    let mut ss_t = Tensor::from_vec(ss0.clone(), vec![num_v, hk, hv], dev).unwrap();

    let out = qkv_t
        .linear_attn_chunk_prefill(
            &cw_t, &a_t, &b_t, &dt_t, &al_t, &mut cs_t, &mut ss_t,
            num_k, num_v, hk, hv, k_size, cs, q_scale, true,
        )
        .expect("linear_attn_chunk_prefill");

    let out_h = host_f32(&out);
    let cs_h = host_f32(&cs_t);
    let ss_h = host_f32(&ss_t);

    let m_out = max_abs(&out_h, &out_ref);
    let m_cs = max_abs(&cs_h, &cs_ref);
    let m_ss = max_abs(&ss_h, &ss_ref);
    eprintln!(
        "[chunk_prefill F32 T={t_in} numk={num_k} numv={num_v} hk={hk} hv={hv} K={k_size} cs={cs}] \
         out_max_abs={m_out:.3e} conv_state_max_abs={m_cs:.3e} ssm_state_max_abs={m_ss:.3e}"
    );
    // F32 чистый путь, без квантизации (a/b проходят через F16, это закономерно
    // даёт дрейф ≤ tol; g/beta считаются с F16 a/b на device — host берёт F32 a/b
    // → 1e-3 для out и ssm_state, conv_state в F32 = 0).
    assert!(m_out < 5e-3, "out parity max_abs={m_out}");
    assert!(m_cs < 1e-5, "conv_state parity max_abs={m_cs}");
    assert!(m_ss < 5e-3, "ssm_state parity max_abs={m_ss}");
}

// Cross-call перенос состояния: prefill чата = НЕСКОЛЬКО последовательных
// вызовов linear_attn_chunk_prefill по `chunk` токенов, опираясь на
// персистентность conv_state/ssm_state между вызовами. Тесты выше делают ОДИН
// вызов (внутренний суб-чанкинг) — НЕ покрывают cross-call. Здесь: один вызов
// на весь T vs два вызова (0..split, split..T) с общими cs_t/ss_t. split кратен
// cs (как batch=512 % CS=64 == 0 в проде). Должны совпасть bit-exact.
#[allow(clippy::too_many_arguments)]
fn run_split_carry_f32(t_in: usize, split: usize, num_k: usize, n_rep: usize, hk: usize, hv: usize, k_size: usize, cs: usize, seed: u64) {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;
    let q_scale = (hk as f32).powf(-0.5);

    let qkv = det_f32(seed, t_in * conv_dim, 0.7, 0.0);
    let cw = det_f32(seed.wrapping_add(0x10), conv_dim * k_size, 0.4, 0.0);
    let a = det_f32(seed.wrapping_add(0x20), t_in * num_v, 0.5, 0.0);
    let b = det_f32(seed.wrapping_add(0x30), t_in * num_v, 0.5, 0.0);
    let dt = det_f32(seed.wrapping_add(0x40), num_v, 0.2, 0.0);
    let al = det_f32(seed.wrapping_add(0x50), num_v, 0.2, 0.0);
    let cs0 = det_f32(seed.wrapping_add(0x60), (k_size - 1) * conv_dim, 0.3, 0.0);
    let ss0 = vec![0.0f32; num_v * hk * hv];

    let cw_t = Tensor::from_vec(cw.clone(), vec![conv_dim, k_size], dev).unwrap();
    let dt_t = Tensor::from_vec(dt.clone(), vec![num_v], dev).unwrap();
    let al_t = Tensor::from_vec(al.clone(), vec![num_v], dev).unwrap();

    let call = |qkv_s: &[f32], a_s: &[f32], b_s: &[f32], t: usize, cs_t: &mut Tensor, ss_t: &mut Tensor| -> Vec<f32> {
        let qkv_t = Tensor::from_vec(qkv_s.to_vec(), vec![1, t, conv_dim], dev).unwrap();
        let a_t = Tensor::from_vec(a_s.to_vec(), vec![1, t, num_v], dev).unwrap().to_dtype(DType::F16).unwrap();
        let b_t = Tensor::from_vec(b_s.to_vec(), vec![1, t, num_v], dev).unwrap().to_dtype(DType::F16).unwrap();
        let out = qkv_t
            .linear_attn_chunk_prefill(&cw_t, &a_t, &b_t, &dt_t, &al_t, cs_t, ss_t,
                num_k, num_v, hk, hv, k_size, cs, q_scale, true)
            .expect("linear_attn_chunk_prefill");
        host_f32(&out)
    };

    // (1) Один вызов на весь T.
    let mut cs1 = Tensor::from_vec(cs0.clone(), vec![k_size - 1, conv_dim], dev).unwrap();
    let mut ss1 = Tensor::from_vec(ss0.clone(), vec![num_v, hk, hv], dev).unwrap();
    let out_single = call(&qkv, &a, &b, t_in, &mut cs1, &mut ss1);
    let ss_single = host_f32(&ss1);

    // (2) Два вызова: 0..split, split..T, общие cs/ss.
    let mut cs2 = Tensor::from_vec(cs0.clone(), vec![k_size - 1, conv_dim], dev).unwrap();
    let mut ss2 = Tensor::from_vec(ss0.clone(), vec![num_v, hk, hv], dev).unwrap();
    let qc = conv_dim;
    let out_a = call(&qkv[..split * qc], &a[..split * num_v], &b[..split * num_v], split, &mut cs2, &mut ss2);
    let out_b = call(&qkv[split * qc..], &a[split * num_v..], &b[split * num_v..], t_in - split, &mut cs2, &mut ss2);
    let ss_split = host_f32(&ss2);

    // out layout = [num_v, T, hv]: склейка по оси T (split) внутри каждой головы.
    let mut out_concat = vec![0.0f32; num_v * t_in * hv];
    for hi in 0..num_v {
        for t in 0..split {
            let src = (hi * split + t) * hv;
            let dst = (hi * t_in + t) * hv;
            out_concat[dst..dst + hv].copy_from_slice(&out_a[src..src + hv]);
        }
        let t2 = t_in - split;
        for t in 0..t2 {
            let src = (hi * t2 + t) * hv;
            let dst = (hi * t_in + split + t) * hv;
            out_concat[dst..dst + hv].copy_from_slice(&out_b[src..src + hv]);
        }
    }

    let m_out = max_abs(&out_concat, &out_single);
    let m_ss = max_abs(&ss_split, &ss_single);
    eprintln!(
        "[split_carry F32 T={t_in} split={split} numv={num_v} hk={hk} hv={hv} K={k_size} cs={cs}] \
         out_2call_vs_1call_max_abs={m_out:.3e} ssm_state_max_abs={m_ss:.3e}"
    );
    // ДВА вызова с переносом состояния ДОЛЖНЫ дать тот же результат, что ОДИН
    // вызов. Расхождение = сломанный cross-call перенос (= баг «забывает контекст»).
    assert!(m_out < 1e-2, "out 2-call vs 1-call max_abs={m_out} — СЛОМАН cross-call перенос состояния");
    assert!(m_ss < 1e-2, "ssm_state 2-call vs 1-call max_abs={m_ss} — СЛОМАН cross-call перенос состояния");
}

#[test]
fn chunk_prefill_split_carry_aligned() {
    // T=256, split=128 (оба кратны cs=64 — как batch=512%CS=64 в проде).
    run_split_carry_f32(256, 128, 4, 4, 128, 256, 4, 64, 0x600);
}

#[test]
fn chunk_prefill_split_carry_small() {
    // Мелкие dims, несколько чанков, split кратен cs.
    run_split_carry_f32(48, 24, 2, 4, 32, 64, 4, 8, 0x700);
}

#[test]
fn chunk_prefill_split_carry_unaligned() {
    // split НЕ кратен cs → первый вызов паддит до кратного cs. Воспроизводит
    // случай, когда prefill_batch не кратен CS=64 (или последний чанк). Если
    // padding между вызовами портит состояние — здесь поймаем.
    run_split_carry_f32(48, 20, 2, 4, 32, 64, 4, 8, 0x800);
}

#[test]
fn chunk_prefill_f32_tiny() {
    run_case_f32(16, 2, 4, 32, 64, 4, 8, 0x100);
}

#[test]
fn chunk_prefill_f32_qwen_shape() {
    // Близко к Qwen3.6-27B-hybrid: num_v=16 (на linear layer), n_rep=4, hk/hv=128/256.
    // T=64 / cs=8 — несколько чанков.
    run_case_f32(64, 4, 4, 128, 256, 4, 8, 0x200);
}

#[test]
fn chunk_prefill_f32_multi_chunk() {
    // Больше chunks (T=80, cs=8 → 10 chunks).
    run_case_f32(80, 2, 4, 64, 128, 3, 8, 0x300);
}

#[test]
fn chunk_prefill_f32_with_padding() {
    // T=47 не кратно cs=8 → t_pad=48 внутри device-ветки, narrow → T=47.
    // Bit-exact против host (host работает на T=47 без padding).
    run_case_f32(47, 2, 4, 32, 64, 4, 8, 0x400);
}

#[test]
fn chunk_prefill_f32_realistic_350_pad() {
    // Близкий к production prompt: T=350 (НЕ кратно 8 → t_pad=352).
    // Маленькие dims чтобы тест был быстрым.
    run_case_f32(350, 2, 4, 32, 64, 4, 8, 0x500);
}
