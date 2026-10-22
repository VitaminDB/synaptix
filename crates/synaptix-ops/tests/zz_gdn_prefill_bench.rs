//! Бенч GDN-префилла (`linear_attn_chunk_prefill`) на формах Qwen3.8-27B:
//! кусок T токенов, 16 K-голов × 3 (48 V-голов), 128×128, свёртка 4, CS=64.
//! Запуск: cargo test -p synaptix-ops --profile fast-release
//!   --test zz_gdn_prefill_bench -- --ignored --nocapture   (BENCH_T — длина куска)

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((x >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
#[ignore]
fn bench_gdn_prefill() {
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    let t_in = std::env::var("BENCH_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2048usize);
    let (num_k, n_rep, hk, hv, k_size, cs) = (16usize, 3usize, 128usize, 128usize, 4usize, 64usize);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;
    let dt = DType::BF16;
    let qkv = Tensor::from_vec(det(1, t_in * conv_dim, 0.7), vec![1, t_in, conv_dim], dev).unwrap().to_dtype(dt).unwrap();
    let cw = Tensor::from_vec(det(2, conv_dim * k_size, 0.5), vec![conv_dim, k_size], dev).unwrap().to_dtype(dt).unwrap();
    let a = Tensor::from_vec(det(3, t_in * num_v, 0.5), vec![1, t_in, num_v], dev).unwrap().to_dtype(DType::F16).unwrap();
    let b = Tensor::from_vec(det(4, t_in * num_v, 0.5), vec![1, t_in, num_v], dev).unwrap().to_dtype(DType::F16).unwrap();
    let dtb = Tensor::from_vec(det(5, num_v, 0.2), vec![num_v], dev).unwrap();
    let al = Tensor::from_vec(det(6, num_v, 0.2), vec![num_v], dev).unwrap();
    let mut cst = Tensor::zeros(vec![k_size - 1, conv_dim], dt, dev).unwrap();
    let mut sst = Tensor::zeros(vec![num_v, hk, hv], DType::F32, dev).unwrap();
    let run = |cst: &mut Tensor, sst: &mut Tensor| {
        qkv.linear_attn_chunk_prefill(&cw, &a, &b, &dtb, &al, cst, sst, num_k, num_v, hk, hv, k_size, cs, 1.0 / (hk as f32).sqrt(), true)
            .expect("linear_attn_chunk_prefill")
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    for _ in 0..3 {
        let _ = run(&mut cst, &mut sst);
    }
    stream.synchronize().unwrap();
    let it = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..it {
        let _ = run(&mut cst, &mut sst);
    }
    stream.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
    eprintln!("[gdn prefill] T={t_in}: {ms:.3} мс на слой, ×48 слоёв = {:.1} мс на кусок", ms * 48.0);
}
