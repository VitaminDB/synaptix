#![cfg(feature = "cuda")]

use std::hint::black_box;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::{Linear, Module};
use synaptix_ops::activation::gelu_exact;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

const D: usize = 1280;
const NH: usize = 20;
const HD: usize = 64;
const FFN: usize = 5120;
const VOCAB: usize = 51866;
const CROSS_LEN: usize = 1500;
const CACHE_LEN: usize = 256;
const LAYERS: usize = 4;

fn dev_f16(shape: Vec<usize>, device: Device) -> Tensor {
    Tensor::randn(shape, Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(device)
        .unwrap()
}

fn lin(out: usize, inf: usize, bias: bool, device: Device) -> Linear {
    let b = if bias { Some(dev_f16(vec![out], device)) } else { None };
    Linear::new(dev_f16(vec![out, inf], device), b).unwrap()
}

fn split_heads(x: &Tensor) -> Tensor {
    let d = x.dims();
    let (b, s) = (d[0], d[1]);
    x.reshape(vec![b, s, NH, HD]).unwrap().permute(vec![0, 2, 1, 3]).unwrap().contiguous().unwrap()
}

fn merge_heads(x: &Tensor) -> Tensor {
    let d = x.dims();
    let (b, h, s, dh) = (d[0], d[1], d[2], d[3]);
    x.permute(vec![0, 2, 1, 3]).unwrap().contiguous().unwrap().reshape(vec![b, s, h * dh]).unwrap()
}

fn sync() {
    synaptix_core::device::cuda::synchronize(0).unwrap();
}

fn bench(name: &str, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    sync();
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    sync();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {name:<28} {ms:8.4} ms");
    ms
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let _ng = synaptix_core::grad::NoGradGuard::new();
    let device = Device::Cuda(0);
    synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let iters: usize = std::env::var("SYN_PROF_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let scale = (HD as f32).powf(-0.5);

    let h = dev_f16(vec![1, 1, D], device);
    let ln_w = dev_f16(vec![D], device);
    let ln_b = dev_f16(vec![D], device);

    let q_lin = lin(D, D, true, device);
    let k_lin = lin(D, D, false, device);
    let v_lin = lin(D, D, true, device);
    let out_lin = lin(D, D, true, device);
    let fc1 = lin(FFN, D, true, device);
    let fc2 = lin(D, FFN, true, device);
    let lm_head = lin(VOCAB, D, false, device);

    let cross_k = dev_f16(vec![1, NH, CROSS_LEN, HD], device);
    let cross_v = dev_f16(vec![1, NH, CROSS_LEN, HD], device);
    let cross_k_f32 = cross_k.to_dtype(DType::F32).unwrap();
    let cross_v_f32 = cross_v.to_dtype(DType::F32).unwrap();

    let mut k_buf = Tensor::zeros(vec![1, NH, CACHE_LEN, HD], DType::F16, device).unwrap();
    let mut v_buf = Tensor::zeros(vec![1, NH, CACHE_LEN, HD], DType::F16, device).unwrap();
    let pos_dev = Tensor::from_vec(vec![(CACHE_LEN - 1) as u32], vec![1usize], device).unwrap();
    let tcache_dev = Tensor::from_vec(vec![CACHE_LEN as u32], vec![1usize], device).unwrap();

    let embed_tok = dev_f16(vec![VOCAB, D], device);
    let embed_pos = dev_f16(vec![448, D], device);
    let tok_ids = Tensor::from_vec(vec![1234u32], vec![1usize], device).unwrap();
    let pos_id = Tensor::from_vec(vec![10u32], vec![1usize], device).unwrap();

    println!("=== Whisper decode-step per-op profile (CUDA F16, cache_len={CACHE_LEN}) ===");
    println!("-- эмбеддинг (раз/шаг) --");
    let t_embtok = bench("embed_gather token", iters, || {
        black_box(embed_tok.embed_gather(&tok_ids).unwrap());
    });
    let t_embpos = bench("embed_gather pos", iters, || {
        black_box(embed_pos.embed_gather(&pos_id).unwrap());
    });

    println!("-- per-слой (×{LAYERS}) --");
    let t_ln = bench("layer_norm", iters, || {
        black_box(layer_norm(&h, Some(&ln_w), Some(&ln_b), 1e-5).unwrap());
    });
    let t_proj = bench("linear 1280->1280 (q/k/v/out)", iters, || {
        black_box(q_lin.forward(&h).unwrap());
    });

    let hn = layer_norm(&h, Some(&ln_w), Some(&ln_b), 1e-5).unwrap();
    let t_self = bench("self-attn (proj+append+flash+out)", iters, || {
        let q = split_heads(&q_lin.forward(&hn).unwrap());
        let k = split_heads(&k_lin.forward(&hn).unwrap());
        let v = split_heads(&v_lin.forward(&hn).unwrap());
        k_buf.kv_append_dev(&k, &pos_dev).unwrap();
        v_buf.kv_append_dev(&v, &pos_dev).unwrap();
        let a = q.flash_attention_dev(&k_buf, &v_buf, &tcache_dev, scale, true).unwrap();
        black_box(out_lin.forward(&merge_heads(&a)).unwrap());
    });
    let t_flash = bench("  └ flash_attention_dev only", iters, || {
        let q = split_heads(&q_lin.forward(&hn).unwrap());
        black_box(q.flash_attention_dev(&k_buf, &v_buf, &tcache_dev, scale, true).unwrap());
    });

    let q_cross = split_heads(&q_lin.forward(&hn).unwrap());
    let t_cross = bench("cross-attn scaled_dot (F16 in)", iters, || {
        let a = scaled_dot_attention(&q_cross, &cross_k, &cross_v, scale, None).unwrap();
        black_box(out_lin.forward(&merge_heads(&a)).unwrap());
    });
    let t_cross_upcast = bench("  └ cross K/V upcast F16->F32", iters, || {
        black_box(cross_k.to_dtype(DType::F32).unwrap());
        black_box(cross_v.to_dtype(DType::F32).unwrap());
    });
    let q_cross_f32 = q_cross.to_dtype(DType::F32).unwrap();
    let t_cross_pre = bench("  └ scaled_dot pre-F32 K/V", iters, || {
        let a = scaled_dot_attention(&q_cross_f32, &cross_k_f32, &cross_v_f32, scale, None).unwrap();
        black_box(merge_heads(&a));
    });
    let cross_len_dev = Tensor::from_vec(vec![CROSS_LEN as u32], vec![1usize], device).unwrap();
    let _t_cross_flash = bench("cross-attn FLASH F16 (новый путь)", iters, || {
        let q = split_heads(&q_lin.forward(&hn).unwrap());
        let a = q.flash_attention_dev(&cross_k, &cross_v, &cross_len_dev, scale, false).unwrap();
        black_box(out_lin.forward(&merge_heads(&a)).unwrap());
    });

    let t_ffn = bench("ffn (fc1+gelu+fc2)", iters, || {
        let m = fc1.forward(&hn).unwrap();
        let m = gelu_exact(&m).unwrap();
        black_box(fc2.forward(&m).unwrap());
    });

    println!("-- финал (раз/шаг) --");
    let last = dev_f16(vec![1, D], device);
    let t_lm = bench("lm_head 1280->51866", iters, || {
        black_box(lm_head.forward(&last).unwrap());
    });

    println!("-- ЭНКОДЕР self-attn (раз/сегмент, ×32 слоя) --");
    let enc_q = dev_f16(vec![1, NH, CROSS_LEN, HD], device);
    let enc_k = dev_f16(vec![1, NH, CROSS_LEN, HD], device);
    let enc_v = dev_f16(vec![1, NH, CROSS_LEN, HD], device);
    let enc_len = Tensor::from_vec(vec![CROSS_LEN as u32], vec![1usize], device).unwrap();
    let _t_enc_sdpa = bench("enc scaled_dot 1500x1500 F32", iters.min(60), || {
        black_box(scaled_dot_attention(&enc_q, &enc_k, &enc_v, scale, None).unwrap());
    });
    let _ = &enc_len;
    let _t_enc_flash = bench("enc FA-4 HD64 (Tq=1500, non-causal)", iters.min(60), || {
        black_box(enc_q.flash_attention(&enc_k, &enc_v, scale, false).unwrap());
    });

    let per_layer = t_ln * 3.0 + t_proj * 4.0 + t_self + t_cross + t_ffn;
    let total = (t_embtok + t_embpos) + per_layer * LAYERS as f64 + t_ln + t_lm;
    println!("\n=== СВОДКА (модель: {LAYERS} слоя) ===");
    println!("  per-layer (ln×3+proj×4+self+cross+ffn) = {per_layer:.4} ms");
    println!("  итого/шаг (оценка)                      = {total:.4} ms");
    println!("  cross×{LAYERS} (с upcast)                     = {:.4} ms", t_cross * LAYERS as f64);
    println!("    из них upcast×{LAYERS}                      = {:.4} ms", t_cross_upcast * LAYERS as f64);
    println!("  ffn×{LAYERS}                                  = {:.4} ms", t_ffn * LAYERS as f64);
    println!("  self×{LAYERS} (flash-only {:.4})            = {:.4} ms", t_flash * LAYERS as f64, t_self * LAYERS as f64);
    println!("  lm_head                                 = {t_lm:.4} ms");
    let _ = t_cross_pre;
}
