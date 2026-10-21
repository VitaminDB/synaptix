//! Лосс и градиенты LoRA на фиксированной упакованной последовательности —
//! для сверки с PyTorch/PEFT (`scripts/train_grad_ref.py`).
//!
//! `cargo run --profile fast-release -p synaptix-llm-train --example grad_ref -- <model.syn> <out.safetensors> [bf16|nvfp4]`

use std::collections::BTreeMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_train::lora::LoraConfig;
use synaptix_llm_train::model::{Batch, TrainModel};

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let out = &args[2];
    let quant = args.get(3).map(|s| s.as_str()).unwrap_or("bf16");
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let mut prec = PrecisionConfig::dense(if quant == "f32" { DType::F32 } else { DType::BF16 });
    if quant == "nvfp4" {
        prec.attn_w = DType::NVFP4;
        prec.mlp_w = DType::NVFP4;
    }
    let pipe = Qwen3Pipeline::load_with_precision(model_path, dev, prec, Some(4096)).expect("load");
    let cfg = LoraConfig { r: 8, alpha: 16.0, seed: 7, ..LoraConfig::default() };
    let mut model = TrainModel::new(&pipe.model, cfg).expect("train model");
    // B не нулём: иначе градиент по A равен нулю и сверять нечего.
    let zero_b = std::env::var("GRAD_REF_ZERO_B").is_ok();
    for (i, l) in model.layers_mut().enumerate().filter(|_| !zero_b) {
        let n = l.b.numel();
        let v: Vec<f32> = (0..n).map(|j| (((j * 7919 + i * 104729) % 2001) as f32 / 1000.0 - 1.0) * 0.02).collect();
        l.b = Tensor::from_vec(v, l.b.dims().to_vec(), dev).unwrap();
    }

    let text_a = "Синтаксис Rust строится вокруг владения и заимствования: у каждого значения один владелец, а ссылки проверяются при компиляции.";
    let text_b = "The quick brown fox jumps over the lazy dog while the compiler checks every borrow.";
    let ta = pipe.encode(text_a).unwrap();
    let tb = pipe.encode(text_b).unwrap();
    let mut batch = Batch::default();
    for (seg, toks) in [ta, tb].iter().enumerate() {
        let n = toks.len();
        for t in 0..n {
            batch.tokens.push(toks[t]);
            batch.labels.push(if t + 1 < n && t >= 4 { toks[t + 1] as i32 } else { -100 });
            batch.positions.push(t as u32);
            batch.seg.push(seg as i32);
        }
    }
    let n_targets = batch.n_targets();
    let stats = model.forward_backward(&batch, 1.0 / n_targets as f32).expect("step");
    let loss = stats.loss_sum / stats.n_targets as f64;
    eprintln!("T = {}, целей {}, loss = {loss:.6}", batch.len(), stats.n_targets);

    let mut blobs: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let ints = |v: &[i64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let t = batch.len();
    blobs.insert("tokens".into(), (vec![t], ints(&batch.tokens.iter().map(|&x| x as i64).collect::<Vec<_>>())));
    blobs.insert("labels".into(), (vec![t], ints(&batch.labels.iter().map(|&x| x as i64).collect::<Vec<_>>())));
    blobs.insert("positions".into(), (vec![t], ints(&batch.positions.iter().map(|&x| x as i64).collect::<Vec<_>>())));
    blobs.insert("seg".into(), (vec![t], ints(&batch.seg.iter().map(|&x| x as i64).collect::<Vec<_>>())));
    blobs.insert("loss".into(), (vec![1], vec![loss as f32]));
    for l in model.layers() {
        let m = &l.module;
        blobs.insert(format!("{m}.lora_A"), (l.a.dims().to_vec(), host(&l.a)));
        blobs.insert(format!("{m}.lora_B"), (l.b.dims().to_vec(), host(&l.b)));
        blobs.insert(format!("{m}.grad_A"), (l.a.dims().to_vec(), host(l.ga.as_ref().unwrap())));
        blobs.insert(format!("{m}.grad_B"), (l.b.dims().to_vec(), host(l.gb.as_ref().unwrap())));
    }
    let bytes: BTreeMap<String, (Vec<usize>, Vec<u8>)> = blobs
        .into_iter()
        .map(|(k, (s, v))| (k, (s, v.iter().flat_map(|x| x.to_le_bytes()).collect())))
        .collect();
    let views: Vec<_> = bytes
        .iter()
        .map(|(k, (s, b))| (k.clone(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b).unwrap()))
        .collect();
    std::fs::write(out, safetensors::serialize(views, None).unwrap()).unwrap();
    eprintln!("записано: {out}");
}
