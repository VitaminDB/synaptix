//! Отладка: лосс обучающего прохода против инференса synaptix на одном примере.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_train::lora::LoraConfig;
use synaptix_llm_train::model::{Batch, TrainModel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let dt = if args.get(2).map(|s| s.as_str()) == Some("f32") { DType::F32 } else { DType::BF16 };
    let pipe = Qwen3Pipeline::load_with_precision(&args[1], dev, PrecisionConfig::dense(dt), Some(4096)).unwrap();
    let toks = pipe.encode("The quick brown fox jumps over the lazy dog while the compiler checks every borrow.").unwrap();
    let n = toks.len();
    let model = TrainModel::new(&pipe.model, LoraConfig { targets: vec![], ..LoraConfig::default() }).unwrap();
    let batch = Batch {
        tokens: toks.clone(),
        labels: (0..n).map(|t| if t + 1 < n && t >= 4 { toks[t + 1] as i32 } else { -100 }).collect(),
        positions: (0..n as u32).collect(),
        seg: vec![0; n],
    };
    let s = model.eval_loss(&batch).unwrap();
    println!("train-проход: {:.6}", s.loss_sum / s.n_targets as f64);
    let ids = Tensor::from_vec(toks.clone(), vec![1, n], dev).unwrap();
    let mut kv = pipe.model.make_kv_cache(1, 4096).unwrap();
    let hidden = pipe.model.forward_trunk(&ids, &mut kv).unwrap();
    let logits = pipe.model.heads_all(&hidden).unwrap().to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap();
    let v = logits.dims()[1];
    let lg = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut sum, mut cnt) = (0f64, 0);
    for t in 4..n - 1 {
        let row = &lg[t * v..(t + 1) * v];
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let z: f64 = row.iter().map(|x| ((x - m) as f64).exp()).sum();
        sum += m as f64 + z.ln() - row[toks[t + 1] as usize] as f64;
        cnt += 1;
    }
    println!("инференс:     {:.6}", sum / cnt as f64);
    println!("токены: {:?}", &toks[..8.min(n)]);
    if let Some(dump) = args.get(3) {
        let hs = model.hidden_states(&batch).unwrap();
        let mut blobs = Vec::new();
        for (i, h) in hs.iter().enumerate() {
            let v = h.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            blobs.push((format!("h{i}"), h.dims().to_vec(), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()));
        }
        let views: Vec<_> = blobs.iter().map(|(k, s, b)| (k.clone(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b).unwrap())).collect();
        std::fs::write(dump, safetensors::serialize(views, None).unwrap()).unwrap();
    }
}
