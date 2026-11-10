//! Где парный шаг «токен + драфт» расходится с декодом по одному токену:
//! послойное сравнение скрытых состояний на первой паре после префилла.
//!
//!   pair_trace <bundle.syn> <текст.txt> [префилл, 128]
//!
//! A: два шага по одному токену с трассировкой; B: те же два токена парой.
//! По каждому слою печатается относительное расхождение выхода смесителя
//! (внимание) и выхода слоя для обеих строк пары.

use std::collections::HashMap;
use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn host(t: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(t.to_device(Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        return Err("usage: pair_trace <bundle.syn> <текст> [префилл, 128]".into());
    }
    let text = String::from_utf8_lossy(&std::fs::read(&a[2])?).into_owned();
    let pre: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipeline = Qwen4ExpPipeline::load_with_precision(&PathBuf::from(&a[1]), Device::Cuda(0), PrecisionConfig::nvfp4(), Some(pre + 64))?;
    let ids = pipeline.encode(&text)?;
    let m = &pipeline.model;
    let map = |v: Vec<(String, Tensor)>| -> Result<HashMap<String, Vec<f32>>, Box<dyn std::error::Error>> {
        let mut out = HashMap::new();
        for (k, t) in v {
            if k.starts_with("mixer_out_") || k.starts_with("layer_out_") || k.starts_with("in_") {
                out.insert(k, host(&t)?);
            }
        }
        Ok(out)
    };
    let mut ca = pipeline.make_cache(pre + 16)?;
    m.forward(&ids[..pre], &mut ca)?;
    let a0 = map(m.forward_traced(&[ids[pre]], &mut ca, true)?.1)?;
    let a1 = map(m.forward_traced(&[ids[pre + 1]], &mut ca, true)?.1)?;
    drop(ca);
    let mut cb = pipeline.make_cache(pre + 16)?;
    m.forward(&ids[..pre], &mut cb)?;
    let b = map(m.forward_pair_traced(&[ids[pre], ids[pre + 1]], &mut cb)?.1)?;
    println!("слой  смеситель: строка 0 / строка 1      выход слоя: строка 0 / строка 1");
    for l in 0..pipeline.config.num_hidden_layers {
        let mut line = format!("{l:>3}  ");
        for key in [format!("mixer_out_{l}"), format!("layer_out_{l}")] {
            let (Some(x0), Some(x1), Some(y)) = (a0.get(&key), a1.get(&key), b.get(&key)) else {
                line.push_str("      —      ");
                continue;
            };
            let h = y.len() / 2;
            line.push_str(&format!("   {:.5} / {:.5}        ", rel(x0, &y[..h]), rel(x1, &y[h..])));
        }
        println!("{line}");
    }
    Ok(())
}
