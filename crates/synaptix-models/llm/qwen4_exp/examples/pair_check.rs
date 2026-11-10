//! Проверка спекулятивного шага: логиты парного прохода «токен + драфт» (и
//! отката после непринятого драфта) против обычного декода по одному токену.
//!
//!   pair_check <bundle.syn> <текст.txt> [токенов, 448] [префилл, 128]
//!
//! Три прогона по одному и тому же тексту после общего префилла:
//!   A  — по одному токену (эталон пути);
//!   B1 — парами; каждая четвёртая пара — с заведомо неверным драфтом и
//!        откатом снимка, как в декоде при несовпадении;
//!   B2 — то же ещё раз (детерминизм парного пути).
//! Печатает KL(A ‖ B1) и совпадение top-1 отдельно для первой строки пары,
//! второй строки и первой строки после отката, худшие позиции и наибольшее
//! расхождение B1 с B2 (должно быть нулевым).

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::RopePositions;
use synaptix_llm_qwen4_exp::{ModelCache, Qwen4ExpPipeline};

fn host(t: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(t.to_device(Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn log_softmax(mut v: Vec<f32>) -> Vec<f32> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = v.iter().map(|x| ((x - m) as f64).exp()).sum();
    let l = m + s.ln() as f32;
    for x in v.iter_mut() {
        *x -= l;
    }
    v
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0usize, |b, (i, x)| if *x > v[b] { i } else { b })
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    First,
    Second,
    AfterReject,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        return Err("usage: pair_check <bundle.syn> <текст> [токенов, 448] [префилл, 128]".into());
    }
    let text = String::from_utf8_lossy(&std::fs::read(&a[2])?).into_owned();
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(448);
    let pre: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipeline = Qwen4ExpPipeline::load_with_precision(&PathBuf::from(&a[1]), Device::Cuda(0), PrecisionConfig::nvfp4(), Some(n + 64))?;
    let mut ids = pipeline.encode(&text)?;
    ids.truncate(n);
    let n = ids.len();
    let vocab = pipeline.config.vocab_size as u32;
    let m = &pipeline.model;
    let prefill = |cache: &mut ModelCache| -> Result<(), Box<dyn std::error::Error>> {
        m.forward(&ids[..pre], cache)?;
        Ok(())
    };

    // A: по одному токену.
    let mut ca = pipeline.make_cache(n + 8)?;
    prefill(&mut ca)?;
    let mut la: Vec<Vec<f32>> = Vec::with_capacity(n - pre);
    for i in pre..n {
        la.push(log_softmax(host(&m.forward_last(&[ids[i]], &mut ca)?)?));
    }
    drop(ca);

    // B: парами, каждая четвёртая — с откатом.
    let run_pairs = || -> Result<(Vec<Vec<f32>>, Vec<Kind>), Box<dyn std::error::Error>> {
        let mut cb = pipeline.make_cache(n + 8)?;
        prefill(&mut cb)?;
        let mut lb: Vec<Vec<f32>> = Vec::with_capacity(n - pre);
        let mut kinds = Vec::with_capacity(n - pre);
        let row = |t: &Tensor, r: usize| -> Result<Tensor, Box<dyn std::error::Error>> { Ok(t.narrow(0, r, 1)?.contiguous()?) };
        let (mut i, mut pair, mut after_reject) = (pre, 0usize, false);
        while i < n {
            if i + 1 >= n {
                lb.push(log_softmax(host(&m.forward_last(&[ids[i]], &mut cb)?)?));
                kinds.push(Kind::First);
                break;
            }
            pair += 1;
            let reject = pair % 4 == 0;
            let draft = if reject { (ids[i + 1] + 1) % vocab } else { ids[i + 1] };
            let (hidden, _stream, snap) = m.forward_pair(&[ids[i], draft], &mut cb, RopePositions::Sequential)?;
            lb.push(log_softmax(host(&m.lm_head_forward(&row(&hidden, 0)?)?)?));
            kinds.push(if after_reject { Kind::AfterReject } else { Kind::First });
            if reject {
                cb.restore(&snap)?;
                after_reject = true;
                i += 1;
            } else {
                lb.push(log_softmax(host(&m.lm_head_forward(&row(&hidden, 1)?)?)?));
                kinds.push(Kind::Second);
                after_reject = false;
                i += 2;
            }
        }
        Ok((lb, kinds))
    };
    let (b1, kinds) = run_pairs()?;
    let (b2, _) = run_pairs()?;

    let kl = |p: &[f32], q: &[f32]| -> f64 { p.iter().zip(q).map(|(lp, lq)| (lp.exp() as f64) * ((lp - lq) as f64)).sum::<f64>().max(0.0) };
    let mut worst: Vec<(f64, usize)> = Vec::new();
    for kind in [Kind::First, Kind::Second, Kind::AfterReject] {
        let pos: Vec<usize> = (0..la.len()).filter(|i| kinds[*i] == kind).collect();
        if pos.is_empty() {
            continue;
        }
        let kls: Vec<f64> = pos.iter().map(|i| kl(&la[*i], &b1[*i])).collect();
        let agree = pos.iter().filter(|i| argmax(&la[**i]) == argmax(&b1[**i])).count();
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        let max = kls.iter().copied().fold(0f64, f64::max);
        println!("{kind:?}: позиций {}, KL(одиночный ‖ парный) сред. {mean:.6}, макс. {max:.5}, top-1 совпал в {} ({:.1} %)", pos.len(), agree, 100.0 * agree as f64 / pos.len() as f64);
        worst.extend(pos.iter().zip(&kls).map(|(i, k)| (*k, *i)));
    }
    worst.sort_by(|x, y| y.0.total_cmp(&x.0));
    let line: Vec<String> = worst.iter().take(6).map(|(k, i)| format!("{}:{:?}:{k:.4}", pre + i, kinds[*i])).collect();
    println!("худшие позиции: {}", line.join(" "));
    let det = b1.iter().zip(&b2).map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
    let det_pos = b1.iter().zip(&b2).filter(|(x, y)| x != y).count();
    println!("детерминизм парного пути: наибольшее расхождение логарифмов вероятностей {det:.6}, позиций с отличием {det_pos} из {}", b1.len());
    Ok(())
}
