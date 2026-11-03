//! Точность логитов OmniVoice: один проход cond+uncond в F32 (эталон) и в
//! BF16/F16 — с головами в той же точности и в F32 (`SYNAPTIX_OV_HEAD_F32`).
//! Сравниваются решения шага раскрытия: argmax CFG-лог-вероятностей и выбор
//! позиций по уверенности. Состояния: всё замаскировано и половина кодов
//! раскрыта (коды — из `codes.json` прогона audit_run).
//!
//! `logit_prec <omnivoice.syn> <codes.json> <text> <instruct> [bf16|f16]`

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline;
use synaptix_tts_omnivoice::text::TextFrontend;

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let s: f64 = x.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let l = s.ln() as f32 + m;
    x.iter().map(|&v| v - l).collect()
}

/// (argmax, max) CFG-лог-вероятностей по всем позициям цели [8·T].
fn decide(c: &[f32], u: &[f32], c_len: usize, t: usize, v: usize) -> Vec<(usize, f32)> {
    (0..8 * t)
        .map(|idx| {
            let (cb, j) = (idx / t, idx % t);
            let co = (cb * c_len + c_len - t + j) * v;
            let uo = (cb * t + j) * v;
            let (cl, ul) = (log_softmax(&c[co..co + v]), log_softmax(&u[uo..uo + v]));
            let mix: Vec<f32> = (0..v).map(|i| cl[i] + 2.0 * (cl[i] - ul[i])).collect();
            let mut lp = log_softmax(&mix);
            lp[1024] = f32::NEG_INFINITY;
            let (a, m) = lp.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
            (a, m - cb as f32 * 5.0)
        })
        .collect()
}

fn run(p: &OmniVoicePipeline, ids: &[i64], mask: &[u8], c_len: usize, t: usize) -> Vec<(usize, f32)> {
    let to = |x: synaptix_core::tensor::Tensor| x.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut uid = vec![0i64; 8 * t];
    for c in 0..8 {
        uid[c * t..(c + 1) * t].copy_from_slice(&ids[c * c_len + c_len - t..(c + 1) * c_len]);
    }
    let um = vec![1u8; t];
    let c = to(p.backbone().forward_padded(&[(ids, mask, c_len)], c_len).unwrap());
    let u = to(p.backbone().forward_padded(&[(&uid, &um, t)], t).unwrap());
    decide(&c, &u, c_len, t, 1025)
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    let codes: Vec<i64> = serde_json::from_value(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&a[2]).unwrap()).unwrap()["codes"].clone()).unwrap();
    let t = codes.len() / 8;
    let low = if a.get(5).map(String::as_str) == Some("f16") { DType::F16 } else { DType::BF16 };
    let bundle = synaptix_bundle::Bundle::open(std::path::Path::new(&a[1])).unwrap();
    let tf = TextFrontend::from_tokenizer_bytes(&bundle.read_file("tokenizer.json").unwrap(), 8, 1024).unwrap();
    let pr = tf.prepare_inference_inputs(&a[3], t, None, None, Some("en"), Some(&a[4]), true).unwrap();
    let ids0 = pr.input_ids.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    let mask = pr.audio_mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let c_len = mask.len();
    // Состояние 2: раскрыта каждая вторая позиция (детерминированно).
    let mut ids1 = ids0.clone();
    for c in 0..8 {
        for j in 0..t {
            if (j * 7 + c * 3) % 2 == 0 {
                ids1[c * c_len + c_len - t + j] = codes[c * t + j];
            }
        }
    }
    let states = [("всё под маской", ids0), ("половина раскрыта", ids1)];
    let refs: Vec<Vec<(usize, f32)>> = {
        let p = OmniVoicePipeline::from_syn(&a[1], Device::Cpu, DType::F32).unwrap();
        states.iter().map(|(_, ids)| run(&p, ids, &mask, c_len, t)).collect()
    };
    for head in ["0", "1"] {
        std::env::set_var("SYNAPTIX_OV_HEAD_F32", head);
        let p = OmniVoicePipeline::from_syn(&a[1], Device::Cpu, low).unwrap();
        for ((name, ids), r) in states.iter().zip(&refs) {
            let o = run(&p, ids, &mask, c_len, t);
            let masked: Vec<usize> = (0..8 * t).filter(|&i| ids[(i / t) * c_len + c_len - t + i % t] == 1024).collect();
            let agree = masked.iter().filter(|&&i| o[i].0 == r[i].0).count() as f32 / masked.len() as f32;
            let dscore = masked.iter().map(|&i| (o[i].1 - r[i].1).abs()).sum::<f32>() / masked.len() as f32;
            // выбор позиций: верхние 10 % по оценке
            let k = (masked.len() / 10).max(1);
            let top = |s: &Vec<(usize, f32)>| {
                let mut m = masked.clone();
                m.sort_by(|&x, &y| s[y].1.total_cmp(&s[x].1));
                m.truncate(k);
                m.sort();
                m
            };
            let (tr, to) = (top(r), top(&o));
            let inter = to.iter().filter(|i| tr.binary_search(i).is_ok()).count() as f32 / k as f32;
            println!("{low:?} головы F32={head} | {name}: совпадение argmax {:.1} %, |Δ оценки| {dscore:.4}, совпадение верхних 10 % позиций {:.1} %", agree * 100.0, inter * 100.0);
        }
    }
}
