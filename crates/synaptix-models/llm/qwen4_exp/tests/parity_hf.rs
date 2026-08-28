use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_qwen4_exp::{Qwen4ExpModel, Qwen4ExpWeights};

fn ref_dir() -> Option<PathBuf> {
    let p = std::env::var("SYN_QWEN4EXP_REF").ok()?;
    let p = PathBuf::from(p);
    p.join("reference.safetensors").exists().then_some(p)
}

fn read_vec(loader: &SafetensorsLoader, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = loader.load_to(name, Device::Cpu, DType::F32).expect(name);
    let dims = t.dims().to_vec();
    (t.flatten_all().unwrap().to_vec1::<f32>().unwrap(), dims)
}

fn diff(got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq += (d as f64) * (d as f64);
        ref_sq += (*w as f64) * (*w as f64);
    }
    (max_abs, (sum_sq / ref_sq.max(1e-12)).sqrt() as f32)
}

fn compare_rows(tag: &str, got: &[f32], want: &[f32], rows: usize, skip: &[usize], tol: f32) {
    assert_eq!(got.len(), want.len(), "{tag}: длина {} vs {}", got.len(), want.len());
    let width = got.len() / rows;
    let mut worst = 0f32;
    let mut checked = 0usize;
    for row in 0..rows {
        if skip.contains(&row) {
            continue;
        }
        let (max_abs, _) = diff(&got[row * width..(row + 1) * width], &want[row * width..(row + 1) * width]);
        if max_abs > worst {
            worst = max_abs;
        }
        checked += 1;
    }
    eprintln!("{tag}: строк {checked}/{rows}, max_abs={worst:.3e}");
    assert!(worst < tol, "{tag}: max_abs={worst:.3e} > {tol:.3e}");
}

fn compare(tag: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{tag}: длина {} vs {}", got.len(), want.len());
    let (max_abs, rel) = diff(got, want);
    eprintln!("{tag}: max_abs={max_abs:.3e} rel_l2={rel:.3e}");
    assert!(max_abs < tol, "{tag}: max_abs={max_abs:.3e} > {tol:.3e}");
}

fn tokens_of(dir: &PathBuf) -> Vec<u32> {
    let raw = std::fs::read(dir.join("tokens.json")).expect("tokens.json");
    let v: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    v["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn build(dir: &PathBuf) -> (Qwen4ExpWeights, Qwen4ExpModel) {
    synaptix_kernels_cpu::ensure_registered();
    let weights = Qwen4ExpWeights::open(dir, Device::Cpu, DType::F32).expect("open");
    let cfg = weights.config.clone();
    let model = Qwen4ExpModel::build(
        &cfg,
        &weights,
        Device::Cpu,
        DType::F32,
        DType::F32,
        cfg.max_position_embeddings.min(4096),
        &|layer| weights.ngram_rows(layer),
    )
    .expect("build");
    (weights, model)
}

#[test]
fn matches_hf_reference() {
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    let (_weights, model) = build(&dir);
    let cfg = model.config.clone();
    let tokens = tokens_of(&dir);

    let mut cache = model.make_cache(tokens.len() + 8).expect("cache");
    let (hidden, traced) = model.forward_traced(&tokens, &mut cache, true).expect("forward");
    let logits = model.lm_head_forward(&hidden).expect("lm_head");
    let got = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let refs = SafetensorsLoader::open(dir.join("reference.safetensors")).expect("reference");

    let mut ambiguous_rows: Vec<usize> = Vec::new();
    for (name, t) in traced.iter() {
        if name.starts_with("index_mask_") {
            if !refs.contains("index_mask") {
                continue;
            }
            let (want, dims) = read_vec(&refs, "index_mask");
            let kv = dims[1];
            let got_mask = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for row in 0..dims[0] {
                let same = (0..kv).all(|j| {
                    let visible_ref = want[row * kv + j] == 0.0;
                    let visible_got = got_mask[row * kv + j] != 0.0;
                    visible_ref == visible_got
                });
                if !same {
                    ambiguous_rows.push(row);
                }
            }
            eprintln!(
                "index_mask: строк с иным выбором блоков {} из {}",
                ambiguous_rows.len(),
                dims[0]
            );
            assert!(
                ambiguous_rows.len() * 10 <= dims[0],
                "выбор индексатора разошёлся на {} строках из {}",
                ambiguous_rows.len(),
                dims[0]
            );
            continue;
        }
        let key = match name.strip_prefix("in_") {
            Some(idx) => format!("hidden_{idx}"),
            None if name == "final" => "hidden_4".to_string(),
            None => name.clone(),
        };
        if !refs.contains(&key) || key.starts_with("mixer_out_") {
            continue;
        }
        let (want, _) = read_vec(&refs, &key);
        let got = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        compare_rows(&key, &got, &want, tokens.len(), &ambiguous_rows, 2e-3);
    }

    let (want, dims) = read_vec(&refs, "logits");
    assert_eq!(dims, vec![tokens.len(), cfg.vocab_size]);
    compare_rows("logits", &got, &want, tokens.len(), &ambiguous_rows, 2e-3);
    assert!(ambiguous_rows.len() * 10 <= tokens.len());
}

#[test]
fn decode_matches_prefill() {
    let Some(dir) = ref_dir() else {
        return;
    };
    let (_weights, model) = build(&dir);
    let tokens = tokens_of(&dir);

    let mut prefill_cache = model.make_cache(tokens.len() + 8).expect("cache");
    let prefill = model
        .forward(&tokens, &mut prefill_cache)
        .expect("prefill")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut cache = model.make_cache(tokens.len() + 8).expect("cache");
    let mut step_logits = Vec::new();
    for token in &tokens {
        let out = model.forward(&[*token], &mut cache).expect("step");
        step_logits.extend(out.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
    compare("decode-vs-prefill", &step_logits, &prefill, 1e-3);
}
