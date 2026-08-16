use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_muse_glimmer::preprocess::ImageGrid;
use synaptix_llm_muse_glimmer::{BundleVisionWeights, MuseConfig, MusePipeline, VisionTower};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from)
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("to_f32")
}

fn compare(name: &str, ours: &[f32], reference: &[f32], min_cosine: f64) {
    assert_eq!(ours.len(), reference.len(), "{name}: len mismatch");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max_abs = 0f64;
    for (a, b) in ours.iter().zip(reference) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    eprintln!("[{name}] cosine={cosine:.6} max_abs_diff={max_abs:.4}");
    assert!(cosine > min_cosine, "{name}: cosine {cosine} < {min_cosine}");
}

#[test]
fn vision_matches_reference() {
    let Some(bundle) = env_path("SYN_MUSE_BUNDLE") else { return };
    let Some(ref_dir) = env_path("SYN_MUSE_REF") else { return };
    let ref_path = ref_dir.join("vision_ref.safetensors");
    if !ref_path.exists() {
        eprintln!("skip: {} отсутствует", ref_path.display());
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);

    let refs = SafetensorsLoader::open(&ref_path).expect("open ref");
    let pixel_values = refs.load_to("pixel_values", device, DType::F32).expect("pixel_values");
    let grid_raw = refs
        .load_to("grid_thw", Device::Cpu, DType::I64)
        .expect("grid")
        .to_vec1::<i64>()
        .expect("grid vec");
    let grid = ImageGrid {
        t: grid_raw[0] as usize,
        h: grid_raw[1] as usize,
        w: grid_raw[2] as usize,
    };
    let features_ref = to_f32(&refs.load_to("features", Device::Cpu, DType::F32).expect("features"));

    let cfg_bytes = synaptix_bundle::Bundle::open(&bundle)
        .and_then(|b| b.read_file("config.json").map(|c| c.into_owned()))
        .expect("config.json");
    let cfg = MuseConfig::from_hf_bytes(&cfg_bytes).expect("config");
    let dtype = match std::env::var("SYN_MUSE_VISION_DTYPE").as_deref() {
        Ok("f32") => DType::F32,
        _ => DType::BF16,
    };
    let weights = BundleVisionWeights::open(&bundle, device).expect("vision weights");
    let tower = VisionTower::build(
        cfg.vision.clone().expect("vision cfg"),
        cfg.rms_norm_eps,
        &weights,
        device,
        dtype,
    )
    .expect("tower");

    let tower_ref = to_f32(&refs.load_to("tower_out", Device::Cpu, DType::F32).expect("tower_out"));
    let merged = tower.forward_tower(&pixel_values, grid).expect("forward_tower");
    compare("vision tower_out", &to_f32(&merged), &tower_ref, 0.995);

    let tower_ref_t = Tensor::from_vec(
        tower_ref.clone(),
        vec![merged.dims()[0], merged.dims()[1]],
        tower.device,
    )
    .and_then(|t| t.to_dtype(tower.dtype))
    .expect("ref tensor");
    let feats_from_ref = tower.project(&tower_ref_t).expect("project(ref)");
    compare("vision project(ref tower_out)", &to_f32(&feats_from_ref), &features_ref, 0.9995);

    let out = tower.project(&merged).expect("project");
    let ours = to_f32(&out);
    if let Ok(dump) = std::env::var("SYN_MUSE_DUMP") {
        let bytes: Vec<u8> = ours.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dump, bytes).expect("dump");
    }
    compare("vision features", &ours, &features_ref, 0.998);
}

#[test]
fn vision_stage_debug() {
    let Some(bundle) = env_path("SYN_MUSE_BUNDLE") else { return };
    let Some(dbg) = env_path("SYN_MUSE_VISION_DEBUG") else { return };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);

    let refs = SafetensorsLoader::open(&dbg).expect("open debug ref");
    let pixel_values = refs.load_to("pixel_values", device, DType::F32).expect("pixel_values");
    let grid_raw = refs
        .load_to("grid_thw", Device::Cpu, DType::I64)
        .expect("grid")
        .to_vec1::<i64>()
        .expect("grid vec");
    let grid = ImageGrid {
        t: grid_raw[0] as usize,
        h: grid_raw[1] as usize,
        w: grid_raw[2] as usize,
    };
    let window_ref: Vec<i64> = refs
        .load_to("window_index", Device::Cpu, DType::I64)
        .expect("window_index")
        .to_vec1::<i64>()
        .expect("wi vec");
    let cu_ref: Vec<i64> = refs
        .load_to("cu_window", Device::Cpu, DType::I64)
        .expect("cu_window")
        .to_vec1::<i64>()
        .expect("cu vec");

    let plan = synaptix_llm_muse_glimmer::vision::window_plan(grid, 32);
    let ours_wi: Vec<i64> = plan.index.iter().map(|x| *x as i64).collect();
    assert_eq!(ours_wi, window_ref, "window_index mismatch");
    let ours_cu: Vec<i64> = plan.cu_windows.iter().map(|x| *x as i64).collect();
    assert_eq!(ours_cu, cu_ref, "cu_window mismatch");
    eprintln!("[debug] window plan OK");

    let cfg_bytes = synaptix_bundle::Bundle::open(&bundle)
        .and_then(|b| b.read_file("config.json").map(|c| c.into_owned()))
        .expect("config.json");
    let cfg = MuseConfig::from_hf_bytes(&cfg_bytes).expect("config");
    let weights = BundleVisionWeights::open(&bundle, device).expect("vision weights");
    let tower = VisionTower::build(
        cfg.vision.clone().expect("vision cfg"),
        cfg.rms_norm_eps,
        &weights,
        device,
        DType::BF16,
    )
    .expect("tower");

    let mut probe: Vec<(String, Tensor)> = Vec::new();
    let _ = tower
        .forward_tower_probed(&pixel_values, grid, Some(&mut probe))
        .expect("probed forward");

    for (name, t) in &probe {
        let ref_name = match name.strip_prefix("hidden_") {
            Some(i) => format!("hidden_{}", i.parse::<usize>().unwrap() + 1),
            None => name.clone(),
        };
        if let Ok(r) = refs.load_to(&ref_name, Device::Cpu, DType::F32) {
            let rv = to_f32(&r);
            let ov = to_f32(t);
            if rv.len() == ov.len() {
                let mut dot = 0f64;
                let mut na = 0f64;
                let mut nb = 0f64;
                let mut max_abs = 0f64;
                for (a, b) in ov.iter().zip(&rv) {
                    let (a, b) = (*a as f64, *b as f64);
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                    max_abs = max_abs.max((a - b).abs());
                }
                let cosine = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                eprintln!("[debug] {name}: cosine={cosine:.6} max_abs={max_abs:.4}");
                if let Ok(dir) = std::env::var("SYN_MUSE_DUMP_DIR") {
                    let bytes: Vec<u8> = ov.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/ours_{name}.bin"), bytes).expect("dump");
                }
            }
        }
    }
}

#[test]
fn text_logits_and_greedy_match_reference() {
    let Some(bundle) = env_path("SYN_MUSE_BUNDLE") else { return };
    let Some(ref_dir) = env_path("SYN_MUSE_REF") else { return };
    let ref_path = ref_dir.join("text_ref.safetensors");
    if !ref_path.exists() {
        eprintln!("skip: {} отсутствует", ref_path.display());
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);

    let refs = SafetensorsLoader::open(&ref_path).expect("open ref");
    let ids: Vec<u32> = refs
        .load_to("input_ids", Device::Cpu, DType::I64)
        .expect("ids")
        .to_vec1::<i64>()
        .expect("ids vec")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let logits_ref = to_f32(&refs.load_to("logits_last", Device::Cpu, DType::F32).expect("logits"));
    let greedy_ref: Vec<u32> = refs
        .load_to("greedy_ids", Device::Cpu, DType::I64)
        .expect("greedy")
        .to_vec1::<i64>()
        .expect("greedy vec")
        .into_iter()
        .map(|x| x as u32)
        .collect();

    synaptix_llm_muse_glimmer::pipeline::set_offload_mode_for_tests();
    let pipeline = MusePipeline::load_with_precision(
        &bundle,
        device,
        PrecisionConfig::dense(DType::BF16),
        Some(512),
    )
    .expect("load bf16");

    let mut kv = pipeline.model.make_kv_cache(1, 512).expect("kv");
    let input = Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], device).expect("input");
    let logits = synaptix_core::grad::no_grad(|| pipeline.model.forward(&input, &mut kv))
        .expect("forward");
    let ours = to_f32(&logits);
    compare("text logits(last)", &ours, &logits_ref, 0.995);

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(
        argmax(&ours),
        argmax(&logits_ref),
        "argmax первого сгенерированного токена должен совпасть"
    );

    let (greedy, _) = pipeline
        .generate(
            &ids,
            synaptix_llm_muse_glimmer::GenerationConfig {
                max_new_tokens: greedy_ref.len(),
                temperature: 0.0,
                max_seq: Some(512),
                ..Default::default()
            },
        )
        .expect("greedy");
    let common = greedy
        .iter()
        .zip(&greedy_ref)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[greedy] ours={greedy:?}\n         ref ={greedy_ref:?}\n         префикс {common}/{}",
        greedy_ref.len()
    );
    assert!(
        common * 2 >= greedy_ref.len(),
        "greedy-префикс слишком короткий: {common}/{}",
        greedy_ref.len()
    );
}

/// Ядро flash-window в bidirectional-режиме (`causal=false`) — путь DFlash:
/// Tq=16 (диффузионное окно) против Tkv=контекст+окно, band |i-j| < window.
/// Сверка с reference sdpa + явной band-маской.
#[test]
fn flash_window_bidirectional_matches_reference() {
    if std::env::var("SYN_MUSE_KERNEL").is_err() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let (nh, nkv, hd) = (32usize, 8usize, 128usize);
    let tq = 16usize;

    for (tkv, window) in [(48usize, 2048usize), (600, 128), (4096, 2048)] {
        let mk = |n: usize, t: usize, seed: u64| -> Tensor {
            let mut s = seed;
            let v: Vec<f32> = (0..n * t * hd)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.6
                })
                .collect();
            Tensor::from_vec(v, vec![1, n, t, hd], device)
                .and_then(|t| t.to_dtype(DType::F16))
                .expect("mk")
        };
        let q = mk(nh, tq, 1);
        let k = mk(nkv, tkv, 2);
        let v = mk(nkv, tkv, 3);
        let scale = 1.0 / (hd as f32).sqrt();

        let got = q
            .flash_attention_window(&k, &v, scale, (window - 1) as i32, false)
            .expect("flash window bidirectional");

        // reference: repeat_kv + маска band |i-j| < window, q — последние tq позиций
        let group = nh / nkv;
        let rep = |x: &Tensor| -> Tensor {
            let reps = Tensor::zeros(vec![1, nkv, group, tkv, hd], x.dtype(), device).unwrap();
            x.unsqueeze(2)
                .and_then(|t| t.broadcast_add(&reps))
                .and_then(|t| t.reshape(vec![1, nh, tkv, hd]))
                .unwrap()
        };
        let base = tkv - tq;
        let mut mask = vec![0f32; tq * tkv];
        for i in 0..tq {
            for j in 0..tkv {
                if ((base + i) as i64 - j as i64).unsigned_abs() as usize >= window {
                    mask[i * tkv + j] = -1.0e4;
                }
            }
        }
        let mask = Tensor::from_vec(mask, vec![tq, tkv], device)
            .and_then(|t| t.to_dtype(DType::F16))
            .unwrap();
        let want = synaptix_ops::attention::softmax::scaled_dot_attention(
            &q,
            &rep(&k),
            &rep(&v),
            scale,
            Some(&mask),
        )
        .expect("reference sdpa");

        compare(
            &format!("flash_window bidir tkv={tkv} win={window}"),
            &to_f32(&got),
            &to_f32(&want),
            0.9995,
        );
    }
}

/// Паритет DFlash-драфтера с transformers: один draft-forward на контексте
/// промпта. Сверяются hidden блока, логиты кандидатов и сами id кандидатов.
#[test]
fn dflash_matches_reference() {
    let Some(bundle) = env_path("SYN_MUSE_BUNDLE") else { return };
    let Some(ref_dir) = env_path("SYN_MUSE_REF") else { return };
    let ref_path = ref_dir.join("dflash_ref.safetensors");
    if !ref_path.exists() {
        eprintln!("skip: {} отсутствует", ref_path.display());
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);

    let refs = SafetensorsLoader::open(&ref_path).expect("open ref");
    let ids: Vec<u32> = refs
        .load_to("input_ids", Device::Cpu, DType::I64)
        .expect("ids")
        .to_vec1::<i64>()
        .expect("ids vec")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let anchor = refs
        .load_to("anchor", Device::Cpu, DType::I64)
        .expect("anchor")
        .to_vec1::<i64>()
        .expect("anchor vec")[0] as u32;
    let cand_ref: Vec<u32> = refs
        .load_to("candidate_ids", Device::Cpu, DType::I64)
        .expect("cand")
        .to_vec1::<i64>()
        .expect("cand vec")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let logits_ref = to_f32(
        &refs
            .load_to("candidate_logits", Device::Cpu, DType::F32)
            .expect("cand logits"),
    );

    let precision = PrecisionConfig::dense(DType::BF16);
    synaptix_llm_muse_glimmer::pipeline::set_offload_mode_for_tests();
    let mut pipeline = MusePipeline::load_with_precision(&bundle, device, precision, Some(1024))
        .expect("load muse");
    assert!(
        pipeline.load_dflash(&bundle, precision).expect("load dflash"),
        "в бандле нет компонента dflash"
    );
    let dflash = pipeline.dflash.as_ref().expect("dflash");

    let mut kv = pipeline.model.make_kv_cache(1, 1024).expect("kv");
    let input = Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], device).expect("input");
    let taps = dflash.config.target_layer_ids.clone();
    let (_, tapped) = synaptix_core::grad::no_grad(|| {
        pipeline.model.forward_trunk_tapped(&input, &mut kv, &taps)
    })
    .expect("prefill tapped");

    let mut dcache = dflash.make_cache().expect("dflash cache");
    let logits = synaptix_core::grad::no_grad(|| {
        dflash.draft_logits(&pipeline.model, &mut dcache, &tapped, 0, anchor)
    })
    .expect("draft");

    // Мы прогоняем драфт-логиты через тот же lm_head-эпилог, что и основную
    // модель (output_multiplier + tanh-softcap); HF в DFlash-генераторе зовёт
    // голый lm_head. Для argmax это эквивалентно, но для сверки значений
    // приводим эталон к тому же виду.
    let softcap: Vec<f32> = logits_ref
        .iter()
        .map(|x| {
            let m = 0.196_116_14f32;
            let cap = 20.0f32;
            cap * (x * m / cap).tanh()
        })
        .collect();
    compare("dflash candidate logits", &to_f32(&logits), &softcap, 0.99);

    let dims = logits.dims().to_vec();
    let vocab = dims[dims.len() - 1];
    let v = to_f32(&logits);
    let ours: Vec<u32> = (0..dims[0])
        .map(|r| {
            v[r * vocab..(r + 1) * vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        })
        .collect();
    let same = ours.iter().zip(&cand_ref).filter(|(a, b)| a == b).count();
    eprintln!("[dflash] ours={ours:?}\n         ref ={cand_ref:?}\n         совпало {same}/{}", cand_ref.len());
    assert!(
        same * 4 >= cand_ref.len() * 3,
        "кандидаты DFlash расходятся с эталоном: {same}/{}",
        cand_ref.len()
    );
}
