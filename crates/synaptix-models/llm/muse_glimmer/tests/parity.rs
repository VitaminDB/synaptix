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
