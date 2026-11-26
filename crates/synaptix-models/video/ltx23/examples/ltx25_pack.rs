//! Сборка однофайлового чекпойнта LTX-2.5 в раскладке LTX-2.3 (её читает
//! движок): DiT (`model.diffusion_model.*`) — из файла трансформера 2.5,
//! VAE видео — из свёрточного VAE 2.5 с префиксом `vae.`, звук и вокодер —
//! из звукового VAE 2.5 (имена как есть), `text_embedding_projection.*` — из
//! файла текстового кодера. Конфиг: секции VAE/звука — из чекпойнта 2.3
//! (VAE побайтно тот же), `transformer` и `scheduler` — из 2.5,
//! `model_version` — из 2.5.
//!
//! `ltx25_pack DIT.safetensors VAE_CONV.safetensors AUDIO.safetensors TE.safetensors REF23.syn OUT.safetensors`
use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype, TensorView};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader as _;

fn open(p: &str) -> SafetensorsLoader {
    if p.ends_with(".syn") {
        SafetensorsLoader::open_bundle(Path::new(p), None).expect("bundle")
    } else {
        SafetensorsLoader::open(Path::new(p)).expect("safetensors")
    }
}

fn st_dtype(d: synaptix_core::dtype::DType) -> Dtype {
    use synaptix_core::dtype::DType as D;
    match d {
        D::BF16 => Dtype::BF16,
        D::F16 => Dtype::F16,
        D::F32 => Dtype::F32,
        D::U8 => Dtype::U8,
        other => panic!("dtype {other:?}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dit, vae, audio, te, r23, out) = (open(&a[0]), open(&a[1]), open(&a[2]), open(&a[3]), open(&a[4]), &a[5]);
    // (имя в выходе, источник, имя в источнике)
    let mut plan: Vec<(String, &SafetensorsLoader, String)> = Vec::new();
    for n in dit.names() {
        if n.starts_with("model.diffusion_model.") {
            plan.push((n.to_string(), &dit, n.to_string()));
        }
    }
    for n in vae.names() {
        plan.push((format!("vae.{n}"), &vae, n.to_string()));
    }
    for n in audio.names() {
        plan.push((n.to_string(), &audio, n.to_string()));
    }
    for n in te.names() {
        if n.starts_with("text_embedding_projection.") {
            plan.push((n.to_string(), &te, n.to_string()));
        }
    }
    plan.sort_by(|x, y| x.0.cmp(&y.0));
    let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(plan.len());
    for (name, src, sname) in &plan {
        let (bytes, dt, shape) = src.raw_bytes(sname).unwrap_or_else(|| panic!("нет {sname}"));
        views.push((name.clone(), TensorView::new(st_dtype(dt), shape.to_vec(), bytes).expect("view")));
    }
    // Конфиг: 2.3 + transformer/scheduler из 2.5.
    let mut cfg: serde_json::Value = serde_json::from_str(r23.metadata().get("config").expect("config 2.3")).expect("json 2.3");
    let c25: serde_json::Value = serde_json::from_str(dit.metadata().get("config").expect("config 2.5")).expect("json 2.5");
    cfg["transformer"] = c25["transformer"].clone();
    cfg["scheduler"] = c25["scheduler"].clone();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("config".into(), cfg.to_string());
    meta.insert("model_version".into(), dit.metadata().get("model_version").cloned().unwrap_or_else(|| "2.5.0".into()));
    if let Some(g) = dit.metadata().get("gemma_source_checkpoint") {
        meta.insert("gemma_source_checkpoint".into(), g.clone());
    }
    println!("тензоров {}, пишу {out}", views.len());
    safetensors::serialize_to_file(views, Some(meta), Path::new(out)).expect("запись");
    println!("готово");
}
