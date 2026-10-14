use std::path::{Path, PathBuf};

use super::params::Task;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPlan {
    pub id: String,
    pub title: String,
    pub task: Task,
    pub arch: String,
    pub description: String,
    pub components: Vec<(String, PathBuf)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CatalogPlan {
    pub models: Vec<ModelPlan>,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug, Clone)]
struct File {
    path: PathBuf,
    stem: String,
    arch: String,
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

fn pretty(stem: &str) -> String {
    let s = stem.to_lowercase();
    let known: &[(&str, &str)] = &[
        ("flux.1-dev", "FLUX.1 dev"),
        ("flux.2-dev", "FLUX.2 dev"),
        ("flux.2-klein-4b", "FLUX.2 klein 4B"),
        ("flux.2-klein-9b", "FLUX.2 klein 9B"),
        ("flux.2-klein-base-4b", "FLUX.2 klein base 4B"),
        ("sdxl-base-1.0", "SDXL 1.0"),
        ("qwen-image-2.1", "Qwen-Image 2.1"),
        ("qwen-image-edit", "Qwen-Image Edit"),
        ("qwen-image-edit-2511", "Qwen-Image Edit 2511"),
        ("ltx-2.3-22b-dev", "LTX-2.3 22B"),
        ("ltx-2.3-22b-distilled-1.1", "LTX-2.3 22B distilled"),
        ("minimax-h3-fl2va", "MiniMax H3 (кадры → видео)"),
        ("minimax-h3-ref2va", "MiniMax H3 (по образцу)"),
        ("acestep_v15_xl_base", "ACE-Step 1.5 XL"),
        ("acestep_v15_xl_turbo", "ACE-Step 1.5 XL turbo"),
        ("yue2-3b", "YuE 2 3B"),
        ("omnivoice", "OmniVoice"),
        ("vibevoice-1.5b", "VibeVoice 1.5B"),
        ("vibevoice-7b", "VibeVoice 7B"),
        ("voxcpm2", "VoxCPM 2"),
        ("whisper-large-v3-turbo", "Whisper large v3 turbo"),
        ("whisper-tiny", "Whisper tiny"),
        ("gigaam-v3", "GigaAM v3"),
        ("bge-m3", "BGE-M3"),
        ("qwen3-embedding-0.6b", "Qwen3 Embedding 0.6B"),
        ("gemma-4-26b-a4b-it", "Gemma 4 26B A4B"),
        ("gemma-4-e2b-it", "Gemma 4 E2B"),
        ("gemma-4-e4b-it", "Gemma 4 E4B"),
        ("qwen3-1.7b", "Qwen3 1.7B"),
        ("muse-glimmer-30b", "Muse Glimmer 30B"),
        ("qwen3.8-27b", "Qwen3.8 27B"),
        ("qwen3.8-flash-next", "Qwen3.8 Flash Next"),
        ("qwen3.6 27b", "Qwen3.6 27B"),
    ];
    known.iter().find(|(k, _)| *k == s).map(|(_, v)| v.to_string()).unwrap_or_else(|| stem.to_string())
}

fn single(f: &File, task: Task, arch: &str, description: &str) -> ModelPlan {
    ModelPlan {
        id: slug(&f.stem),
        title: pretty(&f.stem),
        task,
        arch: arch.to_string(),
        description: description.to_string(),
        components: vec![("model".into(), f.path.clone())],
    }
}

pub fn plan_dir(dir: &Path) -> CatalogPlan {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "syn")).collect())
        .unwrap_or_default();
    paths.sort();
    plan_files(&paths)
}

pub fn plan_files(paths: &[PathBuf]) -> CatalogPlan {
    let files: Vec<File> = paths
        .iter()
        .map(|p| File {
            path: p.clone(),
            stem: p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            arch: super::arch::arch_key(p).unwrap_or_default().to_lowercase(),
        })
        .collect();
    plan_parsed(&files)
}

fn plan_parsed(files: &[File]) -> CatalogPlan {
    let mut out = CatalogPlan::default();
    let name = |f: &File| f.stem.to_lowercase();
    let find = |pred: &dyn Fn(&File) -> bool| files.iter().find(|f| pred(f)).map(|f| f.path.clone());

    let ltx_encoder = find(&|f| f.arch == "gemma3" && name(f).contains("ltx"));
    let ltx_upscaler = find(&|f| f.arch == "ltx-2.3" && name(f).contains("upscaler"));
    let ltx_distill_lora = find(&|f| f.arch == "ltx-2.3" && name(f).contains("distilled-lora"));
    let ltx_ic: Vec<(String, PathBuf)> = files
        .iter()
        .filter(|f| f.arch == "ltx-2.3" && name(f).contains("ic-lora"))
        .map(|f| {
            let n = name(f);
            let role = if n.contains("lipdub") {
                "ic_lora_lipdub"
            } else if n.contains("motion") {
                "ic_lora_motion"
            } else if n.contains("union") {
                "ic_lora_union"
            } else {
                "ic_lora"
            };
            (role.to_string(), f.path.clone())
        })
        .collect();
    let h3_encoder = find(&|f| f.arch == "qwen3_vl" && name(f).contains("minimax"));
    let ace_vae = find(&|f| f.arch == "ace-step-vae");
    let ace_lm_big = find(&|f| name(f).starts_with("acestep_5hz_lm") && name(f).contains("4b"));
    let ace_lm_small = find(&|f| name(f).starts_with("acestep_5hz_lm") && !name(f).contains("4b"));
    let yue_vae = find(&|f| f.arch == "yue2_vae");

    let is_component = |f: &File| {
        Some(&f.path) == ltx_encoder.as_ref()
            || Some(&f.path) == ltx_upscaler.as_ref()
            || Some(&f.path) == ltx_distill_lora.as_ref()
            || ltx_ic.iter().any(|(_, p)| *p == f.path)
            || Some(&f.path) == h3_encoder.as_ref()
            || Some(&f.path) == ace_vae.as_ref()
            || Some(&f.path) == ace_lm_big.as_ref()
            || Some(&f.path) == ace_lm_small.as_ref()
            || Some(&f.path) == yue_vae.as_ref()
    };

    for f in files {
        if is_component(f) {
            continue;
        }
        let n = name(f);
        let plan = match f.arch.as_str() {
            "gemma4" | "gemma3" | "qwen3_5" | "qwen4_exp" | "muse_glimmer" | "llama" | "qwen3_next" if !n.contains("embedding") => {
                Some(single(f, Task::Llm, &f.arch, "Языковая модель"))
            }
            "qwen3" if n.contains("embedding") => Some(single(f, Task::Embed, "qwen3_embedding", "Эмбеддинги для поиска по смыслу")),
            "qwen3" => Some(single(f, Task::Llm, "qwen3", "Языковая модель")),
            "xlm-roberta" if n.contains("reranker") => Some(single(f, Task::Rerank, "bge_reranker", "Переранжирование найденного для поиска")),
            "xlm-roberta" => Some(single(f, Task::Embed, "bge_m3", "Эмбеддинги для поиска по смыслу")),
            "whisper" => Some(single(f, Task::Asr, "whisper", "Распознавание речи")),
            "conformer" => Some(single(f, Task::Asr, "gigaam", "Распознавание русской речи")),
            "sortformer" => Some(single(f, Task::Diarize, "sortformer", "Кто когда говорит (до 4 голосов)")),
            "omnivoice" => Some(single(f, Task::Tts, "omnivoice", "Многоязычный синтез речи, голос по описанию")),
            "vibevoice" => Some(single(f, Task::Tts, "vibevoice", "Синтез речи и диалогов")),
            "voxcpm2" => Some(single(f, Task::Tts, "voxcpm2", "Синтез речи")),
            "flux.1" => Some(single(f, Task::Image, "flux", "Генерация картинок")),
            "flux.2" => Some(single(f, Task::Image, "flux2", "Генерация и правка картинок")),
            "sdxl" => Some(single(f, Task::Image, "sdxl", "Генерация картинок")),
            "qwen-image-2.1" => Some(single(f, Task::Image, "qwen_image21", "Генерация картинок с текстом")),
            "qwen-image-edit" => Some(single(f, Task::Image, "qwen_image", "Правка картинок по описанию")),
            "ltx-2.3" => {
                let mut p = single(f, Task::Video, "ltx23", "Видео со звуком по тексту и картинке");
                p.components.extend(ltx_encoder.iter().map(|x| ("text_encoder".to_string(), x.clone())));
                p.components.extend(ltx_upscaler.iter().map(|x| ("upscaler".to_string(), x.clone())));
                if !n.contains("distilled") {
                    p.components.extend(ltx_distill_lora.iter().map(|x| ("distilled_lora".to_string(), x.clone())));
                }
                p.components.extend(ltx_ic.iter().cloned());
                Some(p)
            }
            // `syn-pack` по каталогу Diffusers пишет имя класса из `model_index.json`.
            "minimax-h3" | "minimaxh3pipeline" => {
                let mut p = single(f, Task::Video, "minimax_h3", "Видео по кадрам и образцу");
                p.components.extend(h3_encoder.iter().map(|x| ("encoder".to_string(), x.clone())));
                Some(p)
            }
            "acestep" => {
                let mut p = single(f, Task::Music, "acestep", "Музыка и песни по описанию и тексту");
                p.components.extend(ace_lm_big.iter().map(|x| ("lm".to_string(), x.clone())));
                p.components.extend(ace_lm_small.iter().map(|x| ("lm_small".to_string(), x.clone())));
                p.components.extend(ace_vae.iter().map(|x| ("vae".to_string(), x.clone())));
                Some(p)
            }
            "yue2" => {
                let mut p = single(f, Task::Music, "yue2", "Песни с вокалом по тексту");
                p.components.extend(yue_vae.iter().map(|x| ("vae".to_string(), x.clone())));
                Some(p)
            }
            _ => None,
        };
        match plan {
            Some(p) => out.models.push(p),
            None => out.skipped.push(Skipped {
                path: f.path.clone(),
                reason: if f.arch.is_empty() { "архитектура не определена".into() } else { format!("служба не знает архитектуру «{}»", f.arch) },
            }),
        }
    }
    for f in files.iter().filter(|f| is_component(f)) {
        if !out.models.iter().any(|m| m.components.iter().any(|(_, p)| *p == f.path)) {
            out.skipped.push(Skipped { path: f.path.clone(), reason: "часть модели, основного файла нет".into() });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(stem: &str, arch: &str) -> File {
        File { path: PathBuf::from(format!("/m/{stem}.syn")), stem: stem.into(), arch: arch.into() }
    }

    #[test]
    fn groups_components() {
        let files = vec![
            f("ltx-2.3-22b-dev", "ltx-2.3"),
            f("ltx-2.3-22b-distilled-1.1", "ltx-2.3"),
            f("ltx-2.3-22b-distilled-lora-384-1.1", "ltx-2.3"),
            f("ltx-2.3-spatial-upscaler-x2-1.1", "ltx-2.3"),
            f("ltx-gemma-3-12b-qat", "gemma3"),
            f("acestep_v15_xl_turbo", "acestep"),
            f("acestep_5hz_lm_4b", "qwen3"),
            f("acestep_vae", "ace-step-vae"),
            f("bge-reranker-v2-m3", "xlm-roberta"),
            f("gemma-4-26b-a4b-it", "gemma4"),
        ];
        let p = plan_parsed(&files);
        let ids: Vec<&str> = p.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["ltx-2-3-22b-dev", "ltx-2-3-22b-distilled-1-1", "acestep-v15-xl-turbo", "gemma-4-26b-a4b-it"]);
        let dev = &p.models[0];
        assert!(dev.components.iter().any(|(r, _)| r == "text_encoder"));
        assert!(dev.components.iter().any(|(r, _)| r == "distilled_lora"));
        assert!(!p.models[1].components.iter().any(|(r, _)| r == "distilled_lora"));
        assert_eq!(p.models[2].components.len(), 3);
        assert_eq!(p.skipped.len(), 1);
    }
}
