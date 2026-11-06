//! Сверка порта Leokani G2P с эталонами `g2p/goldens.json` (как велит его поле `note`):
//! pieces — `normalize(piece, pauses=False)` и `piece`, lines/edge — `normalize(text)` и `phonemize`.
//!
//! Пути: LEOKANI_STAGE (каталог с leokani-g2p.json и g2p_nets/{context,enc,dec}.safetensors),
//! LEOKANI_GOLDENS (goldens.json). Без файлов тест пропускается.
//!
//! cargo test --profile fast-release -p synaptix-tts-leokani --test g2p_goldens -- --nocapture

use std::path::{Path, PathBuf};

use serde_json::Value;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_io::{SafetensorsLoader, WeightLoader};
use synaptix_tts_leokani::g2p::LeokaniG2P;
use synaptix_tts_leokani::LkError;

const STAGE: &str = "/home/master/Storage/Нейросети/Модели/_hf/leokani-stage";
const GOLDENS: &str = "/home/master/.cache/huggingface/hub/models--milliwattlabs--leokani-tts-7M/snapshots/ba0beb24f7b124dc1eb297ffc28d01db892462bd/g2p/goldens.json";

fn env_path(var: &str, def: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(def))
}

fn load_g2p(stage: &Path) -> LeokaniG2P {
    synaptix_kernels_cpu::ensure_registered();
    let doc = std::fs::read(stage.join("leokani-g2p.json")).expect("leokani-g2p.json");
    let loaders: Vec<SafetensorsLoader> = ["context", "enc", "dec"]
        .iter()
        .map(|n| SafetensorsLoader::open(stage.join("g2p_nets").join(format!("{n}.safetensors"))).expect("g2p_nets"))
        .collect();
    let get = |name: &str| {
        let l = loaders.iter().find(|l| l.contains(name)).ok_or_else(|| LkError::Load(format!("нет веса {name}")))?;
        l.load_to(name, Device::Cpu, DType::F32).map_err(|e| LkError::Load(format!("{name}: {e}")))
    };
    LeokaniG2P::new(&doc, &get).expect("LeokaniG2P::new")
}

struct Miss {
    id: String,
    what: &'static str,
    input: String,
    got: String,
    want: String,
}

/// Сверка разделов файла формата goldens; возвращает (число расхождений, отчёт по разделам).
fn check(g2p: &LeokaniG2P, gold: &Value) -> (usize, Vec<String>) {
    let mut total_bad = 0;
    let mut report = Vec::new();
    for section in ["pieces", "lines", "edge"] {
        let Some(items) = gold.get(section).and_then(|v| v.as_array()) else { continue };
        // Куски раздела считаются параллельно: G2P общий (кэш OOV под Mutex).
        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let chunk = items.len().div_ceil(n_threads).max(1);
        let misses: Vec<Miss> = std::thread::scope(|s| {
            let hs: Vec<_> = items
                .chunks(chunk)
                .enumerate()
                .map(|(ci, part)| {
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for (k, e) in part.iter().enumerate() {
                            let id = e
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                                .unwrap_or_else(|| format!("#{}", ci * chunk + k));
                            let (input, norm, ph) = if section == "pieces" {
                                let p = e["piece"].as_str().unwrap();
                                (p.to_string(), g2p.normalize(p, false), g2p.piece(p))
                            } else {
                                let t = e["text"].as_str().unwrap();
                                (t.to_string(), g2p.normalize(t, true), g2p.phonemize(t))
                            };
                            let (wn, wp) = (e["normalized"].as_str().unwrap(), e["leokani"].as_str().unwrap());
                            if norm != wn {
                                out.push(Miss { id: id.clone(), what: "normalized", input: input.clone(), got: norm, want: wn.into() });
                            }
                            if ph != wp {
                                out.push(Miss { id, what: "leokani", input, got: ph, want: wp.into() });
                            }
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let bad: std::collections::HashSet<&str> = misses.iter().map(|m| m.id.as_str()).collect();
        let ok = items.len() - bad.len();
        report.push(format!("{section}: {ok}/{} ({:.2} %)", items.len(), 100.0 * ok as f64 / items.len() as f64));
        for m in misses.iter().take(60) {
            eprintln!("[{section} {} {}] {}\n   got:  {}\n   want: {}", m.id, m.what, m.input, m.got, m.want);
        }
        total_bad += bad.len();
    }
    (total_bad, report)
}

#[test]
fn g2p_goldens() {
    let stage = env_path("LEOKANI_STAGE", STAGE);
    let goldens = env_path("LEOKANI_GOLDENS", GOLDENS);
    if !stage.join("leokani-g2p.json").exists() || !goldens.exists() {
        eprintln!("пропуск: нет {} или {}", stage.display(), goldens.display());
        return;
    }
    let t0 = std::time::Instant::now();
    let g2p = load_g2p(&stage);
    eprintln!("загрузка: {:.1} с", t0.elapsed().as_secs_f64());
    let gold: Value = serde_json::from_slice(&std::fs::read(&goldens).unwrap()).unwrap();
    let t1 = std::time::Instant::now();
    let (bad, report) = check(&g2p, &gold);
    eprintln!("сверка: {:.1} с", t1.elapsed().as_secs_f64());
    for r in &report {
        eprintln!("{r}");
    }
    assert_eq!(bad, 0, "расхождения с goldens: {}", report.join(", "));
}

/// Дополнительная сверка с файлом того же формата (LEOKANI_EXTRA), например выгрузкой эталонного
/// leokani_g2p.py на произвольных строках и псевдословах. Без переменной — пропуск.
#[test]
fn g2p_extra() {
    let Some(extra) = std::env::var_os("LEOKANI_EXTRA") else { return };
    let stage = env_path("LEOKANI_STAGE", STAGE);
    let g2p = load_g2p(&stage);
    let gold: Value = serde_json::from_slice(&std::fs::read(&extra).unwrap()).unwrap();
    let (bad, report) = check(&g2p, &gold);
    for r in &report {
        eprintln!("{r}");
    }
    assert_eq!(bad, 0, "расхождения: {}", report.join(", "));
}
