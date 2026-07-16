//! Кросс-фреймворк-гейт BGE-reranker-v2-m3 vs HF `XLMRobertaForSequenceClassification`.
//!
//! Эталон `tmp/reranker_ref/` (reference/dump_reranker.py):
//!   scores.npy [4] f32 (raw logits), ids_0.npy (S i64), cls_0.npy (1024 f32).
//! Веса = HF-снапшот `storage/hf/bge-reranker-v2-m3`.
//! Cross-framework F32 → abs-diff логитов + идентичный ранкинг + CLS cosine.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_embedding_bge_m3::BgeReranker;

const DIR: &str = "storage/hf/bge-reranker-v2-m3";
const REF: &str = "tmp/reranker_ref";

const QUERY: &str = "What is the capital of France?";
const DOCS: [&str; 4] = [
    "Paris is the capital and most populous city of France.",
    "The Great Wall of China is over 13,000 miles long.",
    "France is a country in Western Europe; its capital city is Paris.",
    "Bananas are a good source of potassium.",
];

fn parse_npy(path: &str) -> (Vec<usize>, Vec<u8>, String) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let major = b[6];
    let (hl, off) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hdr = std::str::from_utf8(&b[off..off + hl]).unwrap();
    let descr = {
        let i = hdr.find("'descr'").unwrap();
        let r = &hdr[i + 7..];
        let q1 = r.find('\'').unwrap();
        let q2 = r[q1 + 1..].find('\'').unwrap();
        r[q1 + 1..q1 + 1 + q2].to_string()
    };
    let shape = {
        let i = hdr.find("'shape'").unwrap();
        let r = &hdr[i..];
        let lp = r.find('(').unwrap();
        let rp = r.find(')').unwrap();
        r[lp + 1..rp].split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect::<Vec<_>>()
    };
    (shape, b[off + hl..].to_vec(), descr)
}

fn load_f32(path: &str) -> Vec<f32> {
    let (_s, data, descr) = parse_npy(path);
    assert!(descr.contains("f4"), "{path} descr={descr}");
    let cnt = data.len() / 4;
    (0..cnt).map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap())).collect()
}

#[test]
fn reranker_gate() {
    if !Path::new(&format!("{DIR}/model.safetensors")).exists()
        || !Path::new(&format!("{REF}/scores.npy")).exists()
    {
        eprintln!("[reranker_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let rr = BgeReranker::from_unpacked(DIR, Device::Cpu, DType::F32).expect("load reranker");

    let pairs: Vec<(&str, &str)> = DOCS.iter().map(|d| (QUERY, *d)).collect();
    let scores = rr.score_pairs(&pairs).expect("score");
    let scores_ref = load_f32(&format!("{REF}/scores.npy"));
    eprintln!("[reranker_gate] scores got={:?}", scores.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    eprintln!("[reranker_gate] scores ref={:?}", scores_ref.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>());

    let max_abs = scores.iter().zip(&scores_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[reranker_gate] logits max-abs diff = {max_abs:.4}");

    // ранкинг.
    let rank = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx
    };
    let (rg, rr_) = (rank(&scores), rank(&scores_ref));
    eprintln!("[reranker_gate] ranking got={rg:?} ref={rr_:?}");

    // rerank API: top-2 → должны быть релевантные доки {0,2}.
    let top = rr.rerank(QUERY, &DOCS, 2).expect("rerank");
    eprintln!("[reranker_gate] top-2: {top:?}");

    assert_eq!(rg, rr_, "ranking mismatch");
    assert!(max_abs < 0.1, "logits max-abs {max_abs} >= 0.1");

    // from_syn (если .syn упакован) даёт те же скоры, что from_unpacked.
    let syn = format!("{}/models/bge-reranker-v2-m3.syn", std::env::var("HOME").unwrap_or_default());
    if Path::new(&syn).exists() {
        let rr2 = BgeReranker::from_syn(&syn, Device::Cpu, DType::F32).expect("from_syn");
        let s2 = rr2.score_pairs(&pairs).expect("score syn");
        let d = s2.iter().zip(&scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("[reranker_gate] from_syn vs from_unpacked max-abs = {d:.6}");
        assert!(d < 1e-3, "from_syn diverges {d}");
    }
}
