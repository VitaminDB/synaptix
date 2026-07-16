//! GPU-смок BGE-reranker: CUDA F16/F32 скоринг пар + ранкинг.
//! cargo run --profile fast-release -p synaptix-embedding-bge-m3 --features cuda --example reranker_gpu

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_embedding_bge_m3::BgeReranker;

const SYN: &str = "models/bge-reranker-v2-m3.syn";
const QUERY: &str = "What is the capital of France?";
const DOCS: [&str; 4] = [
    "Paris is the capital and most populous city of France.",
    "The Great Wall of China is over 13,000 miles long.",
    "France is a country in Western Europe; its capital city is Paris.",
    "Bananas are a good source of potassium.",
];

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pairs: Vec<(&str, &str)> = DOCS.iter().map(|d| (QUERY, *d)).collect();
    for (label, dt) in [("CUDA-F32", DType::F32), ("CUDA-F16", DType::F16)] {
        let rr = BgeReranker::from_syn(SYN, Device::Cuda(0), dt).expect("load");
        let t = std::time::Instant::now();
        let scores = rr.score_pairs(&pairs).expect("score");
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        eprintln!(
            "[rr_gpu] {label} {:?} scores={:?} ranking={idx:?}",
            t.elapsed(),
            scores.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
}
