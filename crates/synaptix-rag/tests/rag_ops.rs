//! synaptix-rag — аналитические тесты chunking / index / retrieval.

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_rag::chunking::markdown_aware::markdown_aware_chunk;
use synaptix_rag::chunking::recursive::recursive_chunk;
use synaptix_rag::chunking::semantic::semantic_chunk;
use synaptix_rag::embedder::Embedder;
use synaptix_rag::error::Result;
use synaptix_rag::index::flat::FlatIndex;
use synaptix_rag::index::hnsw::HnswIndex;
use synaptix_rag::index::ivf::IvfIndex;
use synaptix_rag::retrieval::colbert::ColBert;
use synaptix_rag::retrieval::dense::DenseRetriever;
use synaptix_rag::retrieval::hybrid::HybridRetriever;
use synaptix_rag::retrieval::rag_fusion::RagFusion;
use synaptix_kernels_cpu::ensure_registered;

fn setup() { ensure_registered(); }

fn vec_t(v: &[f32]) -> Tensor {
    Tensor::from_vec::<_, f32>(v.to_vec(), vec![v.len()], Device::Cpu).unwrap()
}

fn mat_t(rows: &[&[f32]]) -> Tensor {
    let dim = rows[0].len();
    let mut data = Vec::new();
    for r in rows { data.extend_from_slice(r); }
    Tensor::from_vec::<_, f32>(data, vec![rows.len(), dim], Device::Cpu).unwrap()
}

// --- chunking -------------------------------------------------------------

#[test]
fn t50_1_recursive_respects_size_and_overlap() {
    let text = "alpha beta gamma. delta epsilon zeta. eta theta iota.";
    let chunks = recursive_chunk(text, 20, 5);
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.chars().count() <= 20, "chunk too long: {:?} ({})", c, c.chars().count());
    }
    // Покрытие: каждое слово исходника встречается хотя бы в одном чанке.
    for word in ["alpha", "zeta", "iota"] {
        assert!(chunks.iter().any(|c| c.contains(word)), "missing {word}");
    }
}

#[test]
fn t50_2_recursive_empty() {
    assert!(recursive_chunk("", 10, 2).is_empty());
    assert!(recursive_chunk("abc", 0, 0).is_empty());
}

#[test]
fn t50_3_markdown_sections() {
    let md = "# Title\nintro line\n\n## Section A\nbody a\n\n## Section B\nbody b";
    let chunks = markdown_aware_chunk(md, 1000, 0);
    // Большой chunk_size → секции не дробятся: 3 секции (Title+intro, A, B).
    assert_eq!(chunks.len(), 3);
    assert!(chunks[0].contains("# Title"));
    assert!(chunks[1].contains("## Section A"));
    assert!(chunks[2].contains("## Section B"));
}

#[test]
fn t50_4_markdown_keeps_code_fence_intact() {
    let md = "## Code\n```\n# not a heading\nline2\n```\ntail";
    let chunks = markdown_aware_chunk(md, 1000, 0);
    // `# not a heading` внутри fence не создаёт новой секции.
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].contains("# not a heading"));
}

struct KwEmbedder;
impl Embedder for KwEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Tensor> {
        let mut data = Vec::new();
        for t in texts {
            let (a, b) = if t.contains("cat") || t.contains("feline") {
                (1.0, 0.0)
            } else if t.contains("dog") || t.contains("canine") {
                (0.0, 1.0)
            } else {
                (0.5, 0.5)
            };
            data.push(a);
            data.push(b);
        }
        Ok(Tensor::from_vec::<_, f32>(data, vec![texts.len(), 2], Device::Cpu).unwrap())
    }
    fn dim(&self) -> usize { 2 }
}

#[test]
fn t50_5_semantic_splits_on_topic_change() {
    setup();
    let text = "I love cats. Cats are feline. Dogs bark loudly. Canine friends are loyal.";
    let chunks = semantic_chunk(text, &KwEmbedder, 0.9).unwrap();
    assert_eq!(chunks.len(), 2, "got {chunks:?}");
    assert!(chunks[0].contains("cats") && chunks[0].contains("feline"));
    assert!(chunks[1].contains("Dogs") && chunks[1].contains("Canine"));
}

// --- index ----------------------------------------------------------------

#[test]
fn t50_6_flat_cosine_ranking() {
    setup();
    let mut idx = FlatIndex::new(2);
    idx.add("x".into(), vec_t(&[1.0, 0.0]));
    idx.add("y".into(), vec_t(&[0.0, 1.0]));
    idx.add("diag".into(), vec_t(&[1.0, 1.0]));
    let res = idx.search(&vec_t(&[1.0, 0.1]), 3).unwrap();
    assert_eq!(res[0].0, "x", "query close to x");
    // Косинусы: x≈0.995, diag≈0.778, y≈0.0995 → строгий порядок.
    assert_eq!(res.iter().map(|r| r.0.clone()).collect::<Vec<_>>(), vec!["x", "diag", "y"]);
}

#[test]
fn t50_7_ivf_full_probe_matches_flat() {
    setup();
    let pts: [(&str, [f32; 2]); 6] = [
        ("a", [1.0, 0.0]), ("b", [0.9, 0.1]), ("c", [0.0, 1.0]),
        ("d", [0.1, 0.9]), ("e", [1.0, 1.0]), ("f", [-1.0, 0.2]),
    ];
    let mut flat = FlatIndex::new(2);
    let mut ivf = IvfIndex::with_lists(2, 2, 2); // nprobe = n_lists → точный
    for (id, p) in &pts {
        flat.add((*id).into(), vec_t(p));
        ivf.add((*id).into(), vec_t(p));
    }
    ivf.build(20);
    let q = vec_t(&[0.8, 0.2]);
    let fr: Vec<String> = flat.search(&q, 4).unwrap().into_iter().map(|x| x.0).collect();
    let ir: Vec<String> = ivf.search(&q, 4).unwrap().into_iter().map(|x| x.0).collect();
    assert_eq!(fr, ir, "full-probe IVF must equal brute force");
}

#[test]
fn t50_8_hnsw_large_ef_matches_flat() {
    setup();
    let pts: [(&str, [f32; 2]); 6] = [
        ("a", [1.0, 0.0]), ("b", [0.9, 0.1]), ("c", [0.0, 1.0]),
        ("d", [0.1, 0.9]), ("e", [1.0, 1.0]), ("f", [-1.0, 0.2]),
    ];
    let mut flat = FlatIndex::new(2);
    let mut hnsw = HnswIndex::with_params(2, 4, 100); // ef >> N → точный
    for (id, p) in &pts {
        flat.add((*id).into(), vec_t(p));
        hnsw.add((*id).into(), vec_t(p));
    }
    let q = vec_t(&[0.8, 0.2]);
    let fr: Vec<String> = flat.search(&q, 3).unwrap().into_iter().map(|x| x.0).collect();
    let hr: Vec<String> = hnsw.search(&q, 3).unwrap().into_iter().map(|x| x.0).collect();
    assert_eq!(fr, hr, "ef≥N HNSW must equal brute force top-3");
}

// --- retrieval -------------------------------------------------------------

#[test]
fn t50_9_dense_retriever_cosine() {
    setup();
    let mut r = DenseRetriever::new(2);
    r.add("apple".into(), vec_t(&[1.0, 0.0]));
    r.add("banana".into(), vec_t(&[0.0, 1.0]));
    let res = r.search(&vec_t(&[0.9, 0.1]), 2).unwrap();
    assert_eq!(res[0].0, "apple");
    assert!(res[0].1 > res[1].1);
}

#[test]
fn t50_10_colbert_maxsim() {
    setup();
    let mut cb = ColBert::new(2);
    // doc1 покрывает оба токена запроса, doc2 — только один.
    cb.add_doc("doc1".into(), &mat_t(&[&[1.0, 0.0], &[0.0, 1.0]])).unwrap();
    cb.add_doc("doc2".into(), &mat_t(&[&[1.0, 0.0], &[1.0, 0.0]])).unwrap();
    let query = mat_t(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let res = cb.search(&query, 2).unwrap();
    assert_eq!(res[0].0, "doc1", "doc1 матчит оба токена → выше");
    assert!(res[0].1 > res[1].1);
}

#[test]
fn t50_11_hybrid_fuse_weighting() {
    let h = HybridRetriever::new(1.0, 0.0); // только dense
    let dense = vec![("a".to_string(), 10.0), ("b".to_string(), 0.0)];
    let sparse = vec![("b".to_string(), 10.0), ("a".to_string(), 0.0)];
    let res = h.fuse(&dense, &sparse, 2).unwrap();
    assert_eq!(res[0].0, "a", "dense_weight=1 → побеждает топ dense");

    let balanced = HybridRetriever::new(0.5, 0.5);
    let res = balanced.fuse(&dense, &sparse, 2).unwrap();
    // a: 0.5*1 + 0.5*0 = 0.5; b: 0.5*0 + 0.5*1 = 0.5 → ничья, оба присутствуют.
    assert_eq!(res.len(), 2);
    assert!((res[0].1 - 0.5).abs() < 1e-6 && (res[1].1 - 0.5).abs() < 1e-6);
}

#[test]
fn t50_12_rag_fusion_rrf() {
    let rf = RagFusion::new(2);
    let l1 = vec![("a".to_string(), 0.0), ("b".to_string(), 0.0), ("c".to_string(), 0.0)];
    let l2 = vec![("b".to_string(), 0.0), ("a".to_string(), 0.0), ("c".to_string(), 0.0)];
    let res = rf.reciprocal_rank_fusion(&[l1, l2], 3).unwrap();
    // a: 1/61 + 1/62; b: 1/62 + 1/61 → равны и выше c (1/63+1/63).
    assert_eq!(res.len(), 3);
    assert_eq!(res[2].0, "c", "c всегда на 3-м месте в обоих списках → ниже");
    assert!(res[0].1 > res[2].1);
}
