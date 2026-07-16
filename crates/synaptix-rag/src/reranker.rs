use crate::error::Result;

pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, candidates: &[String]) -> Result<Vec<f32>>;
}

pub struct MockReranker;

impl Reranker for MockReranker {
    fn rerank(&self, _query: &str, candidates: &[String]) -> Result<Vec<f32>> {
        Ok(vec![0.0; candidates.len()])
    }
}
