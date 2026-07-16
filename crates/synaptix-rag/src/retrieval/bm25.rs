use std::collections::HashMap;

pub struct Bm25 {
    pub k1: f32,
    pub b: f32,
    docs: Vec<Vec<String>>,
    df: HashMap<String, usize>,
    avgdl: f32,
}

impl Bm25 {
    pub fn new() -> Self { Self { k1: 1.5, b: 0.75, docs: Vec::new(), df: HashMap::new(), avgdl: 0.0 } }

    pub fn add_doc(&mut self, tokens: Vec<String>) {
        for t in &tokens { *self.df.entry(t.clone()).or_insert(0) += 1; }
        self.avgdl = (self.avgdl * self.docs.len() as f32 + tokens.len() as f32) / (self.docs.len() + 1) as f32;
        self.docs.push(tokens);
    }

    pub fn score(&self, query_tokens: &[String], doc_idx: usize) -> f32 {
        let doc = &self.docs[doc_idx];
        let dl = doc.len() as f32;
        let mut score = 0.0_f32;
        for q in query_tokens {
            let tf = doc.iter().filter(|t| *t == q).count() as f32;
            let df = *self.df.get(q).unwrap_or(&1) as f32;
            let idf = ((self.docs.len() as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl.max(1.0)));
        }
        score
    }

    pub fn search(&self, query_tokens: &[String], top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = (0..self.docs.len()).map(|i| (i, self.score(query_tokens, i))).collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(top_k).collect()
    }
}

impl Default for Bm25 { fn default() -> Self { Self::new() } }
