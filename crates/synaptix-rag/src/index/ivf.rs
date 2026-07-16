use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::metric::{cosine, l2_sq, tensor_to_vec, top_k_desc};

/// IVF (inverted-file) индекс: вектора кластеризуются k-means по `n_lists`
/// центроидам; поиск пробит `nprobe` ближайших списков. При `nprobe >= n_lists`
/// (или до `build`) поиск точный (brute-force).
pub struct IvfIndex {
    pub dim: usize,
    pub n_lists: usize,
    pub nprobe: usize,
    ids: Vec<String>,
    vecs: Vec<Vec<f32>>,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
    trained: bool,
}

impl IvfIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, n_lists: 100, nprobe: 1, ids: Vec::new(), vecs: Vec::new(), centroids: Vec::new(), lists: Vec::new(), trained: false }
    }

    pub fn with_lists(dim: usize, n_lists: usize, nprobe: usize) -> Self {
        Self { dim, n_lists, nprobe, ids: Vec::new(), vecs: Vec::new(), centroids: Vec::new(), lists: Vec::new(), trained: false }
    }

    pub fn add(&mut self, id: String, emb: Tensor) {
        if let Ok(v) = tensor_to_vec(&emb) {
            self.ids.push(id);
            self.vecs.push(v);
            self.trained = false; // требуется перестройка
        }
    }

    /// Построить центроиды (k-means, `iters` итераций) и разложить вектора по спискам.
    pub fn build(&mut self, iters: usize) {
        if self.vecs.is_empty() {
            return;
        }
        let k = self.n_lists.min(self.vecs.len()).max(1);
        let (centroids, assign) = kmeans(&self.vecs, k, iters.max(1));
        let mut lists = vec![Vec::new(); centroids.len()];
        for (i, &c) in assign.iter().enumerate() {
            lists[c].push(i);
        }
        self.centroids = centroids;
        self.lists = lists;
        self.trained = true;
    }

    pub fn search(&self, query: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        let q = tensor_to_vec(query)?;
        let candidates: Vec<usize> = if self.trained && !self.centroids.is_empty() {
            // nprobe ближайших центроидов (по L2).
            let mut cs: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(c, ce)| (c, l2_sq(&q, ce)))
                .collect();
            cs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let probe = self.nprobe.clamp(1, self.centroids.len());
            cs.iter().take(probe).flat_map(|(c, _)| self.lists[*c].iter().copied()).collect()
        } else {
            (0..self.vecs.len()).collect()
        };
        let scored: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|i| (self.ids[i].clone(), cosine(&q, &self.vecs[i])))
            .collect();
        Ok(top_k_desc(scored, top_k))
    }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
}

/// Детерминированный k-means (Lloyd): init = равномерно выбранные точки.
fn kmeans(vecs: &[Vec<f32>], k: usize, iters: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let n = vecs.len();
    let dim = vecs[0].len();
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|c| vecs[c * n / k].clone()).collect();
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, v) in vecs.iter().enumerate() {
            assign[i] = (0..k)
                .min_by(|&a, &b| l2_sq(v, &centroids[a]).partial_cmp(&l2_sq(v, &centroids[b])).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
        }
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vecs.iter().enumerate() {
            let c = assign[i];
            for d in 0..dim {
                sums[c][d] += v[d];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }
    (centroids, assign)
}
