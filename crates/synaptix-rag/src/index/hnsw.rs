use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::metric::{cosine, tensor_to_vec, top_k_desc};

/// Граф навигации (base-layer NSW, основа HNSW): каждый новый узел соединяется
/// с `m` ближайшими по косинусу, рёбра двунаправленные с обрезкой степени до
/// `m`. Поиск — жадный beam-обход с шириной `ef`. При `ef >= N` обход покрывает
/// весь связный граф (точный результат); иерархия верхних слоёв HNSW здесь не
/// строится.
pub struct HnswIndex {
    pub dim: usize,
    pub m: usize,
    pub ef: usize,
    ids: Vec<String>,
    vecs: Vec<Vec<f32>>,
    neighbors: Vec<Vec<usize>>,
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, m: 16, ef: 64, ids: Vec::new(), vecs: Vec::new(), neighbors: Vec::new() }
    }

    pub fn with_params(dim: usize, m: usize, ef: usize) -> Self {
        Self { dim, m, ef, ids: Vec::new(), vecs: Vec::new(), neighbors: Vec::new() }
    }

    pub fn add(&mut self, id: String, emb: Tensor) {
        let v = match tensor_to_vec(&emb) {
            Ok(v) => v,
            Err(_) => return,
        };
        let new_idx = self.vecs.len();

        // m ближайших уже существующих узлов.
        let mut sims: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine(&v, e)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let m_neighbors: Vec<usize> = sims.iter().take(self.m).map(|(i, _)| *i).collect();

        self.ids.push(id);
        self.vecs.push(v);
        self.neighbors.push(m_neighbors.clone());

        // Двунаправленные рёбра + обрезка степени соседей.
        for nb in m_neighbors {
            self.neighbors[nb].push(new_idx);
            self.prune(nb);
        }
    }

    /// Оставить у узла `node` не более `m` ближайших соседей.
    fn prune(&mut self, node: usize) {
        if self.neighbors[node].len() <= self.m {
            return;
        }
        let v = self.vecs[node].clone();
        let mut nbs = std::mem::take(&mut self.neighbors[node]);
        nbs.sort_by(|&a, &b| {
            cosine(&v, &self.vecs[b])
                .partial_cmp(&cosine(&v, &self.vecs[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nbs.truncate(self.m);
        self.neighbors[node] = nbs;
    }

    pub fn search(&self, query: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        if self.vecs.is_empty() {
            return Ok(Vec::new());
        }
        let q = tensor_to_vec(query)?;
        let ef = self.ef.max(top_k).max(1);
        let n = self.vecs.len();
        let mut visited = vec![false; n];
        let mut frontier: Vec<(usize, f32)> = Vec::new(); // кандидаты к раскрытию
        let mut found: Vec<(usize, f32)> = Vec::new();

        let s0 = cosine(&q, &self.vecs[0]);
        visited[0] = true;
        frontier.push((0, s0));
        found.push((0, s0));

        while !frontier.is_empty() {
            // Лучший нераскрытый кандидат.
            frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let (c, c_sim) = frontier.pop().unwrap();

            // Жадная остановка: набрали ef и текущий хуже худшего из них.
            if found.len() >= ef {
                let worst = found.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
                if c_sim < worst {
                    break;
                }
            }

            for &nb in &self.neighbors[c] {
                if !visited[nb] {
                    visited[nb] = true;
                    let s = cosine(&q, &self.vecs[nb]);
                    frontier.push((nb, s));
                    found.push((nb, s));
                }
            }
        }

        let scored: Vec<(String, f32)> = found.into_iter().map(|(i, s)| (self.ids[i].clone(), s)).collect();
        Ok(top_k_desc(scored, top_k))
    }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
}
