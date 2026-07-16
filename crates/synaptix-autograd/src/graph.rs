//! Лёгкая обёртка над `synaptix_core::grad` — хранит узлы графа и их рёбра, а реальный
//! backward делегирует в `synaptix_core::grad::backward` (он уже реализует topological sort
//! + chain rule через `GradMeta`/`GradFn`-trait, см. `synaptix-core/src/grad.rs`).
//!
//! `ComputeGraph` тут — это аналитический view графа (для визуализации / отладки),
//! не самостоятельный движок дифференцирования. Тензоры в `nodes` должны иметь
//! grad_meta (т.е. быть результатом операций над requires_grad-листьями).

use std::collections::HashMap;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub struct ComputeGraph {
    nodes: Vec<Tensor>,
    edges: Vec<(usize, usize)>,
    grads: HashMap<usize, Tensor>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), grads: HashMap::new() }
    }

    pub fn add_node(&mut self, t: Tensor) -> usize {
        let id = self.nodes.len();
        self.nodes.push(t);
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    pub fn nodes(&self) -> &[Tensor] { &self.nodes }
    pub fn edges(&self) -> &[(usize, usize)] { &self.edges }

    /// Делегирует в `synaptix_core::grad::backward(self.nodes[loss])`. Реальный topo+chain
    /// rule сделан там через grad_meta (атомарный mark/visit + chain rule по grad_fn).
    /// После вызова `Tensor::grad()` каждого лиственного тензора с `requires_grad=true`
    /// будет содержать накопленный градиент; мы заодно копируем эти градиенты в
    /// `self.grads` под node_id для быстрого доступа.
    pub fn backward(&mut self, loss: usize) -> Result<()> {
        let loss_tensor = self
            .nodes
            .get(loss)
            .ok_or_else(|| synaptix_core::error::SynaptixError::Other(
                format!("ComputeGraph::backward: node id {loss} out of range (have {})", self.nodes.len()),
            ))?
            .clone();
        synaptix_core::grad::backward(&loss_tensor)?;

        self.grads.clear();
        for (id, t) in self.nodes.iter().enumerate() {
            if let Some(g) = t.grad() {
                self.grads.insert(id, g);
            }
        }
        Ok(())
    }

    pub fn grad(&self, node_id: usize) -> Option<&Tensor> { self.grads.get(&node_id) }

    /// Топологический порядок узлов (предки → потомки) на основе `edges`. Полезно для
    /// отладочной визуализации графа независимо от tensor metadata. Использует DFS.
    pub fn topological_order(&self) -> Result<Vec<usize>> {
        let n = self.nodes.len();
        let mut in_deg = vec![0usize; n];
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(from, to) in &self.edges {
            if from >= n || to >= n {
                return Err(synaptix_core::error::SynaptixError::Other(
                    format!("ComputeGraph::topological_order: edge ({from}, {to}) out of range"),
                ));
            }
            succ[from].push(to);
            in_deg[to] += 1;
        }
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for (i, &d) in in_deg.iter().enumerate() {
            if d == 0 { queue.push_back(i); }
        }
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &succ[u] {
                in_deg[v] -= 1;
                if in_deg[v] == 0 { queue.push_back(v); }
            }
        }
        if order.len() != n {
            return Err(synaptix_core::error::SynaptixError::Other(
                format!(
                    "ComputeGraph::topological_order: graph has a cycle ({} of {} nodes visited)",
                    order.len(), n,
                ),
            ));
        }
        Ok(order)
    }
}

impl Default for ComputeGraph {
    fn default() -> Self { Self::new() }
}
