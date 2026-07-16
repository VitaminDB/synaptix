use synaptix_core::tensor::Tensor;
use super::radix_tree::RadixTree;

pub struct PrefixCacheStore {
    tree: RadixTree,
    storage: Vec<Vec<(Tensor, Tensor)>>,
    max_entries: usize,
}

impl PrefixCacheStore {
    pub fn new(max_entries: usize) -> Self {
        Self { tree: RadixTree::new(), storage: Vec::new(), max_entries }
    }

    pub fn insert(&mut self, tokens: &[u32], kv_layers: Vec<(Tensor, Tensor)>) {
        if self.storage.len() >= self.max_entries { return; }
        let idx = self.storage.len();
        self.storage.push(kv_layers);
        self.tree.insert(tokens, idx);
    }

    pub fn lookup(&self, tokens: &[u32]) -> (usize, Option<&Vec<(Tensor, Tensor)>>) {
        let (matched, idx) = self.tree.lookup(tokens);
        (matched, idx.and_then(|i| self.storage.get(i)))
    }

    pub fn len(&self) -> usize { self.storage.len() }
    pub fn is_empty(&self) -> bool { self.storage.is_empty() }
}
