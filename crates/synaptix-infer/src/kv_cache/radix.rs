use std::collections::HashMap;
use synaptix_core::tensor::Tensor;

pub struct RadixNode {
    pub key: Vec<u32>,
    pub kv_offset: Option<usize>,
    pub children: HashMap<u32, RadixNode>,
    pub ref_count: usize,
}

impl RadixNode {
    fn new(key: Vec<u32>) -> Self {
        Self { key, kv_offset: None, children: HashMap::new(), ref_count: 0 }
    }
}

pub struct RadixKvCache {
    root: RadixNode,
    storage: Vec<Vec<(Tensor, Tensor)>>,
    num_layers: usize,
}

impl RadixKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self { root: RadixNode::new(Vec::new()), storage: Vec::new(), num_layers }
    }

    pub fn insert(&mut self, tokens: &[u32], kv_layers: Vec<(Tensor, Tensor)>) {
        let idx = self.storage.len();
        self.storage.push(kv_layers);
        let mut node = &mut self.root;
        for &tok in tokens {
            node = node.children.entry(tok).or_insert_with(|| RadixNode::new(vec![tok]));
        }
        node.kv_offset = Some(idx);
        node.ref_count += 1;
    }

    pub fn lookup(&self, tokens: &[u32]) -> (usize, Option<&Vec<(Tensor, Tensor)>>) {
        let mut node = &self.root;
        let mut matched = 0usize;
        let mut last_offset: Option<usize> = None;
        for &tok in tokens {
            match node.children.get(&tok) {
                Some(child) => {
                    matched += 1;
                    if child.kv_offset.is_some() {
                        last_offset = child.kv_offset;
                    }
                    node = child;
                }
                None => break,
            }
        }
        let kv = last_offset.and_then(|i| self.storage.get(i));
        (matched, kv)
    }

    pub fn evict_lru(&mut self) {
        Self::evict_leaf(&mut self.root);
    }

    fn evict_leaf(node: &mut RadixNode) -> bool {
        if node.children.is_empty() {
            return node.ref_count == 0;
        }
        let keys: Vec<u32> = node.children.keys().copied().collect();
        for k in keys {
            let should_remove = {
                let child = node.children.get_mut(&k).unwrap();
                Self::evict_leaf(child)
            };
            if should_remove {
                node.children.remove(&k);
                return false;
            }
        }
        false
    }
}
