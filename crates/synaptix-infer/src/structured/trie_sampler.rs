use std::collections::HashMap;

pub struct TrieNode {
    pub children: HashMap<u8, TrieNode>,
    pub terminal: bool,
    pub token_id: Option<u32>,
}

impl TrieNode {
    pub fn new() -> Self { Self { children: HashMap::new(), terminal: false, token_id: None } }
}

pub struct TrieSampler {
    root: TrieNode,
    vocab_size: usize,
}

impl TrieSampler {
    pub fn new(vocab_size: usize) -> Self {
        Self { root: TrieNode::new(), vocab_size }
    }

    pub fn insert_token(&mut self, bytes: &[u8], token_id: u32) {
        let mut node = &mut self.root;
        for &b in bytes {
            node = node.children.entry(b).or_insert_with(TrieNode::new);
        }
        node.terminal = true;
        node.token_id = Some(token_id);
    }

    pub fn allowed_next(&self, prefix_bytes: &[u8]) -> Vec<u32> {
        let mut node = &self.root;
        for &b in prefix_bytes {
            match node.children.get(&b) {
                Some(n) => node = n,
                None => return Vec::new(),
            }
        }
        collect_terminals(node)
    }
}

fn collect_terminals(node: &TrieNode) -> Vec<u32> {
    let mut out = Vec::new();
    if let Some(id) = node.token_id { out.push(id); }
    for child in node.children.values() {
        out.extend(collect_terminals(child));
    }
    out
}
