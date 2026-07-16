use std::collections::HashMap;

pub struct TrieNode {
    pub children: HashMap<u32, TrieNode>,
    pub value: Option<usize>,
    pub ref_count: usize,
}

impl TrieNode {
    pub fn new() -> Self { Self { children: HashMap::new(), value: None, ref_count: 0 } }
}

impl Default for TrieNode {
    fn default() -> Self { Self::new() }
}

/// Префиксное (radix) дерево с подсчётом ссылок для prefix-cache.
///
/// `ref_count` инкрементируется на **каждом** узле вдоль пути при `insert` —
/// узел, разделяемый `N` последовательностями, имеет `ref_count == N`.
/// `release` симметрично декрементирует путь и удаляет ставшие лишними узлы
/// (`ref_count == 0` И без детей), поддерживая `total_nodes` (число узлов, не
/// считая корень).
pub struct RadixTree {
    pub root: TrieNode,
    pub total_nodes: usize,
}

impl RadixTree {
    pub fn new() -> Self { Self { root: TrieNode::new(), total_nodes: 0 } }

    pub fn insert(&mut self, tokens: &[u32], value: usize) {
        let mut node = &mut self.root;
        for &tok in tokens {
            let created = !node.children.contains_key(&tok);
            node = node.children.entry(tok).or_insert_with(TrieNode::new);
            if created {
                self.total_nodes += 1;
            }
            node.ref_count += 1;
        }
        if !tokens.is_empty() {
            node.value = Some(value);
        }
    }

    pub fn lookup(&self, tokens: &[u32]) -> (usize, Option<usize>) {
        let mut node = &self.root;
        let mut matched = 0;
        for &tok in tokens {
            match node.children.get(&tok) {
                Some(child) => { node = child; matched += 1; }
                None => break,
            }
        }
        (matched, node.value)
    }

    /// Декремент `ref_count` вдоль пути `tokens` + эвикция узлов, у которых после
    /// декремента `ref_count == 0` и нет детей. Возвращает число удалённых узлов.
    /// Если путь отсутствует целиком — ничего не делает (возвращает 0).
    pub fn release(&mut self, tokens: &[u32]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        // Проверяем, что путь существует полностью.
        {
            let mut node = &self.root;
            for &tok in tokens {
                match node.children.get(&tok) {
                    Some(c) => node = c,
                    None => return 0,
                }
            }
        }
        let removed = Self::release_rec(&mut self.root, tokens);
        self.total_nodes -= removed;
        removed
    }

    /// Рекурсивно спускается по `path`, декрементирует `ref_count` посещаемого
    /// ребёнка и удаляет его, если он стал лишним (ref==0 и нет детей).
    /// Возвращает число удалённых поддеревом узлов.
    fn release_rec(node: &mut TrieNode, path: &[u32]) -> usize {
        let tok = path[0];
        let mut removed = 0;
        if let Some(child) = node.children.get_mut(&tok) {
            if child.ref_count > 0 {
                child.ref_count -= 1;
            }
            if path.len() > 1 {
                removed += Self::release_rec(child, &path[1..]);
            } else {
                // Конец пути: тот, кто держал value, отпускает его при ref==0.
                if child.ref_count == 0 {
                    child.value = None;
                }
            }
            // Эвикция лишнего узла: нет ссылок и нет детей.
            if child.ref_count == 0 && child.children.is_empty() {
                node.children.remove(&tok);
                removed += 1;
            }
        }
        removed
    }
}

impl Default for RadixTree {
    fn default() -> Self { Self::new() }
}
