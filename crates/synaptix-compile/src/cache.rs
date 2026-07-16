use std::collections::HashMap;

use crate::ir::IrGraph;

pub struct CompileCache {
    entries: HashMap<u64, IrGraph>,
}

impl CompileCache {
    pub fn new() -> Self { Self { entries: HashMap::new() } }
    pub fn get(&self, key: u64) -> Option<&IrGraph> { self.entries.get(&key) }
    pub fn insert(&mut self, key: u64, graph: IrGraph) { self.entries.insert(key, graph); }
    pub fn clear(&mut self) { self.entries.clear(); }
    pub fn len(&self) -> usize { self.entries.len() }
}

impl Default for CompileCache { fn default() -> Self { Self::new() } }
