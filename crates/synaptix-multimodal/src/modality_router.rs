use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Video,
}

pub struct ModalityRouter {
    routes: HashMap<Modality, String>,
}

impl ModalityRouter {
    pub fn new() -> Self {
        Self { routes: HashMap::new() }
    }

    pub fn route(&mut self, modality: Modality, encoder: impl Into<String>) -> &mut Self {
        self.routes.insert(modality, encoder.into());
        self
    }

    pub fn lookup(&self, modality: Modality) -> Option<&str> {
        self.routes.get(&modality).map(|s| s.as_str())
    }

    pub fn modalities(&self) -> Vec<Modality> {
        self.routes.keys().copied().collect()
    }
}

impl Default for ModalityRouter {
    fn default() -> Self {
        Self::new()
    }
}
