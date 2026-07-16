use crate::session::InferSession;

pub struct InferBatch {
    pub sessions: Vec<InferSession>,
    pub max_seq_len: usize,
}

impl InferBatch {
    pub fn new() -> Self {
        Self { sessions: Vec::new(), max_seq_len: 0 }
    }

    pub fn add(&mut self, session: InferSession) {
        let len = session.request.prompt_tokens.len() + session.generated_tokens.len();
        if len > self.max_seq_len { self.max_seq_len = len; }
        self.sessions.push(session);
    }

    pub fn len(&self) -> usize { self.sessions.len() }
    pub fn is_empty(&self) -> bool { self.sessions.is_empty() }

    pub fn remove_finished(&mut self) -> Vec<InferSession> {
        let mut finished = Vec::new();
        let mut i = 0;
        while i < self.sessions.len() {
            if self.sessions[i].is_finished() {
                finished.push(self.sessions.remove(i));
            } else {
                i += 1;
            }
        }
        finished
    }

    pub fn token_ids(&self) -> Vec<Vec<u32>> {
        self.sessions.iter().map(|s| s.all_tokens()).collect()
    }

    pub fn last_tokens(&self) -> Vec<u32> {
        self.sessions.iter()
            .map(|s| {
                s.generated_tokens.last()
                    .copied()
                    .unwrap_or_else(|| *s.request.prompt_tokens.last().unwrap_or(&0))
            })
            .collect()
    }
}

impl Default for InferBatch {
    fn default() -> Self { Self::new() }
}
