use std::collections::VecDeque;
use crate::session::{InferSession, SessionState};
use crate::batch::InferBatch;
use crate::error::Result;
use super::Scheduler;

pub struct ContinuousBatchScheduler {
    waiting: VecDeque<InferSession>,
    running: Vec<InferSession>,
    max_running: usize,
    max_total_tokens: usize,
}

impl ContinuousBatchScheduler {
    pub fn new(max_running: usize, max_total_tokens: usize) -> Self {
        Self { waiting: VecDeque::new(), running: Vec::new(), max_running, max_total_tokens }
    }

    fn total_tokens_running(&self) -> usize {
        self.running.iter().map(|s| s.request.prompt_tokens.len() + s.generated_tokens.len()).sum()
    }
}

impl Scheduler for ContinuousBatchScheduler {
    fn add_request(&mut self, session: InferSession) { self.waiting.push_back(session); }

    fn schedule(&mut self) -> Result<InferBatch> {
        while self.total_tokens_running() > self.max_total_tokens && !self.running.is_empty() {
            let evicted = self.running.remove(self.running.len() - 1);
            self.waiting.push_front(evicted);
        }
        while self.running.len() < self.max_running {
            let tokens_if_admit = self.total_tokens_running()
                + self.waiting.front().map(|s| s.request.prompt_tokens.len()).unwrap_or(0);
            if tokens_if_admit > self.max_total_tokens { break; }
            if let Some(s) = self.waiting.pop_front() { self.running.push(s); }
            else { break; }
        }
        let mut batch = InferBatch::new();
        for s in self.running.drain(..) { batch.add(s); }
        Ok(batch)
    }

    fn on_step_complete(&mut self, _finished: Vec<InferSession>) {}
    fn pending_count(&self) -> usize { self.waiting.len() }
    fn running_count(&self) -> usize { self.running.len() }
}
