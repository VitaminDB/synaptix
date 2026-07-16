use std::collections::VecDeque;
use crate::session::InferSession;
use crate::batch::InferBatch;
use crate::error::Result;
use super::Scheduler;

pub struct FcfsScheduler {
    queue: VecDeque<InferSession>,
    running: Vec<InferSession>,
    max_batch_size: usize,
    max_tokens_per_batch: usize,
}

impl FcfsScheduler {
    pub fn new(max_batch_size: usize, max_tokens_per_batch: usize) -> Self {
        Self { queue: VecDeque::new(), running: Vec::new(), max_batch_size, max_tokens_per_batch }
    }
}

impl Scheduler for FcfsScheduler {
    fn add_request(&mut self, session: InferSession) { self.queue.push_back(session); }

    fn schedule(&mut self) -> Result<InferBatch> {
        let mut batch = InferBatch::new();
        for s in self.running.drain(..) { batch.add(s); }
        while batch.len() < self.max_batch_size {
            if let Some(s) = self.queue.pop_front() { batch.add(s); }
            else { break; }
        }
        Ok(batch)
    }

    fn on_step_complete(&mut self, _finished: Vec<InferSession>) {}

    fn pending_count(&self) -> usize { self.queue.len() }
    fn running_count(&self) -> usize { self.running.len() }
}
