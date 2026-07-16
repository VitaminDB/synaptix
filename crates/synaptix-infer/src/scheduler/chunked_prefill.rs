pub struct ChunkedPrefillScheduler {
    pub chunk_size: usize,
    inner: super::fcfs::FcfsScheduler,
}

impl ChunkedPrefillScheduler {
    pub fn new(chunk_size: usize, max_batch_size: usize) -> Self {
        Self { chunk_size, inner: super::fcfs::FcfsScheduler::new(max_batch_size, usize::MAX) }
    }
}

impl super::Scheduler for ChunkedPrefillScheduler {
    fn add_request(&mut self, s: crate::session::InferSession) { self.inner.add_request(s); }
    fn schedule(&mut self) -> crate::error::Result<crate::batch::InferBatch> { self.inner.schedule() }
    fn on_step_complete(&mut self, f: Vec<crate::session::InferSession>) { self.inner.on_step_complete(f); }
    fn pending_count(&self) -> usize { self.inner.pending_count() }
    fn running_count(&self) -> usize { self.inner.running_count() }
}
