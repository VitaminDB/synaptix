pub struct SloAwareScheduler {
    pub target_latency_ms: u64,
    inner: super::fcfs::FcfsScheduler,
}

impl SloAwareScheduler {
    pub fn new(target_latency_ms: u64, max_batch_size: usize) -> Self {
        Self { target_latency_ms, inner: super::fcfs::FcfsScheduler::new(max_batch_size, usize::MAX) }
    }
}

impl super::Scheduler for SloAwareScheduler {
    fn add_request(&mut self, s: crate::session::InferSession) { self.inner.add_request(s); }
    fn schedule(&mut self) -> crate::error::Result<crate::batch::InferBatch> { self.inner.schedule() }
    fn on_step_complete(&mut self, f: Vec<crate::session::InferSession>) { self.inner.on_step_complete(f); }
    fn pending_count(&self) -> usize { self.inner.pending_count() }
    fn running_count(&self) -> usize { self.inner.running_count() }
}
