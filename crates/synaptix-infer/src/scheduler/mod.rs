pub mod chunked_prefill;
pub mod continuous_batch;
pub mod disaggregated;
pub mod fcfs;
pub mod slo_aware;

use crate::session::{InferSession, SessionState};
use crate::batch::InferBatch;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    Fcfs,
    SloAware,
    ChunkedPrefill,
}

pub trait Scheduler: Send {
    fn add_request(&mut self, session: InferSession);
    fn schedule(&mut self) -> Result<InferBatch>;
    fn on_step_complete(&mut self, finished: Vec<InferSession>);
    fn pending_count(&self) -> usize;
    fn running_count(&self) -> usize;
    fn is_idle(&self) -> bool { self.pending_count() == 0 && self.running_count() == 0 }
}

pub use chunked_prefill::ChunkedPrefillScheduler;
pub use continuous_batch::ContinuousBatchScheduler;
pub use fcfs::FcfsScheduler;
