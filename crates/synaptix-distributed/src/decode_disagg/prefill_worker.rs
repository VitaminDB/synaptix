pub struct PrefillWorker {
    pub worker_id: usize,
}

impl PrefillWorker {
    pub fn new(worker_id: usize) -> Self { Self { worker_id } }
}
