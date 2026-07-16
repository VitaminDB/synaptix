pub struct DecodeWorker {
    pub worker_id: usize,
}

impl DecodeWorker {
    pub fn new(worker_id: usize) -> Self { Self { worker_id } }
}
