pub trait Callback {
    fn on_step(&mut self, step: usize, loss: f64);
    fn on_epoch_end(&mut self, epoch: usize);
}

pub struct LogCallback { pub log_every: usize }

impl Callback for LogCallback {
    fn on_step(&mut self, step: usize, loss: f64) {
        if step % self.log_every == 0 {
            eprintln!("step={step} loss={loss:.4}");
        }
    }
    fn on_epoch_end(&mut self, epoch: usize) {
        eprintln!("epoch={epoch} done");
    }
}
