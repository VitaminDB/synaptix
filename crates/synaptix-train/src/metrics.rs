pub struct Metrics { pub step: usize, pub loss: f64, pub lr: f64, pub grad_norm: f64 }

pub fn log_metrics(m: &Metrics) {
    eprintln!("step={} loss={:.4} lr={:.2e}", m.step, m.loss, m.lr);
}
