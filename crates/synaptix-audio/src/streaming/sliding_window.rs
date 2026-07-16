pub struct SlidingWindow {
    buffer: Vec<f32>,
    capacity: usize,
}

impl SlidingWindow {
    pub fn new(capacity_samples: usize) -> Self {
        Self { buffer: Vec::with_capacity(capacity_samples), capacity: capacity_samples }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
        if self.buffer.len() > self.capacity {
            let excess = self.buffer.len() - self.capacity;
            self.buffer.drain(..excess);
        }
    }

    pub fn snapshot(&self) -> Vec<f32> {
        self.buffer.clone()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_keeps_last_n() {
        let mut w = SlidingWindow::new(4);
        w.push(&[1.0, 2.0, 3.0]);
        assert_eq!(w.snapshot(), vec![1.0, 2.0, 3.0]);
        w.push(&[4.0, 5.0, 6.0]);
        assert_eq!(w.snapshot(), vec![3.0, 4.0, 5.0, 6.0]);
    }
}
