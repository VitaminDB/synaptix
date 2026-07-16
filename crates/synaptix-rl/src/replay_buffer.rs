use synaptix_core::tensor::Tensor;
use std::collections::VecDeque;

pub struct Transition { pub obs: Tensor, pub action: Tensor, pub reward: f32, pub next_obs: Tensor, pub done: bool }

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self { Self { buffer: VecDeque::new(), capacity } }
    pub fn push(&mut self, t: Transition) {
        if self.buffer.len() >= self.capacity { self.buffer.pop_front(); }
        self.buffer.push_back(t);
    }
    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn is_empty(&self) -> bool { self.buffer.is_empty() }
    pub fn sample(&self, n: usize) -> Vec<&Transition> { self.buffer.iter().take(n).collect() }
}
