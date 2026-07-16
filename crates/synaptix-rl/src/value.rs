use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub trait ValueFn: Send {
    fn value(&self, obs: &Tensor) -> Result<Tensor>;
}

pub fn td_target(reward: f32, next_val: f32, gamma: f32, done: bool) -> f32 {
    if done { reward } else { reward + gamma * next_val }
}

pub fn gae(rewards: &[f32], values: &[f32], gamma: f32, lambda: f32) -> Vec<f32> {
    let n = rewards.len();
    let mut advantages = vec![0.0_f32; n];
    let mut gae_acc = 0.0;
    for i in (0..n).rev() {
        let next_val = if i + 1 < n { values[i + 1] } else { 0.0 };
        let delta = rewards[i] + gamma * next_val - values[i];
        gae_acc = delta + gamma * lambda * gae_acc;
        advantages[i] = gae_acc;
    }
    advantages
}
