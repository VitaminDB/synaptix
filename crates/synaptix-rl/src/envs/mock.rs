use crate::env::Env;
use crate::error::Result;

pub struct MockEnv { pub obs_dim: usize, pub action_dim: usize, pub step_count: usize }

impl Env for MockEnv {
    type Obs = Vec<f32>;
    type Action = Vec<f32>;
    fn reset(&mut self) -> Result<Vec<f32>> { self.step_count = 0; Ok(vec![0.0; self.obs_dim]) }
    fn step(&mut self, _action: Vec<f32>) -> Result<(Vec<f32>, f32, bool)> {
        self.step_count += 1;
        Ok((vec![0.0; self.obs_dim], 0.0, self.step_count >= 100))
    }
    fn obs_dim(&self) -> usize { self.obs_dim }
    fn action_dim(&self) -> usize { self.action_dim }
}
