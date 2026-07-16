use crate::error::Result;

pub trait Env: Send {
    type Obs;
    type Action;
    fn reset(&mut self) -> Result<Self::Obs>;
    fn step(&mut self, action: Self::Action) -> Result<(Self::Obs, f32, bool)>;
    fn obs_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}
