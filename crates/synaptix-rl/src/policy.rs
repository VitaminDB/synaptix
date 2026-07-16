use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub trait Policy: Send {
    fn act(&self, obs: &Tensor) -> Result<Tensor>;
    fn act_with_logprob(&self, obs: &Tensor) -> Result<(Tensor, Tensor)>;
}

pub struct RandomPolicy { pub action_dim: usize }

impl Policy for RandomPolicy {
    fn act(&self, obs: &Tensor) -> Result<Tensor> { Ok(obs.clone()) }
    fn act_with_logprob(&self, obs: &Tensor) -> Result<(Tensor, Tensor)> { Ok((obs.clone(), obs.clone())) }
}
