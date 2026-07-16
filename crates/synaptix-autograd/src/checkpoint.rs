use std::sync::Arc;

use synaptix_core::error::Result;
use synaptix_core::grad::{GradFn, GradMeta};
use synaptix_core::tensor::Tensor;

use crate::no_grad::no_grad;

pub type CheckpointFn = dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static;

pub fn checkpoint<F>(input: &Tensor, f: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
{
    let saved_input = input.detach();
    let f_arc: Arc<CheckpointFn> = Arc::new(f);
    let f_clone = f_arc.clone();
    let mut output = no_grad(|| f_clone(&saved_input))?;
    if input.requires_grad() {
        let grad_fn: Arc<dyn GradFn> = Arc::new(CheckpointGradFn {
            parents: [input.clone()],
            saved_input,
            forward_fn: f_arc,
        });
        output.set_grad_meta(Some(GradMeta::intermediate(grad_fn)));
    }
    Ok(output)
}

pub struct CheckpointGradFn {
    parents: [Tensor; 1],
    saved_input: Tensor,
    forward_fn: Arc<CheckpointFn>,
}

impl GradFn for CheckpointGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input_replay = self.saved_input.detach().requires_grad_(true);
        let output_replay = (self.forward_fn)(&input_replay)?;
        output_replay.backward_with(output_grad.clone())?;
        Ok(vec![input_replay.grad()])
    }

    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    fn name(&self) -> &'static str {
        "CheckpointGradFn"
    }
}
